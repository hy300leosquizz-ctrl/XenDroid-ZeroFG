/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */
// Adapted for ZeroFG from xenia/ui/vulkan/vulkan_presenter.cc: the output
// pipelines and descriptor layouts below follow Xenia's own, built on ZeroFG's
// device B. This file stays under Xenia's BSD 3-Clause license.

// Host-only device-B execution subset. Native paint/WSI remains on A.
VulkanPresenter::ZeroFGPresenterVulkanContext::ZeroFGPresenterVulkanContext(
    VulkanDevice* source)
    : device(source) {}

VulkanDevice* VulkanPresenter::ZeroFGDevice() const {
  return zerofg_vulkan_context_->device;
}

VulkanPresenter::ZeroFGPresenterVulkanContext::~ZeroFGPresenterVulkanContext() {
  // Generation/Post were drained by the independent presenter callbacks.
  // B is not recreated by surface disconnects.
  if (!owned_device) return;
  const auto& f = device->functions();
  const VkDevice d = device->device();
  device->vkDeviceWaitIdle()(d);
  handoff.reset();
  for (auto pipeline : pipelines)
    if (pipeline) f.vkDestroyPipeline(d, pipeline, nullptr);
  if (render_pass) f.vkDestroyRenderPass(d, render_pass, nullptr);
  for (auto shader : fragments)
    if (shader) f.vkDestroyShaderModule(d, shader, nullptr);
  if (vertex) f.vkDestroyShaderModule(d, vertex, nullptr);
  for (auto layout : layouts)
    if (layout) f.vkDestroyPipelineLayout(d, layout, nullptr);
  if (image_layout) f.vkDestroyDescriptorSetLayout(d, image_layout, nullptr);
  samplers.reset();
}

bool VulkanPresenter::InitializeZeroFGDeviceContext() {
  if (!IsZeroFGRequested()) return true;
  auto& context = *zerofg_vulkan_context_;
  // Main Surface Authority needs B with WSI. If that profile cannot be
  // created, ZeroFG stays off and A presents.
  ZeroFGBackendFeatureRequests backend_requests;
  context.owned_device = VulkanDevice::CreateIfSupported(
      vulkan_device_->vulkan_instance(), vulkan_device_->physical_device(),
      false, true,
      VulkanDevice::CreationProfile::kZeroFGMainSurfacePresenter, backend_requests);
  if (!context.owned_device) {
    XELOGE("ZeroFGDeviceSplit activation_failed=device_B split_armed=false");
    XELOGE("ZeroFGMainSurface refused reason=no_wsi_profile fallback=native_A");
    return false;
  }
  context.device = context.owned_device.get();
  context.samplers = UISamplers::Create(context.device);
  if (!context.samplers) return false;
  const auto& dfn = context.device->functions();
  const VkDevice device = context.device->device();
  VkDescriptorSetLayoutBinding guest_output_image_sampler_bindings[2];
  guest_output_image_sampler_bindings[0].binding = 0;
  guest_output_image_sampler_bindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  guest_output_image_sampler_bindings[0].descriptorCount = 1;
  guest_output_image_sampler_bindings[0].stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_image_sampler_bindings[0].pImmutableSamplers = nullptr;
  const VkSampler sampler_linear_clamp =
      context.samplers->samplers()[UISamplers::kSamplerIndexLinearClampToEdge];
  guest_output_image_sampler_bindings[1].binding = 1;
  guest_output_image_sampler_bindings[1].descriptorType =
      VK_DESCRIPTOR_TYPE_SAMPLER;
  guest_output_image_sampler_bindings[1].descriptorCount = 1;
  guest_output_image_sampler_bindings[1].stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_image_sampler_bindings[1].pImmutableSamplers =
      &sampler_linear_clamp;
  VkDescriptorSetLayoutCreateInfo image_layout_create_info;
  image_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  image_layout_create_info.pNext = nullptr;
  image_layout_create_info.flags = 0;
  image_layout_create_info.bindingCount =
      uint32_t(xe::countof(guest_output_image_sampler_bindings));
  image_layout_create_info.pBindings = guest_output_image_sampler_bindings;
  if (dfn.vkCreateDescriptorSetLayout(device, &image_layout_create_info,
                                      nullptr,
                                      &context.image_layout) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create the guest output image descriptor "
        "set layout");
    return false;
  }

  VkPushConstantRange guest_output_paint_push_constant_ranges[2];
  VkPushConstantRange& guest_output_paint_push_constant_range_rect =
      guest_output_paint_push_constant_ranges[0];
  guest_output_paint_push_constant_range_rect.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT;
  guest_output_paint_push_constant_range_rect.offset = 0;
  guest_output_paint_push_constant_range_rect.size =
      sizeof(GuestOutputPaintRectangleConstants);
  VkPushConstantRange& guest_output_paint_push_constant_range_ffx =
      guest_output_paint_push_constant_ranges[1];
  guest_output_paint_push_constant_range_ffx.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_paint_push_constant_range_ffx.offset =
      guest_output_paint_push_constant_ranges[0].offset +
      guest_output_paint_push_constant_ranges[0].size;
  VkPipelineLayoutCreateInfo guest_output_paint_pipeline_layout_create_info;
  guest_output_paint_pipeline_layout_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  guest_output_paint_pipeline_layout_create_info.pNext = nullptr;
  guest_output_paint_pipeline_layout_create_info.flags = 0;
  guest_output_paint_pipeline_layout_create_info.setLayoutCount = 1;
  guest_output_paint_pipeline_layout_create_info.pSetLayouts =
      &context.image_layout;
  guest_output_paint_pipeline_layout_create_info.pPushConstantRanges =
      guest_output_paint_push_constant_ranges;
  for (size_t i = 0; i < size_t(kGuestOutputPaintPipelineLayoutCount); ++i) {
    switch (GuestOutputPaintPipelineLayoutIndex(i)) {
      case kGuestOutputPaintPipelineLayoutIndexBilinear:
        guest_output_paint_push_constant_range_ffx.size =
            sizeof(BilinearConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexCasSharpen:
        guest_output_paint_push_constant_range_ffx.size =
            sizeof(CasSharpenConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexCasResample:
        guest_output_paint_push_constant_range_ffx.size =
            sizeof(CasResampleConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexFsrEasu:
        guest_output_paint_push_constant_range_ffx.size =
            sizeof(FsrEasuConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexFsrRcas:
        guest_output_paint_push_constant_range_ffx.size =
            sizeof(FsrRcasConstants);
        break;
      case kGuestOutputPaintPipelineLayoutIndexSgsr:
        guest_output_paint_push_constant_range_ffx.size = sizeof(SgsrConstants);
        break;
      default:
        assert_unhandled_case(GuestOutputPaintPipelineLayoutIndex(i));
        continue;
    }
    guest_output_paint_pipeline_layout_create_info.pushConstantRangeCount =
        1 + uint32_t(guest_output_paint_push_constant_range_ffx.size != 0);
    if (dfn.vkCreatePipelineLayout(
            device, &guest_output_paint_pipeline_layout_create_info, nullptr,
            &context.layouts[i]) != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to create a guest output presentation "
          "pipeline layout with {} bytes of push constants",
          guest_output_paint_push_constant_range_rect.size +
              guest_output_paint_push_constant_range_ffx.size);
      return false;
    }
  }

  VkShaderModuleCreateInfo shader_module_create_info;
  shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_create_info.pNext = nullptr;
  shader_module_create_info.flags = 0;
  shader_module_create_info.codeSize =
      sizeof(shaders::guest_output_triangle_strip_rect_vs);
  shader_module_create_info.pCode =
      shaders::guest_output_triangle_strip_rect_vs;
  if (dfn.vkCreateShaderModule(device, &shader_module_create_info, nullptr,
                               &context.vertex) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create the guest output presentation "
        "vertex shader module");
    return false;
  }
  for (size_t i = 0; i < size_t(GuestOutputPaintEffect::kCount); ++i) {
    GuestOutputPaintEffect guest_output_paint_effect =
        GuestOutputPaintEffect(i);
    switch (guest_output_paint_effect) {
      case GuestOutputPaintEffect::kBilinear:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_bilinear_ps);
        shader_module_create_info.pCode = shaders::guest_output_bilinear_ps;
        break;
      case GuestOutputPaintEffect::kBilinearDither:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_bilinear_dither_ps);
        shader_module_create_info.pCode =
            shaders::guest_output_bilinear_dither_ps;
        break;
      case GuestOutputPaintEffect::kCasSharpen:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_cas_sharpen_ps);
        shader_module_create_info.pCode =
            shaders::guest_output_ffx_cas_sharpen_ps;
        break;
      case GuestOutputPaintEffect::kCasSharpenDither:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_cas_sharpen_dither_ps);
        shader_module_create_info.pCode =
            shaders::guest_output_ffx_cas_sharpen_dither_ps;
        break;
      case GuestOutputPaintEffect::kCasResample:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_cas_resample_ps);
        shader_module_create_info.pCode =
            shaders::guest_output_ffx_cas_resample_ps;
        break;
      case GuestOutputPaintEffect::kCasResampleDither:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_cas_resample_dither_ps);
        shader_module_create_info.pCode =
            shaders::guest_output_ffx_cas_resample_dither_ps;
        break;
      case GuestOutputPaintEffect::kFsrEasu:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_fsr_easu_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_fsr_easu_ps;
        break;
      case GuestOutputPaintEffect::kFsrRcas:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_fsr_rcas_ps);
        shader_module_create_info.pCode = shaders::guest_output_ffx_fsr_rcas_ps;
        break;
      case GuestOutputPaintEffect::kFsrRcasDither:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_ffx_fsr_rcas_dither_ps);
        shader_module_create_info.pCode =
            shaders::guest_output_ffx_fsr_rcas_dither_ps;
        break;
      case GuestOutputPaintEffect::kSgsr:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_sgsr_ps);
        shader_module_create_info.pCode = shaders::guest_output_sgsr_ps;
        break;
      case GuestOutputPaintEffect::kSgsrEdgeDirection:
        shader_module_create_info.codeSize =
            sizeof(shaders::guest_output_sgsr_edge_direction_ps);
        shader_module_create_info.pCode =
            shaders::guest_output_sgsr_edge_direction_ps;
        break;
      default:
        // Not supported by this implementation.
        continue;
    }
    if (dfn.vkCreateShaderModule(device, &shader_module_create_info, nullptr,
                                 &context.fragments[i]) != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to create the guest output painting shader "
          "module for effect {}",
          i);
      return false;
    }
  }

  VkAttachmentDescription intermediate_render_pass_attachment;
  intermediate_render_pass_attachment.flags = 0;
  intermediate_render_pass_attachment.format = kGuestOutputFormat;
  intermediate_render_pass_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  intermediate_render_pass_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  intermediate_render_pass_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  intermediate_render_pass_attachment.stencilLoadOp =
      VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  intermediate_render_pass_attachment.stencilStoreOp =
      VK_ATTACHMENT_STORE_OP_DONT_CARE;
  intermediate_render_pass_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  intermediate_render_pass_attachment.finalLayout =
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference intermediate_render_pass_color_attachment;
  intermediate_render_pass_color_attachment.attachment = 0;
  intermediate_render_pass_color_attachment.layout =
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkSubpassDescription intermediate_render_pass_subpass = {};
  intermediate_render_pass_subpass.pipelineBindPoint =
      VK_PIPELINE_BIND_POINT_GRAPHICS;
  intermediate_render_pass_subpass.colorAttachmentCount = 1;
  intermediate_render_pass_subpass.pColorAttachments =
      &intermediate_render_pass_color_attachment;
  VkSubpassDependency intermediate_render_pass_dependencies[2];
  intermediate_render_pass_dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  intermediate_render_pass_dependencies[0].dstSubpass = 0;
  intermediate_render_pass_dependencies[0].srcStageMask =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  intermediate_render_pass_dependencies[0].dstStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  intermediate_render_pass_dependencies[0].srcAccessMask =
      VK_ACCESS_SHADER_READ_BIT;
  intermediate_render_pass_dependencies[0].dstAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  intermediate_render_pass_dependencies[0].dependencyFlags = 0;
  intermediate_render_pass_dependencies[1].srcSubpass = 0;
  intermediate_render_pass_dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  intermediate_render_pass_dependencies[1].srcStageMask =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  intermediate_render_pass_dependencies[1].dstStageMask =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  intermediate_render_pass_dependencies[1].srcAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  intermediate_render_pass_dependencies[1].dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT;
  intermediate_render_pass_dependencies[1].dependencyFlags = 0;
  VkRenderPassCreateInfo intermediate_render_pass_create_info;
  intermediate_render_pass_create_info.sType =
      VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  intermediate_render_pass_create_info.pNext = nullptr;
  intermediate_render_pass_create_info.flags = 0;
  intermediate_render_pass_create_info.attachmentCount = 1;
  intermediate_render_pass_create_info.pAttachments =
      &intermediate_render_pass_attachment;
  intermediate_render_pass_create_info.subpassCount = 1;
  intermediate_render_pass_create_info.pSubpasses =
      &intermediate_render_pass_subpass;
  intermediate_render_pass_create_info.dependencyCount =
      uint32_t(xe::countof(intermediate_render_pass_dependencies));
  intermediate_render_pass_create_info.pDependencies =
      intermediate_render_pass_dependencies;
  if (dfn.vkCreateRenderPass(device, &intermediate_render_pass_create_info,
                             nullptr, &context.render_pass) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create the guest output intermediate image "
        "render pass");
    return false;
  }
  for (size_t i = 0; i < context.pipelines.size(); ++i) {
    const auto effect = GuestOutputPaintEffect(i);
    if (!CanGuestOutputPaintEffectBeIntermediate(effect) ||
        context.fragments[i] == VK_NULL_HANDLE)
      continue;
    context.pipelines[i] = CreateGuestOutputPaintPipeline(
        effect, context.render_pass, nullptr, true);
    if (!context.pipelines[i]) return false;
  }
  context.handoff =
      std::make_unique<ZeroFGDeviceHandoff>(vulkan_device_, context.device);
  return context.handoff->SelfTest();
}
