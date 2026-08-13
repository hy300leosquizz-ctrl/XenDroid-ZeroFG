#include "zerofg/zerofg.h"
#include "vulkan_dispatch.h"
#include "vulkan_image.h"
#include "luma_spv.h"
#include "motion_spv.h"
#include "synth_spv.h"

#include <cmath>
#include <utility>

namespace zerofg {

class Interpolator::Impl {
 public:
  explicit Impl(const CreateInfo& create_info)
      : vulkan_(create_info.vulkan) {}

  ~Impl() {
    previous_luma_image_.Destroy(vulkan_, dispatch_);
    current_luma_image_.Destroy(vulkan_, dispatch_);
    synth_image_.Destroy(vulkan_, dispatch_);
    motion_image_.Destroy(vulkan_, dispatch_);

    if (motion_descriptor_pool_ != VK_NULL_HANDLE) {
      dispatch_.destroy_descriptor_pool(
          vulkan_.device,
          motion_descriptor_pool_,
          vulkan_.allocator);
      motion_descriptor_pool_ = VK_NULL_HANDLE;
      motion_descriptor_set_ = VK_NULL_HANDLE;
    }

    if (luma_descriptor_pool_ != VK_NULL_HANDLE) {
      dispatch_.destroy_descriptor_pool(
          vulkan_.device,
          luma_descriptor_pool_,
          vulkan_.allocator);
      luma_descriptor_pool_ = VK_NULL_HANDLE;
      luma_descriptor_set_ = VK_NULL_HANDLE;
      previous_luma_descriptor_set_ = VK_NULL_HANDLE;
    }

    if (luma_sampler_ != VK_NULL_HANDLE) {
      dispatch_.destroy_sampler(
          vulkan_.device, luma_sampler_, vulkan_.allocator);
      luma_sampler_ = VK_NULL_HANDLE;
    }

    if (synth_pipeline_ != VK_NULL_HANDLE) {
      dispatch_.destroy_pipeline(
          vulkan_.device,
          synth_pipeline_,
          vulkan_.allocator);
      synth_pipeline_ = VK_NULL_HANDLE;
    }

    if (synth_pipeline_layout_ != VK_NULL_HANDLE) {
      dispatch_.destroy_pipeline_layout(
          vulkan_.device,
          synth_pipeline_layout_,
          vulkan_.allocator);
      synth_pipeline_layout_ = VK_NULL_HANDLE;
    }

    if (synth_descriptor_pool_ != VK_NULL_HANDLE) {
      dispatch_.destroy_descriptor_pool(
          vulkan_.device,
          synth_descriptor_pool_,
          vulkan_.allocator);
      synth_descriptor_pool_ = VK_NULL_HANDLE;
      synth_descriptor_set_ = VK_NULL_HANDLE;
    }

    if (synth_sampler_ != VK_NULL_HANDLE) {
      dispatch_.destroy_sampler(
          vulkan_.device,
          synth_sampler_,
          vulkan_.allocator);
      synth_sampler_ = VK_NULL_HANDLE;
    }

    if (synth_descriptor_set_layout_ != VK_NULL_HANDLE) {
      dispatch_.destroy_descriptor_set_layout(
          vulkan_.device,
          synth_descriptor_set_layout_,
          vulkan_.allocator);
      synth_descriptor_set_layout_ = VK_NULL_HANDLE;
    }

    if (synth_shader_module_ != VK_NULL_HANDLE) {
      dispatch_.destroy_shader_module(
          vulkan_.device,
          synth_shader_module_,
          vulkan_.allocator);
      synth_shader_module_ = VK_NULL_HANDLE;
    }

    if (motion_pipeline_ != VK_NULL_HANDLE) {
      dispatch_.destroy_pipeline(
          vulkan_.device,
          motion_pipeline_,
          vulkan_.allocator);
      motion_pipeline_ = VK_NULL_HANDLE;
    }

    if (motion_pipeline_layout_ != VK_NULL_HANDLE) {
      dispatch_.destroy_pipeline_layout(
          vulkan_.device,
          motion_pipeline_layout_,
          vulkan_.allocator);
      motion_pipeline_layout_ = VK_NULL_HANDLE;
    }

    if (motion_descriptor_set_layout_ != VK_NULL_HANDLE) {
      dispatch_.destroy_descriptor_set_layout(
          vulkan_.device,
          motion_descriptor_set_layout_,
          vulkan_.allocator);
      motion_descriptor_set_layout_ = VK_NULL_HANDLE;
    }

    if (luma_pipeline_ != VK_NULL_HANDLE) {
      dispatch_.destroy_pipeline(
          vulkan_.device, luma_pipeline_, vulkan_.allocator);
      luma_pipeline_ = VK_NULL_HANDLE;
    }

    if (luma_pipeline_layout_ != VK_NULL_HANDLE) {
      dispatch_.destroy_pipeline_layout(
          vulkan_.device,
          luma_pipeline_layout_,
          vulkan_.allocator);
      luma_pipeline_layout_ = VK_NULL_HANDLE;
    }

    if (luma_descriptor_set_layout_ != VK_NULL_HANDLE) {
      dispatch_.destroy_descriptor_set_layout(
          vulkan_.device,
          luma_descriptor_set_layout_,
          vulkan_.allocator);
      luma_descriptor_set_layout_ = VK_NULL_HANDLE;
    }

    if (motion_shader_module_ != VK_NULL_HANDLE) {
      dispatch_.destroy_shader_module(
          vulkan_.device,
          motion_shader_module_,
          vulkan_.allocator);
      motion_shader_module_ = VK_NULL_HANDLE;
    }

    if (luma_shader_module_ != VK_NULL_HANDLE) {
      dispatch_.destroy_shader_module(
          vulkan_.device,
          luma_shader_module_,
          vulkan_.allocator);
      luma_shader_module_ = VK_NULL_HANDLE;
    }
  }

  bool Initialize() {
    if (!dispatch_.Load(
            vulkan_.instance,
            vulkan_.device,
            vulkan_.get_instance_proc_addr)) {
      return false;
    }

    VkShaderModuleCreateInfo shader_info{};

    VkShaderModuleCreateInfo motion_shader_info{};
    motion_shader_info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    motion_shader_info.codeSize = sizeof(kMotionSpv);
    motion_shader_info.pCode = kMotionSpv;

    if (dispatch_.create_shader_module(
            vulkan_.device,
            &motion_shader_info,
            vulkan_.allocator,
            &motion_shader_module_) != VK_SUCCESS) {
      return false;
    }

    VkShaderModuleCreateInfo synth_shader_info{};
    synth_shader_info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    synth_shader_info.codeSize =
        sizeof(kSynthSpv);
    synth_shader_info.pCode =
        kSynthSpv;

    if (dispatch_.create_shader_module(
            vulkan_.device,
            &synth_shader_info,
            vulkan_.allocator,
            &synth_shader_module_) != VK_SUCCESS) {
      return false;
    }

    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = shaders::kLumaSpvSize;
    shader_info.pCode = shaders::kLumaSpv;

    if (dispatch_.create_shader_module(
            vulkan_.device,
            &shader_info,
            vulkan_.allocator,
            &luma_shader_module_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorSetLayoutBinding bindings[2]{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;

    if (dispatch_.create_descriptor_set_layout(
            vulkan_.device,
            &layout_info,
            vulkan_.allocator,
            &luma_descriptor_set_layout_) != VK_SUCCESS) {
      return false;
    }

    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = sizeof(uint32_t) * 2;

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &luma_descriptor_set_layout_;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;

    if (dispatch_.create_pipeline_layout(
            vulkan_.device,
            &pipeline_layout_info,
            vulkan_.allocator,
            &luma_pipeline_layout_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorSetLayoutBinding motion_bindings[3]{};

    motion_bindings[0].binding = 0;
    motion_bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    motion_bindings[0].descriptorCount = 1;
    motion_bindings[0].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    motion_bindings[1].binding = 1;
    motion_bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    motion_bindings[1].descriptorCount = 1;
    motion_bindings[1].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    motion_bindings[2].binding = 2;
    motion_bindings[2].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    motion_bindings[2].descriptorCount = 1;
    motion_bindings[2].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo motion_set_layout_info{};
    motion_set_layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    motion_set_layout_info.bindingCount = 3;
    motion_set_layout_info.pBindings = motion_bindings;

    if (dispatch_.create_descriptor_set_layout(
            vulkan_.device,
            &motion_set_layout_info,
            vulkan_.allocator,
            &motion_descriptor_set_layout_) != VK_SUCCESS) {
      return false;
    }

    VkPushConstantRange motion_push_constant{};
    motion_push_constant.stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    motion_push_constant.offset = 0;
    motion_push_constant.size = sizeof(uint32_t) * 2;

    VkPipelineLayoutCreateInfo motion_pipeline_layout_info{};
    motion_pipeline_layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    motion_pipeline_layout_info.setLayoutCount = 1;
    motion_pipeline_layout_info.pSetLayouts =
        &motion_descriptor_set_layout_;
    motion_pipeline_layout_info.pushConstantRangeCount = 1;
    motion_pipeline_layout_info.pPushConstantRanges =
        &motion_push_constant;

    if (dispatch_.create_pipeline_layout(
            vulkan_.device,
            &motion_pipeline_layout_info,
            vulkan_.allocator,
            &motion_pipeline_layout_) != VK_SUCCESS) {
      return false;
    }

    VkPipelineShaderStageCreateInfo motion_stage_info{};
    motion_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    motion_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    motion_stage_info.module = motion_shader_module_;
    motion_stage_info.pName = "main";

    VkComputePipelineCreateInfo motion_pipeline_info{};
    motion_pipeline_info.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    motion_pipeline_info.stage = motion_stage_info;
    motion_pipeline_info.layout = motion_pipeline_layout_;

    if (dispatch_.create_compute_pipelines(
            vulkan_.device,
            VK_NULL_HANDLE,
            1,
            &motion_pipeline_info,
            vulkan_.allocator,
            &motion_pipeline_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorSetLayoutBinding synth_bindings[4]{};

    synth_bindings[0].binding = 0;
    synth_bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    synth_bindings[0].descriptorCount = 1;
    synth_bindings[0].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    synth_bindings[1].binding = 1;
    synth_bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    synth_bindings[1].descriptorCount = 1;
    synth_bindings[1].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    synth_bindings[2].binding = 2;
    synth_bindings[2].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    synth_bindings[2].descriptorCount = 1;
    synth_bindings[2].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    synth_bindings[3].binding = 3;
    synth_bindings[3].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    synth_bindings[3].descriptorCount = 1;
    synth_bindings[3].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo synth_set_layout_info{};
    synth_set_layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    synth_set_layout_info.bindingCount = 4;
    synth_set_layout_info.pBindings =
        synth_bindings;

    if (dispatch_.create_descriptor_set_layout(
            vulkan_.device,
            &synth_set_layout_info,
            vulkan_.allocator,
            &synth_descriptor_set_layout_) != VK_SUCCESS) {
      return false;
    }

    VkPushConstantRange synth_push_constant{};
    synth_push_constant.stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT;
    synth_push_constant.offset = 0;
    synth_push_constant.size =
        sizeof(uint32_t) * 2 + sizeof(float);

    VkPipelineLayoutCreateInfo synth_pipeline_layout_info{};
    synth_pipeline_layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    synth_pipeline_layout_info.setLayoutCount = 1;
    synth_pipeline_layout_info.pSetLayouts =
        &synth_descriptor_set_layout_;
    synth_pipeline_layout_info.pushConstantRangeCount = 1;
    synth_pipeline_layout_info.pPushConstantRanges =
        &synth_push_constant;

    if (dispatch_.create_pipeline_layout(
            vulkan_.device,
            &synth_pipeline_layout_info,
            vulkan_.allocator,
            &synth_pipeline_layout_) != VK_SUCCESS) {
      return false;
    }

    VkPipelineShaderStageCreateInfo synth_stage_info{};
    synth_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    synth_stage_info.stage =
        VK_SHADER_STAGE_COMPUTE_BIT;
    synth_stage_info.module =
        synth_shader_module_;
    synth_stage_info.pName =
        "main";

    VkComputePipelineCreateInfo synth_pipeline_info{};
    synth_pipeline_info.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    synth_pipeline_info.stage =
        synth_stage_info;
    synth_pipeline_info.layout =
        synth_pipeline_layout_;

    if (dispatch_.create_compute_pipelines(
            vulkan_.device,
            VK_NULL_HANDLE,
            1,
            &synth_pipeline_info,
            vulkan_.allocator,
            &synth_pipeline_) != VK_SUCCESS) {
      return false;
    }

    VkSamplerCreateInfo synth_sampler_info{};
    synth_sampler_info.sType =
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    synth_sampler_info.magFilter =
        VK_FILTER_LINEAR;
    synth_sampler_info.minFilter =
        VK_FILTER_LINEAR;
    synth_sampler_info.mipmapMode =
        VK_SAMPLER_MIPMAP_MODE_NEAREST;
    synth_sampler_info.addressModeU =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    synth_sampler_info.addressModeV =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    synth_sampler_info.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    synth_sampler_info.minLod = 0.0f;
    synth_sampler_info.maxLod = 0.0f;
    synth_sampler_info.maxAnisotropy = 1.0f;

    if (dispatch_.create_sampler(
            vulkan_.device,
            &synth_sampler_info,
            vulkan_.allocator,
            &synth_sampler_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorPoolSize synth_pool_sizes[2]{};

    synth_pool_sizes[0].type =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    synth_pool_sizes[0].descriptorCount = 2;

    synth_pool_sizes[1].type =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    synth_pool_sizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo synth_pool_info{};
    synth_pool_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    synth_pool_info.maxSets = 1;
    synth_pool_info.poolSizeCount = 2;
    synth_pool_info.pPoolSizes =
        synth_pool_sizes;

    if (dispatch_.create_descriptor_pool(
            vulkan_.device,
            &synth_pool_info,
            vulkan_.allocator,
            &synth_descriptor_pool_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorSetAllocateInfo synth_set_info{};
    synth_set_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    synth_set_info.descriptorPool =
        synth_descriptor_pool_;
    synth_set_info.descriptorSetCount = 1;
    synth_set_info.pSetLayouts =
        &synth_descriptor_set_layout_;

    if (dispatch_.allocate_descriptor_sets(
            vulkan_.device,
            &synth_set_info,
            &synth_descriptor_set_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorPoolSize motion_pool_size{};
    motion_pool_size.type =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    motion_pool_size.descriptorCount = 3;

    VkDescriptorPoolCreateInfo motion_pool_info{};
    motion_pool_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    motion_pool_info.maxSets = 1;
    motion_pool_info.poolSizeCount = 1;
    motion_pool_info.pPoolSizes = &motion_pool_size;

    if (dispatch_.create_descriptor_pool(
            vulkan_.device,
            &motion_pool_info,
            vulkan_.allocator,
            &motion_descriptor_pool_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorSetAllocateInfo motion_set_info{};
    motion_set_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    motion_set_info.descriptorPool =
        motion_descriptor_pool_;
    motion_set_info.descriptorSetCount = 1;
    motion_set_info.pSetLayouts =
        &motion_descriptor_set_layout_;

    if (dispatch_.allocate_descriptor_sets(
            vulkan_.device,
            &motion_set_info,
            &motion_descriptor_set_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorPoolSize pool_sizes[2]{};
    pool_sizes[0].type =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount = 2;
    pool_sizes[1].type =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 2;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;

    if (dispatch_.create_descriptor_pool(
            vulkan_.device,
            &pool_info,
            vulkan_.allocator,
            &luma_descriptor_pool_) != VK_SUCCESS) {
      return false;
    }

    VkDescriptorSetAllocateInfo set_allocate_info{};
    set_allocate_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_allocate_info.descriptorPool = luma_descriptor_pool_;
    set_allocate_info.descriptorSetCount = 1;
    set_allocate_info.pSetLayouts =
        &luma_descriptor_set_layout_;

    if (dispatch_.allocate_descriptor_sets(
            vulkan_.device,
            &set_allocate_info,
            &luma_descriptor_set_) != VK_SUCCESS) {
      return false;
    }

    if (dispatch_.allocate_descriptor_sets(
            vulkan_.device,
            &set_allocate_info,
            &previous_luma_descriptor_set_) != VK_SUCCESS) {
      return false;
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_NEAREST;
    sampler_info.minFilter = VK_FILTER_NEAREST;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;
    sampler_info.maxAnisotropy = 1.0f;

    if (dispatch_.create_sampler(
            vulkan_.device,
            &sampler_info,
            vulkan_.allocator,
            &luma_sampler_) != VK_SUCCESS) {
      return false;
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = luma_shader_module_;
    stage_info.pName = "main";

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType =
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = luma_pipeline_layout_;

    return dispatch_.create_compute_pipelines(
               vulkan_.device,
               VK_NULL_HANDLE,
               1,
               &pipeline_info,
               vulkan_.allocator,
               &luma_pipeline_) == VK_SUCCESS;
  }

  Status Resize(uint32_t width,
                uint32_t height,
                VkFormat input_format,
                VkFormat output_format) {
    if (width == 0 || height == 0 ||
        input_format == VK_FORMAT_UNDEFINED ||
        output_format == VK_FORMAT_UNDEFINED) {
      return Status::kInvalidArgument;
    }

    const bool supported_mvp_output_format =
        output_format == VK_FORMAT_R8G8B8A8_UNORM ||
        output_format == VK_FORMAT_R8G8B8A8_SRGB ||
        output_format == VK_FORMAT_B8G8R8A8_UNORM ||
        output_format == VK_FORMAT_B8G8R8A8_SRGB ||
        output_format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
        output_format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
        output_format == VK_FORMAT_R16G16B16A16_SFLOAT;

    if (!supported_mvp_output_format) {
      return Status::kUnsupported;
    }

    VkFormatProperties synth_format_properties{};
    dispatch_.get_physical_device_format_properties(
        vulkan_.physical_device,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        &synth_format_properties);

    VkFormatProperties output_format_properties{};
    dispatch_.get_physical_device_format_properties(
        vulkan_.physical_device,
        output_format,
        &output_format_properties);

    if ((synth_format_properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_BLIT_SRC_BIT) == 0 ||
        (output_format_properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_BLIT_DST_BIT) == 0) {
      return Status::kUnsupported;
    }

    initialized_ = false;

    const Status previous_luma_status = previous_luma_image_.Create(
        vulkan_,
        dispatch_,
        width,
        height,
        VK_FORMAT_R8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    if (previous_luma_status != Status::kSuccess) {
      return previous_luma_status;
    }

    const Status motion_status = motion_image_.Create(
        vulkan_,
        dispatch_,
        ((width + 7) / 8),
        ((height + 7) / 8),
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    if (motion_status != Status::kSuccess) {
      motion_image_.Destroy(vulkan_, dispatch_);
      previous_luma_image_.Destroy(vulkan_, dispatch_);
      return motion_status;
    }

    const Status synth_status = synth_image_.Create(
        vulkan_,
        dispatch_,
        width,
        height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    if (synth_status != Status::kSuccess) {
      synth_image_.Destroy(vulkan_, dispatch_);
      motion_image_.Destroy(vulkan_, dispatch_);
      previous_luma_image_.Destroy(vulkan_, dispatch_);
      return synth_status;
    }

    const Status current_luma_status = current_luma_image_.Create(
        vulkan_,
        dispatch_,
        width,
        height,
        VK_FORMAT_R8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    if (current_luma_status != Status::kSuccess) {
      current_luma_image_.Destroy(vulkan_, dispatch_);
      synth_image_.Destroy(vulkan_, dispatch_);
      motion_image_.Destroy(vulkan_, dispatch_);
      previous_luma_image_.Destroy(vulkan_, dispatch_);
      return current_luma_status;
    }

    UpdateLumaOutputDescriptor(
        luma_descriptor_set_,
        current_luma_image_.view());

    UpdateLumaOutputDescriptor(
        previous_luma_descriptor_set_,
        previous_luma_image_.view());

    VkDescriptorImageInfo motion_images[3]{};

    motion_images[0].imageView =
        previous_luma_image_.view();
    motion_images[0].imageLayout =
        VK_IMAGE_LAYOUT_GENERAL;

    motion_images[1].imageView =
        current_luma_image_.view();
    motion_images[1].imageLayout =
        VK_IMAGE_LAYOUT_GENERAL;

    motion_images[2].imageView =
        motion_image_.view();
    motion_images[2].imageLayout =
        VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet motion_writes[3]{};

    for (uint32_t i = 0; i < 3; ++i) {
      motion_writes[i].sType =
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      motion_writes[i].dstSet =
          motion_descriptor_set_;
      motion_writes[i].dstBinding = i;
      motion_writes[i].descriptorCount = 1;
      motion_writes[i].descriptorType =
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      motion_writes[i].pImageInfo =
          &motion_images[i];
    }

    dispatch_.update_descriptor_sets(
        vulkan_.device,
        3,
        motion_writes,
        0,
        nullptr);

    previous_luma_initialized_ = false;
    current_luma_initialized_ = false;

    motion_initialized_ = false;
    synth_initialized_ = false;
    width_ = width;
    height_ = height;
    input_format_ = input_format;
    output_format_ = output_format;
    initialized_ = true;

    return Status::kSuccess;
  }

  Status Interpolate(VkCommandBuffer command_buffer,
                     const Image& previous,
                     const Image& current,
                     float phase,
                     const Image& output) {
    if (!initialized_ ||
        command_buffer == VK_NULL_HANDLE ||
        previous.image == VK_NULL_HANDLE ||
        previous.view == VK_NULL_HANDLE ||
        current.image == VK_NULL_HANDLE ||
        current.view == VK_NULL_HANDLE ||
        output.image == VK_NULL_HANDLE ||
        output.view == VK_NULL_HANDLE) {
      return Status::kInvalidArgument;
    }

    if (!std::isfinite(phase) || phase < 0.0f || phase > 1.0f) {
      return Status::kInvalidArgument;
    }

    if (previous.width != width_ || previous.height != height_ ||
        current.width != width_ || current.height != height_ ||
        output.width != width_ || output.height != height_) {
      return Status::kInvalidArgument;
    }

    if (previous.format != input_format_ ||
        current.format != input_format_ ||
        output.format != output_format_) {
      return Status::kInvalidArgument;
    }

    if ((previous.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0 ||
        (current.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0 ||
        (output.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
      return Status::kInvalidArgument;
    }

    if (previous.layout == VK_IMAGE_LAYOUT_UNDEFINED ||
        current.layout == VK_IMAGE_LAYOUT_UNDEFINED ||
        output.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
      return Status::kInvalidArgument;
    }

    if (output.image == previous.image ||
        output.image == current.image) {
      return Status::kInvalidArgument;
    }


    // ZeroFG MVP initially targets exactly one synthetic midpoint frame.
    if (std::abs(phase - 0.5f) > 0.0001f) {
      return Status::kUnsupported;
    }

    // Vulkan compute implementation comes next.
    UpdateLumaInputDescriptor(
        luma_descriptor_set_, current);

    TransitionImage(
        command_buffer,
        current.image,
        current.layout,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    const VkImageLayout luma_old_layout =
        current_luma_initialized_
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_UNDEFINED;

    const VkAccessFlags luma_src_access =
        current_luma_initialized_
            ? VK_ACCESS_SHADER_WRITE_BIT
            : 0;

    const VkPipelineStageFlags luma_src_stage =
        current_luma_initialized_
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    TransitionImage(
        command_buffer,
        current_luma_image_.image(),
        luma_old_layout,
        VK_IMAGE_LAYOUT_GENERAL,
        luma_src_access,
        VK_ACCESS_SHADER_WRITE_BIT,
        luma_src_stage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    dispatch_.cmd_bind_pipeline(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        luma_pipeline_);

    dispatch_.cmd_bind_descriptor_sets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        luma_pipeline_layout_,
        0,
        1,
        &luma_descriptor_set_,
        0,
        nullptr);

    const uint32_t luma_extent[2] = {
        width_,
        height_,
    };

    dispatch_.cmd_push_constants(
        command_buffer,
        luma_pipeline_layout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(luma_extent),
        luma_extent);

    dispatch_.cmd_dispatch(
        command_buffer,
        (width_ + 15) / 16,
        (height_ + 15) / 16,
        1);

    TransitionImage(
        command_buffer,
        current_luma_image_.image(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    current_luma_initialized_ = true;

    TransitionImage(
        command_buffer,
        current.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        current.layout,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    UpdateLumaInputDescriptor(
        previous_luma_descriptor_set_, previous);

    TransitionImage(
        command_buffer,
        previous.image,
        previous.layout,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    const VkImageLayout previous_luma_old_layout =
        previous_luma_initialized_
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_UNDEFINED;

    const VkAccessFlags previous_luma_src_access =
        previous_luma_initialized_
            ? VK_ACCESS_SHADER_WRITE_BIT
            : 0;

    const VkPipelineStageFlags previous_luma_src_stage =
        previous_luma_initialized_
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    TransitionImage(
        command_buffer,
        previous_luma_image_.image(),
        previous_luma_old_layout,
        VK_IMAGE_LAYOUT_GENERAL,
        previous_luma_src_access,
        VK_ACCESS_SHADER_WRITE_BIT,
        previous_luma_src_stage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    dispatch_.cmd_bind_pipeline(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        luma_pipeline_);

    dispatch_.cmd_bind_descriptor_sets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        luma_pipeline_layout_,
        0,
        1,
        &previous_luma_descriptor_set_,
        0,
        nullptr);

    dispatch_.cmd_push_constants(
        command_buffer,
        luma_pipeline_layout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(luma_extent),
        luma_extent);

    dispatch_.cmd_dispatch(
        command_buffer,
        (width_ + 15) / 16,
        (height_ + 15) / 16,
        1);

    TransitionImage(
        command_buffer,
        previous_luma_image_.image(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    previous_luma_initialized_ = true;

    TransitionImage(
        command_buffer,
        previous.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        previous.layout,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    const VkImageLayout motion_old_layout =
        motion_initialized_
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_UNDEFINED;

    const VkAccessFlags motion_src_access =
        motion_initialized_
            ? (VK_ACCESS_SHADER_READ_BIT |
               VK_ACCESS_SHADER_WRITE_BIT)
            : 0;

    const VkPipelineStageFlags motion_src_stage =
        motion_initialized_
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    TransitionImage(
        command_buffer,
        motion_image_.image(),
        motion_old_layout,
        VK_IMAGE_LAYOUT_GENERAL,
        motion_src_access,
        VK_ACCESS_SHADER_WRITE_BIT,
        motion_src_stage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    dispatch_.cmd_bind_pipeline(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        motion_pipeline_);

    dispatch_.cmd_bind_descriptor_sets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        motion_pipeline_layout_,
        0,
        1,
        &motion_descriptor_set_,
        0,
        nullptr);

    const uint32_t motion_extent[2] = {
        width_,
        height_,
    };

    dispatch_.cmd_push_constants(
        command_buffer,
        motion_pipeline_layout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(motion_extent),
        motion_extent);

    const uint32_t motion_blocks_x =
        (width_ + 7) / 8;
    const uint32_t motion_blocks_y =
        (height_ + 7) / 8;

    dispatch_.cmd_dispatch(
        command_buffer,
        (motion_blocks_x + 7) / 8,
        (motion_blocks_y + 7) / 8,
        1);

    TransitionImage(
        command_buffer,
        motion_image_.image(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    motion_initialized_ = true;

    VkDescriptorImageInfo synth_images[4]{};

    synth_images[0].sampler =
        synth_sampler_;
    synth_images[0].imageView =
        previous.view;
    synth_images[0].imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    synth_images[1].sampler =
        synth_sampler_;
    synth_images[1].imageView =
        current.view;
    synth_images[1].imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    synth_images[2].imageView =
        motion_image_.view();
    synth_images[2].imageLayout =
        VK_IMAGE_LAYOUT_GENERAL;

    synth_images[3].imageView =
        synth_image_.view();
    synth_images[3].imageLayout =
        VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet synth_writes[4]{};

    for (uint32_t i = 0; i < 4; ++i) {
      synth_writes[i].sType =
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      synth_writes[i].dstSet =
          synth_descriptor_set_;
      synth_writes[i].dstBinding = i;
      synth_writes[i].descriptorCount = 1;
      synth_writes[i].descriptorType =
          i < 2
              ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      synth_writes[i].pImageInfo =
          &synth_images[i];
    }

    dispatch_.update_descriptor_sets(
        vulkan_.device,
        4,
        synth_writes,
        0,
        nullptr);

    TransitionImage(
        command_buffer,
        previous.image,
        previous.layout,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    TransitionImage(
        command_buffer,
        current.image,
        current.layout,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_MEMORY_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    const VkImageLayout synth_old_layout =
        synth_initialized_
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_UNDEFINED;

    const VkAccessFlags synth_src_access =
        synth_initialized_
            ? (VK_ACCESS_SHADER_READ_BIT |
               VK_ACCESS_SHADER_WRITE_BIT)
            : 0;

    const VkPipelineStageFlags synth_src_stage =
        synth_initialized_
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    TransitionImage(
        command_buffer,
        synth_image_.image(),
        synth_old_layout,
        VK_IMAGE_LAYOUT_GENERAL,
        synth_src_access,
        VK_ACCESS_SHADER_WRITE_BIT,
        synth_src_stage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    dispatch_.cmd_bind_pipeline(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        synth_pipeline_);

    dispatch_.cmd_bind_descriptor_sets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        synth_pipeline_layout_,
        0,
        1,
        &synth_descriptor_set_,
        0,
        nullptr);

    struct SynthPushConstants {
      uint32_t extent[2];
      float phase;
    };

    const SynthPushConstants synth_push = {
        {width_, height_},
        static_cast<float>(phase),
    };

    static_assert(
        sizeof(SynthPushConstants) == 12,
        "unexpected synth push constant size");

    dispatch_.cmd_push_constants(
        command_buffer,
        synth_pipeline_layout_,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(synth_push),
        &synth_push);

    dispatch_.cmd_dispatch(
        command_buffer,
        (width_ + 15) / 16,
        (height_ + 15) / 16,
        1);

    TransitionImage(
        command_buffer,
        synth_image_.image(),
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);

    synth_initialized_ = true;

    const bool output_is_present =
        output.layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    const VkAccessFlags output_src_access =
        output_is_present
            ? 0
            : (VK_ACCESS_MEMORY_READ_BIT |
               VK_ACCESS_MEMORY_WRITE_BIT);

    TransitionImage(
        command_buffer,
        output.image,
        output.layout,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        output_src_access,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageBlit blit_region{};

    blit_region.srcSubresource.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;
    blit_region.srcSubresource.mipLevel = 0;
    blit_region.srcSubresource.baseArrayLayer = 0;
    blit_region.srcSubresource.layerCount = 1;

    blit_region.srcOffsets[0] = {
        0,
        0,
        0};

    blit_region.srcOffsets[1] = {
        static_cast<int32_t>(width_),
        static_cast<int32_t>(height_),
        1};

    blit_region.dstSubresource.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;
    blit_region.dstSubresource.mipLevel = 0;
    blit_region.dstSubresource.baseArrayLayer = 0;
    blit_region.dstSubresource.layerCount = 1;

    blit_region.dstOffsets[0] = {
        0,
        0,
        0};

    blit_region.dstOffsets[1] = {
        static_cast<int32_t>(width_),
        static_cast<int32_t>(height_),
        1};

    dispatch_.cmd_blit_image(
        command_buffer,
        synth_image_.image(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        output.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blit_region,
        VK_FILTER_NEAREST);

    TransitionImage(
        command_buffer,
        output.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        output.layout,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        output_is_present
            ? 0
            : (VK_ACCESS_MEMORY_READ_BIT |
               VK_ACCESS_MEMORY_WRITE_BIT),
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    TransitionImage(
        command_buffer,
        synth_image_.image(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    TransitionImage(
        command_buffer,
        previous.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        previous.layout,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    TransitionImage(
        command_buffer,
        current.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        current.layout,
        VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);


    return Status::kSuccess;
  }

 private:
  void TransitionImage(
      VkCommandBuffer command_buffer,
      VkImage image,
      VkImageLayout old_layout,
      VkImageLayout new_layout,
      VkAccessFlags src_access,
      VkAccessFlags dst_access,
      VkPipelineStageFlags src_stage,
      VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    dispatch_.cmd_pipeline_barrier(
        command_buffer,
        src_stage,
        dst_stage,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier);
  }

  void UpdateLumaOutputDescriptor(
      VkDescriptorSet descriptor_set,
      VkImageView image_view) {
    VkDescriptorImageInfo output_info{};
    output_info.imageView = image_view;
    output_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet output_write{};
    output_write.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    output_write.dstSet = descriptor_set;
    output_write.dstBinding = 1;
    output_write.dstArrayElement = 0;
    output_write.descriptorCount = 1;
    output_write.descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    output_write.pImageInfo = &output_info;

    dispatch_.update_descriptor_sets(
        vulkan_.device,
        1,
        &output_write,
        0,
        nullptr);
  }

  void UpdateLumaInputDescriptor(
      VkDescriptorSet descriptor_set,
      const Image& input) {
    VkDescriptorImageInfo input_info{};
    input_info.sampler = luma_sampler_;
    input_info.imageView = input.view;
    input_info.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet input_write{};
    input_write.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    input_write.dstSet = descriptor_set;
    input_write.dstBinding = 0;
    input_write.dstArrayElement = 0;
    input_write.descriptorCount = 1;
    input_write.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    input_write.pImageInfo = &input_info;

    dispatch_.update_descriptor_sets(
        vulkan_.device,
        1,
        &input_write,
        0,
        nullptr);
  }

  VulkanContext vulkan_;
  VulkanDispatch dispatch_;

  VkShaderModule luma_shader_module_ = VK_NULL_HANDLE;
  VkShaderModule motion_shader_module_ = VK_NULL_HANDLE;
  VkShaderModule synth_shader_module_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout synth_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout synth_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline synth_pipeline_ = VK_NULL_HANDLE;
  VkSampler synth_sampler_ = VK_NULL_HANDLE;
  VkDescriptorPool synth_descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet synth_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout motion_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout motion_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline motion_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool motion_descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet motion_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout luma_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout luma_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline luma_pipeline_ = VK_NULL_HANDLE;
  VkSampler luma_sampler_ = VK_NULL_HANDLE;
  VkDescriptorPool luma_descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet luma_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSet previous_luma_descriptor_set_ = VK_NULL_HANDLE;
  bool previous_luma_initialized_ = false;
  bool current_luma_initialized_ = false;

  bool motion_initialized_ = false;
  bool synth_initialized_ = false;
  OwnedImage previous_luma_image_;
  OwnedImage current_luma_image_;
  OwnedImage motion_image_;
  OwnedImage synth_image_;

  uint32_t width_ = 0;
  uint32_t height_ = 0;

  VkFormat input_format_ = VK_FORMAT_UNDEFINED;
  VkFormat output_format_ = VK_FORMAT_UNDEFINED;

  bool initialized_ = false;
};

std::unique_ptr<Interpolator> Interpolator::Create(
    const CreateInfo& create_info,
    Status* status) {
  if (create_info.vulkan.instance == VK_NULL_HANDLE ||
      create_info.vulkan.physical_device == VK_NULL_HANDLE ||
      create_info.vulkan.device == VK_NULL_HANDLE ||
      create_info.vulkan.get_instance_proc_addr == nullptr) {
    if (status) {
      *status = Status::kInvalidArgument;
    }
    return nullptr;
  }

  if (create_info.frame_context_count == 0 ||
      create_info.frame_context_count > 8) {
    if (status) {
      *status = Status::kInvalidArgument;
    }
    return nullptr;
  }

  std::vector<std::unique_ptr<Impl>> impls;
  impls.reserve(create_info.frame_context_count);

  for (uint32_t i = 0;
       i < create_info.frame_context_count;
       ++i) {
    auto impl = std::make_unique<Impl>(create_info);

    if (!impl->Initialize()) {
      if (status) {
        *status = Status::kUnsupported;
      }
      return nullptr;
    }

    impls.push_back(std::move(impl));
  }

  auto interpolator =
      std::unique_ptr<Interpolator>(
          new Interpolator(std::move(impls)));

  if (status) {
    *status = Status::kSuccess;
  }

  return interpolator;
}

Interpolator::Interpolator(
    std::vector<std::unique_ptr<Impl>> impls)
    : impls_(std::move(impls)) {}

Interpolator::~Interpolator() = default;

Status Interpolator::Resize(uint32_t width,
                            uint32_t height,
                            VkFormat input_format,
                            VkFormat output_format) {
  if (impls_.empty()) {
    return Status::kInvalidArgument;
  }

  for (const auto& impl : impls_) {
    const Status status =
        impl->Resize(
            width,
            height,
            input_format,
            output_format);

    if (status != Status::kSuccess) {
      return status;
    }
  }

  return Status::kSuccess;
}

Status Interpolator::Interpolate(VkCommandBuffer command_buffer,
    uint32_t frame_context_index,
                                 const Image& previous,
                                 const Image& current,
                                 float phase,
                                 const Image& output) {
  if (frame_context_index >= impls_.size()) {
    return Status::kInvalidArgument;
  }

  return impls_[frame_context_index]->Interpolate(
      command_buffer,
      previous,
      current,
      phase,
      output);
}

}  // namespace zerofg
