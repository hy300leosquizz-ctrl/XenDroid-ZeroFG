/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/vulkan_presenter.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "third_party/fmt/include/fmt/format.h"
#include "zerofg/integration/xenia/zerofg_xenia_adapter.h"
#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/frame_stats.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/base/platform.h"
#include "xenia/gpu/gpu_flags.h"
#include "xenia/ui/vulkan/vulkan_util.h"
#include "xenia/ui/vulkan/zerofg_completion_owner.h"
#include "xenia/ui/vulkan/zerofg_config.h"
#include "xenia/ui/vulkan/zerofg_device_handoff.h"

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
#include <errno.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <android/hardware_buffer.h>

#include "xenia/ui/surface_android.h"
#endif
#if XE_PLATFORM_MAC
#include "xenia/ui/surface_mac.h"
#endif
#if XE_PLATFORM_GNU_LINUX
#include "xenia/ui/surface_gnulinux.h"
#endif
#if XE_PLATFORM_WIN32
#include "xenia/ui/surface_win.h"
#endif

DECLARE_bool(adrenotools_force_max_clocks);
DECLARE_uint32(gpu_stall_spin_iterations);
DECLARE_bool(host_present_from_non_ui_thread);
DECLARE_double(time_scalar);
DECLARE_string(turnip_debug);
DECLARE_bool(use_50Hz_mode);
DECLARE_int32(vulkan_mid_frame_submission_draws);

// Note: If the priorities in the description are changed, update the actual
// present mode selection logic.
DEFINE_bool(
    vulkan_allow_present_mode_immediate, true,
    "When available, allow the immediate presentation mode (1st priority), "
    "offering the lowest latency with the possibility of tearing in certain "
    "cases, and, depending on the configuration, variable refresh rate.",
    "Vulkan");
DEFINE_bool(
    vulkan_allow_present_mode_mailbox, true,
    "When available, allow the mailbox presentation mode (2nd priority), "
    "offering low latency without the possibility of tearing.",
    "Vulkan");
DEFINE_bool(
    vulkan_allow_present_mode_fifo_relaxed, true,
    "When available, allow the relaxed first-in-first-out presentation mode "
    "(3rd priority), which causes waiting for host display vertical sync, but "
    "may present with tearing if frames don't meet the host display refresh "
    "rate.",
    "Vulkan");
DEFINE_bool(
    zerofg_frame_generation, false,
    "Enable ZeroFG through the independent Android presenter. "
    "Initialization falls back to the native Vulkan presenter on failure.",
    "Vulkan");
DEFINE_string(
    zerofg_mode, "",
    "ZeroFG frame generation: off, zero or reallyzero (the economy tier). "
    "Read when a game starts.",
    "Vulkan");
#if XE_PLATFORM_MAC
DEFINE_bool(vulkan_presenter_use_backing_scale, false,
            "Use the macOS view backing scale factor for MoltenVK drawables.",
            "Vulkan");
#endif  // XE_PLATFORM_MAC

namespace xe {
namespace ui {
namespace vulkan {

namespace {

uint64_t GetZeroFGMonotonicTimeNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

constexpr uint64_t kNativePresentTelemetryIntervalNs = 2500000000ull;
// The two timestamps around each Generation: its GPU service time.
constexpr uint32_t kZeroFGTimestampCount = 2;
// Bounded scheduler seed for the Synthetic GPU cost until it is measured.
constexpr uint64_t kZeroFGSyntheticCostSeedNs = 8000000ull;

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
constexpr bool kZeroFGSyncFdSupported = true;
#else
constexpr bool kZeroFGSyncFdSupported = false;
#endif

}  // namespace

// Generated with `xb buildshaders`.
namespace shaders {
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_bilinear_dither_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_bilinear_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_ffx_cas_resample_dither_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_ffx_cas_resample_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_ffx_cas_sharpen_dither_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_ffx_cas_sharpen_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_ffx_fsr_easu_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_ffx_fsr_rcas_dither_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_ffx_fsr_rcas_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_sgsr_edge_direction_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_sgsr_ps.h"
#include "xenia/ui/shaders/bytecode/vulkan_spirv/guest_output_triangle_strip_rect_vs.h"
}  // namespace shaders

VulkanPresenter::VulkanPresenter(HostGpuLossCallback host_gpu_loss_callback,
                                 VulkanDevice* vulkan_device,
                                 const UISamplers* ui_samplers)
    : Presenter(host_gpu_loss_callback),
      vulkan_device_(vulkan_device),
      ui_samplers_(ui_samplers),
      zerofg_vulkan_context_(
          std::make_unique<ZeroFGPresenterVulkanContext>(vulkan_device)),
      guest_output_image_refresher_completion_timeline_(vulkan_device,
                                                        "guest-refresher"),
      ui_completion_timeline_(vulkan_device, "ui"),
      paint_context_(vulkan_device) {
  assert_not_null(vulkan_device);
  assert_not_null(ui_samplers);
  xe::RequestFrameTelemetryReset();
}

VulkanPresenter::PaintContext::Submission::~Submission() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  if (draw_command_pool_ != VK_NULL_HANDLE) {
    dfn.vkDestroyCommandPool(device, draw_command_pool_, nullptr);
  }

  if (acquire_semaphore_ != VK_NULL_HANDLE) {
    dfn.vkDestroySemaphore(device, acquire_semaphore_, nullptr);
  }
}

bool VulkanPresenter::PaintContext::Submission::Initialize() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkSemaphoreCreateInfo semaphore_create_info;
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext = nullptr;
  semaphore_create_info.flags = 0;
  if (dfn.vkCreateSemaphore(device, &semaphore_create_info, nullptr,
                            &acquire_semaphore_) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create a swapchain image acquisition "
        "semaphore");
    return false;
  }

  VkCommandPoolCreateInfo command_pool_create_info;
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.pNext = nullptr;
  command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  command_pool_create_info.queueFamilyIndex =
      vulkan_device_->queue_family_graphics_compute();
  if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr,
                              &draw_command_pool_) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create a command pool for drawing to a "
        "swapchain");
    return false;
  }
  VkCommandBufferAllocateInfo command_buffer_allocate_info;
  command_buffer_allocate_info.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_allocate_info.pNext = nullptr;
  command_buffer_allocate_info.commandPool = draw_command_pool_;
  command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_allocate_info.commandBufferCount = 1;
  if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info,
                                   &draw_command_buffer_) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to allocate a command buffer for drawing to a "
        "swapchain");
    return false;
  }

  return true;
}

VulkanPresenter::~VulkanPresenter() {
  xe::RequestFrameTelemetryReset();
  BeginZeroFGSurfaceDisconnect();
  if (zerofg_independent_presenter_) {
    // Main Surface Authority: B may hold a swapchain on A's VkSurfaceKHR,
    // which must be gone before the surface is destroyed below.
    guest_output_image_refresher_completion_timeline_.AwaitAllSubmissions();
    DestroyZeroFGSurfaceResourcesAfterSourceIdle();
  }
  // Destroy the swapchain after its images are not used for drawing anymore.
  // This is a confusing part in Vulkan, as vkQueuePresentKHR doesn't signal a
  // fence clearly indicating when it's safe to destroy a swapchain, so we
  // assume that its lifetime is tracked internally in the WSI. This is also
  // done before destroying the semaphore awaited by vkQueuePresentKHR, hoping
  // that it will prevent the destruction during the semaphore wait in
  // vkQueuePresentKHR execution (or between the vkQueueSubmit semaphore signal
  // and the vkQueuePresentKHR semaphore wait).
  // This will await completion of all paint submissions also.
  paint_context_.DestroySwapchainAndVulkanSurface();

  // Await completion of the usage of everything before destroying anything
  // (paint submission completion already awaited).
  // From most likely the latest to most likely the earliest to be signaled, so
  // just one sleep will likely be needed.
  ui_completion_timeline_.AwaitAllSubmissions();
  guest_output_image_refresher_completion_timeline_.AwaitAllSubmissions();
  DestroyZeroFGSurfaceResourcesAfterSourceIdle();

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  paint_context_.completion_timeline.AwaitAllSubmissions();

  if (paint_context_.swapchain_render_pass != VK_NULL_HANDLE) {
    dfn.vkDestroyRenderPass(device, paint_context_.swapchain_render_pass,
                            nullptr);
  }

  for (const PaintContext::UISetupCommandBuffer& ui_setup_command_buffer :
       paint_context_.ui_setup_command_buffers) {
    dfn.vkDestroyCommandPool(device, ui_setup_command_buffer.command_pool,
                             nullptr);
  }

  for (VkFramebuffer& framebuffer :
       paint_context_.guest_output_intermediate_framebuffers) {
    util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device, framebuffer);
  }
  util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device,
                             paint_context_.guest_output_descriptor_pool);
  for (PaintContext::GuestOutputPaintPipeline& guest_output_paint_pipeline :
       paint_context_.guest_output_paint_pipelines) {
    util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                               guest_output_paint_pipeline.swapchain_pipeline);
    util::DestroyAndNullHandle(
        dfn.vkDestroyPipeline, device,
        guest_output_paint_pipeline.intermediate_pipeline);
  }

  util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device,
                             guest_output_intermediate_render_pass_);
  for (VkShaderModule& shader_module : guest_output_paint_fs_) {
    util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device,
                               shader_module);
  }
  util::DestroyAndNullHandle(dfn.vkDestroyShaderModule, device,
                             guest_output_paint_vs_);
  for (VkPipelineLayout& pipeline_layout :
       guest_output_paint_pipeline_layouts_) {
    util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                               pipeline_layout);
  }
  util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                             guest_output_paint_image_descriptor_set_layout_);
}

Surface::TypeFlags VulkanPresenter::GetSurfaceTypesSupportedByInstance(
    const VulkanInstance::Extensions& instance_extensions) {
  if (!instance_extensions.ext_KHR_surface) {
    return 0;
  }
  Surface::TypeFlags type_flags = 0;
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (instance_extensions.ext_KHR_android_surface) {
    type_flags |= Surface::kTypeFlag_AndroidNativeWindow;
  }
#endif
#if XE_PLATFORM_MAC
  if (instance_extensions.ext_EXT_metal_surface) {
    type_flags |= Surface::kTypeFlag_MacNSView;
  }
#endif
#if XE_PLATFORM_GNU_LINUX
  if (instance_extensions.ext_KHR_wayland_surface) {
    type_flags |= Surface::kTypeFlag_WaylandWindow;
  }
  if (instance_extensions.ext_KHR_xcb_surface) {
    type_flags |= Surface::kTypeFlag_XcbWindow;
  }
#endif
#if XE_PLATFORM_WIN32
  if (instance_extensions.ext_KHR_win32_surface) {
    type_flags |= Surface::kTypeFlag_Win32Hwnd;
  }
#endif
  return type_flags;
}

Surface::TypeFlags VulkanPresenter::GetSupportedSurfaceTypes() const {
  if (!vulkan_device_->extensions().ext_KHR_swapchain) {
    return 0;
  }
  return GetSurfaceTypesSupportedByInstance(
      vulkan_device_->vulkan_instance()->extensions());
}

bool VulkanPresenter::CaptureGuestOutput(RawImage& image_out) {
  std::shared_ptr<GuestOutputImage> guest_output_image;
  {
    uint32_t guest_output_mailbox_index;
    std::unique_lock<std::mutex> guest_output_consumer_lock(
        ConsumeGuestOutput(guest_output_mailbox_index, nullptr, nullptr));
    if (guest_output_mailbox_index != UINT32_MAX) {
      assert_true(guest_output_images_[guest_output_mailbox_index]
                      .ever_successfully_refreshed);
      guest_output_image =
          guest_output_images_[guest_output_mailbox_index].image;
    }
    // Incremented the reference count of the guest output image - safe to leave
    // the consumer critical section now.
  }
  if (!guest_output_image) {
    return false;
  }

  VkExtent2D image_extent = guest_output_image->extent();
  size_t pixel_count = size_t(image_extent.width) * image_extent.height;
  VkDeviceSize buffer_size = VkDeviceSize(sizeof(uint32_t) * pixel_count);
  VkBuffer buffer;
  VkDeviceMemory buffer_memory;
  if (!util::CreateDedicatedAllocationBuffer(
          vulkan_device_, buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          util::MemoryPurpose::kReadback, buffer, buffer_memory)) {
    XELOGE("VulkanPresenter: Failed to create the guest output capture buffer");
    return false;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  {
    VkCommandPoolCreateInfo command_pool_create_info;
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.pNext = nullptr;
    command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    command_pool_create_info.queueFamilyIndex =
        vulkan_device_->queue_family_graphics_compute();
    VkCommandPool command_pool;
    if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr,
                                &command_pool) != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to create the guest output capturing "
          "command pool");
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    VkCommandBufferAllocateInfo command_buffer_allocate_info;
    command_buffer_allocate_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.pNext = nullptr;
    command_buffer_allocate_info.commandPool = command_pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1;
    VkCommandBuffer command_buffer;
    if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info,
                                     &command_buffer) != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to allocate the guest output capturing "
          "command buffer");
      dfn.vkDestroyCommandPool(device, command_pool, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    VkCommandBufferBeginInfo command_buffer_begin_info;
    command_buffer_begin_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.pNext = nullptr;
    command_buffer_begin_info.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    command_buffer_begin_info.pInheritanceInfo = nullptr;
    if (dfn.vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info) !=
        VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to begin recording the guest output "
          "capturing command buffer");
      dfn.vkDestroyCommandPool(device, command_pool, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    VkImageMemoryBarrier image_memory_barrier;
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.pNext = nullptr;
    image_memory_barrier.srcAccessMask = kGuestOutputInternalAccessMask;
    image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    image_memory_barrier.oldLayout = kGuestOutputInternalLayout;
    image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image_memory_barrier.image = guest_output_image->image();
    image_memory_barrier.subresourceRange = util::InitializeSubresourceRange();
    dfn.vkCmdPipelineBarrier(command_buffer, kGuestOutputInternalStageMask,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &image_memory_barrier);

    VkBufferImageCopy buffer_image_copy = {};
    buffer_image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    buffer_image_copy.imageSubresource.layerCount = 1;
    buffer_image_copy.imageExtent.width = image_extent.width;
    buffer_image_copy.imageExtent.height = image_extent.height;
    buffer_image_copy.imageExtent.depth = 1;
    dfn.vkCmdCopyImageToBuffer(command_buffer, guest_output_image->image(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                               &buffer_image_copy);

    // A fence doesn't guarantee host visibility and availability.
    VkBufferMemoryBarrier buffer_memory_barrier;
    buffer_memory_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    buffer_memory_barrier.pNext = nullptr;
    buffer_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    buffer_memory_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    buffer_memory_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_memory_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_memory_barrier.buffer = buffer;
    buffer_memory_barrier.offset = 0;
    buffer_memory_barrier.size = VK_WHOLE_SIZE;
    std::swap(image_memory_barrier.srcAccessMask,
              image_memory_barrier.dstAccessMask);
    std::swap(image_memory_barrier.oldLayout, image_memory_barrier.newLayout);
    dfn.vkCmdPipelineBarrier(
        command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | kGuestOutputInternalStageMask, 0, 0,
        nullptr, 1, &buffer_memory_barrier, 1, &image_memory_barrier);

    if (dfn.vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to end recording the guest output capturing "
          "command buffer");
      dfn.vkDestroyCommandPool(device, command_pool, nullptr);
      dfn.vkDestroyBuffer(device, buffer, nullptr);
      dfn.vkFreeMemory(device, buffer_memory, nullptr);
      return false;
    }

    {
      VulkanGPUCompletionTimeline completion_timeline(vulkan_device_,
                                                      "guest-capture");
      VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit_info.commandBufferCount = 1;
      submit_info.pCommandBuffers = &command_buffer;
      const VkResult submit_result = completion_timeline.AcquireFenceAndSubmit(
          vulkan_device_->queue_family_graphics_compute(), 0, 1, &submit_info);
      if (submit_result != VK_SUCCESS) {
        XELOGE(
            "VulkanPresenter: Failed to submit the guest output capturing "
            "command buffer: {}",
            vk::to_string(vk::Result(submit_result)));
        dfn.vkDestroyCommandPool(device, command_pool, nullptr);
        dfn.vkDestroyBuffer(device, buffer, nullptr);
        dfn.vkFreeMemory(device, buffer_memory, nullptr);
        return false;
      }
      // Destroying the completion timeline causes the submission to be awaited.
    }

    dfn.vkDestroyCommandPool(device, command_pool, nullptr);
  }

  // Don't need the buffer anymore, just its memory.
  dfn.vkDestroyBuffer(device, buffer, nullptr);

  void* mapping;
  if (dfn.vkMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, &mapping) !=
      VK_SUCCESS) {
    XELOGE("VulkanPresenter: Failed to map the guest output capture memory");
    dfn.vkFreeMemory(device, buffer_memory, nullptr);
    return false;
  }

  image_out.width = image_extent.width;
  image_out.height = image_extent.height;
  image_out.stride = sizeof(uint32_t) * image_extent.width;
  image_out.data.resize(size_t(buffer_size));
  uint32_t* image_out_pixels =
      reinterpret_cast<uint32_t*>(image_out.data.data());
  for (size_t i = 0; i < pixel_count; ++i) {
    image_out_pixels[i] = Packed10bpcRGBTo8bpcBytes(
        reinterpret_cast<const uint32_t*>(mapping)[i]);
  }

  // Unmapping will be done by freeing.
  dfn.vkFreeMemory(device, buffer_memory, nullptr);

  return true;
}

VkCommandBuffer VulkanPresenter::AcquireUISetupCommandBufferFromUIThread() {
  if (paint_context_.ui_setup_command_buffer_current_index != SIZE_MAX) {
    return paint_context_
        .ui_setup_command_buffers[paint_context_
                                      .ui_setup_command_buffer_current_index]
        .command_buffer;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = nullptr;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  command_buffer_begin_info.pInheritanceInfo = nullptr;

  // Try to reuse an existing command buffer.
  if (!paint_context_.ui_setup_command_buffers.empty()) {
    const uint64_t submission_index_completed =
        ui_completion_timeline_.UpdateAndGetCompletedSubmission();
    for (size_t i = 0; i < paint_context_.ui_setup_command_buffers.size();
         ++i) {
      PaintContext::UISetupCommandBuffer& ui_setup_command_buffer =
          paint_context_.ui_setup_command_buffers[i];
      if (ui_setup_command_buffer.last_usage_submission_index >
          submission_index_completed) {
        continue;
      }
      if (dfn.vkResetCommandPool(device, ui_setup_command_buffer.command_pool,
                                 0) != VK_SUCCESS) {
        XELOGE("VulkanPresenter: Failed to reset a UI setup command pool");
        return VK_NULL_HANDLE;
      }
      if (dfn.vkBeginCommandBuffer(ui_setup_command_buffer.command_buffer,
                                   &command_buffer_begin_info) != VK_SUCCESS) {
        XELOGE(
            "VulkanPresenter: Failed to begin UI setup command buffer "
            "recording");
        return VK_NULL_HANDLE;
      }
      paint_context_.ui_setup_command_buffer_current_index = i;
      ui_setup_command_buffer.last_usage_submission_index =
          ui_completion_timeline_.GetUpcomingSubmission();
      return ui_setup_command_buffer.command_buffer;
    }
  }

  // Create a new command buffer.
  VkCommandPoolCreateInfo command_pool_create_info;
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.pNext = nullptr;
  command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  command_pool_create_info.queueFamilyIndex =
      vulkan_device_->queue_family_graphics_compute();
  VkCommandPool new_command_pool;
  if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr,
                              &new_command_pool) != VK_SUCCESS) {
    XELOGE("VulkanPresenter: Failed to create a UI setup command pool");
    return VK_NULL_HANDLE;
  }
  VkCommandBufferAllocateInfo command_buffer_allocate_info;
  command_buffer_allocate_info.sType =
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_allocate_info.pNext = nullptr;
  command_buffer_allocate_info.commandPool = new_command_pool;
  command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_allocate_info.commandBufferCount = 1;
  VkCommandBuffer new_command_buffer;
  if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info,
                                   &new_command_buffer) != VK_SUCCESS) {
    XELOGE("VulkanPresenter: Failed to allocate a UI setup command buffer");
    dfn.vkDestroyCommandPool(device, new_command_pool, nullptr);
    return VK_NULL_HANDLE;
  }
  if (dfn.vkBeginCommandBuffer(new_command_buffer,
                               &command_buffer_begin_info) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to begin UI setup command buffer recording");
    dfn.vkDestroyCommandPool(device, new_command_pool, nullptr);
    return VK_NULL_HANDLE;
  }
  paint_context_.ui_setup_command_buffer_current_index =
      paint_context_.ui_setup_command_buffers.size();
  paint_context_.ui_setup_command_buffers.emplace_back(
      new_command_pool, new_command_buffer,
      ui_completion_timeline_.GetUpcomingSubmission());
  return new_command_buffer;
}

Presenter::SurfacePaintConnectResult
VulkanPresenter::ConnectOrReconnectPaintingToSurfaceFromUIThread(
    Surface& new_surface, uint32_t new_surface_width,
    uint32_t new_surface_height, bool was_paintable,
    bool& is_vsync_implicit_out) {
  xe::RequestFrameTelemetryReset();
  BeginZeroFGSurfaceDisconnect();
  guest_output_image_refresher_completion_timeline_.AwaitAllSubmissions();
  DestroyZeroFGSurfaceResourcesAfterSourceIdle();
  // Main Surface Authority: the ZeroFG teardown above destroyed B's swapchain.
  // If B still produced, A creating one now would make A && B: refuse.
  if (zerofg_independent_presenter_ &&
      !zerofg_independent_presenter_->MainSurfaceAllowsProducerA()) {
    XELOGE(
        "ZeroFGMainSurface dual_producer_refused: A swapchain requested while "
        "B still produces");
    return SurfacePaintConnectResult::kFailure;
  }
  const VulkanInstance* const vulkan_instance =
      vulkan_device_->vulkan_instance();
  const VulkanInstance::Functions& ifn = vulkan_instance->functions();
  const VkInstance instance = vulkan_instance->instance();
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkFormat new_swapchain_format;

  // ConnectOrReconnectToSurfaceFromUIThread may be called only for the
  // ui::Surface of the current swapchain or when the old swapchain and
  // VkSurface have, if the ui::Surface is the same, try using the existing
  // VkSurface and creating the swapchain smoothly from the existing one - if
  // this doesn't succeed, start from scratch.
  // The retirement or destruction of the swapchain here will also cause
  // awaiting completion of the usage of the swapchain and the surface on the
  // GPU.
  if (paint_context_.vulkan_surface != VK_NULL_HANDLE) {
#if XE_PLATFORM_MAC
    if (new_surface.GetType() == Surface::kTypeIndex_MacNSView) {
      auto& mac_nsview_surface =
          static_cast<const MacNSViewSurface&>(new_surface);
      const double contents_scale = cvars::vulkan_presenter_use_backing_scale
                                        ? mac_nsview_surface.GetBackingScale()
                                        : 1.0;
      mac_nsview_surface.ConfigureMetalLayer(
          new_surface_width, new_surface_height, contents_scale);
    }
#endif  // XE_PLATFORM_MAC
    VkSwapchainKHR old_swapchain =
        paint_context_.PrepareForSwapchainRetirement();
    bool surface_unusable;
    paint_context_.swapchain = PaintContext::CreateSwapchainForVulkanSurface(
        vulkan_device_, paint_context_.vulkan_surface, new_surface_width,
        new_surface_height, old_swapchain, paint_context_.present_queue_family,
        new_swapchain_format, paint_context_.swapchain_extent,
        paint_context_.swapchain_is_fifo,
        paint_context_.swapchain_requested_image_count,
        paint_context_.swapchain_present_mode, surface_unusable);
    // Destroy the old swapchain that may be retired now.
    if (old_swapchain != VK_NULL_HANDLE) {
      dfn.vkDestroySwapchainKHR(device, old_swapchain, nullptr);
    }
    if (paint_context_.swapchain == VK_NULL_HANDLE) {
      // Couldn't create the swapchain for the existing surface - start over.
      paint_context_.DestroySwapchainAndVulkanSurface();
    }
  }

  // If failed to create the swapchain for the previous surface, recreate the
  // surface and create the new swapchain.
  if (paint_context_.swapchain == VK_NULL_HANDLE) {
    // DestroySwapchainAndVulkanSurface should have been called previously.
    assert_true(paint_context_.vulkan_surface == VK_NULL_HANDLE);
    Surface::TypeIndex surface_type = new_surface.GetType();
    // Check if the surface type is supported according to the Vulkan
    // extensions.
    if (!(GetSupportedSurfaceTypes() &
          (Surface::TypeFlags(1) << surface_type))) {
      XELOGE(
          "VulkanPresenter: Tried to create a Vulkan surface for an "
          "unsupported Xenia surface type");
      return SurfacePaintConnectResult::kFailureSurfaceUnusable;
    }
    VkResult vulkan_surface_create_result = VK_ERROR_UNKNOWN;
    switch (surface_type) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
      case Surface::kTypeIndex_AndroidNativeWindow: {
        auto& android_native_window_surface =
            static_cast<const AndroidNativeWindowSurface&>(new_surface);
        VkAndroidSurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType =
            VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.window = android_native_window_surface.window();
        vulkan_surface_create_result = ifn.vkCreateAndroidSurfaceKHR(
            instance, &surface_create_info, nullptr,
            &paint_context_.vulkan_surface);
      } break;
#endif
#if XE_PLATFORM_GNU_LINUX
      case Surface::kTypeIndex_WaylandWindow: {
        auto& wayland_window_surface =
            static_cast<const WaylandWindowSurface&>(new_surface);
        VkWaylandSurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType =
            VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.display = wayland_window_surface.display();
        surface_create_info.surface = wayland_window_surface.surface();
        vulkan_surface_create_result = ifn.vkCreateWaylandSurfaceKHR(
            instance, &surface_create_info, nullptr,
            &paint_context_.vulkan_surface);
      } break;
      case Surface::kTypeIndex_XcbWindow: {
        auto& xcb_window_surface =
            static_cast<const XcbWindowSurface&>(new_surface);
        VkXcbSurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType =
            VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.connection = xcb_window_surface.connection();
        surface_create_info.window = xcb_window_surface.window();
        vulkan_surface_create_result =
            ifn.vkCreateXcbSurfaceKHR(instance, &surface_create_info, nullptr,
                                      &paint_context_.vulkan_surface);
      } break;
#endif
#if XE_PLATFORM_MAC
      case Surface::kTypeIndex_MacNSView: {
        auto& mac_nsview_surface =
            static_cast<const MacNSViewSurface&>(new_surface);
        const double contents_scale = cvars::vulkan_presenter_use_backing_scale
                                          ? mac_nsview_surface.GetBackingScale()
                                          : 1.0;
        mac_nsview_surface.ConfigureMetalLayer(
            new_surface_width, new_surface_height, contents_scale);
        CAMetalLayer* const metal_layer =
            mac_nsview_surface.GetOrCreateMetalLayer();
        if (!metal_layer) {
          XELOGE(
              "VulkanPresenter: Failed to create a CAMetalLayer for MoltenVK");
          return SurfacePaintConnectResult::kFailureSurfaceUnusable;
        }
        VkMetalSurfaceCreateInfoEXT surface_create_info;
        surface_create_info.sType =
            VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.pLayer = metal_layer;
        vulkan_surface_create_result =
            ifn.vkCreateMetalSurfaceEXT(instance, &surface_create_info, nullptr,
                                        &paint_context_.vulkan_surface);
      } break;
#endif  // XE_PLATFORM_MAC
#if XE_PLATFORM_WIN32
      case Surface::kTypeIndex_Win32Hwnd: {
        auto& win32_hwnd_surface =
            static_cast<const Win32HwndSurface&>(new_surface);
        VkWin32SurfaceCreateInfoKHR surface_create_info;
        surface_create_info.sType =
            VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        surface_create_info.pNext = nullptr;
        surface_create_info.flags = 0;
        surface_create_info.hinstance = win32_hwnd_surface.hinstance();
        surface_create_info.hwnd = win32_hwnd_surface.hwnd();
        vulkan_surface_create_result =
            ifn.vkCreateWin32SurfaceKHR(instance, &surface_create_info, nullptr,
                                        &paint_context_.vulkan_surface);
      } break;
#endif
      default:
        assert_unhandled_case(surface_type);
        XELOGE(
            "VulkanPresenter: Tried to create a Vulkan surface for an "
            "unknown Xenia surface type");
        return SurfacePaintConnectResult::kFailureSurfaceUnusable;
    }
    if (vulkan_surface_create_result != VK_SUCCESS) {
      XELOGE("VulkanPresenter: Failed to create a Vulkan surface");
      return SurfacePaintConnectResult::kFailure;
    }
    bool surface_unusable;
    paint_context_.swapchain = PaintContext::CreateSwapchainForVulkanSurface(
        vulkan_device_, paint_context_.vulkan_surface, new_surface_width,
        new_surface_height, VK_NULL_HANDLE, paint_context_.present_queue_family,
        new_swapchain_format, paint_context_.swapchain_extent,
        paint_context_.swapchain_is_fifo,
        paint_context_.swapchain_requested_image_count,
        paint_context_.swapchain_present_mode, surface_unusable);
    if (paint_context_.swapchain == VK_NULL_HANDLE) {
      // Failed to create the swapchain for the new Vulkan surface - destroy the
      // Vulkan surface.
      ifn.vkDestroySurfaceKHR(instance, paint_context_.vulkan_surface, nullptr);
      paint_context_.vulkan_surface = VK_NULL_HANDLE;
      return surface_unusable
                 ? SurfacePaintConnectResult::kFailureSurfaceUnusable
                 : SurfacePaintConnectResult::kFailure;
    }
    // Successfully attached (at least for now).
  }

  // From now on, in case of failure,
  // paint_context_.DestroySwapchainAndVulkanSurface must be called before
  // returning.

  // Update the render pass to the new format.
  if (paint_context_.swapchain_render_pass_format != new_swapchain_format) {
    util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device,
                               paint_context_.swapchain_render_pass);
    paint_context_.swapchain_render_pass_format = new_swapchain_format;
  }
  if (paint_context_.swapchain_render_pass == VK_NULL_HANDLE) {
    VkAttachmentDescription render_pass_attachment;
    render_pass_attachment.flags = 0;
    render_pass_attachment.format = new_swapchain_format;
    render_pass_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    render_pass_attachment.loadOp = cvars::present_render_pass_clear
                                        ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                        : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    render_pass_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    render_pass_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    render_pass_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    render_pass_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    render_pass_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference render_pass_color_attachment;
    render_pass_color_attachment.attachment = 0;
    render_pass_color_attachment.layout =
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription render_pass_subpass = {};
    render_pass_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    render_pass_subpass.colorAttachmentCount = 1;
    render_pass_subpass.pColorAttachments = &render_pass_color_attachment;
    VkSubpassDependency render_pass_dependencies[2];
    render_pass_dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    render_pass_dependencies[0].dstSubpass = 0;
    // srcStageMask is the semaphore wait stage.
    render_pass_dependencies[0].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    render_pass_dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    render_pass_dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    // The main target can be used for UI drawing at any moment, which requires
    // blending, so VK_ACCESS_COLOR_ATTACHMENT_READ_BIT is also included.
    render_pass_dependencies[0].dstAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    render_pass_dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    render_pass_dependencies[1].srcSubpass = 0;
    render_pass_dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    render_pass_dependencies[1].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Semaphores are signaled at VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
    render_pass_dependencies[1].dstStageMask =
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    render_pass_dependencies[1].srcAccessMask =
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    render_pass_dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    render_pass_dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    VkRenderPassCreateInfo render_pass_create_info;
    render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_create_info.pNext = nullptr;
    render_pass_create_info.flags = 0;
    render_pass_create_info.attachmentCount = 1;
    render_pass_create_info.pAttachments = &render_pass_attachment;
    render_pass_create_info.subpassCount = 1;
    render_pass_create_info.pSubpasses = &render_pass_subpass;
    render_pass_create_info.dependencyCount =
        uint32_t(xe::countof(render_pass_dependencies));
    render_pass_create_info.pDependencies = render_pass_dependencies;
    VkRenderPass new_render_pass;
    if (dfn.vkCreateRenderPass(device, &render_pass_create_info, nullptr,
                               &new_render_pass) != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to create the render pass for drawing to "
          "swapchain images");
      paint_context_.DestroySwapchainAndVulkanSurface();
      return SurfacePaintConnectResult::kFailure;
    }
    paint_context_.swapchain_render_pass = new_render_pass;
    paint_context_.swapchain_render_pass_format = new_swapchain_format;
    paint_context_.swapchain_render_pass_clear_load_op =
        render_pass_attachment.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
  }

  // Get the swapchain images.
  paint_context_.swapchain_images.clear();
  VkResult swapchain_images_get_result;
  for (;;) {
    uint32_t swapchain_image_count =
        uint32_t(paint_context_.swapchain_images.size());
    bool swapchain_images_were_empty = !swapchain_image_count;
    swapchain_images_get_result = dfn.vkGetSwapchainImagesKHR(
        device, paint_context_.swapchain, &swapchain_image_count,
        swapchain_images_were_empty ? nullptr
                                    : paint_context_.swapchain_images.data());
    // If the original swapchain image count was 0 (first call), SUCCESS is
    // returned, not INCOMPLETE.
    if (swapchain_images_get_result == VK_SUCCESS ||
        swapchain_images_get_result == VK_INCOMPLETE) {
      paint_context_.swapchain_images.resize(swapchain_image_count);
      if (swapchain_images_get_result == VK_SUCCESS &&
          (!swapchain_images_were_empty || !swapchain_image_count)) {
        break;
      }
    } else {
      break;
    }
  }
  if (swapchain_images_get_result != VK_SUCCESS) {
    XELOGE("VulkanPresenter: Failed to get swapchain images");
    paint_context_.DestroySwapchainAndVulkanSurface();
    return SurfacePaintConnectResult::kFailure;
  }

  // Create the image views and the framebuffers.
  assert_true(paint_context_.swapchain_framebuffers.empty());
  paint_context_.swapchain_framebuffers.reserve(
      paint_context_.swapchain_images.size());
  VkImageViewCreateInfo image_view_create_info;
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.pNext = nullptr;
  image_view_create_info.flags = 0;
  image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format = new_swapchain_format;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_COLOR_BIT;
  image_view_create_info.subresourceRange.baseMipLevel = 0;
  image_view_create_info.subresourceRange.levelCount = 1;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount = 1;
  VkImageView image_view;
  VkFramebufferCreateInfo framebuffer_create_info;
  framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_create_info.pNext = nullptr;
  framebuffer_create_info.flags = 0;
  framebuffer_create_info.renderPass = paint_context_.swapchain_render_pass;
  framebuffer_create_info.attachmentCount = 1;
  framebuffer_create_info.pAttachments = &image_view;
  framebuffer_create_info.width = paint_context_.swapchain_extent.width;
  framebuffer_create_info.height = paint_context_.swapchain_extent.height;
  framebuffer_create_info.layers = 1;
  for (VkImage image : paint_context_.swapchain_images) {
    image_view_create_info.image = image;
    if (dfn.vkCreateImageView(device, &image_view_create_info, nullptr,
                              &image_view) != VK_SUCCESS) {
      XELOGE("VulkanPresenter: Failed to create a swapchain image view");
      paint_context_.DestroySwapchainAndVulkanSurface();
      return SurfacePaintConnectResult::kFailure;
    }
    VkFramebuffer framebuffer;
    if (dfn.vkCreateFramebuffer(device, &framebuffer_create_info, nullptr,
                                &framebuffer) != VK_SUCCESS) {
      XELOGE("VulkanPresenter: Failed to create a swapchain framebuffer");
      dfn.vkDestroyImageView(device, image_view, nullptr);
      paint_context_.DestroySwapchainAndVulkanSurface();
      return SurfacePaintConnectResult::kFailure;
    }
    paint_context_.swapchain_framebuffers.emplace_back(image_view, framebuffer);
  }

  // Create per-swapchain-image present semaphores to avoid
  // VUID-vkQueueSubmit-pSignalSemaphores-00067 (semaphore reuse before the
  // previous present completes).
  paint_context_.swapchain_image_present_semaphores.reserve(
      paint_context_.swapchain_images.size());
  VkSemaphoreCreateInfo present_semaphore_create_info;
  present_semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  present_semaphore_create_info.pNext = nullptr;
  present_semaphore_create_info.flags = 0;
  for (size_t i = 0; i < paint_context_.swapchain_images.size(); ++i) {
    VkSemaphore present_semaphore;
    if (dfn.vkCreateSemaphore(device, &present_semaphore_create_info, nullptr,
                              &present_semaphore) != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to create a per-swapchain-image present "
          "semaphore");
      paint_context_.DestroySwapchainAndVulkanSurface();
      return SurfacePaintConnectResult::kFailure;
    }
    paint_context_.swapchain_image_present_semaphores.push_back(
        present_semaphore);
  }

  is_vsync_implicit_out = paint_context_.swapchain_is_fifo;
  if (zerofg_independent_presenter_) {
    // A's swapchain exists again: A owns the Surface (handback complete).
    zerofg_independent_presenter_->NoteMainSurfaceProducedByA();
  }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (zerofg_independent_presenter_ &&
      new_surface.GetType() == Surface::kTypeIndex_AndroidNativeWindow) {
    auto& android_native_window_surface =
        static_cast<const AndroidNativeWindowSurface&>(new_surface);
    bool generation_context_ready = true;
    if (IsZeroFGRequested()) {
      ZeroFGIndependentPresenter::GenerationResult prepare_result;
      uint64_t prepare_ns = 0;
      bool created = false;
      generation_context_ready = PrepareZeroFGGenerationContext(
          prepare_result, prepare_ns, created);
      if (!generation_context_ready) {
        XELOGW(
            "ZeroFGC0: generation prewarm failed stage={} "
            "status={} vk_result={} valid={}; preserving normal presentation",
            uint32_t(prepare_result.failure_stage),
            prepare_result.zerofg_status, int32_t(prepare_result.vk_result),
            prepare_result.vk_result_valid);
      } else if (created) {
        XELOGI("ZeroFGC0: generation pipelines prewarmed in {} us",
               prepare_ns / 1000);
      }
    }
    // Main Surface Authority borrows A's VkSurfaceKHR.
    const bool connected = generation_context_ready &&
                           zerofg_independent_presenter_->Connect(
                               android_native_window_surface.window(),
                               paint_context_.swapchain_extent,
                               paint_context_.swapchain_render_pass_format,
                               paint_context_.vulkan_surface);
    if (generation_context_ready && !connected) {
      XELOGW("ZeroFGC0: independent surface unavailable; preserving the "
             "normal XenDroid presenter");
    }
    if (zerofg_vulkan_context_->handoff && connected) {
      zerofg_vulkan_context_->handoff->Arm();
    }
  }
#endif
  return SurfacePaintConnectResult::kSuccess;
}

void VulkanPresenter::DisconnectPaintingFromSurfaceFromUIThreadImpl() {
  xe::RequestFrameTelemetryReset();
  BeginZeroFGSurfaceDisconnect();
  guest_output_image_refresher_completion_timeline_.AwaitAllSubmissions();
  DestroyZeroFGSurfaceResourcesAfterSourceIdle();
  paint_context_.DestroySwapchainAndVulkanSurface();
  if (zerofg_independent_presenter_) {
    // B's swapchain went with the ZeroFG teardown: the Surface lifecycle is
    // A's again.
    zerofg_independent_presenter_->NoteMainSurfaceProducedByA();
  }
}

bool VulkanPresenter::RefreshGuestOutputImpl(
    uint32_t mailbox_index, uint32_t frontbuffer_width,
    uint32_t frontbuffer_height,
    std::function<bool(GuestOutputRefreshContext& context)> refresher,
    bool& is_8bpc_out_ref) {
  assert_not_zero(frontbuffer_width);
  assert_not_zero(frontbuffer_height);
  VkExtent2D max_framebuffer_extent =
      util::GetMax2DFramebufferExtent(vulkan_device_->properties());
  if (frontbuffer_width > max_framebuffer_extent.width ||
      frontbuffer_height > max_framebuffer_extent.height) {
    // Writing the guest output isn't supposed to rescale, and a guest texture
    // exceeding the maximum size won't be loadable anyway.
#if XE_PLATFORM_xendroid
    XELOGE(
        "VulkanPresenter: guest output {}x{} exceeds the maximum framebuffer "
        "extent {}x{} - dropping the guest frame",
        frontbuffer_width, frontbuffer_height, max_framebuffer_extent.width,
        max_framebuffer_extent.height);
#endif
    return false;
  }

  GuestOutputImageInstance& image_instance =
      guest_output_images_[mailbox_index];
  if (image_instance.image &&
      (image_instance.image->extent().width != frontbuffer_width ||
       image_instance.image->extent().height != frontbuffer_height)) {
    guest_output_image_refresher_completion_timeline_
        .AwaitSubmissionAndUpdateCompleted(
            image_instance.last_refresher_submission);
    image_instance.image.reset();
  }
  if (!image_instance.image) {
    std::unique_ptr<GuestOutputImage> new_image = GuestOutputImage::Create(
        vulkan_device_, frontbuffer_width, frontbuffer_height);
    if (!new_image) {
      return false;
    }
    image_instance.SetToNewImage(std::move(new_image),
                                 guest_output_image_next_version_++);
  }

  VulkanGuestOutputRefreshContext context(
      is_8bpc_out_ref, image_instance.image->image(),
      image_instance.image->view(), image_instance.version,
      image_instance.ever_successfully_refreshed);
  bool refresher_succeeded = refresher(context);
  if (refresher_succeeded) {
    image_instance.ever_successfully_refreshed = true;
  }
  // Even if the refresher has returned false, it still might have submitted
  // some commands referencing the image. It's better to put an excessive
  // signal and wait slightly longer, for nothing important, while shutting down
  // than to destroy the image while it's still in use.
  image_instance.last_refresher_submission =
      guest_output_image_refresher_completion_timeline_.GetUpcomingSubmission();
  // No need to make the refresher signal the fence by itself - signal it here
  // instead to have more control:
  // "Fence signal operations that are defined by vkQueueSubmit additionally
  //  include in the first synchronization scope all commands that occur earlier
  //  in submission order."
  const VkSubmitInfo* handoff_submit = nullptr;
  if (zerofg_vulkan_context_->handoff) {
    if (refresher_succeeded && zerofg_independent_presenter_ &&
        zerofg_independent_presenter_->accepting()) {
      handoff_submit = zerofg_vulkan_context_->handoff->PrepareSource(
          mailbox_index, image_instance.image->image(),
          image_instance.image->extent());
    }
  }
  uint64_t handoff_queue_ns = 0, handoff_submit_ns = 0;
  const VkResult submit_result =
      guest_output_image_refresher_completion_timeline_.AcquireFenceAndSubmit(
          vulkan_device_->queue_family_graphics_compute(), 0,
          handoff_submit ? 1 : 0, handoff_submit,
          handoff_submit ? &handoff_queue_ns : nullptr,
          handoff_submit ? &handoff_submit_ns : nullptr);
  if (handoff_submit) {
    // AcquireFenceAndSubmit has released A's queue lock before fd export.
    zerofg_vulkan_context_->handoff->SourceSubmitted(
        submit_result, handoff_queue_ns, handoff_submit_ns);
  }
  if (submit_result != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to submit the guest output image refresh "
        "fence signal: {}",
        vk::to_string(vk::Result(submit_result)));
  }
  return refresher_succeeded;
}

VkSwapchainKHR VulkanPresenter::PaintContext::CreateSwapchainForVulkanSurface(
    const VulkanDevice* vulkan_device, VkSurfaceKHR surface, uint32_t width,
    uint32_t height, VkSwapchainKHR old_swapchain,
    uint32_t& present_queue_family_out, VkFormat& image_format_out,
    VkExtent2D& image_extent_out, bool& is_fifo_out,
    uint32_t& requested_image_count_out, VkPresentModeKHR& present_mode_out,
    bool& ui_surface_unusable_out) {
  ui_surface_unusable_out = false;

  const VulkanInstance::Functions& ifn =
      vulkan_device->vulkan_instance()->functions();
  const VkPhysicalDevice physical_device = vulkan_device->physical_device();
  const VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Get surface capabilities.
  VkSurfaceCapabilitiesKHR surface_capabilities;
  if (ifn.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
          physical_device, surface, &surface_capabilities) != VK_SUCCESS) {
    XELOGE("VulkanPresenter: Failed to get Vulkan surface capabilities");
    // Some strange error, try again later.
    return VK_NULL_HANDLE;
  }

  // First, check if the surface is not zero-area because in this case, the rest
  // of the fields in theory may not be informative as the surface doesn't need
  // to go to presentation anyway, thus there's no need to return real
  // information, and ui_surface_unusable_out (from which it may not be possible
  // to recover on certain platforms at all) may be set to true spuriously if
  // any checks set it to true. VkSurfaceKHR may have zero-area bounds in some
  // window state cases on Windows, for example. Also clamp the requested size
  // to the maximum supported by the physical device as long as minImageExtent
  // in the instance's WSI allows that (if not, there's no way to satisfy both
  // requirements - the maximum 2D framebuffer size on the specific physical
  // device, and the minimum swap chain size on the whole instance - fail to
  // create until the surface becomes smaller).
  VkExtent2D max_framebuffer_extent =
      util::GetMax2DFramebufferExtent(vulkan_device->properties());
  VkExtent2D image_extent;
  image_extent.width =
      std::min(std::max(std::min(width, max_framebuffer_extent.width),
                        surface_capabilities.minImageExtent.width),
               surface_capabilities.maxImageExtent.width);
  image_extent.height =
      std::min(std::max(std::min(height, max_framebuffer_extent.height),
                        surface_capabilities.minImageExtent.height),
               surface_capabilities.maxImageExtent.height);
  if (!image_extent.width || !image_extent.height ||
      image_extent.width > max_framebuffer_extent.width ||
      image_extent.height > max_framebuffer_extent.height) {
    return VK_NULL_HANDLE;
  }

  // Get the queue family for presentation.
  uint32_t queue_family_index_present = UINT32_MAX;
  const std::vector<VulkanDevice::QueueFamily>& queue_families =
      vulkan_device->queue_families();
  VkBool32 queue_family_present_supported;
  // First try the graphics and compute queue, prefer it to avoid the concurrent
  // image sharing mode.
  uint32_t queue_family_index_graphics_compute =
      vulkan_device->queue_family_graphics_compute();
  const VulkanDevice::QueueFamily& queue_family_graphics_compute =
      queue_families[queue_family_index_graphics_compute];
  if (queue_family_graphics_compute.may_support_presentation &&
      !queue_family_graphics_compute.queues.empty() &&
      ifn.vkGetPhysicalDeviceSurfaceSupportKHR(
          physical_device, queue_family_index_graphics_compute, surface,
          &queue_family_present_supported) == VK_SUCCESS &&
      queue_family_present_supported) {
    queue_family_index_present = queue_family_index_graphics_compute;
  } else {
    for (uint32_t i = 0; i < uint32_t(queue_families.size()); ++i) {
      const VulkanDevice::QueueFamily& queue_family = queue_families[i];
      if (!queue_family.queues.empty() &&
          queue_family.may_support_presentation &&
          ifn.vkGetPhysicalDeviceSurfaceSupportKHR(
              physical_device, i, surface, &queue_family_present_supported) ==
              VK_SUCCESS &&
          queue_family_present_supported) {
        queue_family_index_present = i;
        break;
      }
    }
  }
  if (queue_family_index_present == UINT32_MAX) {
    // Not unusable though - may become presentable if the window (with the same
    // surface) moved to a different display, for instance.
    return VK_NULL_HANDLE;
  }

  // TODO(Triang3l): Support transforms.
  if (!(surface_capabilities.supportedTransforms &
        (VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR |
         VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR))) {
    XELOGE(
        "VulkanPresenter: The surface doesn't support identity or "
        "window-system-controlled transform");
    return VK_NULL_HANDLE;
  }

  // Get the surface format.
  std::vector<VkSurfaceFormatKHR> surface_formats;
  VkResult surface_formats_get_result;
  for (;;) {
    uint32_t surface_format_count = uint32_t(surface_formats.size());
    bool surface_formats_were_empty = !surface_format_count;
    surface_formats_get_result = ifn.vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device, surface, &surface_format_count,
        surface_formats_were_empty ? nullptr : surface_formats.data());
    // If the original presentation mode count was 0 (first call), SUCCESS is
    // returned, not INCOMPLETE.
    if (surface_formats_get_result == VK_SUCCESS ||
        surface_formats_get_result == VK_INCOMPLETE) {
      surface_formats.resize(surface_format_count);
      if (surface_formats_get_result == VK_SUCCESS &&
          (!surface_formats_were_empty || !surface_format_count)) {
        break;
      }
    } else {
      break;
    }
  }
  if (surface_formats_get_result != VK_SUCCESS) {
    // Assuming any format in case of an error (or as some fallback in case of
    // specification violation).
    surface_formats.clear();
  }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  // Android uses R8G8B8A8.
  static constexpr VkFormat kFormat8888Primary = VK_FORMAT_R8G8B8A8_UNORM;
  static constexpr VkFormat kFormat8888Secondary = VK_FORMAT_B8G8R8A8_UNORM;
#else
  // GNU/Linux X11 and Windows DWM use B8G8R8A8.
  static constexpr VkFormat kFormat8888Primary = VK_FORMAT_B8G8R8A8_UNORM;
  static constexpr VkFormat kFormat8888Secondary = VK_FORMAT_R8G8B8A8_UNORM;
#endif
  VkSurfaceFormatKHR image_format;
  if (surface_formats.empty() ||
      (surface_formats.size() == 1 ||
       surface_formats[0].format == VK_FORMAT_UNDEFINED)) {
    // Can choose any format if the implementation specifies only UNDEFINED.
    image_format.format = kFormat8888Primary;
    image_format.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  } else {
    // Pick the sRGB 8888 format preferred by the OS, fall back to any sRGB
    // 8888, then to 8888 with an unknown color space, and then to the first
    // sRGB available, and then to the first.
    auto format_8888_primary_it = surface_formats.cend();
    auto format_8888_secondary_it = surface_formats.cend();
    auto any_non_8888_srgb_it = surface_formats.cend();
    for (auto it = surface_formats.cbegin(); it != surface_formats.cend();
         ++it) {
      if (it->format != kFormat8888Primary &&
          it->format != kFormat8888Secondary) {
        if (it->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
            any_non_8888_srgb_it == surface_formats.cend()) {
          any_non_8888_srgb_it = it;
        }
        continue;
      }
      auto& preferred_8888_it = it->format == kFormat8888Primary
                                    ? format_8888_primary_it
                                    : format_8888_secondary_it;
      if (preferred_8888_it == surface_formats.cend()) {
        // First primary or secondary 8888 encounter.
        preferred_8888_it = it;
        continue;
      }
      // Is this a better primary or secondary 8888, that is, this is sRGB,
      // while the previous encounter was not?
      if (it->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR &&
          preferred_8888_it->colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        preferred_8888_it = it;
      }
    }
    if (format_8888_primary_it != surface_formats.cend() &&
        format_8888_secondary_it != surface_formats.cend()) {
      // Both the primary and the secondary 8888 formats are available - prefer
      // sRGB, if both are sRGB or not, prefer the primary format.
      if (format_8888_primary_it->colorSpace ==
              VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
          format_8888_secondary_it->colorSpace !=
              VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        image_format = *format_8888_primary_it;
      } else {
        image_format = *format_8888_secondary_it;
      }
    } else if (format_8888_primary_it != surface_formats.cend()) {
      // Only primary 8888.
      image_format = *format_8888_primary_it;
    } else if (format_8888_secondary_it != surface_formats.cend()) {
      // Only secondary 8888.
      image_format = *format_8888_secondary_it;
    } else if (any_non_8888_srgb_it != surface_formats.cend()) {
      // No 8888, but some sRGB format is available.
      image_format = *any_non_8888_srgb_it;
    } else {
      // Just pick any format.
      image_format = surface_formats.front();
    }
  }

  // Get presentation modes.
  std::vector<VkPresentModeKHR> present_modes;
  VkResult present_modes_get_result;
  for (;;) {
    uint32_t present_mode_count = uint32_t(present_modes.size());
    bool present_modes_were_empty = !present_mode_count;
    present_modes_get_result = ifn.vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &present_mode_count,
        present_modes_were_empty ? nullptr : present_modes.data());
    // If the original presentation mode count was 0 (first call), SUCCESS is
    // returned, not INCOMPLETE.
    if (present_modes_get_result == VK_SUCCESS ||
        present_modes_get_result == VK_INCOMPLETE) {
      present_modes.resize(present_mode_count);
      if (present_modes_get_result == VK_SUCCESS &&
          (!present_modes_were_empty || !present_mode_count)) {
        break;
      }
    } else {
      break;
    }
  }
  if (present_modes_get_result != VK_SUCCESS) {
    // Assuming FIFO only (required everywhere) in case of an error.
    present_modes.clear();
  }

  // Create the swapchain.
  VkSwapchainCreateInfoKHR swapchain_create_info;
  swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchain_create_info.pNext = nullptr;

#if XE_PLATFORM_WIN32
  // On Windows, use VK_EXT_full_screen_exclusive to explicitly disallow
  // fullscreen exclusive mode. This prevents HDR state corruption when
  // entering/exiting fullscreen, as the Windows compositor remains in control
  // of the display state throughout the transition.
  VkSurfaceFullScreenExclusiveInfoEXT full_screen_exclusive_info;
  if (vulkan_device->extensions().ext_EXT_full_screen_exclusive) {
    full_screen_exclusive_info.sType =
        VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
    full_screen_exclusive_info.pNext = nullptr;
    full_screen_exclusive_info.fullScreenExclusive =
        VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT;
    swapchain_create_info.pNext = &full_screen_exclusive_info;
  }
#endif
  swapchain_create_info.flags = 0;
  swapchain_create_info.surface = surface;
  swapchain_create_info.minImageCount =
      std::max(kSubmissionCount, surface_capabilities.minImageCount);
  if (surface_capabilities.maxImageCount) {
    swapchain_create_info.minImageCount =
        std::min(swapchain_create_info.minImageCount,
                 surface_capabilities.maxImageCount);
  }
  swapchain_create_info.imageFormat = image_format.format;
  swapchain_create_info.imageColorSpace = image_format.colorSpace;
  swapchain_create_info.imageExtent = image_extent;
  swapchain_create_info.imageArrayLayers = 1;
  swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  uint32_t swapchain_queue_family_indices[2];
  if (queue_family_index_graphics_compute != queue_family_index_present) {
    // Using concurrent sharing mode to avoid an explicit ownership transfer
    // before presenting, which would require an additional command buffer
    // submission with the acquire barrier and a semaphore between the graphics
    // queue and the present queue. Different queues are an extremely rare case,
    // and Xenia only uses the swapchain for the final guest output and the
    // internal UI, so keeping framebuffer compression is not worth the
    // additional submission complexity and possibly latency.
    swapchain_queue_family_indices[0] = queue_family_index_graphics_compute;
    swapchain_queue_family_indices[1] = queue_family_index_present;
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    swapchain_create_info.queueFamilyIndexCount = 2;
    swapchain_create_info.pQueueFamilyIndices = swapchain_queue_family_indices;
  } else {
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_create_info.queueFamilyIndexCount = 0;
    swapchain_create_info.pQueueFamilyIndices = nullptr;
  }
  swapchain_create_info.preTransform =
      (surface_capabilities.supportedTransforms &
       VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
          ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
          : VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR;
  // Prefer opaque to avoid blending in the window system, or let that be
  // specified via the window system if it can't be forced. As a last resort,
  // just pick any - guest output will write alpha of 1 anyway.
  if (surface_capabilities.supportedCompositeAlpha &
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  } else if (surface_capabilities.supportedCompositeAlpha &
             VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  } else {
    uint32_t composite_alpha_shift;
    if (!xe::bit_scan_forward(
            uint32_t(surface_capabilities.supportedCompositeAlpha),
            &composite_alpha_shift)) {
      // Against the Vulkan specification, but breaks the logic here.
      XELOGE(
          "VulkanPresenter: The surface doesn't support any composite alpha "
          "mode");
      return VK_NULL_HANDLE;
    }
    swapchain_create_info.compositeAlpha =
        VkCompositeAlphaFlagBitsKHR(uint32_t(1) << composite_alpha_shift);
  }
  // As presentation is usually controlled by the GPU command processor, it's
  // better to use modes that allow as quick acquisition as possible to avoid
  // interfering with GPU command processing, and also to allow tearing so
  // variable refresh rate may be used where it's available.
  // Note: If the priorities here are changes, update the cvar descriptions.
  if (cvars::vulkan_allow_present_mode_immediate &&
      std::find(present_modes.cbegin(), present_modes.cend(),
                VK_PRESENT_MODE_IMMEDIATE_KHR) != present_modes.cend()) {
    // Allowing tearing to reduce latency, and possibly variable refresh rate
    // (though on Windows with borderless fullscreen, GDI copying is used
    // instead of independent flip, so it's not supported there).
    swapchain_create_info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
  } else if (cvars::vulkan_allow_present_mode_mailbox &&
             std::find(present_modes.cbegin(), present_modes.cend(),
                       VK_PRESENT_MODE_MAILBOX_KHR) != present_modes.cend()) {
    // Allowing dropping frames to reduce latency, but no tearing.
    swapchain_create_info.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
  } else if (cvars::vulkan_allow_present_mode_fifo_relaxed &&
             std::find(present_modes.cbegin(), present_modes.cend(),
                       VK_PRESENT_MODE_FIFO_RELAXED_KHR) !=
                 present_modes.cend()) {
    // Limiting the frame rate, but lets too long frames cause tearing not to
    // make the latency even worse.
    swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
  } else {
    // Highest latency (but always guaranteed to be available).
    swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  }
  swapchain_create_info.clipped = VK_TRUE;
  swapchain_create_info.oldSwapchain = old_swapchain;
  VkSwapchainKHR swapchain;
  VkResult swapchain_create_result = dfn.vkCreateSwapchainKHR(
      device, &swapchain_create_info, nullptr, &swapchain);
  if (swapchain_create_result != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create a swapchain (VkResult {}, "
        "extent {}x{}, format {}, color space {}, present mode {}, "
        "minImageCount {})",
        int32_t(swapchain_create_result),
        swapchain_create_info.imageExtent.width,
        swapchain_create_info.imageExtent.height,
        uint32_t(swapchain_create_info.imageFormat),
        uint32_t(swapchain_create_info.imageColorSpace),
        uint32_t(swapchain_create_info.presentMode),
        swapchain_create_info.minImageCount);
    return VK_NULL_HANDLE;
  }
  requested_image_count_out = swapchain_create_info.minImageCount;
  present_mode_out = swapchain_create_info.presentMode;
  XELOGI(
      "VulkanPresenter: Created {}x{} swapchain with format {}, color space "
      "{}, presentation mode {}",
      swapchain_create_info.imageExtent.width,
      swapchain_create_info.imageExtent.height,
      uint32_t(swapchain_create_info.imageFormat),
      uint32_t(swapchain_create_info.imageColorSpace),
      uint32_t(swapchain_create_info.presentMode));

  present_queue_family_out = queue_family_index_present;
  image_format_out = swapchain_create_info.imageFormat;
  image_extent_out = swapchain_create_info.imageExtent;
  is_fifo_out =
      swapchain_create_info.presentMode == VK_PRESENT_MODE_FIFO_KHR ||
      swapchain_create_info.presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
  return swapchain;
}

VkSwapchainKHR VulkanPresenter::PaintContext::PrepareForSwapchainRetirement() {
  if (swapchain != VK_NULL_HANDLE) {
    completion_timeline.AwaitAllSubmissions();
    // Also wait for the presentation queue since vkQueuePresentKHR doesn't
    // signal a fence, and the present semaphores may still be in use.
    if (present_queue_family != UINT32_MAX) {
      const VulkanDevice::Queue::Acquisition queue_acquisition =
          vulkan_device->AcquireQueue(present_queue_family, 0);
      vulkan_device->functions().vkQueueWaitIdle(queue_acquisition.queue());
    }
  }
  const VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (const SwapchainFramebuffer& framebuffer : swapchain_framebuffers) {
    dfn.vkDestroyFramebuffer(device, framebuffer.framebuffer, nullptr);
    dfn.vkDestroyImageView(device, framebuffer.image_view, nullptr);
  }
  swapchain_framebuffers.clear();
  for (VkSemaphore present_semaphore : swapchain_image_present_semaphores) {
    dfn.vkDestroySemaphore(device, present_semaphore, nullptr);
  }
  swapchain_image_present_semaphores.clear();
  swapchain_images.clear();
  swapchain_extent.width = 0;
  swapchain_extent.height = 0;
  // The old swapchain must be destroyed externally.
  VkSwapchainKHR old_swapchain = swapchain;
  swapchain = nullptr;
  return old_swapchain;
}

void VulkanPresenter::PaintContext::DestroySwapchainAndVulkanSurface() {
  VkSwapchainKHR old_swapchain = PrepareForSwapchainRetirement();
  if (old_swapchain != VK_NULL_HANDLE) {
    vulkan_device->functions().vkDestroySwapchainKHR(vulkan_device->device(),
                                                     old_swapchain, nullptr);
  }
  present_queue_family = UINT32_MAX;
  if (vulkan_surface != VK_NULL_HANDLE) {
    const VulkanInstance* vulkan_instance = vulkan_device->vulkan_instance();
    vulkan_instance->functions().vkDestroySurfaceKHR(
        vulkan_instance->instance(), vulkan_surface, nullptr);
    vulkan_surface = VK_NULL_HANDLE;
  }
}


VulkanPresenter::GuestOutputImage::~GuestOutputImage() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();
  if (view_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, view_, nullptr);
  }
  if (image_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImage(device, image_, nullptr);
  }
  if (memory_ != VK_NULL_HANDLE) {
    dfn.vkFreeMemory(device, memory_, nullptr);
  }
}

bool VulkanPresenter::GuestOutputImage::Initialize() {
  VkImageCreateInfo image_create_info;
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.pNext = nullptr;
  image_create_info.flags = 0;
  image_create_info.imageType = VK_IMAGE_TYPE_2D;
  image_create_info.format = kGuestOutputFormat;
  image_create_info.extent.width = extent_.width;
  image_create_info.extent.height = extent_.height;
  image_create_info.extent.depth = 1;
  image_create_info.mipLevels = 1;
  image_create_info.arrayLayers = 1;
  image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
      additional_usage_;
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.queueFamilyIndexCount = 0;
  image_create_info.pQueueFamilyIndices = nullptr;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          vulkan_device_, image_create_info,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal, image_, memory_)) {
    XELOGE("VulkanPresenter: Failed to create a guest output image");
    return false;
  }

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkImageViewCreateInfo image_view_create_info;
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.pNext = nullptr;
  image_view_create_info.flags = 0;
  image_view_create_info.image = image_;
  image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format = kGuestOutputFormat;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_COLOR_BIT;
  image_view_create_info.subresourceRange.baseMipLevel = 0;
  image_view_create_info.subresourceRange.levelCount = 1;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount = 1;
  if (dfn.vkCreateImageView(device, &image_view_create_info, nullptr, &view_) !=
      VK_SUCCESS) {
    XELOGE("VulkanPresenter: Failed to create a guest output image view");
    return false;
  }

  return true;
}

struct VulkanPresenter::ZeroFGPostContext {
  struct JobContext {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMaxGuestOutputPaintEffects> descriptor_sets = {};
    std::array<std::unique_ptr<GuestOutputImage>,
               kMaxGuestOutputPaintEffects - 1>
        intermediate_images;
    std::array<VkFramebuffer, kMaxGuestOutputPaintEffects - 1>
        intermediate_framebuffers = {};
    std::unique_ptr<GuestOutputImage> logical_input;
    bool logical_input_ever_written = false;
    VkFramebuffer final_framebuffer = VK_NULL_HANDLE;
    VkImageView final_view = VK_NULL_HANDLE;
    // Binary semaphore whose pending signal is exported as an Android
    // sync_file, the presenter's completion authority for this Post. One
    // semaphore is tied to each physical FinalOutput job context and is reused
    // only after the presenter retires its fd.
    VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
    // Completion owner of this FinalOutput slot. Its timeline exists only when
    // The effective physical profile keeps D2 per-slot completion on. Its
    // previous point belongs to the slot's previous Post, which the presenter
    // observed complete before releasing the slot, so cleanup on signal finds
    // that point retired instead of waiting on pending GPU work.
    ZeroFGCompletionOwner completion;
    uint64_t signal_value = 0;
    uint64_t submit_time_ns = 0;
    bool submitted = false;
  };

  std::array<JobContext,
             ZeroFGIndependentPresenter::kFinalOutputPoolSize>
      jobs;
  VkSemaphore completion_timeline = VK_NULL_HANDLE;
  uint64_t next_timeline_value = 1;
  // Posts signal their job's own timeline, created with the job.
  // Completion/ownership falsifiers of the Post path, reported with every
  // Post result.
  ZeroFGCompletionOwnerStats owner_stats;
  VkRenderPass final_render_pass = VK_NULL_HANDLE;
  VkFormat final_format = VK_FORMAT_UNDEFINED;
  std::array<VkPipeline, size_t(GuestOutputPaintEffect::kCount)>
      final_pipelines = {};
};

struct VulkanPresenter::ZeroFGGenerationContext {
  static constexpr uint32_t kContextCount =
      ZeroFGIndependentPresenter::kSyntheticPoolSize;
  struct CommandContext {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    uint64_t signal_value = 0;
    uint64_t submit_time_ns = 0;
    uint64_t submit_call_ns = 0;
    bool profiling_sample_pending = false;
    bool submitted = false;
    // Completion owner of this Generation context.
    // - Timeline (effective V2-1 per-context profile): its
    //   previous point belongs to this context's previous Generation, which
    //   retired before ReleaseZeroFGSynthetic handed the context back.
    // - sync_fd (effective D3 profile): signaled by the
    //   same submit. An orphaned or dropped Generation is observed with a
    //   non-waiting poll instead of a timeline counter query, which waits on
    //   pending GPU work on Turnip/KGSL. Export failure and reuse follow the
    //   capture sync_fd (C2).
    ZeroFGCompletionOwner completion;
  };

  std::unique_ptr<zerofg::xenia::Adapter> adapter;
  std::array<CommandContext, kContextCount> commands = {};
  std::array<std::unique_ptr<GuestOutputImage>, kContextCount> outputs;
  std::array<bool, kContextCount> output_initialized = {};
  std::array<bool, kContextCount> output_busy = {};
  VkSemaphore completion_timeline = VK_NULL_HANDLE;
  VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
  uint32_t timestamp_valid_bits = 0;
  double timestamp_period_ns = 0.0;
  uint32_t timestamp_query_count_per_context = 0;
  uint64_t next_timeline_value = 1;
  VkExtent2D extent = {};
  uint32_t pool_high_water = 0;
  // D3 sync_fd observation (Android only). Generations signal their command
  // context's own timeline, or completion_timeline if it failed to create.
  bool sync_fd_observation = false;
  // Completion/ownership falsifiers of the Generation path, reported with
  // every Generation result.
  ZeroFGCompletionOwnerStats owner_stats;
};

#include "xenia/ui/vulkan/zerofg_device_handoff.inc"
#include "xenia/ui/vulkan/vulkan_presenter_zerofg_device_context.inc"
#include "xenia/ui/vulkan/vulkan_presenter_zerofg_source_adapter.inc"

void VulkanPresenter::RecordNativePresentHostOperation(
    NativePresentHostOperationStats& stats, const uint64_t duration_ns) {
  ++stats.calls;
  stats.total_ns += duration_ns;
  stats.max_ns = std::max(stats.max_ns, duration_ns);
  stats.over_1ms += duration_ns >= 1000000ull;
  stats.over_4ms += duration_ns >= 4000000ull;
}

void VulkanPresenter::MaybeLogNativePresentHostTelemetry() {
  if (!cvars::vulkan_completion_wait_telemetry) {
    return;
  }
  const uint64_t now_ns = GetZeroFGMonotonicTimeNs();
  if (!native_present_host_telemetry_.interval_start_ns) {
    native_present_host_telemetry_.interval_start_ns = now_ns;
    return;
  }
  if (now_ns - native_present_host_telemetry_.interval_start_ns <
      kNativePresentTelemetryIntervalNs) {
    return;
  }
  const auto log_operation = [](const char* name,
                                const NativePresentHostOperationStats& stats) {
    XELOGI(
        "VulkanNativePresentHost operation={} calls={} total/avg/max_us="
        "{}/{}/{} over_1ms/4ms={}/{}",
        name, stats.calls, stats.total_ns / 1000,
        stats.calls ? stats.total_ns / stats.calls / 1000 : 0,
        stats.max_ns / 1000, stats.over_1ms, stats.over_4ms);
  };
  log_operation("completion", native_present_host_telemetry_.completion);
  log_operation("acquire", native_present_host_telemetry_.acquire);
  log_operation("submit", native_present_host_telemetry_.submit);
  log_operation("present_queue_lock",
                native_present_host_telemetry_.present_queue_lock);
  log_operation("present", native_present_host_telemetry_.present);
  native_present_host_telemetry_ = NativePresentHostTelemetry{};
  native_present_host_telemetry_.interval_start_ns = now_ns;
}

void VulkanPresenter::RetireSwapchainForMainSurfaceAuthority() {
  if (paint_context_.swapchain == VK_NULL_HANDLE) {
    return;
  }
  // Waits for A's paint submissions and its present queue, as every swapchain
  // retirement does, then destroys A's swapchain. The VkSurfaceKHR stays: B
  // creates its own swapchain on it once the producer state leaves A.
  VkSwapchainKHR old_swapchain = paint_context_.PrepareForSwapchainRetirement();
  if (old_swapchain != VK_NULL_HANDLE) {
    vulkan_device_->functions().vkDestroySwapchainKHR(
        vulkan_device_->device(), old_swapchain, nullptr);
  }
  XELOGI("ZeroFGMainSurface A retired its swapchain; surface kept for B");
  zerofg_independent_presenter_->MainSurfaceReleasedByA();
}

Presenter::PaintResult VulkanPresenter::PaintAndPresentImpl(
    bool execute_ui_drawers) {
  if (zerofg_independent_presenter_ &&
      zerofg_independent_presenter_->main_surface_reconnect_requested()) {
    // Main Surface Authority: the Surface went out of date under B. The UI
    // thread reconnect tears ZeroFG down and rebuilds A's swapchain.
    return PaintResult::kNotPresentedConnectionOutdated;
  }
  if (zerofg_independent_presenter_ &&
      zerofg_independent_presenter_->owns_final_output()) {
    // ZeroFG owns final presentation. Ordinary paint ticks must not become a
    // second producer.
    if (zerofg_independent_presenter_->main_surface_handoff_pending()) {
      // Main Surface Authority: B becomes the Surface's producer. A retires
      // only its swapchain, under painting ownership, and keeps the
      // VkSurfaceKHR that B's swapchain is created on.
      RetireSwapchainForMainSurfaceAuthority();
    }
    return PaintResult::kNotPresented;
  }
  if (paint_context_.swapchain == VK_NULL_HANDLE) {
    // Main Surface Authority handback: B has given the Surface back. A's
    // swapchain is recreated only by the UI-thread reconnect, never lazily
    // here.
    return PaintResult::kNotPresentedConnectionOutdated;
  }
  MaybeLogNativePresentHostTelemetry();
  // Begin the submission in place of the one not currently potentially used on
  // the GPU.
  const uint64_t current_paint_submission_index =
      paint_context_.completion_timeline.GetUpcomingSubmission();
  uint64_t paint_submission_count = uint64_t(paint_context_.submissions.size());
  const uint64_t completion_begin_ns =
      cvars::vulkan_completion_wait_telemetry ? GetZeroFGMonotonicTimeNs() : 0;
  paint_context_.completion_timeline
      .AwaitMaxSubmissionsPendingAndUpdateCompleted(paint_submission_count);
  if (cvars::vulkan_completion_wait_telemetry) {
    RecordNativePresentHostOperation(
        native_present_host_telemetry_.completion,
        GetZeroFGMonotonicTimeNs() - completion_begin_ns);
  }
  const PaintContext::Submission& paint_submission =
      *paint_context_.submissions[current_paint_submission_index %
                                  paint_submission_count];

  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  VkCommandPool draw_command_pool = paint_submission.draw_command_pool();
  if (dfn.vkResetCommandPool(device, draw_command_pool, 0) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to reset a command buffer for drawing to the "
        "swapchain");
    return PaintResult::kNotPresented;
  }

  VkCommandBuffer draw_command_buffer = paint_submission.draw_command_buffer();
  VkCommandBufferBeginInfo command_buffer_begin_info;
  command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_buffer_begin_info.pNext = nullptr;
  command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  command_buffer_begin_info.pInheritanceInfo = nullptr;
  if (dfn.vkBeginCommandBuffer(draw_command_buffer,
                               &command_buffer_begin_info) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to being recording the command buffer for "
        "drawing to the swapchain");
    return PaintResult::kNotPresented;
  }
  // vkResetCommandPool resets from both initial and recording states, still
  // safe to return early from this function in case of an error.

  VkSemaphore acquire_semaphore = paint_submission.acquire_semaphore();

  uint32_t swapchain_image_index;
  // Finite timeout: an infinite acquire holds paint_mode_mutex_ against the
  // surface-teardown path. On timeout just drop the frame - no connection
  // state change, so a merely-slow compositor can't trigger a rebuild loop.
  constexpr uint64_t kAcquireTimeoutNs = 1000000000ull;  // 1 second
  const uint64_t acquire_begin_ns =
      cvars::vulkan_completion_wait_telemetry ? GetZeroFGMonotonicTimeNs() : 0;
  VkResult acquire_result = dfn.vkAcquireNextImageKHR(
      device, paint_context_.swapchain, kAcquireTimeoutNs, acquire_semaphore,
      VK_NULL_HANDLE, &swapchain_image_index);
  if (cvars::vulkan_completion_wait_telemetry) {
    RecordNativePresentHostOperation(native_present_host_telemetry_.acquire,
                                     GetZeroFGMonotonicTimeNs() -
                                         acquire_begin_ns);
  }
  switch (acquire_result) {
    case VK_SUCCESS:
    case VK_SUBOPTIMAL_KHR:
      break;
    case VK_TIMEOUT:
    case VK_NOT_READY:
      XELOGI(
          "VulkanPresenter: Swapchain image acquire timed out; dropping the "
          "frame");
      return PaintResult::kNotPresented;
    case VK_ERROR_DEVICE_LOST:
      XELOGE(
          "VulkanPresenter: Failed to acquire the swapchain image as the "
          "device has been lost");
      return PaintResult::kGpuLostResponsible;
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      // Not an error, reporting just as info (may normally occur while resizing
      // on some platforms).
      XELOGI(
          "VulkanPresenter: Presentation to the swapchain image has been "
          "dropped as the swapchain or the surface has become outdated");
      return PaintResult::kNotPresentedConnectionOutdated;
    default:
      XELOGE("VulkanPresenter: Failed to acquire the swapchain image");
      return PaintResult::kNotPresented;
  }

  // Non-zero extents needed for both the viewport (width must not be zero) and
  // the guest output rectangle.
  assert_not_zero(paint_context_.swapchain_extent.width);
  assert_not_zero(paint_context_.swapchain_extent.height);

  bool swapchain_image_clear_needed = true;
  VkClearAttachment swapchain_image_clear_attachment;
  swapchain_image_clear_attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  swapchain_image_clear_attachment.colorAttachment = 0;
  swapchain_image_clear_attachment.clearValue.color.float32[0] = 0.0f;
  swapchain_image_clear_attachment.clearValue.color.float32[1] = 0.0f;
  swapchain_image_clear_attachment.clearValue.color.float32[2] = 0.0f;
  swapchain_image_clear_attachment.clearValue.color.float32[3] = 1.0f;

  VkRenderPassBeginInfo swapchain_render_pass_begin_info;
  swapchain_render_pass_begin_info.sType =
      VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  swapchain_render_pass_begin_info.pNext = nullptr;
  swapchain_render_pass_begin_info.renderPass =
      paint_context_.swapchain_render_pass;
  swapchain_render_pass_begin_info.framebuffer =
      paint_context_.swapchain_framebuffers[swapchain_image_index].framebuffer;
  swapchain_render_pass_begin_info.renderArea.offset.x = 0;
  swapchain_render_pass_begin_info.renderArea.offset.y = 0;
  swapchain_render_pass_begin_info.renderArea.extent =
      paint_context_.swapchain_extent;
  swapchain_render_pass_begin_info.clearValueCount = 0;
  swapchain_render_pass_begin_info.pClearValues = nullptr;
  if (paint_context_.swapchain_render_pass_clear_load_op) {
    swapchain_render_pass_begin_info.clearValueCount = 1;
    swapchain_render_pass_begin_info.pClearValues =
        &swapchain_image_clear_attachment.clearValue;
    swapchain_image_clear_needed = false;
  }

  bool swapchain_image_pass_begun = false;

  GuestOutputProperties guest_output_properties;
  GuestOutputPaintConfig guest_output_paint_config;
  std::shared_ptr<GuestOutputImage> guest_output_image;
  {
    uint32_t guest_output_mailbox_index;
    std::unique_lock<std::mutex> guest_output_consumer_lock(
        ConsumeGuestOutput(guest_output_mailbox_index, &guest_output_properties,
                           &guest_output_paint_config));
    if (guest_output_mailbox_index != UINT32_MAX) {
      assert_true(guest_output_images_[guest_output_mailbox_index]
                      .ever_successfully_refreshed);
      guest_output_image =
          guest_output_images_[guest_output_mailbox_index].image;
    }
    // Incremented the reference count of the guest output image - safe to leave
    // the consumer critical section now as everything here either will be using
    // the new reference or is exclusively owned by main target painting (and
    // multiple threads can't paint the main target at the same time).
  }

  if (guest_output_image) {
    VkExtent2D max_framebuffer_extent =
        util::GetMax2DFramebufferExtent(vulkan_device_->properties());
    GuestOutputPaintFlow guest_output_flow = GetGuestOutputPaintFlow(
        guest_output_properties, paint_context_.swapchain_extent.width,
        paint_context_.swapchain_extent.height, max_framebuffer_extent.width,
        max_framebuffer_extent.height, guest_output_paint_config);
    if (guest_output_flow.effect_count) {
      // Store the main target reference to the guest output image so it's not
      // destroyed while it's still potentially in use by main target painting
      // queued on the GPU.
      size_t guest_output_image_paint_ref_index = SIZE_MAX;
      size_t guest_output_image_paint_ref_new_index = SIZE_MAX;
      // Try to find the existing reference to the same image, or an already
      // released (or a taken, but never actually used) slot.
      for (size_t i = 0;
           i < paint_context_.guest_output_image_paint_refs.size(); ++i) {
        const std::pair<uint64_t, std::shared_ptr<GuestOutputImage>>&
            guest_output_image_paint_ref =
                paint_context_.guest_output_image_paint_refs[i];
        if (guest_output_image_paint_ref.second == guest_output_image) {
          guest_output_image_paint_ref_index = i;
          break;
        }
        if (guest_output_image_paint_ref_new_index == SIZE_MAX &&
            (!guest_output_image_paint_ref.second ||
             !guest_output_image_paint_ref.first)) {
          guest_output_image_paint_ref_new_index = i;
        }
      }
      if (guest_output_image_paint_ref_index == SIZE_MAX) {
        // New image - store the reference and create the descriptors.
        if (guest_output_image_paint_ref_new_index == SIZE_MAX) {
          // Replace the earliest used reference.
          guest_output_image_paint_ref_new_index = 0;
          for (size_t i = 1;
               i < paint_context_.guest_output_image_paint_refs.size(); ++i) {
            if (paint_context_.guest_output_image_paint_refs[i].first <
                paint_context_
                    .guest_output_image_paint_refs
                        [guest_output_image_paint_ref_new_index]
                    .first) {
              guest_output_image_paint_ref_new_index = i;
            }
          }
          // Await the completion of the usage of the old guest output image and
          // its descriptors.
          paint_context_.completion_timeline.AwaitSubmissionAndUpdateCompleted(
              paint_context_
                  .guest_output_image_paint_refs
                      [guest_output_image_paint_ref_new_index]
                  .first);
        }
        guest_output_image_paint_ref_index =
            guest_output_image_paint_ref_new_index;
        // The actual submission index will be set if the image is actually
        // used, not dropped due to some error.
        paint_context_
            .guest_output_image_paint_refs[guest_output_image_paint_ref_index] =
            std::make_pair(uint64_t(0), guest_output_image);
        // Create the descriptors of the new image.
        VkDescriptorImageInfo guest_output_image_descriptor_image_info;
        guest_output_image_descriptor_image_info.sampler = VK_NULL_HANDLE;
        guest_output_image_descriptor_image_info.imageView =
            guest_output_image->view();
        guest_output_image_descriptor_image_info.imageLayout =
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet guest_output_image_descriptor_write;
        guest_output_image_descriptor_write.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        guest_output_image_descriptor_write.pNext = nullptr;
        guest_output_image_descriptor_write.dstSet =
            paint_context_.guest_output_descriptor_sets
                [PaintContext::kGuestOutputDescriptorSetGuestOutput0Sampled +
                 guest_output_image_paint_ref_index];
        guest_output_image_descriptor_write.dstBinding = 0;
        guest_output_image_descriptor_write.dstArrayElement = 0;
        guest_output_image_descriptor_write.descriptorCount = 1;
        guest_output_image_descriptor_write.descriptorType =
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        guest_output_image_descriptor_write.pImageInfo =
            &guest_output_image_descriptor_image_info;
        guest_output_image_descriptor_write.pBufferInfo = nullptr;
        guest_output_image_descriptor_write.pTexelBufferView = nullptr;
        dfn.vkUpdateDescriptorSets(
            device, 1, &guest_output_image_descriptor_write, 0, nullptr);
      }

      // Make sure intermediate textures of the needed size are available, and
      // unneeded intermediate textures are destroyed.
      for (size_t i = 0; i < kMaxGuestOutputPaintEffects - 1; ++i) {
        std::pair<uint32_t, uint32_t> intermediate_needed_size(0, 0);
        if (i + 1 < guest_output_flow.effect_count) {
          intermediate_needed_size = guest_output_flow.effect_output_sizes[i];
        }
        std::unique_ptr<GuestOutputImage>& intermediate_image_ptr_ref =
            paint_context_.guest_output_intermediate_images[i];
        VkExtent2D intermediate_current_extent(
            intermediate_image_ptr_ref ? intermediate_image_ptr_ref->extent()
                                       : VkExtent2D{});
        if (intermediate_current_extent.width !=
                intermediate_needed_size.first ||
            intermediate_current_extent.height !=
                intermediate_needed_size.second) {
          if (intermediate_needed_size.first &&
              intermediate_needed_size.second) {
            // Need to replace immediately as a new image with the requested
            // size is needed.
            if (intermediate_image_ptr_ref) {
              paint_context_.completion_timeline
                  .AwaitSubmissionAndUpdateCompleted(
                      paint_context_
                          .guest_output_intermediate_image_last_submission);
              intermediate_image_ptr_ref.reset();
              util::DestroyAndNullHandle(
                  dfn.vkDestroyFramebuffer, device,
                  paint_context_.guest_output_intermediate_framebuffers[i]);
            }
            // Image.
            intermediate_image_ptr_ref = GuestOutputImage::Create(
                vulkan_device_, intermediate_needed_size.first,
                intermediate_needed_size.second);
            if (!intermediate_image_ptr_ref) {
              // Don't display the guest output, and don't try to create more
              // intermediate textures (only destroy them).
              guest_output_flow.effect_count = 0;
              continue;
            }
            // Framebuffer.
            VkImageView intermediate_framebuffer_attachment =
                intermediate_image_ptr_ref->view();
            VkFramebufferCreateInfo intermediate_framebuffer_create_info;
            intermediate_framebuffer_create_info.sType =
                VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            intermediate_framebuffer_create_info.pNext = nullptr;
            intermediate_framebuffer_create_info.flags = 0;
            intermediate_framebuffer_create_info.renderPass =
                guest_output_intermediate_render_pass_;
            intermediate_framebuffer_create_info.attachmentCount = 1;
            intermediate_framebuffer_create_info.pAttachments =
                &intermediate_framebuffer_attachment;
            intermediate_framebuffer_create_info.width =
                intermediate_needed_size.first;
            intermediate_framebuffer_create_info.height =
                intermediate_needed_size.second;
            intermediate_framebuffer_create_info.layers = 1;
            if (dfn.vkCreateFramebuffer(
                    device, &intermediate_framebuffer_create_info, nullptr,
                    &paint_context_
                         .guest_output_intermediate_framebuffers[i]) !=
                VK_SUCCESS) {
              XELOGE(
                  "VulkanPresenter: Failed to create a guest output "
                  "intermediate framebuffer");
              // Don't display the guest output, and don't try to create more
              // intermediate textures (only destroy them).
              guest_output_flow.effect_count = 0;
              continue;
            }
            // Descriptors.
            VkDescriptorImageInfo intermediate_descriptor_image_info;
            intermediate_descriptor_image_info.sampler = VK_NULL_HANDLE;
            intermediate_descriptor_image_info.imageView =
                intermediate_image_ptr_ref->view();
            intermediate_descriptor_image_info.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet intermediate_descriptor_write;
            intermediate_descriptor_write.sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            intermediate_descriptor_write.pNext = nullptr;
            intermediate_descriptor_write.dstSet =
                paint_context_.guest_output_descriptor_sets
                    [PaintContext ::
                         kGuestOutputDescriptorSetIntermediate0Sampled +
                     i];
            intermediate_descriptor_write.dstBinding = 0;
            intermediate_descriptor_write.dstArrayElement = 0;
            intermediate_descriptor_write.descriptorCount = 1;
            intermediate_descriptor_write.descriptorType =
                VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            intermediate_descriptor_write.pImageInfo =
                &intermediate_descriptor_image_info;
            intermediate_descriptor_write.pBufferInfo = nullptr;
            intermediate_descriptor_write.pTexelBufferView = nullptr;
            dfn.vkUpdateDescriptorSets(
                device, 1, &intermediate_descriptor_write, 0, nullptr);
          } else {
            // Was previously needed, but not anymore - destroy when possible.
            if (intermediate_image_ptr_ref &&
                paint_context_.completion_timeline
                        .UpdateAndGetCompletedSubmission() >=
                    paint_context_
                        .guest_output_intermediate_image_last_submission) {
              intermediate_image_ptr_ref.reset();
              util::DestroyAndNullHandle(
                  dfn.vkDestroyFramebuffer, device,
                  paint_context_.guest_output_intermediate_framebuffers[i]);
            }
          }
        }
      }

      if (guest_output_flow.effect_count) {
        // Check if all the intermediate effects are supported by the
        // implementation.
        for (size_t i = 0; i + 1 < guest_output_flow.effect_count; ++i) {
          if (paint_context_
                  .guest_output_paint_pipelines[size_t(
                      guest_output_flow.effects[i])]
                  .intermediate_pipeline == VK_NULL_HANDLE) {
            guest_output_flow.effect_count = 0;
            break;
          }
        }
        // Ensure the pipeline exists for the final effect drawing to the
        // swapchain, for the render pass with the up-to-date image format.
        GuestOutputPaintEffect swapchain_effect =
            guest_output_flow.effects[guest_output_flow.effect_count - 1];
        PaintContext::GuestOutputPaintPipeline& swapchain_effect_pipeline =
            paint_context_
                .guest_output_paint_pipelines[size_t(swapchain_effect)];
        if (swapchain_effect_pipeline.swapchain_pipeline != VK_NULL_HANDLE &&
            swapchain_effect_pipeline.swapchain_format !=
                paint_context_.swapchain_render_pass_format) {
          paint_context_.completion_timeline.AwaitSubmissionAndUpdateCompleted(
              paint_context_.guest_output_image_paint_last_submission);
          util::DestroyAndNullHandle(
              dfn.vkDestroyPipeline, device,
              swapchain_effect_pipeline.swapchain_pipeline);
        }
        if (swapchain_effect_pipeline.swapchain_pipeline == VK_NULL_HANDLE) {
          assert_true(CanGuestOutputPaintEffectBeFinal(swapchain_effect));
          assert_true(guest_output_paint_fs_[size_t(swapchain_effect)] !=
                      VK_NULL_HANDLE);
          swapchain_effect_pipeline.swapchain_pipeline =
              CreateGuestOutputPaintPipeline(
                  swapchain_effect, paint_context_.swapchain_render_pass);
          if (swapchain_effect_pipeline.swapchain_pipeline == VK_NULL_HANDLE) {
            guest_output_flow.effect_count = 0;
          } else {
            // Record the format this pipeline was created for, so the check
            // above rebuilds it only when the swapchain format changes.
            swapchain_effect_pipeline.swapchain_format =
                paint_context_.swapchain_render_pass_format;
          }
        }
      }

      if (guest_output_flow.effect_count) {
        // Actually draw the guest output.
        paint_context_
            .guest_output_image_paint_refs[guest_output_image_paint_ref_index]
            .first = current_paint_submission_index;
        paint_context_.guest_output_image_paint_last_submission =
            current_paint_submission_index;
        VkViewport guest_output_viewport;
        guest_output_viewport.x = 0.0f;
        guest_output_viewport.y = 0.0f;
        guest_output_viewport.minDepth = 0.0f;
        guest_output_viewport.maxDepth = 1.0f;
        VkRect2D guest_output_scissor;
        guest_output_scissor.offset.x = 0;
        guest_output_scissor.offset.y = 0;
        if (guest_output_flow.effect_count > 1) {
          paint_context_.guest_output_intermediate_image_last_submission =
              current_paint_submission_index;
        }
        for (size_t i = 0; i < guest_output_flow.effect_count; ++i) {
          bool is_final_effect = i + 1 >= guest_output_flow.effect_count;

          int32_t effect_rect_x, effect_rect_y;
          std::pair<uint32_t, uint32_t> effect_rect_size =
              guest_output_flow.effect_output_sizes[i];
          if (is_final_effect) {
            effect_rect_x = guest_output_flow.output_x;
            effect_rect_y = guest_output_flow.output_y;
            dfn.vkCmdBeginRenderPass(draw_command_buffer,
                                     &swapchain_render_pass_begin_info,
                                     VK_SUBPASS_CONTENTS_INLINE);
            swapchain_image_pass_begun = true;
            guest_output_viewport.width =
                float(paint_context_.swapchain_extent.width);
            guest_output_viewport.height =
                float(paint_context_.swapchain_extent.height);
            guest_output_scissor.extent.width =
                paint_context_.swapchain_extent.width;
            guest_output_scissor.extent.height =
                paint_context_.swapchain_extent.height;
          } else {
            effect_rect_x = 0;
            effect_rect_y = 0;
            VkRenderPassBeginInfo intermediate_render_pass_begin_info;
            intermediate_render_pass_begin_info.sType =
                VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            intermediate_render_pass_begin_info.pNext = nullptr;
            intermediate_render_pass_begin_info.renderPass =
                guest_output_intermediate_render_pass_;
            intermediate_render_pass_begin_info.framebuffer =
                paint_context_.guest_output_intermediate_framebuffers[i];
            intermediate_render_pass_begin_info.renderArea.offset.x = 0;
            intermediate_render_pass_begin_info.renderArea.offset.y = 0;
            intermediate_render_pass_begin_info.renderArea.extent.width =
                effect_rect_size.first;
            intermediate_render_pass_begin_info.renderArea.extent.height =
                effect_rect_size.second;
            intermediate_render_pass_begin_info.clearValueCount = 0;
            intermediate_render_pass_begin_info.pClearValues = nullptr;
            dfn.vkCmdBeginRenderPass(draw_command_buffer,
                                     &intermediate_render_pass_begin_info,
                                     VK_SUBPASS_CONTENTS_INLINE);
            guest_output_viewport.width = float(effect_rect_size.first);
            guest_output_viewport.height = float(effect_rect_size.second);
            guest_output_scissor.extent.width = effect_rect_size.first;
            guest_output_scissor.extent.height = effect_rect_size.second;
          }
          dfn.vkCmdSetViewport(draw_command_buffer, 0, 1,
                               &guest_output_viewport);
          dfn.vkCmdSetScissor(draw_command_buffer, 0, 1, &guest_output_scissor);

          GuestOutputPaintEffect effect = guest_output_flow.effects[i];

          const PaintContext::GuestOutputPaintPipeline& effect_pipeline =
              paint_context_.guest_output_paint_pipelines[size_t(effect)];
          VkPipeline effect_vulkan_pipeline =
              is_final_effect ? effect_pipeline.swapchain_pipeline
                              : effect_pipeline.intermediate_pipeline;
          assert_true(effect_vulkan_pipeline != VK_NULL_HANDLE);
          dfn.vkCmdBindPipeline(draw_command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                effect_vulkan_pipeline);

          GuestOutputPaintPipelineLayoutIndex
              guest_output_paint_pipeline_layout_index =
                  GetGuestOutputPaintPipelineLayoutIndex(effect);
          VkPipelineLayout guest_output_paint_pipeline_layout =
              guest_output_paint_pipeline_layouts_
                  [guest_output_paint_pipeline_layout_index];

          PaintContext::GuestOutputDescriptorSet effect_src_descriptor_set;
          if (i) {
            effect_src_descriptor_set = PaintContext::GuestOutputDescriptorSet(
                PaintContext::kGuestOutputDescriptorSetIntermediate0Sampled +
                (i - 1));
          } else {
            effect_src_descriptor_set = PaintContext::GuestOutputDescriptorSet(
                PaintContext::kGuestOutputDescriptorSetGuestOutput0Sampled +
                guest_output_image_paint_ref_index);
          }
          dfn.vkCmdBindDescriptorSets(
              draw_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
              guest_output_paint_pipeline_layout, 0, 1,
              &paint_context_
                   .guest_output_descriptor_sets[effect_src_descriptor_set],
              0, nullptr);

          GuestOutputPaintRectangleConstants effect_rect_constants;
          float effect_x_to_ndc = 2.0f / guest_output_viewport.width;
          float effect_y_to_ndc = 2.0f / guest_output_viewport.height;
          effect_rect_constants.x =
              -1.0f + float(effect_rect_x) * effect_x_to_ndc;
          effect_rect_constants.y =
              -1.0f + float(effect_rect_y) * effect_y_to_ndc;
          effect_rect_constants.width =
              float(effect_rect_size.first) * effect_x_to_ndc;
          effect_rect_constants.height =
              float(effect_rect_size.second) * effect_y_to_ndc;
          dfn.vkCmdPushConstants(
              draw_command_buffer, guest_output_paint_pipeline_layout,
              VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(effect_rect_constants),
              &effect_rect_constants);

          uint32_t effect_constants_size = 0;
          union {
            BilinearConstants bilinear;
            CasSharpenConstants cas_sharpen;
            CasResampleConstants cas_resample;
            FsrEasuConstants fsr_easu;
            FsrRcasConstants fsr_rcas;
            SgsrConstants sgsr;
          } effect_constants;
          switch (guest_output_paint_pipeline_layout_index) {
            case kGuestOutputPaintPipelineLayoutIndexBilinear: {
              effect_constants_size = sizeof(effect_constants.bilinear);
              effect_constants.bilinear.Initialize(guest_output_flow, i);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexCasSharpen: {
              effect_constants_size = sizeof(effect_constants.cas_sharpen);
              effect_constants.cas_sharpen.Initialize(
                  guest_output_flow, i, guest_output_paint_config);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexCasResample: {
              effect_constants_size = sizeof(effect_constants.cas_resample);
              effect_constants.cas_resample.Initialize(
                  guest_output_flow, i, guest_output_paint_config);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexFsrEasu: {
              effect_constants_size = sizeof(effect_constants.fsr_easu);
              effect_constants.fsr_easu.Initialize(guest_output_flow, i);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexFsrRcas: {
              effect_constants_size = sizeof(effect_constants.fsr_rcas);
              effect_constants.fsr_rcas.Initialize(guest_output_flow, i,
                                                   guest_output_paint_config);
            } break;
            case kGuestOutputPaintPipelineLayoutIndexSgsr: {
              effect_constants_size = sizeof(effect_constants.sgsr);
              effect_constants.sgsr.Initialize(guest_output_flow, i);
            } break;
            default:
              break;
          }
          if (effect_constants_size) {
            dfn.vkCmdPushConstants(
                draw_command_buffer, guest_output_paint_pipeline_layout,
                VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(effect_rect_constants),
                effect_constants_size, &effect_constants);
          }

          dfn.vkCmdDraw(draw_command_buffer, 4, 1, 0, 0);

          if (is_final_effect) {
            // Clear the letterbox around the guest output if the guest output
            // doesn't cover the entire swapchain image.
            if (swapchain_image_clear_needed &&
                guest_output_flow.letterbox_clear_rectangle_count) {
              VkClearRect letterbox_clear_vulkan_rectangles
                  [GuestOutputPaintFlow::kMaxClearRectangles];
              for (size_t i = 0;
                   i < guest_output_flow.letterbox_clear_rectangle_count; ++i) {
                VkClearRect& letterbox_clear_vulkan_rectangle =
                    letterbox_clear_vulkan_rectangles[i];
                const GuestOutputPaintFlow::ClearRectangle&
                    letterbox_clear_rectangle =
                        guest_output_flow.letterbox_clear_rectangles[i];
                letterbox_clear_vulkan_rectangle.rect.offset.x =
                    int32_t(letterbox_clear_rectangle.x);
                letterbox_clear_vulkan_rectangle.rect.offset.y =
                    int32_t(letterbox_clear_rectangle.y);
                letterbox_clear_vulkan_rectangle.rect.extent.width =
                    letterbox_clear_rectangle.width;
                letterbox_clear_vulkan_rectangle.rect.extent.height =
                    letterbox_clear_rectangle.height;
                letterbox_clear_vulkan_rectangle.baseArrayLayer = 0;
                letterbox_clear_vulkan_rectangle.layerCount = 1;
              }
              dfn.vkCmdClearAttachments(
                  draw_command_buffer, 1, &swapchain_image_clear_attachment,
                  uint32_t(guest_output_flow.letterbox_clear_rectangle_count),
                  letterbox_clear_vulkan_rectangles);
            }
            swapchain_image_clear_needed = false;
          } else {
            // Still need the swapchain pass to be open for UI drawing.
            dfn.vkCmdEndRenderPass(draw_command_buffer);
          }
        }
      }
    }
  }

  // Release main target guest output image references that aren't needed
  // anymore (this is done after various potential guest-output-related main
  // target completion timeline waits so the completed submission index is the
  // most actual).
  uint64_t completed_paint_submission =
      paint_context_.completion_timeline.UpdateAndGetCompletedSubmission();
  for (std::pair<uint64_t, std::shared_ptr<GuestOutputImage>>&
           guest_output_image_paint_ref :
       paint_context_.guest_output_image_paint_refs) {
    if (!guest_output_image_paint_ref.second ||
        guest_output_image_paint_ref.second == guest_output_image) {
      continue;
    }
    if (completed_paint_submission >= guest_output_image_paint_ref.first) {
      guest_output_image_paint_ref.second.reset();
    }
  }

  // If hasn't presented the guest output, begin the pass to clear and, if
  // needed, to draw the UI.
  if (!swapchain_image_pass_begun) {
    dfn.vkCmdBeginRenderPass(draw_command_buffer,
                             &swapchain_render_pass_begin_info,
                             VK_SUBPASS_CONTENTS_INLINE);
  }
  if (swapchain_image_clear_needed) {
    VkClearRect swapchain_image_clear_rectangle;
    swapchain_image_clear_rectangle.rect.offset.x = 0;
    swapchain_image_clear_rectangle.rect.offset.y = 0;
    swapchain_image_clear_rectangle.rect.extent =
        paint_context_.swapchain_extent;
    swapchain_image_clear_rectangle.baseArrayLayer = 0;
    swapchain_image_clear_rectangle.layerCount = 1;
    dfn.vkCmdClearAttachments(draw_command_buffer, 1,
                              &swapchain_image_clear_attachment, 1,
                              &swapchain_image_clear_rectangle);
    swapchain_image_clear_needed = false;
  }

  if (execute_ui_drawers) {
    // Draw the UI.
    VulkanUIDrawContext ui_draw_context(
        *this, paint_context_.swapchain_extent.width,
        paint_context_.swapchain_extent.height, draw_command_buffer,
        ui_completion_timeline_.GetUpcomingSubmission(),
        ui_completion_timeline_.UpdateAndGetCompletedSubmission(),
        paint_context_.swapchain_render_pass,
        paint_context_.swapchain_render_pass_format);
    ExecuteUIDrawersFromUIThread(ui_draw_context);
  }

  dfn.vkCmdEndRenderPass(draw_command_buffer);

  dfn.vkEndCommandBuffer(draw_command_buffer);

  VkPipelineStageFlags acquire_semaphore_wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkCommandBuffer command_buffers[2];
  uint32_t command_buffer_count = 0;
  // UI setup command buffers must be accessed only if execute_ui_drawers is not
  // null, to identify the UI thread.
  size_t ui_setup_command_buffer_index =
      execute_ui_drawers ? paint_context_.ui_setup_command_buffer_current_index
                         : SIZE_MAX;
  if (ui_setup_command_buffer_index != SIZE_MAX) {
    PaintContext::UISetupCommandBuffer& ui_setup_command_buffer =
        paint_context_.ui_setup_command_buffers[ui_setup_command_buffer_index];
    dfn.vkEndCommandBuffer(ui_setup_command_buffer.command_buffer);
    command_buffers[command_buffer_count++] =
        ui_setup_command_buffer.command_buffer;
    // Release the current UI setup command buffer regardless of submission
    // result. Failed submissions (if the UI submission index wasn't incremented
    // since the previous draw) should be handled by UI drawers themselves by
    // retrying all the failed work if needed.
    paint_context_.ui_setup_command_buffer_current_index = SIZE_MAX;
  }
  command_buffers[command_buffer_count++] = draw_command_buffer;
  VkSemaphore present_semaphore =
      paint_context_.swapchain_image_present_semaphores[swapchain_image_index];
  const VkSemaphore signal_semaphores[] = {present_semaphore,
                                           VK_NULL_HANDLE};
  VkSubmitInfo submit_info;
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;
  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &acquire_semaphore;
  submit_info.pWaitDstStageMask = &acquire_semaphore_wait_stage;
  submit_info.commandBufferCount = command_buffer_count;
  submit_info.pCommandBuffers = command_buffers;
  submit_info.signalSemaphoreCount =
      signal_semaphores[1] != VK_NULL_HANDLE ? 2 : 1;
  submit_info.pSignalSemaphores = signal_semaphores;
  const uint64_t submit_begin_ns =
      cvars::vulkan_completion_wait_telemetry ? GetZeroFGMonotonicTimeNs() : 0;
  const VkResult submit_result =
      paint_context_.completion_timeline.AcquireFenceAndSubmit(
          vulkan_device_->queue_family_graphics_compute(), 0, 1, &submit_info);
  if (cvars::vulkan_completion_wait_telemetry) {
    RecordNativePresentHostOperation(native_present_host_telemetry_.submit,
                                     GetZeroFGMonotonicTimeNs() -
                                         submit_begin_ns);
  }
  if (submit_result != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to submit the presentation command buffer: {} "
        "- submission: {} (completed: {}, in-flight: {}), swapchain "
        "image_index: {}, ui_setup_buffer_index: {}, execute_ui_drawers: {}",
        vk::to_string(vk::Result(submit_result)),
        paint_context_.completion_timeline.GetUpcomingSubmission(),
        paint_context_.completion_timeline
            .GetCompletedSubmissionFromLastUpdate(),
        paint_context_.completion_timeline.pending_submission_count(),
        swapchain_image_index,
        ui_setup_command_buffer_index == SIZE_MAX
            ? -1
            : int64_t(ui_setup_command_buffer_index),
        execute_ui_drawers);
    if (ui_setup_command_buffer_index != SIZE_MAX) {
      // If failed to submit, make the UI setup command buffer available for
      // immediate reuse, as the completed submission index won't be updated to
      // the current index, and failing submissions with setup command buffer
      // over and over will result in never reusing the setup command buffers.
      paint_context_.ui_setup_command_buffers[ui_setup_command_buffer_index]
          .last_usage_submission_index = 0;
    }
    if (submit_result == VK_ERROR_DEVICE_LOST) {
      XELOGE("VulkanPresenter: VK_ERROR_DEVICE_LOST - GPU crashed or hung");
    } else if (submit_result == VK_ERROR_OUT_OF_HOST_MEMORY) {
      XELOGE("VulkanPresenter: VK_ERROR_OUT_OF_HOST_MEMORY");
    } else if (submit_result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
      XELOGE("VulkanPresenter: VK_ERROR_OUT_OF_DEVICE_MEMORY");
    }
    // The image is in an acquired state - but now, it will be in it forever.
    // To avoid that, recreate the swapchain - don't return just kNotPresented.
    return PaintResult::kNotPresentedConnectionOutdated;
  }
  if (execute_ui_drawers) {
    // Also update the completion timeline providing submission indices to UI
    // draw callbacks if submission is successful.
    const VkResult ui_signal_submit_result =
        ui_completion_timeline_.AcquireFenceAndSubmit(
            vulkan_device_->queue_family_graphics_compute(), 0, 0, nullptr);
    if (ui_signal_submit_result != VK_SUCCESS) {
      XELOGE(
          "VulkanPresenter: Failed to submit the UI drawing fence signal: {}",
          vk::to_string(vk::Result(ui_signal_submit_result)));
    }
  }

  VkPresentInfoKHR present_info;
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.pNext = nullptr;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &present_semaphore;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &paint_context_.swapchain;
  present_info.pImageIndices = &swapchain_image_index;
  present_info.pResults = nullptr;
  VkResult present_result;
  {
    const uint64_t present_queue_lock_begin_ns =
        cvars::vulkan_completion_wait_telemetry ? GetZeroFGMonotonicTimeNs()
                                                : 0;
    const VulkanDevice::Queue::Acquisition queue_acquisition =
        vulkan_device_->AcquireQueue(paint_context_.present_queue_family, 0);
    if (cvars::vulkan_completion_wait_telemetry) {
      RecordNativePresentHostOperation(
          native_present_host_telemetry_.present_queue_lock,
          GetZeroFGMonotonicTimeNs() - present_queue_lock_begin_ns);
    }
    const uint64_t present_begin_ns =
        cvars::vulkan_completion_wait_telemetry ? GetZeroFGMonotonicTimeNs()
                                                : 0;
    present_result =
        dfn.vkQueuePresentKHR(queue_acquisition.queue(), &present_info);
    if (cvars::vulkan_completion_wait_telemetry) {
      RecordNativePresentHostOperation(native_present_host_telemetry_.present,
                                       GetZeroFGMonotonicTimeNs() -
                                           present_begin_ns);
    }
  }
  switch (present_result) {
    case VK_SUCCESS:
      return PaintResult::kPresented;
    case VK_SUBOPTIMAL_KHR:
      return PaintResult::kPresentedSuboptimal;
    case VK_ERROR_DEVICE_LOST:
      XELOGE(
          "VulkanPresenter: Failed to present the swapchain image as the "
          "device has been lost (image_index: {}, paint submission: {} "
          "completed: {}, in-flight: {})",
          swapchain_image_index,
          paint_context_.completion_timeline.GetUpcomingSubmission(),
          paint_context_.completion_timeline
              .GetCompletedSubmissionFromLastUpdate(),
          paint_context_.completion_timeline.pending_submission_count());
      return PaintResult::kGpuLostResponsible;
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
    case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
      // Not an error, reporting just as info (may normally occur while resizing
      // on some platforms).
      XELOGI(
          "VulkanPresenter: Presentation to the swapchain image has been "
          "dropped as the swapchain or the surface has become outdated");
      // Note that the semaphore wait (followed by reset) has been enqueued,
      // however, this should have no effect on anything here likely.
      return PaintResult::kNotPresentedConnectionOutdated;
    default:
      XELOGE("VulkanPresenter: Failed to present the swapchain image");
      // The image is in an acquired state - but now, it will be in it forever.
      // To avoid that, recreate the swapchain - don't return just
      // kNotPresented.
      return PaintResult::kNotPresentedConnectionOutdated;
  }
}

bool VulkanPresenter::InitializeSurfaceIndependent() {
  const VulkanDevice::Functions& dfn = vulkan_device_->functions();
  const VkDevice device = vulkan_device_->device();

  const bool baseline_available =
      vulkan_device_->properties().apiVersion >= VK_API_VERSION_1_3 &&
      vulkan_device_->properties().synchronization2;
  const bool device_context_available = InitializeZeroFGDeviceContext();
  if (IsZeroFGRequested() && baseline_available &&
      device_context_available) {
    zerofg_independent_presenter_ = ZeroFGIndependentPresenter::Create(
        ZeroFGDevice(), true,
        kZeroFGSyntheticCostSeedNs,
        [this](
            ZeroFGIndependentPresenter::IngressSourcePublication& publication) {
          return AcquireZeroFGIngressSource(publication);
        },
        [this]() { return GetGuestOutputPaintConfigForExternalThread(); },
        [this](const ZeroFGIndependentPresenter::GenerationRequest& request,
               ZeroFGIndependentPresenter::GenerationResult& result) {
          return ProcessZeroFGGeneration(request, result);
        },
        [this](uint32_t synthetic_index,
               ZeroFGIndependentPresenter::GenerationResult& result,
               bool& ready) {
          return PollZeroFGGeneration(synthetic_index, result, ready);
        },
        [this](uint32_t synthetic_index) {
          ReleaseZeroFGSynthetic(synthetic_index);
        },
        [this]() { DestroyZeroFGGenerationContext(); },
        [this](const ZeroFGIndependentPresenter::PostProcessRequest& request,
               ZeroFGIndependentPresenter::PostProcessResult& result) {
          return ProcessZeroFGPost(request, result);
        },
        [this](uint32_t final_output_index) {
          ReleaseZeroFGPost(final_output_index);
        },
        [this]() { DestroyZeroFGPostContext(); },
        [this]() { return ZeroFGDevice()->DrainZeroFGForTeardown(); });
    if (!zerofg_independent_presenter_) {
      XELOGW("ZeroFGC0: capability initialization failed; using normal "
             "Real-only presentation");
    } else {
      // The egress asks for UI-thread paints while the Surface changes hands:
      // A retires its swapchain there, and recovers it there on handback.
      zerofg_independent_presenter_->SetMainSurfaceUIRequest(
          [this]() { RequestUIThreadPaintFromExternalThread(); });
      XELOGI(
          "ZeroFGOutput path=main_surface_fifo "
          "present_authority=fifo_vblank producer=device_B");
    }
  } else if (IsZeroFGRequested() && !device_context_available) {
    XELOGE(
        "ZeroFGDeviceSplit activation_failed=device_B_qualification "
        "split_armed=false shared_ZeroFG_fallback=false "
        "native_Source_preserved=true");
  } else if (IsZeroFGRequested()) {
    XELOGW(
        "ZeroFG requires effective Vulkan 1.3 with synchronization2 "
        "enabled; preserving the normal XenDroid presenter");
  }

  VkDescriptorSetLayoutBinding guest_output_image_sampler_bindings[2];
  guest_output_image_sampler_bindings[0].binding = 0;
  guest_output_image_sampler_bindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  guest_output_image_sampler_bindings[0].descriptorCount = 1;
  guest_output_image_sampler_bindings[0].stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_image_sampler_bindings[0].pImmutableSamplers = nullptr;
  const VkSampler sampler_linear_clamp =
      ui_samplers_->samplers()[UISamplers::kSamplerIndexLinearClampToEdge];
  guest_output_image_sampler_bindings[1].binding = 1;
  guest_output_image_sampler_bindings[1].descriptorType =
      VK_DESCRIPTOR_TYPE_SAMPLER;
  guest_output_image_sampler_bindings[1].descriptorCount = 1;
  guest_output_image_sampler_bindings[1].stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT;
  guest_output_image_sampler_bindings[1].pImmutableSamplers =
      &sampler_linear_clamp;
  VkDescriptorSetLayoutCreateInfo
      guest_output_paint_image_descriptor_set_layout_create_info;
  guest_output_paint_image_descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  guest_output_paint_image_descriptor_set_layout_create_info.pNext = nullptr;
  guest_output_paint_image_descriptor_set_layout_create_info.flags = 0;
  guest_output_paint_image_descriptor_set_layout_create_info.bindingCount =
      uint32_t(xe::countof(guest_output_image_sampler_bindings));
  guest_output_paint_image_descriptor_set_layout_create_info.pBindings =
      guest_output_image_sampler_bindings;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &guest_output_paint_image_descriptor_set_layout_create_info,
          nullptr,
          &guest_output_paint_image_descriptor_set_layout_) != VK_SUCCESS) {
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
      &guest_output_paint_image_descriptor_set_layout_;
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
        guest_output_paint_push_constant_range_ffx.size =
            sizeof(SgsrConstants);
        break;
      default:
        assert_unhandled_case(GuestOutputPaintPipelineLayoutIndex(i));
        continue;
    }
    guest_output_paint_pipeline_layout_create_info.pushConstantRangeCount =
        1 + uint32_t(guest_output_paint_push_constant_range_ffx.size != 0);
    if (dfn.vkCreatePipelineLayout(
            device, &guest_output_paint_pipeline_layout_create_info, nullptr,
            &guest_output_paint_pipeline_layouts_[i]) != VK_SUCCESS) {
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
                               &guest_output_paint_vs_) != VK_SUCCESS) {
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
                                 &guest_output_paint_fs_[i]) != VK_SUCCESS) {
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
  if (dfn.vkCreateRenderPass(
          device, &intermediate_render_pass_create_info, nullptr,
          &guest_output_intermediate_render_pass_) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create the guest output intermediate image "
        "render pass");
    return false;
  }

  // Initialize connection-independent parts of the painting context.

  for (size_t i = 0; i < paint_context_.submissions.size(); ++i) {
    paint_context_.submissions[i] =
        PaintContext::Submission::Create(vulkan_device_);
    if (!paint_context_.submissions[i]) {
      return false;
    }
  }

  // Guest output paint pipelines drawing to intermediate images, not depending
  // on runtime state unlike ones drawing to the swapchain images as those
  // depend on the swapchain format.
  for (size_t i = 0; i < size_t(GuestOutputPaintEffect::kCount); ++i) {
    if (!CanGuestOutputPaintEffectBeIntermediate(GuestOutputPaintEffect(i)) ||
        guest_output_paint_fs_[i] == VK_NULL_HANDLE) {
      continue;
    }
    VkPipeline guest_output_paint_intermediate_pipeline =
        CreateGuestOutputPaintPipeline(GuestOutputPaintEffect(i),
                                       guest_output_intermediate_render_pass_);
    if (guest_output_paint_intermediate_pipeline == VK_NULL_HANDLE) {
      return false;
    }
    paint_context_.guest_output_paint_pipelines[i].intermediate_pipeline =
        guest_output_paint_intermediate_pipeline;
  }

  // Guest output painting descriptor sets.
  VkDescriptorPoolSize guest_output_paint_descriptor_pool_sizes[2];
  guest_output_paint_descriptor_pool_sizes[0].type =
      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  guest_output_paint_descriptor_pool_sizes[0].descriptorCount =
      PaintContext::kGuestOutputDescriptorSetCount;
  // Required even when using immutable samplers, otherwise failing to allocate
  // descriptor sets (tested on AMD Software: Adrenalin Edition 22.3.2 on
  // Windows 10 on AMD Radeon RX Vega 10 with Vulkan validation enabled).
  guest_output_paint_descriptor_pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
  guest_output_paint_descriptor_pool_sizes[1].descriptorCount =
      PaintContext::kGuestOutputDescriptorSetCount;
  VkDescriptorPoolCreateInfo guest_output_paint_descriptor_pool_create_info;
  guest_output_paint_descriptor_pool_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  guest_output_paint_descriptor_pool_create_info.pNext = nullptr;
  guest_output_paint_descriptor_pool_create_info.flags = 0;
  guest_output_paint_descriptor_pool_create_info.maxSets =
      PaintContext::kGuestOutputDescriptorSetCount;
  guest_output_paint_descriptor_pool_create_info.poolSizeCount =
      uint32_t(xe::countof(guest_output_paint_descriptor_pool_sizes));
  guest_output_paint_descriptor_pool_create_info.pPoolSizes =
      guest_output_paint_descriptor_pool_sizes;
  if (dfn.vkCreateDescriptorPool(
          device, &guest_output_paint_descriptor_pool_create_info, nullptr,
          &paint_context_.guest_output_descriptor_pool) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to create the guest output painting "
        "descriptor pool");
    return false;
  }
  VkDescriptorSetLayout guest_output_paint_descriptor_set_layouts
      [PaintContext::kGuestOutputDescriptorSetCount];
  std::fill(guest_output_paint_descriptor_set_layouts,
            guest_output_paint_descriptor_set_layouts +
                xe::countof(guest_output_paint_descriptor_set_layouts),
            guest_output_paint_image_descriptor_set_layout_);
  VkDescriptorSetAllocateInfo guest_output_paint_descriptor_set_allocate_info;
  guest_output_paint_descriptor_set_allocate_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  guest_output_paint_descriptor_set_allocate_info.pNext = nullptr;
  guest_output_paint_descriptor_set_allocate_info.descriptorPool =
      paint_context_.guest_output_descriptor_pool;
  guest_output_paint_descriptor_set_allocate_info.descriptorSetCount =
      PaintContext::kGuestOutputDescriptorSetCount;
  guest_output_paint_descriptor_set_allocate_info.pSetLayouts =
      guest_output_paint_descriptor_set_layouts;
  if (dfn.vkAllocateDescriptorSets(
          device, &guest_output_paint_descriptor_set_allocate_info,
          paint_context_.guest_output_descriptor_sets) != VK_SUCCESS) {
    XELOGE(
        "VulkanPresenter: Failed to allocate the guest output painting "
        "descriptor sets");
    return false;
  }

  return InitializeCommonSurfaceIndependent();
}

void VulkanPresenter::BeginZeroFGSurfaceDisconnect() {
  if (zerofg_vulkan_context_->handoff) {
    zerofg_vulkan_context_->handoff->DisarmAndQuiesceSource();
  }
  if (zerofg_independent_presenter_) {
    zerofg_independent_presenter_->BeginSurfaceDisconnect();
  }
}

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
namespace {
// An eventfd write never blocks here; retrying EINTR means no wake is lost,
// above all the one that lets the observer exit before its join.
// Returns false on a failure other than EINTR.
}  // namespace
#endif

void VulkanPresenter::DestroyZeroFGSurfaceResourcesAfterSourceIdle() {
  // Make the ordering explicit even when the independent presenter has
  // already been detached: A's marker must be quiesced before any B teardown.
  if (zerofg_vulkan_context_->handoff) {
    zerofg_vulkan_context_->handoff->DisarmAndQuiesceSource();
  }
  guest_output_image_refresher_completion_timeline_.AwaitAllSubmissions();
  if (zerofg_independent_presenter_) {
    zerofg_independent_presenter_->DestroySurfaceResourcesAfterSourceIdle();
  }
  // The independent presenter drains B before invoking its Generation/Post
  // shutdown callbacks. Keep this fallback for a presenter that was never
  // surface-connected, and make the single physical authority obvious here.
  if (ZeroFGDevice()->is_zerofg_presenter_device() &&
      !ZeroFGDevice()->DrainZeroFGForTeardown()) {
    XELOGW("ZeroFGC0: device-B teardown drain failed before context destroy");
  }
  DestroyZeroFGGenerationContext();
  DestroyZeroFGPostContext();
  if (zerofg_vulkan_context_->handoff) {
    zerofg_vulkan_context_->handoff->DestroyAfterIdle();
  }
}

void VulkanPresenter::DestroyZeroFGPostContext() {
  if (!zerofg_vulkan_context_->post) {
    return;
  }
  ZeroFGPostContext& context = *zerofg_vulkan_context_->post;
  const VulkanDevice::Functions& dfn = ZeroFGDevice()->functions();
  const VkDevice device = ZeroFGDevice()->device();
  for (ZeroFGPostContext::JobContext& job : context.jobs) {
    util::DestroyAndNullHandle(dfn.vkDestroySemaphore, device,
                               job.acquire_semaphore);
    job.completion.DestroyTimeline(ZeroFGDevice());
    util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device,
                               job.final_framebuffer);
    for (VkFramebuffer& framebuffer : job.intermediate_framebuffers) {
      util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device,
                                 framebuffer);
    }
    job.logical_input.reset();
    for (auto& image : job.intermediate_images) {
      image.reset();
    }
    util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device,
                               job.descriptor_pool);
    util::DestroyAndNullHandle(dfn.vkDestroyCommandPool, device,
                               job.command_pool);
    job.command_buffer = VK_NULL_HANDLE;
  }
  for (VkPipeline& pipeline : context.final_pipelines) {
    util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device, pipeline);
  }
  util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device,
                             context.final_render_pass);
  util::DestroyAndNullHandle(dfn.vkDestroySemaphore, device,
                             context.completion_timeline);
  zerofg_vulkan_context_->post.reset();
}

void VulkanPresenter::DestroyZeroFGGenerationContext() {
  if (!zerofg_vulkan_context_->generation) {
    return;
  }
  ZeroFGGenerationContext& context = *zerofg_vulkan_context_->generation;
  const VulkanDevice::Functions& dfn = ZeroFGDevice()->functions();
  const VkDevice device = ZeroFGDevice()->device();
  context.outputs = {};
  context.adapter.reset();
  for (ZeroFGGenerationContext::CommandContext& command :
       context.commands) {
    util::DestroyAndNullHandle(dfn.vkDestroyCommandPool, device,
                               command.command_pool);
    command.command_buffer = VK_NULL_HANDLE;
    command.completion.DestroySyncFd(ZeroFGDevice());
    command.completion.DestroyTimeline(ZeroFGDevice());
  }
  util::DestroyAndNullHandle(dfn.vkDestroySemaphore, device,
                             context.completion_timeline);
  util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                             context.timestamp_query_pool);
  zerofg_vulkan_context_->generation.reset();
}

void VulkanPresenter::ReleaseZeroFGSynthetic(uint32_t synthetic_index) {
  if (!zerofg_vulkan_context_->generation ||
      synthetic_index >= ZeroFGGenerationContext::kContextCount) {
    return;
  }
  ZeroFGGenerationContext& context = *zerofg_vulkan_context_->generation;
  ZeroFGGenerationContext::CommandContext& command =
      context.commands[synthetic_index];
  command.submitted = false;
  // The Generation has retired (its Post completed, or a poll observed it).
  // Rule 5 falsifier: while this Generation still holds its own sync_fd,
  // check that claim without waiting before the context is handed back. A
  // pending fd means the next signal of this context's timeline could wait
  // on it.
  if (command.completion.occupied() &&
      command.completion.sync_fd_observable()) {
    ++context.owner_stats.release_checks;
    const ZeroFGCompletionOwner::SyncFdPoll poll_state =
        command.completion.PollSyncFd();
    if (poll_state == ZeroFGCompletionOwner::SyncFdPoll::kPending) {
      ++context.owner_stats.release_unretired;
    } else if (poll_state == ZeroFGCompletionOwner::SyncFdPoll::kError) {
      ++context.owner_stats.sync_fd_errors;
      command.completion.AbandonSyncFd();
    }
  }
  // The sync_fd is no longer needed. A pending recreation is kept: a
  // semaphore whose export failed still holds its signal and is replaced
  // before this context's next Generation submit.
  command.completion.RetireSyncFd();
  command.completion.MarkRetired();
  context.output_busy[synthetic_index] = false;
}

bool VulkanPresenter::PrepareZeroFGGenerationContext(
    ZeroFGIndependentPresenter::GenerationResult& result,
    uint64_t& setup_ns, bool& created_out) {
  using FailureStage =
      ZeroFGIndependentPresenter::GenerationFailureStage;
  created_out = false;
  if (zerofg_vulkan_context_->generation) {
    return true;
  }

  const uint64_t setup_begin_ns = GetZeroFGMonotonicTimeNs();
  const VulkanDevice::Functions& dfn = ZeroFGDevice()->functions();
  const VkDevice device = ZeroFGDevice()->device();
  const uint32_t queue_family = ZeroFGDevice()->queue_family_graphics_compute();
  auto context = std::make_unique<ZeroFGGenerationContext>();
  const VulkanInstance* instance = ZeroFGDevice()->vulkan_instance();
  const bool backend_fallback = zerofg_vulkan_context_->backend_fallback_active;
  if (backend_fallback) {
    XELOGW("ZeroFGBackendFallback active=true: this generation context runs "
           "the Compat backend");
  }
  zerofg::VulkanContext vulkan = {};
  vulkan.instance = instance->instance();
  vulkan.physical_device = ZeroFGDevice()->physical_device();
  vulkan.device = device;
  vulkan.get_instance_proc_addr = instance->functions().vkGetInstanceProcAddr;
  zerofg::Status adapter_status = zerofg::Status::kSuccess;
  zerofg::Capabilities capabilities =
      ZeroFGDevice()->zerofg_backend_capabilities();
  capabilities.effective_api_version = ZeroFGDevice()->properties().apiVersion;
  capabilities.synchronization2_enabled =
      ZeroFGDevice()->properties().synchronization2;
  capabilities.shader_storage_image_extended_formats_enabled =
      ZeroFGDevice()->properties().shaderStorageImageExtendedFormats;
  VkSemaphoreTypeCreateInfo timeline_type = {
      VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo semaphore_info = {
      VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphore_info.pNext = &timeline_type;
  const VkResult timeline_result = dfn.vkCreateSemaphore(
      device, &semaphore_info, nullptr, &context->completion_timeline);
  if (timeline_result != VK_SUCCESS) {
    result.failure_stage = FailureStage::kAdapter;
    result.vk_result = timeline_result;
    result.vk_result_valid = true;
    setup_ns += GetZeroFGMonotonicTimeNs() - setup_begin_ns;
    return false;
  }

  const VulkanDevice::QueueFamily& queue_family_properties =
      ZeroFGDevice()->queue_families()[queue_family];
  context->timestamp_valid_bits =
      queue_family_properties.timestamp_valid_bits;
  context->timestamp_period_ns =
      double(ZeroFGDevice()->properties().timestampPeriod);
  if (context->timestamp_valid_bits && context->timestamp_period_ns > 0.0) {
    context->timestamp_query_count_per_context = kZeroFGTimestampCount;
    VkQueryPoolCreateInfo query_info = {
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = ZeroFGGenerationContext::kContextCount *
                            context->timestamp_query_count_per_context;
    if (dfn.vkCreateQueryPool(device, &query_info, nullptr,
                              &context->timestamp_query_pool) != VK_SUCCESS) {
      context->timestamp_query_pool = VK_NULL_HANDLE;
      context->timestamp_valid_bits = 0;
      context->timestamp_query_count_per_context = 0;
    }
  }
  // Auto: the generic fast paths the device really has (Modern); a failure
  // retries once on Compat (ZeroFGBackendFallback).
  context->adapter = zerofg::xenia::Adapter::Create(
      vulkan, ZeroFGGenerationContext::kContextCount,
      IsReallyZeroRequested() ? zerofg::Mode::kReallyZero : zerofg::Mode::kZero,
      backend_fallback ? zerofg::Backend::kCompat : zerofg::Backend::kAuto,
      capabilities, &adapter_status);
  if (!context->adapter) {
    util::DestroyAndNullHandle(dfn.vkDestroyQueryPool, device,
                               context->timestamp_query_pool);
    util::DestroyAndNullHandle(dfn.vkDestroySemaphore, device,
                               context->completion_timeline);
    if (!backend_fallback) {
      XELOGW("ZeroFGBackendFallback stage=adapter status={}: retrying the "
             "generation context on the Compat backend",
             uint32_t(adapter_status));
      zerofg_vulkan_context_->backend_fallback_active = true;
      context.reset();
      return PrepareZeroFGGenerationContext(result, setup_ns, created_out);
    }
    result.failure_stage = FailureStage::kAdapter;
    result.zerofg_status = uint32_t(adapter_status);
    setup_ns += GetZeroFGMonotonicTimeNs() - setup_begin_ns;
    return false;
  }
  XELOGI("ZeroFGVariant mode={} backend={} timestamp_valid_bits={} "
         "timestamp_period_ns={}",
         ZeroFGSelectionName(), backend_fallback ? "compat" : "auto",
         context->timestamp_valid_bits, context->timestamp_period_ns);

  context->sync_fd_observation = kZeroFGSyncFdSupported;
  for (auto& command : context->commands) {
    VkCommandPoolCreateInfo pool_info = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                      VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = queue_family;
    const VkResult pool_result = dfn.vkCreateCommandPool(
        device, &pool_info, nullptr, &command.command_pool);
    if (pool_result != VK_SUCCESS) {
      result.failure_stage = FailureStage::kAdapter;
      result.vk_result = pool_result;
      result.vk_result_valid = true;
      zerofg_vulkan_context_->generation = std::move(context);
      DestroyZeroFGGenerationContext();
      setup_ns += GetZeroFGMonotonicTimeNs() - setup_begin_ns;
      return false;
    }
    VkCommandBufferAllocateInfo allocate = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = command.command_pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    const VkResult allocate_result = dfn.vkAllocateCommandBuffers(
        device, &allocate, &command.command_buffer);
    if (allocate_result != VK_SUCCESS) {
      result.failure_stage = FailureStage::kAdapter;
      result.vk_result = allocate_result;
      result.vk_result_valid = true;
      zerofg_vulkan_context_->generation = std::move(context);
      DestroyZeroFGGenerationContext();
      setup_ns += GetZeroFGMonotonicTimeNs() - setup_begin_ns;
      return false;
    }
    if (context->sync_fd_observation &&
        command.completion.CreateSyncFd(
            ZeroFGDevice(), "ZeroFG Generation sync_fd") != VK_SUCCESS) {
      // ZeroFG needs the per-command sync_fd semaphore.
      command.completion.SetSyncFdFallback();
      zerofg_vulkan_context_->generation = std::move(context);
      DestroyZeroFGGenerationContext();
      result.failure_stage = FailureStage::kAdapter;
      return false;
    }
    if (command.completion.CreateTimeline(
            ZeroFGDevice(),
            "ZeroFG Generation completion " +
                std::to_string(&command - context->commands.data())) !=
        VK_SUCCESS) {
      ++context->owner_stats.timeline_create_failures;
      zerofg_vulkan_context_->generation = std::move(context);
      DestroyZeroFGGenerationContext();
      result.failure_stage = FailureStage::kAdapter;
      return false;
    }
  }

  zerofg_vulkan_context_->generation = std::move(context);
  setup_ns += GetZeroFGMonotonicTimeNs() - setup_begin_ns;
  created_out = true;
  return true;
}

bool VulkanPresenter::ReadZeroFGPreviousProfilingSample(
    uint32_t synthetic_index,
    ZeroFGIndependentPresenter::GenerationResult& result) {
  if (!zerofg_vulkan_context_->generation ||
      synthetic_index >= ZeroFGGenerationContext::kContextCount) {
    return false;
  }
  ZeroFGGenerationContext& context = *zerofg_vulkan_context_->generation;
  ZeroFGGenerationContext::CommandContext& command =
      context.commands[synthetic_index];
  if (!command.profiling_sample_pending) {
    return true;
  }

  ++result.profile_readback_attempts;
  // The slot is being reused only after its previous submission completed.
  // Read without WAIT and consume the sample before the new query range is
  // reset. A lost/invalid profile must never affect product scheduling.
  command.profiling_sample_pending = false;
  if (context.timestamp_query_pool == VK_NULL_HANDLE ||
      !context.timestamp_query_count_per_context) {
    ++result.profile_readback_errors;
    result.profile_readback_last_error = VK_ERROR_INITIALIZATION_FAILED;
    return false;
  }

  const uint32_t timestamp_count = context.timestamp_query_count_per_context;
  std::array<uint64_t, kZeroFGTimestampCount> timestamps = {};
  const VkResult query_result =
      ZeroFGDevice()->functions().vkGetQueryPoolResults(
          ZeroFGDevice()->device(), context.timestamp_query_pool,
          synthetic_index * timestamp_count, timestamp_count,
          sizeof(uint64_t) * timestamp_count, timestamps.data(),
          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
  if (query_result == VK_NOT_READY) {
    ++result.profile_readback_not_ready;
    return false;
  }
  if (query_result != VK_SUCCESS) {
    ++result.profile_readback_errors;
    result.profile_readback_last_error = query_result;
    return false;
  }

  ++result.profile_readback_successes;
  result.drained_profile_sample_valid = true;
  auto& sample = result.drained_profile_sample;
  sample.submit_call_ns = command.submit_call_ns;

  const auto delta_ticks = [&context](uint64_t begin,
                                      uint64_t end) -> uint64_t {
    if (context.timestamp_valid_bits < 64) {
      const uint64_t mask =
          (uint64_t(1) << context.timestamp_valid_bits) - 1;
      return (end - begin) & mask;
    }
    return end - begin;
  };
  sample.service_gpu_ns = uint64_t(
      double(delta_ticks(timestamps[0], timestamps[1])) *
      context.timestamp_period_ns);
  sample.timestamp_valid = sample.service_gpu_ns != 0;
  return true;
}

bool VulkanPresenter::ProcessZeroFGGeneration(
    const ZeroFGIndependentPresenter::GenerationRequest& request,
    ZeroFGIndependentPresenter::GenerationResult& result) {
  using FailureStage =
      ZeroFGIndependentPresenter::GenerationFailureStage;
  result = {};
  const uint64_t request_begin_ns = GetZeroFGMonotonicTimeNs();
  uint64_t setup_ns = 0;
  const auto finish_failure = [&result, &setup_ns, request_begin_ns]() {
    result.setup_ns = setup_ns;
    result.total_ns = GetZeroFGMonotonicTimeNs() - request_begin_ns;
  };
  const auto fail = [&result, &finish_failure](FailureStage stage) {
    result.failure_stage = stage;
    finish_failure();
    return false;
  };
  const auto fail_status = [&result, &finish_failure](
                               FailureStage stage, zerofg::Status status) {
    result.failure_stage = stage;
    result.zerofg_status = uint32_t(status);
    finish_failure();
    return false;
  };
  const auto fail_vk = [this, &result, &finish_failure](FailureStage stage,
                                                        VkResult vk_result) {
    if (vk_result == VK_ERROR_DEVICE_LOST) {
      ZeroFGDevice()->SetLost();
      XELOGE(
          "ZeroFGDeviceB lost operation=Generation stage={} "
          "Source_A_untouched=true",
          uint32_t(stage));
    }
    result.failure_stage = stage;
    result.vk_result = vk_result;
    result.vk_result_valid = true;
    finish_failure();
    return false;
  };
  const auto input_waits_valid = [&request]() {
    if (!request.input_wait_count ||
        request.input_wait_count > request.input_wait_semaphores.size()) {
      return false;
    }
    for (uint32_t i = 0; i < request.input_wait_count; ++i) {
      if (request.input_wait_semaphores[i] == VK_NULL_HANDLE ||
          !request.input_wait_values[i]) {
        return false;
      }
    }
    // Each semaphore appears once: the presenter folds A/B Residency images
    // on the same timeline into one wait with the higher value.
    return request.input_wait_count < 2 ||
           request.input_wait_semaphores[0] !=
               request.input_wait_semaphores[1];
  };
  if (request.previous_image == VK_NULL_HANDLE ||
      request.previous_view == VK_NULL_HANDLE ||
      request.current_image == VK_NULL_HANDLE ||
      request.current_view == VK_NULL_HANDLE || !request.extent.width ||
      !request.extent.height || !request.source_physical_extent.width ||
      !request.source_physical_extent.height ||
      request.source_physical_extent.width < request.extent.width ||
      request.source_physical_extent.height < request.extent.height ||
      !input_waits_valid() ||
      !request.previous_source_id ||
      // Accepted adjacency forms the pair; an IssueSwap gap is valid.
      request.current_source_id <= request.previous_source_id) {
    return fail(FailureStage::kRequestValidation);
  }

  const VulkanDevice::Functions& dfn = ZeroFGDevice()->functions();
  const VkDevice device = ZeroFGDevice()->device();
  const uint32_t queue_family = ZeroFGDevice()->queue_family_graphics_compute();
  const uint32_t queue_index = ZeroFGDevice()->queue_index_zerofg_presenter();
  bool generation_context_created = false;
  if (!PrepareZeroFGGenerationContext(result, setup_ns,
                                         generation_context_created)) {
    finish_failure();
    return false;
  }
  if (generation_context_created) {
    // Pipeline compilation is first-use-heavy. If prewarming was
    // unavailable, yield after creating the context rather than combining it
    // with Resize, command recording and queue acquisition in one arbiter
    // quantum.
    return fail(FailureStage::kPreparationYield);
  }

  ZeroFGGenerationContext& context = *zerofg_vulkan_context_->generation;
  if (context.extent.width != request.extent.width ||
      context.extent.height != request.extent.height) {
    const uint64_t setup_begin_ns = GetZeroFGMonotonicTimeNs();
    if (std::any_of(context.output_busy.begin(), context.output_busy.end(),
                    [](bool busy) { return busy; })) {
      return fail(FailureStage::kResize);
    }
    const zerofg::Status resize_status = context.adapter->Resize(
        request.extent.width, request.extent.height, kGuestOutputFormat,
        kGuestOutputFormat);
    if (resize_status != zerofg::Status::kSuccess) {
      if (resize_status == zerofg::Status::kVulkanError &&
          !zerofg_vulkan_context_->backend_fallback_active) {
        // A Modern fast path must never take generation down: drop the
        // context and let the next cycle recreate it on Compat.
        XELOGW("ZeroFGBackendFallback stage=resize status={}: recreating the "
               "generation context on the Compat backend",
               uint32_t(resize_status));
        zerofg_vulkan_context_->backend_fallback_active = true;
        DestroyZeroFGGenerationContext();
        return fail(FailureStage::kPreparationYield);
      }
      return fail_status(FailureStage::kResize, resize_status);
    }
    context.outputs = {};
    context.output_initialized = {};
    for (auto& output : context.outputs) {
      output = GuestOutputImage::Create(ZeroFGDevice(), request.extent.width,
                                        request.extent.height,
                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT);
      if (!output) {
        context.outputs = {};
        return fail(FailureStage::kResize);
      }
    }
    context.extent = request.extent;
    setup_ns += GetZeroFGMonotonicTimeNs() - setup_begin_ns;
    // Return Source-critical authority between first-use allocation and the
    // first Generation submission. The same logical retries next cycle.
    return fail(FailureStage::kPreparationYield);
  }

  uint32_t selected = UINT32_MAX;
  for (uint32_t i = 0; i < ZeroFGGenerationContext::kContextCount; ++i) {
    if (!context.output_busy[i]) {
      context.output_busy[i] = true;
      selected = i;
      break;
    }
  }
  if (selected == UINT32_MAX) {
    return fail(FailureStage::kSyntheticPool);
  }
  const auto release_before_submit = [&context, selected]() {
    context.output_busy[selected] = false;
  };
  uint32_t occupancy = 0;
  for (bool busy : context.output_busy) {
    occupancy += busy;
  }
  context.pool_high_water = std::max(context.pool_high_water, occupancy);
  result.pool_occupancy = occupancy;
  result.pool_high_water = context.pool_high_water;

  ZeroFGGenerationContext::CommandContext& command =
      context.commands[selected];
  // Drain the previous generation's profiling queries before resetting this
  // slot for the new command buffer. This is a non-blocking observation only;
  // the slot-reuse contract already proves the prior submission completed.
  ReadZeroFGPreviousProfilingSample(selected, result);
  VkResult vk_result =
      dfn.vkResetCommandPool(device, command.command_pool, 0);
  if (vk_result != VK_SUCCESS) {
    release_before_submit();
    return fail_vk(FailureStage::kCommandReset, vk_result);
  }
  VkCommandBufferBeginInfo begin = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk_result = dfn.vkBeginCommandBuffer(command.command_buffer, &begin);
  if (vk_result != VK_SUCCESS) {
    release_before_submit();
    return fail_vk(FailureStage::kCommandBegin, vk_result);
  }

  VkImageMemoryBarrier real_acquires[2] = {};
  const VkImageLayout real_layouts[2] = {request.previous_layout,
                                         request.current_layout};
  const VkImage real_images[2] = {request.previous_image,
                                  request.current_image};
  for (uint32_t i = 0; i < xe::countof(real_acquires); ++i) {
    VkImageMemoryBarrier& barrier = real_acquires[i];
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.oldLayout = real_layouts[i];
    barrier.newLayout = real_layouts[i];
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.image = real_images[i];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
  }
  dfn.vkCmdPipelineBarrier(command.command_buffer,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, uint32_t(xe::countof(real_acquires)),
                           real_acquires);

  if (!context.output_initialized[selected]) {
    VkImageMemoryBarrier initialize = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    initialize.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    initialize.newLayout = kGuestOutputInternalLayout;
    initialize.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    initialize.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    initialize.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    initialize.image = context.outputs[selected]->image();
    initialize.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    initialize.subresourceRange.levelCount = 1;
    initialize.subresourceRange.layerCount = 1;
    dfn.vkCmdPipelineBarrier(command.command_buffer,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &initialize);
  }

  // `sequence` is the accepted-Real identity (the source id): RC3 uses the
  // previous pair's motion only when this pair starts at the Real the
  // previous pair ended on. 0 = unknown (no temporal prior); other modes
  // ignore it.
  const auto wrap_real = [&request](VkImage image, VkImageView view,
                                    VkImageLayout layout,
                                    uint64_t sequence) {
    zerofg::Image wrapped = {};
    wrapped.image = image;
    wrapped.view = view;
    wrapped.layout = layout;
    wrapped.sequence = sequence;
    wrapped.format = kGuestOutputFormat;
    wrapped.width = request.source_physical_extent.width;
    wrapped.height = request.source_physical_extent.height;
    wrapped.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT;
    // AllocateResidencySlot creates optimal, flags=0, identity 2D views.
    wrapped.creation_facts_known = true;
    wrapped.active_rect = {0, 0, request.extent.width, request.extent.height};
    return wrapped;
  };
  const auto wrap_internal = [&request](const GuestOutputImage& image) {
    zerofg::Image wrapped = {};
    wrapped.image = image.image();
    wrapped.view = image.view();
    wrapped.layout = kGuestOutputInternalLayout;
    wrapped.format = kGuestOutputFormat;
    wrapped.width = request.extent.width;
    wrapped.height = request.extent.height;
    wrapped.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_STORAGE_BIT;
    wrapped.creation_facts_known = true;
    wrapped.active_rect = {0, 0, request.extent.width, request.extent.height};
    return wrapped;
  };
  const uint32_t timestamp_query =
      selected * context.timestamp_query_count_per_context;
  if (context.timestamp_query_pool != VK_NULL_HANDLE) {
    dfn.vkCmdResetQueryPool(command.command_buffer,
                            context.timestamp_query_pool, timestamp_query,
                            context.timestamp_query_count_per_context);
    // The pair around the Generation: its GPU service time.
    dfn.vkCmdWriteTimestamp(command.command_buffer,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            context.timestamp_query_pool, timestamp_query);
  }
  const zerofg::Status interpolate_status = context.adapter->Interpolate(
      command.command_buffer, selected,
      wrap_real(request.previous_image, request.previous_view,
                request.previous_layout, request.previous_source_id),
      wrap_real(request.current_image, request.current_view,
                request.current_layout, request.current_source_id),
      0.5f,
      wrap_internal(*context.outputs[selected]));
  if (interpolate_status != zerofg::Status::kSuccess) {
    dfn.vkResetCommandPool(device, command.command_pool, 0);
    release_before_submit();
    return fail_status(FailureStage::kRecord, interpolate_status);
  }
  if (context.timestamp_query_pool != VK_NULL_HANDLE) {
    dfn.vkCmdWriteTimestamp(command.command_buffer,
                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            context.timestamp_query_pool,
                            timestamp_query + 1);
  }

  VkImageMemoryBarrier real_releases[2] = {};
  for (uint32_t i = 0; i < xe::countof(real_releases); ++i) {
    VkImageMemoryBarrier& barrier = real_releases[i];
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = real_layouts[i];
    barrier.newLayout = real_layouts[i];
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = real_images[i];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
  }
  dfn.vkCmdPipelineBarrier(command.command_buffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                           0, nullptr, uint32_t(xe::countof(real_releases)),
                           real_releases);
  vk_result = dfn.vkEndCommandBuffer(command.command_buffer);
  if (vk_result != VK_SUCCESS) {
    release_before_submit();
    return fail_vk(FailureStage::kCommandEnd, vk_result);
  }

  // V2-1: signal this context's own timeline. Its previous point belongs to
  // this context's previous Generation, which retired before the context was
  // released, so the signal's cleanup never waits on another context's
  // pending Generation. Without it, the shared Generation timeline.
  const bool completion_per_context = command.completion.owns_timeline();
  const VkSemaphore generation_timeline = completion_per_context
                                              ? command.completion.timeline()
                                              : context.completion_timeline;
  const uint64_t signal_value = completion_per_context
                                    ? command.completion.ClaimTimelineValue()
                                    : context.next_timeline_value++;
  if (command.completion.occupied()) {
    ++context.owner_stats.reuse_before_retire;
  }
  // D3: a sync semaphore whose export failed on an earlier use still holds
  // that signal. This context was released, so its previous Generation has
  // retired; replace the semaphore before signaling it again.
  if (command.completion.sync_fd_recreate_pending()) {
    command.completion.RecreateSyncFd(ZeroFGDevice(),
                                      "ZeroFG Generation sync_fd");
  }
  const bool signal_sync_fd =
      context.sync_fd_observation && command.completion.can_signal_sync_fd();
  const uint32_t signal_semaphore_count = signal_sync_fd ? 2 : 1;
  const VkSemaphore signal_semaphores[2] = {
      generation_timeline, command.completion.sync_semaphore()};
  // One value per signal semaphore; the binary sync_fd semaphore's is ignored.
  const uint64_t signal_values[2] = {signal_value, 0};
  // V2-2: one input wait per readiness timeline. The presenter folds the A/B
  // Residency images into one wait when they share a timeline and passes two
  // when each Residency slot owns its readiness timeline.
  const VkPipelineStageFlags input_wait_stages[2] = {
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
  VkTimelineSemaphoreSubmitInfo timeline_submit = {
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline_submit.waitSemaphoreValueCount = request.input_wait_count;
  timeline_submit.pWaitSemaphoreValues = request.input_wait_values.data();
  timeline_submit.signalSemaphoreValueCount = signal_semaphore_count;
  timeline_submit.pSignalSemaphoreValues = signal_values;
  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.pNext = &timeline_submit;
  submit.waitSemaphoreCount = request.input_wait_count;
  submit.pWaitSemaphores = request.input_wait_semaphores.data();
  submit.pWaitDstStageMask = input_wait_stages;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command.command_buffer;
  submit.signalSemaphoreCount = signal_semaphore_count;
  submit.pSignalSemaphores = signal_semaphores;
  const uint64_t queue_request_ns = GetZeroFGMonotonicTimeNs();
  uint64_t submit_begin_ns = 0;
  {
    std::optional<VulkanDevice::Queue::Acquisition> queue =
        ZeroFGDevice()->TryAcquireQueue(queue_family, queue_index);
    while (!queue && request.queue_commit_allowed &&
           request.queue_commit_allowed()) {
      std::this_thread::yield();
      queue = ZeroFGDevice()->TryAcquireQueue(queue_family, queue_index);
    }
    if (queue && request.queue_commit_allowed &&
        !request.queue_commit_allowed()) {
      queue.reset();
    }
    if (!queue) {
      result.queue_wait_ns = GetZeroFGMonotonicTimeNs() - queue_request_ns;
      // Physical Source runway or lifecycle state revoked this S's q0
      // authority. Keep it pending and return to the arbiter without submit.
      dfn.vkResetCommandPool(device, command.command_pool, 0);
      release_before_submit();
      return fail(FailureStage::kQueueBusy);
    }
    if (ZeroFGDevice()->RejectZeroFGSubmitAfterTeardownIdle()) {
      dfn.vkResetCommandPool(device, command.command_pool, 0);
      release_before_submit();
      return fail(FailureStage::kQueueBusy);
    }
    submit_begin_ns = GetZeroFGMonotonicTimeNs();
    result.queue_wait_ns = submit_begin_ns - queue_request_ns;
    vk_result =
        dfn.vkQueueSubmit(queue->queue(), 1, &submit, VK_NULL_HANDLE);
    result.submit_host_ns = GetZeroFGMonotonicTimeNs() - submit_begin_ns;
  }
  if (vk_result != VK_SUCCESS) {
    release_before_submit();
    return fail_vk(FailureStage::kQueueSubmit, vk_result);
  }
  result.submission_accepted = true;
  command.signal_value = signal_value;
  command.submit_time_ns = GetZeroFGMonotonicTimeNs();
  command.submitted = true;
  command.completion.MarkSubmitted(generation_timeline);
  context.owner_stats.CountSubmit(completion_per_context,
                                  result.submit_host_ns);
  // D3: export the sync_fd after queue acceptance, outside the queue lock.
  // Without an fd this Generation falls back to its completion timeline if
  // it ever needs CPU observation.
  command.completion.CloseSyncFd();
  if (signal_sync_fd) {
    const uint64_t export_begin_ns = GetZeroFGMonotonicTimeNs();
    const bool exported = command.completion.ExportSyncFd(ZeroFGDevice());
    result.completion_fd_export_attempted = true;
    result.completion_fd_export_host_ns =
        GetZeroFGMonotonicTimeNs() - export_begin_ns;
    result.completion_fd_exported = exported;
    result.completion_fd_fallback = !exported;
  } else if (context.sync_fd_observation) {
    command.completion.SetSyncFdFallback();
    result.completion_fd_fallback = true;
  }
  context.output_initialized[selected] = true;
  result.image = context.outputs[selected]->image();
  result.view = context.outputs[selected]->view();
  result.extent = context.extent;
  result.synthetic_index = selected;
  result.signal_value = signal_value;
  result.completion_semaphore = generation_timeline;
  result.completion_per_context = completion_per_context;
  result.completion_owner_stats = context.owner_stats;
  result.submit_time_ns = command.submit_time_ns;
  command.submit_call_ns = result.submit_host_ns;
  command.profiling_sample_pending =
      context.timestamp_query_pool != VK_NULL_HANDLE;
  result.setup_ns = setup_ns;
  result.total_ns = GetZeroFGMonotonicTimeNs() - request_begin_ns;
  return true;
}

bool VulkanPresenter::PollZeroFGGeneration(
    uint32_t synthetic_index,
    ZeroFGIndependentPresenter::GenerationResult& result,
    bool& ready_out) {
  using FailureStage =
      ZeroFGIndependentPresenter::GenerationFailureStage;
  result = {};
  ready_out = false;
  if (!zerofg_vulkan_context_->generation ||
      synthetic_index >= ZeroFGGenerationContext::kContextCount) {
    result.failure_stage = FailureStage::kRequestValidation;
    return false;
  }
  ZeroFGGenerationContext& context = *zerofg_vulkan_context_->generation;
  ZeroFGGenerationContext::CommandContext& command =
      context.commands[synthetic_index];
  if (!command.submitted || !command.signal_value) {
    result.failure_stage = FailureStage::kRequestValidation;
    return false;
  }
  // D3: observe an orphaned or dropped Generation through its sync_fd with a
  // non-waiting poll. The counter query on the timeline this Generation
  // signaled (the shared one, or its context's own with V2-1) stays as the
  // fallback when no fd was exported; on Turnip/KGSL it waits on pending GPU
  // work.
  bool complete = false;
  bool observed = false;
  if (ZeroFGDevice()->zerofg_teardown_idle()) {
    complete = observed = true;
  } else if (command.completion.sync_fd_observable()) {
    const uint64_t poll_begin_ns = GetZeroFGMonotonicTimeNs();
    const ZeroFGCompletionOwner::SyncFdPoll poll_state =
        command.completion.PollSyncFd();
    result.completion_fd_poll_host_ns =
        GetZeroFGMonotonicTimeNs() - poll_begin_ns;
    if (poll_state == ZeroFGCompletionOwner::SyncFdPoll::kError) {
      // The fd proves nothing any more; the timeline decides below.
      ++context.owner_stats.sync_fd_errors;
      command.completion.AbandonSyncFd();
    } else {
      result.completion_fd_polled = true;
      observed = true;
      complete = poll_state == ZeroFGCompletionOwner::SyncFdPoll::kSignaled;
    }
  }
  if (!observed) {
    result.failure_stage = FailureStage::kTimelineQuery;
    result.submission_accepted = true;
    return false;  // Fail-open drain owns retirement, never query pending B.
  }
  result.completion_owner_stats = context.owner_stats;
  if (!complete) {
    return true;
  }
  command.completion.CloseSyncFd();
  command.completion.MarkRetired();
  ready_out = true;
  result.submission_accepted = true;
  result.signal_value = command.signal_value;
  result.submit_time_ns = command.submit_time_ns;
  result.submit_to_ready_ns =
      GetZeroFGMonotonicTimeNs() - command.submit_time_ns;
  result.gpu_completion_ns = result.submit_to_ready_ns;
  result.image = context.outputs[synthetic_index]->image();
  result.view = context.outputs[synthetic_index]->view();
  result.extent = context.extent;
  result.synthetic_index = synthetic_index;
  // Profiling queries are drained by ProcessZeroFGGeneration when this slot
  // is reused. Polling here must remain completion/liveness-only: the normal
  // Generation->Post GPU chain deliberately does not require a CPU readback.
  command.submitted = false;
  return true;
}

bool VulkanPresenter::ProcessZeroFGPost(
    const ZeroFGIndependentPresenter::PostProcessRequest& request,
    ZeroFGIndependentPresenter::PostProcessResult& result) {
  using PostFailureStage =
      ZeroFGIndependentPresenter::PostFailureStage;
  result = {};
  const auto fail = [&result](PostFailureStage stage) {
    result.failure_stage = stage;
    return false;
  };
  const auto fail_vk = [this, &result](PostFailureStage stage,
                                       VkResult vk_result) {
    if (vk_result == VK_ERROR_DEVICE_LOST) {
      ZeroFGDevice()->SetLost();
      XELOGE(
          "ZeroFGDeviceB lost operation=Post stage={} Source_A_untouched=true",
          uint32_t(stage));
    }
    result.failure_stage = stage;
    result.vk_result = vk_result;
    result.vk_result_valid = true;
    return false;
  };
  const auto handle_value = [](auto handle) -> uint64_t {
    using Handle = decltype(handle);
    if constexpr (std::is_pointer_v<Handle>) {
      return uint64_t(reinterpret_cast<uintptr_t>(handle));
    } else {
      return uint64_t(handle);
    }
  };
  if (!request.frontbuffer_width || !request.frontbuffer_height ||
      !request.display_aspect_ratio_x || !request.display_aspect_ratio_y ||
      request.candidate_image == VK_NULL_HANDLE ||
      request.candidate_layout == VK_IMAGE_LAYOUT_UNDEFINED ||
      request.candidate_storage_extent.width < request.frontbuffer_width ||
      request.candidate_storage_extent.height < request.frontbuffer_height ||
      request.final_output_image == VK_NULL_HANDLE ||
      request.final_output_view == VK_NULL_HANDLE ||
      !request.final_output_extent.width ||
      !request.final_output_extent.height ||
      request.candidate_image == request.final_output_image ||
      request.final_output_index >=
          ZeroFGIndependentPresenter::kFinalOutputPoolSize ||
      ((request.candidate_wait_semaphore == VK_NULL_HANDLE) !=
       (request.candidate_wait_value == 0)) ||
      request.final_output_format == VK_FORMAT_UNDEFINED) {
    XELOGE(
        "ZeroFGC0PostInvariant kind={} source={} pair_a={} logical={} "
        "candidate=0x{:X} storage={}x{} logical_extent={}x{} "
        "final=0x{:X} final_extent={}x{} reason=request_contract",
        request.candidate_is_synthetic ? "S" : "R", request.source_id,
        request.pair_a_source_id, request.logical_sequence,
        handle_value(request.candidate_image),
        request.candidate_storage_extent.width,
        request.candidate_storage_extent.height, request.frontbuffer_width,
        request.frontbuffer_height, handle_value(request.final_output_image),
        request.final_output_extent.width, request.final_output_extent.height);
    return fail(PostFailureStage::kRequestValidation);
  }
  const VulkanDevice::Functions& dfn = ZeroFGDevice()->functions();
  const VkDevice device = ZeroFGDevice()->device();
  const uint32_t queue_family = ZeroFGDevice()->queue_family_graphics_compute();
  const uint32_t queue_index = ZeroFGDevice()->queue_index_zerofg_presenter();
  if (!zerofg_vulkan_context_->post) {
    auto context = std::make_unique<ZeroFGPostContext>();
    VkCommandPoolCreateInfo pool_info = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family;
    VkSemaphoreTypeCreateInfo timeline_type = {
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &timeline_type;
    const VkResult timeline_result = dfn.vkCreateSemaphore(
        device, &semaphore_info, nullptr, &context->completion_timeline);
    if (timeline_result != VK_SUCCESS) {
      zerofg_vulkan_context_->post = std::move(context);
      DestroyZeroFGPostContext();
      return fail_vk(PostFailureStage::kContextTimeline, timeline_result);
    }
    // The shared layout contains both a sampled image and an immutable
    // sampler. Vulkan still accounts immutable samplers against the descriptor
    // pool on drivers that enforce this strictly, so mirror the normal
    // presenter pool instead of declaring sampled images only.
    VkDescriptorPoolSize descriptor_pool_sizes[2];
    descriptor_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_pool_sizes[0].descriptorCount = kMaxGuestOutputPaintEffects;
    descriptor_pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptor_pool_sizes[1].descriptorCount = kMaxGuestOutputPaintEffects;
    VkDescriptorPoolCreateInfo descriptor_pool_info = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_pool_info.maxSets = kMaxGuestOutputPaintEffects;
    descriptor_pool_info.poolSizeCount = uint32_t(
        xe::countof(descriptor_pool_sizes));
    descriptor_pool_info.pPoolSizes = descriptor_pool_sizes;
    std::array<VkDescriptorSetLayout, kMaxGuestOutputPaintEffects> layouts;
    layouts.fill(zerofg_vulkan_context_->image_layout);
    for (ZeroFGPostContext::JobContext& job : context->jobs) {
      const VkResult command_pool_result = dfn.vkCreateCommandPool(
          device, &pool_info, nullptr, &job.command_pool);
      if (command_pool_result != VK_SUCCESS) {
        zerofg_vulkan_context_->post = std::move(context);
        DestroyZeroFGPostContext();
        return fail_vk(PostFailureStage::kContextCommandPool,
                       command_pool_result);
      }
      VkCommandBufferAllocateInfo command_info = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      command_info.commandPool = job.command_pool;
      command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_info.commandBufferCount = 1;
      const VkResult command_buffer_result = dfn.vkAllocateCommandBuffers(
          device, &command_info, &job.command_buffer);
      if (command_buffer_result != VK_SUCCESS) {
        zerofg_vulkan_context_->post = std::move(context);
        DestroyZeroFGPostContext();
        return fail_vk(PostFailureStage::kContextCommandBuffer,
                       command_buffer_result);
      }
      const VkResult descriptor_pool_result = dfn.vkCreateDescriptorPool(
          device, &descriptor_pool_info, nullptr, &job.descriptor_pool);
      if (descriptor_pool_result != VK_SUCCESS) {
        zerofg_vulkan_context_->post = std::move(context);
        DestroyZeroFGPostContext();
        return fail_vk(PostFailureStage::kDescriptorPool,
                       descriptor_pool_result);
      }
      VkDescriptorSetAllocateInfo descriptor_allocate = {
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
      descriptor_allocate.descriptorPool = job.descriptor_pool;
      descriptor_allocate.descriptorSetCount = uint32_t(layouts.size());
      descriptor_allocate.pSetLayouts = layouts.data();
      const VkResult descriptor_sets_result = dfn.vkAllocateDescriptorSets(
          device, &descriptor_allocate, job.descriptor_sets.data());
      if (descriptor_sets_result != VK_SUCCESS) {
        zerofg_vulkan_context_->post = std::move(context);
        DestroyZeroFGPostContext();
        return fail_vk(PostFailureStage::kDescriptorSets,
                       descriptor_sets_result);
      }
      VkExportSemaphoreCreateInfo export_info = {
          VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
      export_info.handleTypes =
          VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
      VkSemaphoreCreateInfo acquire_info = {
          VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      acquire_info.pNext = &export_info;
      const VkResult acquire_result = dfn.vkCreateSemaphore(
          device, &acquire_info, nullptr, &job.acquire_semaphore);
      if (acquire_result != VK_SUCCESS) {
        zerofg_vulkan_context_->post = std::move(context);
        DestroyZeroFGPostContext();
        return fail_vk(PostFailureStage::kContextTimeline, acquire_result);
      }
      const VkResult job_timeline_result = job.completion.CreateTimeline(
          ZeroFGDevice(), "ZeroFG Post completion " +
                              std::to_string(&job - context->jobs.data()));
      if (job_timeline_result != VK_SUCCESS) {
        zerofg_vulkan_context_->post = std::move(context);
        DestroyZeroFGPostContext();
        return fail_vk(PostFailureStage::kContextTimeline,
                       job_timeline_result);
      }
    }
    VkAttachmentDescription attachment = {};
    attachment.format = request.final_output_format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference attachment_ref = {};
    attachment_ref.attachment = 0;
    attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &attachment_ref;
    VkRenderPassCreateInfo render_pass_info = {
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass_info.attachmentCount = 1;
    render_pass_info.pAttachments = &attachment;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    const VkResult render_pass_result = dfn.vkCreateRenderPass(
        device, &render_pass_info, nullptr, &context->final_render_pass);
    if (render_pass_result != VK_SUCCESS) {
      zerofg_vulkan_context_->post = std::move(context);
      DestroyZeroFGPostContext();
      return fail_vk(PostFailureStage::kFinalRenderPass, render_pass_result);
    }
    context->final_format = request.final_output_format;
    zerofg_vulkan_context_->post = std::move(context);
  }
  ZeroFGPostContext& context = *zerofg_vulkan_context_->post;
  if (context.final_format != request.final_output_format) {
    return fail(PostFailureStage::kFinalFormat);
  }
  ZeroFGPostContext::JobContext& job =
      context.jobs[request.final_output_index];
  if (job.submitted) {
    return fail(PostFailureStage::kCommandReset);
  }

  const VkExtent2D logical_extent = {
      request.frontbuffer_width, request.frontbuffer_height};
  if (!job.logical_input ||
      job.logical_input->extent().width != logical_extent.width ||
      job.logical_input->extent().height != logical_extent.height) {
    job.logical_input = GuestOutputImage::Create(
        ZeroFGDevice(), logical_extent.width, logical_extent.height,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    job.logical_input_ever_written = false;
    if (!job.logical_input) {
      return fail(PostFailureStage::kLogicalInput);
    }
    VkDescriptorImageInfo image_descriptor = {};
    image_descriptor.imageView = job.logical_input->view();
    image_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = job.descriptor_sets[0];
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &image_descriptor;
    dfn.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }

  VkExtent2D max_extent =
      util::GetMax2DFramebufferExtent(ZeroFGDevice()->properties());
  GuestOutputProperties properties;
  properties.frontbuffer_width = request.frontbuffer_width;
  properties.frontbuffer_height = request.frontbuffer_height;
  properties.display_aspect_ratio_x = request.display_aspect_ratio_x;
  properties.display_aspect_ratio_y = request.display_aspect_ratio_y;
  properties.is_8bpc = request.is_8bpc;
  GuestOutputPaintFlow flow = GetGuestOutputPaintFlow(
      properties, request.final_output_extent.width,
      request.final_output_extent.height, max_extent.width, max_extent.height,
      request.config);
  if (!flow.effect_count) {
    return fail(PostFailureStage::kPaintFlow);
  }
  const std::pair<uint32_t, uint32_t>& final_effect_size =
      flow.effect_output_sizes[flow.effect_count - 1];
  const int64_t final_right =
      int64_t(flow.output_x) + int64_t(final_effect_size.first);
  const int64_t final_bottom =
      int64_t(flow.output_y) + int64_t(final_effect_size.second);
  if (!final_effect_size.first || !final_effect_size.second ||
      final_right <= 0 || final_bottom <= 0 ||
      flow.output_x >= int32_t(request.final_output_extent.width) ||
      flow.output_y >= int32_t(request.final_output_extent.height)) {
    XELOGE(
        "ZeroFGC0PostInvariant kind={} source={} pair_a={} logical={} "
        "final_rect={},{},{}x{} final_extent={}x{} reason=paint_flow_rect",
        request.candidate_is_synthetic ? "S" : "R", request.source_id,
        request.pair_a_source_id, request.logical_sequence, flow.output_x,
        flow.output_y, final_effect_size.first, final_effect_size.second,
        request.final_output_extent.width, request.final_output_extent.height);
    return fail(PostFailureStage::kPaintFlow);
  }
  result.effect_count = uint32_t(flow.effect_count);

  for (size_t i = 0; i < kMaxGuestOutputPaintEffects - 1; ++i) {
    std::pair<uint32_t, uint32_t> needed = {};
    if (i + 1 < flow.effect_count) {
      needed = flow.effect_output_sizes[i];
    }
    const VkExtent2D current = job.intermediate_images[i]
                                   ? job.intermediate_images[i]->extent()
                                   : VkExtent2D{};
    if (current.width == needed.first && current.height == needed.second) {
      continue;
    }
    job.intermediate_images[i].reset();
    util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device,
                               job.intermediate_framebuffers[i]);
    if (!needed.first || !needed.second) {
      continue;
    }
    job.intermediate_images[i] =
        GuestOutputImage::Create(ZeroFGDevice(), needed.first, needed.second);
    if (!job.intermediate_images[i]) {
      return fail(PostFailureStage::kIntermediateResource);
    }
    VkImageView attachment = job.intermediate_images[i]->view();
    VkFramebufferCreateInfo framebuffer_info = {
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_info.renderPass = zerofg_vulkan_context_->render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &attachment;
    framebuffer_info.width = needed.first;
    framebuffer_info.height = needed.second;
    framebuffer_info.layers = 1;
    const VkResult framebuffer_result = dfn.vkCreateFramebuffer(
        device, &framebuffer_info, nullptr,
        &job.intermediate_framebuffers[i]);
    if (framebuffer_result != VK_SUCCESS) {
      return fail_vk(PostFailureStage::kIntermediateFramebuffer,
                     framebuffer_result);
    }
    VkDescriptorImageInfo image_descriptor = {};
    image_descriptor.imageView = job.intermediate_images[i]->view();
    image_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = job.descriptor_sets[i + 1];
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &image_descriptor;
    dfn.vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }
  for (size_t i = 0; i + 1 < flow.effect_count; ++i) {
    const size_t effect_index = size_t(flow.effects[i]);
    if (zerofg_vulkan_context_->fragments[effect_index] == VK_NULL_HANDLE ||
        zerofg_vulkan_context_->pipelines[effect_index] == VK_NULL_HANDLE) {
      return fail(PostFailureStage::kIntermediatePipeline);
    }
  }
  const GuestOutputPaintEffect final_effect =
      flow.effects[flow.effect_count - 1];
  if (zerofg_vulkan_context_->fragments[size_t(final_effect)] ==
      VK_NULL_HANDLE) {
    return fail(PostFailureStage::kFinalPipeline);
  }
  VkPipeline& final_pipeline = context.final_pipelines[size_t(final_effect)];
  if (final_pipeline == VK_NULL_HANDLE) {
    VkResult pipeline_result = VK_SUCCESS;
    final_pipeline = CreateGuestOutputPaintPipeline(
        final_effect, context.final_render_pass, &pipeline_result, true);
    if (final_pipeline == VK_NULL_HANDLE) {
      return fail_vk(PostFailureStage::kFinalPipeline, pipeline_result);
    }
  }
  if (job.final_view != request.final_output_view) {
    util::DestroyAndNullHandle(
        dfn.vkDestroyFramebuffer, device,
        job.final_framebuffer);
    VkFramebufferCreateInfo framebuffer_info = {
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer_info.renderPass = context.final_render_pass;
    framebuffer_info.attachmentCount = 1;
    framebuffer_info.pAttachments = &request.final_output_view;
    framebuffer_info.width = request.final_output_extent.width;
    framebuffer_info.height = request.final_output_extent.height;
    framebuffer_info.layers = 1;
    const VkResult framebuffer_result = dfn.vkCreateFramebuffer(
        device, &framebuffer_info, nullptr,
        &job.final_framebuffer);
    if (framebuffer_result != VK_SUCCESS) {
      return fail_vk(PostFailureStage::kFinalFramebuffer, framebuffer_result);
    }
    job.final_view = request.final_output_view;
  }
  const VkResult reset_result =
      dfn.vkResetCommandPool(device, job.command_pool, 0);
  if (reset_result != VK_SUCCESS) {
    return fail_vk(PostFailureStage::kCommandReset, reset_result);
  }
  VkCommandBufferBeginInfo begin = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  const VkResult begin_result =
      dfn.vkBeginCommandBuffer(job.command_buffer, &begin);
  if (begin_result != VK_SUCCESS) {
    return fail_vk(PostFailureStage::kCommandBegin, begin_result);
  }
  VkImageMemoryBarrier acquire_barriers[3] = {};
  for (VkImageMemoryBarrier& barrier : acquire_barriers) {
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
  }
  acquire_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  acquire_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  acquire_barriers[0].oldLayout = request.candidate_layout;
  acquire_barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  acquire_barriers[0].srcAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  acquire_barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  acquire_barriers[0].image = request.candidate_image;
  acquire_barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  acquire_barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  acquire_barriers[1].oldLayout = job.logical_input_ever_written
                                      ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                      : VK_IMAGE_LAYOUT_UNDEFINED;
  acquire_barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  acquire_barriers[1].srcAccessMask = job.logical_input_ever_written
                                         ? VK_ACCESS_SHADER_READ_BIT
                                         : 0;
  acquire_barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  acquire_barriers[1].image = job.logical_input->image();
  // Main Surface Authority: a B-local FinalOutput never leaves device B, and
  // the Post clears all of it, so its previous contents (the egress copy's
  // source, retired before the slot came back) are simply discarded.
  acquire_barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  acquire_barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  acquire_barriers[2].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  acquire_barriers[2].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  acquire_barriers[2].srcAccessMask = 0;
  acquire_barriers[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  acquire_barriers[2].image = request.final_output_image;
  dfn.vkCmdPipelineBarrier(
      job.command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      0, 0, nullptr, 0, nullptr, uint32_t(xe::countof(acquire_barriers)),
      acquire_barriers);
  VkImageCopy copy = {};
  copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.srcSubresource.layerCount = 1;
  copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.dstSubresource.layerCount = 1;
  copy.extent = {logical_extent.width, logical_extent.height, 1};
  dfn.vkCmdCopyImage(job.command_buffer, request.candidate_image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     job.logical_input->image(),
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  VkImageMemoryBarrier input_ready = {
      VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  input_ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  input_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  input_ready.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  input_ready.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  input_ready.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  input_ready.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  input_ready.image = job.logical_input->image();
  input_ready.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  input_ready.subresourceRange.levelCount = 1;
  input_ready.subresourceRange.layerCount = 1;
  dfn.vkCmdPipelineBarrier(job.command_buffer,
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &input_ready);

  VkViewport viewport = {};
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  VkRect2D scissor = {};
  VkClearValue black = {};
  VkClearAttachment clear_attachment = {};
  clear_attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  clear_attachment.colorAttachment = 0;
  for (size_t i = 0; i < flow.effect_count; ++i) {
    const bool final = i + 1 == flow.effect_count;
    const auto effect_size = flow.effect_output_sizes[i];
    int32_t rect_x = final ? flow.output_x : 0;
    int32_t rect_y = final ? flow.output_y : 0;
    VkRenderPassBeginInfo render_begin = {
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render_begin.renderPass = final ? context.final_render_pass
                                    : zerofg_vulkan_context_->render_pass;
    render_begin.framebuffer =
        final ? job.final_framebuffer : job.intermediate_framebuffers[i];
    render_begin.renderArea.extent =
        final ? request.final_output_extent
              : VkExtent2D{effect_size.first, effect_size.second};
    if (final) {
      render_begin.clearValueCount = 1;
      render_begin.pClearValues = &black;
    }
    dfn.vkCmdBeginRenderPass(job.command_buffer, &render_begin,
                             VK_SUBPASS_CONTENTS_INLINE);
    viewport.width = float(render_begin.renderArea.extent.width);
    viewport.height = float(render_begin.renderArea.extent.height);
    scissor.extent = render_begin.renderArea.extent;
    dfn.vkCmdSetViewport(job.command_buffer, 0, 1, &viewport);
    dfn.vkCmdSetScissor(job.command_buffer, 0, 1, &scissor);
    const GuestOutputPaintEffect effect = flow.effects[i];
    const VkPipeline pipeline =
        final ? context.final_pipelines[size_t(effect)]
              : zerofg_vulkan_context_->pipelines[size_t(effect)];
    dfn.vkCmdBindPipeline(job.command_buffer,
                          VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const GuestOutputPaintPipelineLayoutIndex layout_index =
        GetGuestOutputPaintPipelineLayoutIndex(effect);
    const VkPipelineLayout layout =
        zerofg_vulkan_context_->layouts[layout_index];
    // Set 0 samples the exact-size logical input; set i samples intermediate
    // i-1 for every later effect.
    const VkDescriptorSet descriptor = job.descriptor_sets[i];
    dfn.vkCmdBindDescriptorSets(job.command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                                &descriptor, 0, nullptr);
    GuestOutputPaintRectangleConstants rectangle;
    const float x_to_ndc = 2.0f / viewport.width;
    const float y_to_ndc = 2.0f / viewport.height;
    rectangle.x = -1.0f + float(rect_x) * x_to_ndc;
    rectangle.y = -1.0f + float(rect_y) * y_to_ndc;
    rectangle.width = float(effect_size.first) * x_to_ndc;
    rectangle.height = float(effect_size.second) * y_to_ndc;
    dfn.vkCmdPushConstants(job.command_buffer, layout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(rectangle),
                           &rectangle);
    uint32_t constants_size = 0;
    union {
      BilinearConstants bilinear;
      CasSharpenConstants cas_sharpen;
      CasResampleConstants cas_resample;
      FsrEasuConstants fsr_easu;
      FsrRcasConstants fsr_rcas;
      SgsrConstants sgsr;
    } constants;
    switch (layout_index) {
      case kGuestOutputPaintPipelineLayoutIndexBilinear:
        constants_size = sizeof(constants.bilinear);
        constants.bilinear.Initialize(flow, i);
        break;
      case kGuestOutputPaintPipelineLayoutIndexCasSharpen:
        constants_size = sizeof(constants.cas_sharpen);
        constants.cas_sharpen.Initialize(flow, i, request.config);
        break;
      case kGuestOutputPaintPipelineLayoutIndexCasResample:
        constants_size = sizeof(constants.cas_resample);
        constants.cas_resample.Initialize(flow, i, request.config);
        break;
      case kGuestOutputPaintPipelineLayoutIndexFsrEasu:
        constants_size = sizeof(constants.fsr_easu);
        constants.fsr_easu.Initialize(flow, i);
        break;
      case kGuestOutputPaintPipelineLayoutIndexFsrRcas:
        constants_size = sizeof(constants.fsr_rcas);
        constants.fsr_rcas.Initialize(flow, i, request.config);
        break;
      case kGuestOutputPaintPipelineLayoutIndexSgsr:
        constants_size = sizeof(constants.sgsr);
        constants.sgsr.Initialize(flow, i);
        break;
      default:
        break;
    }
    if (constants_size) {
      dfn.vkCmdPushConstants(job.command_buffer, layout,
                             VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(rectangle),
                             constants_size, &constants);
    }
    dfn.vkCmdDraw(job.command_buffer, 4, 1, 0, 0);
    if (final && flow.letterbox_clear_rectangle_count) {
      std::array<VkClearRect, GuestOutputPaintFlow::kMaxClearRectangles> clears;
      for (size_t clear_index = 0;
           clear_index < flow.letterbox_clear_rectangle_count; ++clear_index) {
        const auto& source = flow.letterbox_clear_rectangles[clear_index];
        VkClearRect& clear = clears[clear_index];
        clear.rect.offset = {int32_t(source.x), int32_t(source.y)};
        clear.rect.extent = {source.width, source.height};
        clear.baseArrayLayer = 0;
        clear.layerCount = 1;
      }
      dfn.vkCmdClearAttachments(
          job.command_buffer, 1, &clear_attachment,
          uint32_t(flow.letterbox_clear_rectangle_count), clears.data());
    }
    dfn.vkCmdEndRenderPass(job.command_buffer);
  }
  VkImageMemoryBarrier releases[2] = {};
  for (VkImageMemoryBarrier& barrier : releases) {
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
  }
  releases[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  releases[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  releases[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  releases[0].newLayout = request.candidate_layout;
  releases[0].image = request.candidate_image;
  releases[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  releases[1].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  releases[1].image = request.final_output_image;
  // Main Surface Authority: the FinalOutput stays on device B, ready for the
  // egress copy (same queue family, so no ownership transfer). The copy's wait
  // on this Post's timeline point makes the writes visible to it.
  releases[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  releases[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  releases[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  dfn.vkCmdPipelineBarrier(job.command_buffer,
                           VK_PIPELINE_STAGE_TRANSFER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                           0, nullptr, uint32_t(xe::countof(releases)), releases);
  const VkResult end_result = dfn.vkEndCommandBuffer(job.command_buffer);
  if (end_result != VK_SUCCESS) {
    return fail_vk(PostFailureStage::kCommandEnd, end_result);
  }
  // D2: with per-slot completion the Post signals its own slot's timeline, so
  // the signal's cleanup never waits on another slot's pending Post.
  const bool completion_per_slot = job.completion.owns_timeline();
  const VkSemaphore post_completion_timeline =
      completion_per_slot ? job.completion.timeline()
                          : context.completion_timeline;
  const uint64_t signal_value = completion_per_slot
                                    ? job.completion.ClaimTimelineValue()
                                    : context.next_timeline_value++;
  if (job.completion.occupied()) {
    ++context.owner_stats.reuse_before_retire;
  }
  const uint64_t wait_values[1] = {request.candidate_wait_value};
  const uint64_t signal_values[2] = {signal_value, 0};
  VkTimelineSemaphoreSubmitInfo timeline_submit = {
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline_submit.waitSemaphoreValueCount =
      request.candidate_wait_semaphore != VK_NULL_HANDLE ? 1 : 0;
  timeline_submit.pWaitSemaphoreValues = wait_values;
  timeline_submit.signalSemaphoreValueCount = 2;
  timeline_submit.pSignalSemaphoreValues = signal_values;
  // The candidate's final Generation barrier and layout are part of the
  // dependency, not merely its first transfer read. Keep the chained Post
  // wholly behind the timeline value even if command recording evolves.
  const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  const VkSemaphore signal_semaphores[2] = {post_completion_timeline,
                                            job.acquire_semaphore};
  VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.pNext = &timeline_submit;
  submit.waitSemaphoreCount =
      request.candidate_wait_semaphore != VK_NULL_HANDLE ? 1 : 0;
  submit.pWaitSemaphores = &request.candidate_wait_semaphore;
  submit.pWaitDstStageMask = &wait_stage;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &job.command_buffer;
  submit.signalSemaphoreCount = 2;
  submit.pSignalSemaphores = signal_semaphores;
  // Telemetry only: whether the candidate dependency is known complete when
  // the Post enters q0. With a pending wait it stays unknown: the counter
  // query that could tell waits on pending GPU work on Turnip/KGSL.
  result.input_has_wait = request.candidate_wait_semaphore != VK_NULL_HANDLE;
  if (!result.input_has_wait) {
    result.input_readiness_known = true;
    result.input_complete_at_submit = true;
  }
  const uint64_t queue_request_ns = GetZeroFGMonotonicTimeNs();
  result.submit_request_ns = queue_request_ns;
  VkResult submit_result;
  uint64_t submit_begin_ns;
  {
    const VulkanDevice::Queue::Acquisition queue =
        ZeroFGDevice()->AcquireQueue(queue_family, queue_index);
    if (ZeroFGDevice()->RejectZeroFGSubmitAfterTeardownIdle()) {
      dfn.vkResetCommandPool(device, job.command_pool, 0);
      return fail(PostFailureStage::kQueueSubmit);
    }
    submit_begin_ns = GetZeroFGMonotonicTimeNs();
    result.queue_wait_ns = submit_begin_ns - queue_request_ns;
    submit_result = dfn.vkQueueSubmit(queue.queue(), 1, &submit, VK_NULL_HANDLE);
    result.submit_host_ns = GetZeroFGMonotonicTimeNs() - submit_begin_ns;
  }
  if (submit_result != VK_SUCCESS) {
    return fail_vk(PostFailureStage::kQueueSubmit, submit_result);
  }
  result.submission_accepted = true;
  job.signal_value = signal_value;
  job.submit_time_ns = GetZeroFGMonotonicTimeNs();
  job.submitted = true;
  job.logical_input_ever_written = true;
  job.completion.MarkSubmitted(post_completion_timeline);
  context.owner_stats.CountSubmit(completion_per_slot, result.submit_host_ns);
  result.signal_value = signal_value;
  result.completion_semaphore = post_completion_timeline;
  result.completion_per_slot = completion_per_slot;
  result.completion_owner_stats = context.owner_stats;
  result.submit_time_ns = job.submit_time_ns;
  VkSemaphoreGetFdInfoKHR fd_info = {
      VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  fd_info.semaphore = job.acquire_semaphore;
  fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
  const uint64_t export_begin_ns = GetZeroFGMonotonicTimeNs();
  const VkResult export_result = ZeroFGDevice()->vkGetSemaphoreFdKHR()(
      device, &fd_info, &result.acquire_fence_fd);
  result.export_host_ns = GetZeroFGMonotonicTimeNs() - export_begin_ns;
  if (export_result != VK_SUCCESS) {
    return fail_vk(PostFailureStage::kAcquireFenceExport, export_result);
  }
  result.acquire_fence_exported = true;
  return true;
}

void VulkanPresenter::ReleaseZeroFGPost(uint32_t final_output_index) {
  if (!zerofg_vulkan_context_->post ||
      final_output_index >= ZeroFGIndependentPresenter::kFinalOutputPoolSize) {
    return;
  }
  ZeroFGPostContext::JobContext& job =
      zerofg_vulkan_context_->post->jobs[final_output_index];
  job.submitted = false;
  job.signal_value = 0;
  job.submit_time_ns = 0;
  // The presenter observed this Post complete before releasing the slot.
  job.completion.MarkRetired();
}

VkPipeline VulkanPresenter::CreateGuestOutputPaintPipeline(
    GuestOutputPaintEffect effect, VkRenderPass render_pass,
    VkResult* result_out, bool zerofg) {
  VkPipelineShaderStageCreateInfo stages[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    VkPipelineShaderStageCreateInfo& stage = stages[i];
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = i ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
    stage.pName = "main";
  }
  stages[0].module =
      zerofg ? zerofg_vulkan_context_->vertex : guest_output_paint_vs_;
  stages[1].module = (zerofg ? zerofg_vulkan_context_->fragments
                             : guest_output_paint_fs_)[size_t(effect)];
  assert_true(stages[1].module != VK_NULL_HANDLE);

  VkPipelineVertexInputStateCreateInfo vertex_input_state = {};
  vertex_input_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo input_assembly_state = {};
  input_assembly_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

  VkPipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.viewportCount = 1;
  viewport_state.scissorCount = 1;

  VkPipelineRasterizationStateCreateInfo rasterization_state = {};
  rasterization_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
  rasterization_state.cullMode = VK_CULL_MODE_NONE;
  rasterization_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterization_state.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo multisample_state = {};
  multisample_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
  color_blend_attachment_state.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo color_blend_state = {};
  color_blend_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blend_state.attachmentCount = 1;
  color_blend_state.pAttachments = &color_blend_attachment_state;

  static constexpr VkDynamicState kPipelineDynamicStates[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo dynamic_state = {};
  dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamic_state.dynamicStateCount =
      uint32_t(xe::countof(kPipelineDynamicStates));
  dynamic_state.pDynamicStates = kPipelineDynamicStates;

  VkGraphicsPipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = nullptr;
  pipeline_create_info.flags = 0;
  pipeline_create_info.stageCount = uint32_t(xe::countof(stages));
  pipeline_create_info.pStages = stages;
  pipeline_create_info.pVertexInputState = &vertex_input_state;
  pipeline_create_info.pInputAssemblyState = &input_assembly_state;
  pipeline_create_info.pTessellationState = nullptr;
  pipeline_create_info.pViewportState = &viewport_state;
  pipeline_create_info.pRasterizationState = &rasterization_state;
  pipeline_create_info.pMultisampleState = &multisample_state;
  pipeline_create_info.pDepthStencilState = nullptr;
  pipeline_create_info.pColorBlendState = &color_blend_state;
  pipeline_create_info.pDynamicState = &dynamic_state;
  pipeline_create_info.layout = (zerofg ? zerofg_vulkan_context_->layouts
                                        : guest_output_paint_pipeline_layouts_)
      [GetGuestOutputPaintPipelineLayoutIndex(effect)];
  pipeline_create_info.renderPass = render_pass;
  pipeline_create_info.subpass = 0;
  pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  pipeline_create_info.basePipelineIndex = -1;

  VulkanDevice* pipeline_device = zerofg ? ZeroFGDevice() : vulkan_device_;
  const VulkanDevice::Functions& dfn = pipeline_device->functions();
  const VkDevice device = pipeline_device->device();

  VkPipeline pipeline;
  const VkResult pipeline_result = dfn.vkCreateGraphicsPipelines(
      device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &pipeline);
  if (result_out) {
    *result_out = pipeline_result;
  }
  if (pipeline_result != VK_SUCCESS) {
    if (!result_out) {
      XELOGE(
          "VulkanPresenter: Failed to create the guest output painting "
          "pipeline for effect {}",
          size_t(effect));
    }
    return VK_NULL_HANDLE;
  }
  return pipeline;
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe
