/**
 ******************************************************************************
 * XenDroid ZeroFG V2 Main Surface Authority egress                           *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/zerofg_main_surface_egress.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#include "xenia/base/frame_stats.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
#include <errno.h>
#include <poll.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace xe {
namespace ui {
namespace vulkan {

namespace {

uint64_t EgressMonotonicTimeNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

}  // namespace

ZeroFGMainSurfaceProducer::~ZeroFGMainSurfaceProducer() {
  if (state() == State::kB) {
    xe::SetZeroFGMainSurfaceActive(false);
  }
}

const char* ZeroFGMainSurfaceProducer::Name(State state) {
  switch (state) {
    case State::kA:
      return "A";
    case State::kHandoffGap:
      return "handoff_gap";
    case State::kB:
      return "B";
    case State::kHandbackGap:
      return "handback_gap";
  }
  return "unknown";
}

bool ZeroFGMainSurfaceProducer::Transition(State from, State to,
                                           const char* actor) {
  State expected = from;
  if (!state_.compare_exchange_strong(expected, to, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
    violation_total_.fetch_add(1, std::memory_order_relaxed);
    XELOGE(
        "ZeroFGMainSurfaceProducerViolation actor={} from={} to={} actual={}",
        actor, Name(from), Name(to), Name(expected));
    return false;
  }
  if (from == State::kA && to == State::kHandoffGap) {
    handoff_begin_total_.fetch_add(1, std::memory_order_relaxed);
  } else if (from == State::kHandoffGap && to == State::kB) {
    handoff_done_total_.fetch_add(1, std::memory_order_relaxed);
  } else if (to == State::kHandbackGap) {
    handback_begin_total_.fetch_add(1, std::memory_order_relaxed);
  } else if (to == State::kA) {
    handback_done_total_.fetch_add(1, std::memory_order_relaxed);
  }
  xe::SetZeroFGMainSurfaceActive(to == State::kB);
  XELOGI("ZeroFGMainSurface producer {} -> {} by={}", Name(from), Name(to),
         actor);
  return true;
}

void ZeroFGMainSurfaceProducer::CountViolation(const char* what) {
  violation_total_.fetch_add(1, std::memory_order_relaxed);
  XELOGE("ZeroFGMainSurfaceProducerViolation what={} state={}", what,
         Name(state()));
}

const char* ZeroFGMainSurfaceEgress::FailureName(Failure failure) {
  switch (failure) {
    case Failure::kNone:
      return "none";
    case Failure::kNoSurfaceSupport:
      return "no_surface_support";
    case Failure::kNoTransferDst:
      return "no_transfer_dst";
    case Failure::kNoFormat:
      return "no_format";
    case Failure::kExtentMismatch:
      return "extent_mismatch";
    case Failure::kSwapchainCreate:
      return "swapchain_create";
    case Failure::kSwapchainImages:
      return "swapchain_images";
    case Failure::kResourceCreate:
      return "resource_create";
    case Failure::kProducerOrder:
      return "producer_order";
    case Failure::kAcquire:
      return "acquire";
    case Failure::kRecord:
      return "record";
    case Failure::kSubmit:
      return "submit";
    case Failure::kCopyFenceExport:
      return "copy_fence";
    case Failure::kPresent:
      return "present";
    case Failure::kSlotReuse:
      return "slot_reuse";
    case Failure::kCompletionOverflow:
      return "completion_overflow";
    case Failure::kDeviceLost:
      return "device_lost";
  }
  return "unknown";
}

template <size_t Capacity>
void ZeroFGMainSurfaceEgress::SampleWindow<Capacity>::Add(uint64_t value) {
  values_[next_] = value;
  next_ = (next_ + 1) % Capacity;
  count_ = std::min(count_ + 1, Capacity);
}

template <size_t Capacity>
uint64_t ZeroFGMainSurfaceEgress::SampleWindow<Capacity>::Quantile(
    uint32_t numerator, uint32_t denominator) const {
  if (!count_) {
    return 0;
  }
  std::array<uint64_t, Capacity> sorted = values_;
  std::sort(sorted.begin(), sorted.begin() + count_);
  const size_t index =
      std::min(count_ - 1,
               size_t((uint64_t(count_ - 1) * numerator + denominator - 1) /
                      denominator));
  return sorted[index];
}

uint64_t ZeroFGMainSurfaceEgress::BiasSigned(int64_t value_ns) {
  return uint64_t(std::clamp<int64_t>(value_ns, -kSignedSampleBiasNs,
                                      kSignedSampleBiasNs) +
                  kSignedSampleBiasNs);
}

int64_t ZeroFGMainSurfaceEgress::UnbiasSigned(uint64_t value_ns) {
  return int64_t(value_ns) - kSignedSampleBiasNs;
}

ZeroFGMainSurfaceEgress::ZeroFGMainSurfaceEgress(
    VulkanDevice* device, ZeroFGMainSurfaceProducer* producer,
    VkSurfaceKHR surface, VkExtent2D extent, VkFormat format,
    uint32_t slot_count, bool elevated_priority, bool apocalypse_guard,
    bool free_output, bool vsync_quantizer,
    std::function<void()> wake_owner)
    : device_(device),
      producer_(producer),
      surface_(surface),
      extent_(extent),
      format_(format),
      slot_count_(std::min(slot_count, kMaxSlots)),
      elevated_priority_(elevated_priority),
      wake_owner_(std::move(wake_owner)),
      queue_family_(device->queue_family_graphics_compute()),
      queue_index_(device->queue_index_zerofg_main_surface_present()),
      timing_available_(device->vkGetPastPresentationTimingGOOGLE() !=
                        nullptr),
      apocalypse_guard_(apocalypse_guard && timing_available_),
      free_output_(free_output),
      vsync_quantizer_(vsync_quantizer && timing_available_) {
  counters_.timing_available = timing_available_;
  counters_.apocalypse_guard = apocalypse_guard_;
  counters_.free_output = free_output_;
  counters_.vsync_quantizer = vsync_quantizer_;
  counters_.queue_index = queue_index_;
}

ZeroFGMainSurfaceEgress::~ZeroFGMainSurfaceEgress() { Stop(); }

bool ZeroFGMainSurfaceEgress::Start() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (started_ || stopped_ || !slot_count_ || surface_ == VK_NULL_HANDLE ||
      !device_->vkCmdPipelineBarrier2() || !device_->vkGetSemaphoreFdKHR()) {
    return started_;
  }
  const VulkanDevice::Functions& dfn = device_->functions();
  const VkDevice device = device_->device();
  for (uint32_t i = 0; i < slot_count_; ++i) {
    CopyContext& context = contexts_[i];
    VkCommandPoolCreateInfo pool_info = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = queue_family_;
    if (dfn.vkCreateCommandPool(device, &pool_info, nullptr,
                                &context.command_pool) != VK_SUCCESS) {
      XELOGE("ZeroFGMainSurface copy context {} command pool failed", i);
      return false;
    }
    VkCommandBufferAllocateInfo buffer_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    buffer_info.commandPool = context.command_pool;
    buffer_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    buffer_info.commandBufferCount = 1;
    if (dfn.vkAllocateCommandBuffers(device, &buffer_info,
                                     &context.command_buffer) != VK_SUCCESS) {
      XELOGE("ZeroFGMainSurface copy context {} command buffer failed", i);
      return false;
    }
    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (dfn.vkCreateSemaphore(device, &semaphore_info, nullptr,
                              &context.acquire_semaphore) != VK_SUCCESS) {
      XELOGE("ZeroFGMainSurface copy context {} acquire semaphore failed", i);
      return false;
    }
    VkExportSemaphoreCreateInfo export_info = {
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    VkSemaphoreCreateInfo copy_done_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    copy_done_info.pNext = &export_info;
    if (dfn.vkCreateSemaphore(device, &copy_done_info, nullptr,
                              &context.copy_done_semaphore) != VK_SUCCESS) {
      XELOGE("ZeroFGMainSurface copy context {} copy-done semaphore failed",
             i);
      return false;
    }
  }
  started_ = true;
  thread_ = std::thread(&ZeroFGMainSurfaceEgress::ThreadMain, this);
  XELOGI(
      "ZeroFGMainSurface egress armed queue=B:{} slots={} extent={}x{} "
      "format={} timing_available={} apocalypse_guard={} "
      "waiting_for=A_retirement",
      queue_index_, slot_count_, extent_.width, extent_.height, int(format_),
      timing_available_, apocalypse_guard_);
  return true;
#else
  return false;
#endif
}

void ZeroFGMainSurfaceEgress::Stop() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (stopped_) {
    return;
  }
  stopped_ = true;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_requested_ = true;
  }
  condition_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
  DestroySwapchainAfterIdle();
  const VulkanDevice::Functions& dfn = device_->functions();
  const VkDevice device = device_->device();
  for (CopyContext& context : contexts_) {
    if (context.copy_done_fd >= 0) {
      close(context.copy_done_fd);
      context.copy_done_fd = -1;
    }
    if (context.copy_done_semaphore != VK_NULL_HANDLE) {
      dfn.vkDestroySemaphore(device, context.copy_done_semaphore, nullptr);
    }
    if (context.acquire_semaphore != VK_NULL_HANDLE) {
      dfn.vkDestroySemaphore(device, context.acquire_semaphore, nullptr);
    }
    if (context.command_pool != VK_NULL_HANDLE) {
      dfn.vkDestroyCommandPool(device, context.command_pool, nullptr);
    }
    context = {};
  }
#endif
}

void ZeroFGMainSurfaceEgress::NotifyProducerChanged() {
  // The waiter checks the producer state under this mutex; taking it here
  // orders the state change before the wake-up.
  { std::lock_guard<std::mutex> lock(mutex_); }
  condition_.notify_all();
}

bool ZeroFGMainSurfaceEgress::HasCapacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !stop_requested_ && ring_count_ < slot_count_;
}

ZeroFGMainSurfaceEgress::EnqueueResult ZeroFGMainSurfaceEgress::TryEnqueue(
    const Request& request) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_requested_ ||
        producer_->state() != ZeroFGMainSurfaceProducer::State::kB) {
      return EnqueueResult::kNotProducing;
    }
    if (ring_count_ >= slot_count_) {
      return EnqueueResult::kFull;
    }
    ring_[(ring_head_ + ring_count_) % slot_count_] = request;
    ++ring_count_;
  }
  condition_.notify_all();
  return EnqueueResult::kQueued;
}

bool ZeroFGMainSurfaceEgress::PopCompletion(Completion& completion_out) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!completion_count_) {
    return false;
  }
  completion_out = completions_[completion_head_];
  completion_head_ = (completion_head_ + 1) % uint32_t(completions_.size());
  --completion_count_;
  return true;
}

ZeroFGMainSurfaceEgress::Stats ZeroFGMainSurfaceEgress::SnapshotStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  Stats stats = counters_;
  stats.actual_interval_min_ns = actual_interval_ns_.Quantile(0, 100);
  stats.actual_interval_p10_ns = actual_interval_ns_.Quantile(10, 100);
  stats.actual_interval_p50_ns = actual_interval_ns_.Quantile(50, 100);
  stats.actual_interval_p90_ns = actual_interval_ns_.Quantile(90, 100);
  stats.actual_interval_p99_ns = actual_interval_ns_.Quantile(99, 100);
  stats.actual_interval_max_ns = actual_interval_ns_.Quantile(100, 100);
  stats.desired_to_actual_p50_ns = desired_to_actual_ns_.Quantile(50, 100);
  stats.desired_to_actual_p90_ns = desired_to_actual_ns_.Quantile(90, 100);
  stats.desired_to_actual_p99_ns = desired_to_actual_ns_.Quantile(99, 100);
  stats.desired_to_actual_max_ns = desired_to_actual_ns_.Quantile(100, 100);
  stats.acquire_block_p50_ns = acquire_block_ns_.Quantile(50, 100);
  stats.acquire_block_p90_ns = acquire_block_ns_.Quantile(90, 100);
  stats.acquire_block_max_ns = acquire_block_ns_.Quantile(100, 100);
  stats.copy_residence_p50_ns = copy_residence_ns_.Quantile(50, 100);
  stats.copy_residence_p90_ns = copy_residence_ns_.Quantile(90, 100);
  stats.copy_residence_max_ns = copy_residence_ns_.Quantile(100, 100);
  if (present_call_vs_desired_ns_.count()) {
    stats.present_call_vs_desired_p50_ns =
        UnbiasSigned(present_call_vs_desired_ns_.Quantile(50, 100));
    stats.present_call_vs_desired_p90_ns =
        UnbiasSigned(present_call_vs_desired_ns_.Quantile(90, 100));
    stats.present_call_vs_desired_max_ns =
        UnbiasSigned(present_call_vs_desired_ns_.Quantile(100, 100));
  }
  stats.present_call_p50_ns = present_call_ns_.Quantile(50, 100);
  stats.present_call_p90_ns = present_call_ns_.Quantile(90, 100);
  stats.present_call_max_ns = present_call_ns_.Quantile(100, 100);
  if (ready_vs_desired_ns_.count()) {
    stats.ready_vs_desired_p50_ns =
        UnbiasSigned(ready_vs_desired_ns_.Quantile(50, 100));
    stats.ready_vs_desired_p90_ns =
        UnbiasSigned(ready_vs_desired_ns_.Quantile(90, 100));
    stats.ready_vs_desired_max_ns =
        UnbiasSigned(ready_vs_desired_ns_.Quantile(100, 100));
  }
  stats.ready_to_actual_p50_ns = ready_to_actual_ns_.Quantile(50, 100);
  stats.ready_to_actual_p90_ns = ready_to_actual_ns_.Quantile(90, 100);
  stats.ready_to_actual_max_ns = ready_to_actual_ns_.Quantile(100, 100);
  if (present_margin_ns_.count()) {
    stats.present_margin_p50_ns =
        UnbiasSigned(present_margin_ns_.Quantile(50, 100));
    stats.present_margin_p90_ns =
        UnbiasSigned(present_margin_ns_.Quantile(90, 100));
  }
  if (ready_before_vsync_ns_.count()) {
    stats.ready_before_vsync_p10_ns =
        UnbiasSigned(ready_before_vsync_ns_.Quantile(10, 100));
    stats.ready_before_vsync_p50_ns =
        UnbiasSigned(ready_before_vsync_ns_.Quantile(50, 100));
    stats.ready_before_vsync_p90_ns =
        UnbiasSigned(ready_before_vsync_ns_.Quantile(90, 100));
  }
  if (hit_queue_before_vsync_ns_.count()) {
    stats.hit_queue_before_vsync_min_ns =
        UnbiasSigned(hit_queue_before_vsync_ns_.Quantile(0, 100));
  }
  if (hit_ready_before_vsync_ns_.count()) {
    stats.hit_ready_before_vsync_min_ns =
        UnbiasSigned(hit_ready_before_vsync_ns_.Quantile(0, 100));
  }
  if (late_queue_before_vsync_ns_.count()) {
    stats.late_queue_before_vsync_max_ns =
        UnbiasSigned(late_queue_before_vsync_ns_.Quantile(100, 100));
  }
  if (late_ready_before_vsync_ns_.count()) {
    stats.late_ready_before_vsync_max_ns =
        UnbiasSigned(late_ready_before_vsync_ns_.Quantile(100, 100));
  }
  stats.earliest_gap_p50_ns = earliest_gap_ns_.Quantile(50, 100);
  stats.earliest_gap_p90_ns = earliest_gap_ns_.Quantile(90, 100);
  stats.earliest_gap_max_ns = earliest_gap_ns_.Quantile(100, 100);
  stats.shaper_push_p50_ns = shaper_push_ns_.Quantile(50, 100);
  stats.shaper_push_p90_ns = shaper_push_ns_.Quantile(90, 100);
  stats.shaper_push_max_ns = shaper_push_ns_.Quantile(100, 100);
  return stats;
}

void ZeroFGMainSurfaceEgress::ThreadMain() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  ApplyThreadPriority();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] {
      return stop_requested_ || producer_->state() ==
                                    ZeroFGMainSurfaceProducer::State::kHandoffGap;
    });
    if (stop_requested_) {
      return;
    }
  }
  // A has destroyed its swapchain: only now may B create one.
  if (!CreateSwapchain()) {
    return;
  }
  if (!producer_->Transition(ZeroFGMainSurfaceProducer::State::kHandoffGap,
                             ZeroFGMainSurfaceProducer::State::kB,
                             "B_egress_swapchain")) {
    Fail(Failure::kProducerOrder, 0);
    return;
  }
  if (wake_owner_) {
    wake_owner_();
  }
  Serve();
#endif
}

bool ZeroFGMainSurfaceEgress::CreateSwapchain() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  const VulkanInstance::Functions& ifn =
      device_->vulkan_instance()->functions();
  const VkPhysicalDevice physical_device = device_->physical_device();
  const VulkanDevice::Functions& dfn = device_->functions();
  const VkDevice device = device_->device();
  if (producer_->state() != ZeroFGMainSurfaceProducer::State::kHandoffGap) {
    producer_->CountViolation("B_swapchain_before_A_retired");
    Fail(Failure::kProducerOrder, 0);
    return false;
  }
  VkBool32 supported = VK_FALSE;
  VkResult result = ifn.vkGetPhysicalDeviceSurfaceSupportKHR(
      physical_device, queue_family_, surface_, &supported);
  if (result != VK_SUCCESS || !supported) {
    Fail(Failure::kNoSurfaceSupport, int32_t(result));
    return false;
  }
  VkSurfaceCapabilitiesKHR capabilities;
  result = ifn.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      physical_device, surface_, &capabilities);
  if (result != VK_SUCCESS) {
    Fail(Failure::kNoSurfaceSupport, int32_t(result));
    return false;
  }
  if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
    // No draw fallback in this round: without TRANSFER_DST, MSA is refused.
    Fail(Failure::kNoTransferDst, 0);
    return false;
  }
  // FinalOutput already carries A's clamped swapchain extent; a copy needs the
  // same one, never a stretch.
  if (extent_.width < capabilities.minImageExtent.width ||
      extent_.height < capabilities.minImageExtent.height ||
      extent_.width > capabilities.maxImageExtent.width ||
      extent_.height > capabilities.maxImageExtent.height) {
    Fail(Failure::kExtentMismatch, 0);
    return false;
  }
  if (!(capabilities.supportedTransforms &
        (VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR |
         VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR))) {
    Fail(Failure::kSwapchainCreate, 0);
    return false;
  }
  uint32_t format_count = 0;
  result = ifn.vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface_,
                                                    &format_count, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(format_count);
  if (result == VK_SUCCESS && format_count) {
    result = ifn.vkGetPhysicalDeviceSurfaceFormatsKHR(
        physical_device, surface_, &format_count, formats.data());
    formats.resize(format_count);
  }
  if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
    Fail(Failure::kNoFormat, int32_t(result));
    return false;
  }
  bool format_found = false;
  VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
    format_found = true;
  } else {
    for (const VkSurfaceFormatKHR& surface_format : formats) {
      if (surface_format.format != format_) {
        continue;
      }
      if (!format_found ||
          surface_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        color_space = surface_format.colorSpace;
      }
      format_found = true;
    }
  }
  if (!format_found) {
    Fail(Failure::kNoFormat, int32_t(format_));
    return false;
  }
  uint32_t image_count = std::max(capabilities.minImageCount + 1, 3u);
  if (capabilities.maxImageCount) {
    image_count = std::min(image_count, capabilities.maxImageCount);
  }
  VkSwapchainCreateInfoKHR create_info = {
      VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  create_info.surface = surface_;
  create_info.minImageCount = image_count;
  create_info.imageFormat = format_;
  create_info.imageColorSpace = color_space;
  create_info.imageExtent = extent_;
  create_info.imageArrayLayers = 1;
  // COLOR_ATTACHMENT is guaranteed; it keeps phase 2 (compose on B:1) open.
  create_info.imageUsage =
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  create_info.preTransform = (capabilities.supportedTransforms &
                              VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                                 ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                                 : VK_SURFACE_TRANSFORM_INHERIT_BIT_KHR;
  if (capabilities.supportedCompositeAlpha &
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) {
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  } else if (capabilities.supportedCompositeAlpha &
             VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  } else {
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
  }
  create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = VK_NULL_HANDLE;
  result = dfn.vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain_);
  if (result != VK_SUCCESS) {
    swapchain_ = VK_NULL_HANDLE;
    Fail(Failure::kSwapchainCreate, int32_t(result));
    return false;
  }
  uint32_t swapchain_image_count = 0;
  result = dfn.vkGetSwapchainImagesKHR(device, swapchain_,
                                       &swapchain_image_count, nullptr);
  if (result == VK_SUCCESS && swapchain_image_count) {
    images_.resize(swapchain_image_count);
    result = dfn.vkGetSwapchainImagesKHR(device, swapchain_,
                                         &swapchain_image_count, images_.data());
    images_.resize(swapchain_image_count);
  }
  if (result != VK_SUCCESS || images_.empty()) {
    Fail(Failure::kSwapchainImages, int32_t(result));
    return false;
  }
  // One present-wait semaphore per image: a semaphore is signaled again only
  // once its image has come back from the presentation engine.
  render_done_.assign(images_.size(), VK_NULL_HANDLE);
  VkSemaphoreCreateInfo semaphore_info = {
      VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  for (VkSemaphore& semaphore : render_done_) {
    if (dfn.vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore) !=
        VK_SUCCESS) {
      semaphore = VK_NULL_HANDLE;
      Fail(Failure::kResourceCreate, 0);
      return false;
    }
  }
  uint64_t refresh_cycle_ns = 0;
  if (timing_available_ && device_->vkGetRefreshCycleDurationGOOGLE()) {
    VkRefreshCycleDurationGOOGLE refresh = {};
    if (device_->vkGetRefreshCycleDurationGOOGLE()(device, swapchain_,
                                                   &refresh) == VK_SUCCESS) {
      refresh_cycle_ns = refresh.refreshDuration;
    }
  }
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    counters_.swapchain_ready = true;
    counters_.image_count = uint32_t(images_.size());
    counters_.refresh_cycle_ns = refresh_cycle_ns;
    counters_.refresh_cycle_now_ns = refresh_cycle_ns;
  }
  XELOGI(
      "ZeroFGMainSurface swapchain created mode=FIFO images={} requested={} "
      "extent={}x{} format={} color_space={} usage=transfer_dst|color "
      "queue=B:{} timing_available={} refresh_cycle_us={}",
      images_.size(), image_count, extent_.width, extent_.height,
      int(format_), int(color_space), queue_index_, timing_available_,
      refresh_cycle_ns / 1000);
  return true;
#else
  return false;
#endif
}

void ZeroFGMainSurfaceEgress::DestroySwapchainAfterIdle() {
  const VulkanDevice::Functions& dfn = device_->functions();
  const VkDevice device = device_->device();
  if (swapchain_ != VK_NULL_HANDLE || AnyCopyInFlight()) {
    // Every copy, and the present waits behind them, retire before the
    // swapchain, its semaphores or the FinalOutputs the copies read go away.
    const VulkanDevice::Queue::Acquisition queue =
        device_->AcquireQueue(queue_family_, queue_index_);
    const VkResult idle_result = dfn.vkQueueWaitIdle(queue.queue());
    if (idle_result == VK_ERROR_DEVICE_LOST) {
      device_->SetLost();
    }
  }
  if (image_fence_ != VK_NULL_HANDLE) {
    // A held image's release fence may still be pending: bounded, then gone
    // with the swapchain.
    if (held_image_ != UINT32_MAX) {
      dfn.vkWaitForFences(device, 1, &image_fence_, VK_TRUE, 100000000ull);
    }
    dfn.vkDestroyFence(device, image_fence_, nullptr);
    image_fence_ = VK_NULL_HANDLE;
    held_image_ = UINT32_MAX;
  }
  if (swapchain_ != VK_NULL_HANDLE) {
    dfn.vkDestroySwapchainKHR(device, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }
  for (VkSemaphore semaphore : render_done_) {
    if (semaphore != VK_NULL_HANDLE) {
      dfn.vkDestroySemaphore(device, semaphore, nullptr);
    }
  }
  render_done_.clear();
  images_.clear();
  for (CopyContext& context : contexts_) {
    context.in_flight = false;
  }
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    counters_.swapchain_ready = false;
  }
  // Only now, with B's swapchain gone, does the Surface leave B.
  const ZeroFGMainSurfaceProducer::State state = producer_->state();
  if (state == ZeroFGMainSurfaceProducer::State::kB) {
    producer_->Transition(ZeroFGMainSurfaceProducer::State::kB,
                          ZeroFGMainSurfaceProducer::State::kHandbackGap,
                          "B_egress_stop");
  } else if (state == ZeroFGMainSurfaceProducer::State::kHandoffGap) {
    // A retired but B never produced: A must recreate its swapchain.
    producer_->Transition(ZeroFGMainSurfaceProducer::State::kHandoffGap,
                          ZeroFGMainSurfaceProducer::State::kHandbackGap,
                          "B_egress_stop_before_swapchain");
  }
}

void ZeroFGMainSurfaceEgress::Serve() {
  std::array<Request, kMaxSlots> superseded = {};
  uint32_t superseded_count = 0;
  for (;;) {
    PollCopyCompletions();
    MaybePollTimings(false);
    Request request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stop_requested_) {
        break;
      }
      if (!ring_count_) {
        // Sleep until a request or the stop arrives, but come back for the
        // next copy completion or presentation timing observation.
        const auto timeout = AnyCopyInFlight() ? std::chrono::milliseconds(1)
                                               : std::chrono::milliseconds(4);
        condition_.wait_for(lock, timeout, [this] {
          return stop_requested_ || ring_count_ != 0;
        });
        if (stop_requested_) {
          break;
        }
        if (!ring_count_) {
          continue;
        }
      }
      request = ring_[ring_head_];
      ring_head_ = (ring_head_ + 1) % slot_count_;
      --ring_count_;
      // Outputs that arrived while the last present blocked: only the newest
      // is served. With the vsync quantizer and presents that do not block,
      // each output has a vsync of its own and is served in order: a burst
      // (a late S right before its R) is absorbed by the quantizer's lead
      // instead of losing the S (~5 a second at Rayman 120, 2026-10-07).
      if (free_output_ && (!vsync_quantizer_ || limiter_period_ns_ ||
                           limiter_block_streak_)) {
        while (ring_count_) {
          superseded[superseded_count++] = request;
          request = ring_[ring_head_];
          ring_head_ = (ring_head_ + 1) % slot_count_;
          --ring_count_;
        }
      }
    }
    for (uint32_t i = 0; i < superseded_count; ++i) {
      DropUnpresented(superseded[i], true);
    }
    superseded_count = 0;
    bool probe = false;
    if (free_output_ && !PaceFreePresent(request, probe)) {
      Abandon(request);
      break;
    }
    if (!ServeRequest(request, probe)) {
      break;
    }
  }
  // Whatever is still queued never reaches the swapchain.
  for (;;) {
    Request request;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!ring_count_) {
        break;
      }
      request = ring_[ring_head_];
      ring_head_ = (ring_head_ + 1) % slot_count_;
      --ring_count_;
    }
    Abandon(request);
  }
  // Publish the copies already in flight, bounded: teardown idles the queue
  // afterwards in any case.
  const uint64_t deadline_ns = EgressMonotonicTimeNs() + 250000000ull;
  while (AnyCopyInFlight() && !device_->IsLost() &&
         EgressMonotonicTimeNs() < deadline_ns &&
         failure_.load(std::memory_order_acquire) == Failure::kNone) {
    PollCopyCompletions();
    if (AnyCopyInFlight()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

bool ZeroFGMainSurfaceEgress::ServeRequest(const Request& request,
                                           bool probe) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (request.slot >= slot_count_ || request.image == VK_NULL_HANDLE) {
    Fail(Failure::kSlotReuse, 0);
    Abandon(request);
    return false;
  }
  CopyContext& context = contexts_[request.slot];
  if (context.in_flight) {
    // The FinalOutput came back before its previous copy was observed done.
    Fail(Failure::kSlotReuse, 0);
    Abandon(request);
    return false;
  }
  const VulkanDevice::Functions& dfn = device_->functions();
  const VkDevice device = device_->device();
  if (context.acquire_dirty) {
    // An acquire whose wait was never consumed leaves the semaphore pending.
    dfn.vkDestroySemaphore(device, context.acquire_semaphore, nullptr);
    context.acquire_semaphore = VK_NULL_HANDLE;
    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (dfn.vkCreateSemaphore(device, &semaphore_info, nullptr,
                              &context.acquire_semaphore) != VK_SUCCESS) {
      context.acquire_semaphore = VK_NULL_HANDLE;
      Fail(Failure::kResourceCreate, 0);
      Abandon(request);
      return false;
    }
    context.acquire_dirty = false;
  }

  // Bounded acquire: never UINT64_MAX (it could hold the join), never 0 (a
  // poll). Shutdown, teardown and completions are checked between attempts.
  const uint64_t acquire_begin_ns = EgressMonotonicTimeNs();
  uint32_t image_index = 0;
  bool wait_acquire = true;
  if (free_output_) {
    const FreeImage free_image = TryTakeFreeImage(image_index);
    if (free_image == FreeImage::kStop) {
      Abandon(request);
      return false;
    }
    if (free_image == FreeImage::kBusy) {
      // Every image we could copy into is still the compositor's: this output
      // is not presented, and its FinalOutput returns at once.
      DropUnpresented(request, false);
      return true;
    }
    // Its release fence has signaled: the copy needs no acquire wait.
    wait_acquire = false;
  }
  for (; !free_output_;) {
    if (StopRequested() || device_->IsLost() ||
        device_->zerofg_teardown_idle()) {
      Abandon(request);
      return false;
    }
    const VkResult acquire_result = dfn.vkAcquireNextImageKHR(
        device, swapchain_, kAcquireTimeoutNs, context.acquire_semaphore,
        VK_NULL_HANDLE, &image_index);
    if (acquire_result == VK_SUCCESS || acquire_result == VK_SUBOPTIMAL_KHR) {
      if (acquire_result == VK_SUBOPTIMAL_KHR) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++counters_.suboptimal;
      }
      break;
    }
    if (acquire_result == VK_TIMEOUT || acquire_result == VK_NOT_READY) {
      {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++counters_.acquire_timeouts;
      }
      PollCopyCompletions();
      MaybePollTimings(false);
      continue;
    }
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR ||
        acquire_result == VK_ERROR_SURFACE_LOST_KHR) {
      MarkSurfaceOutdated(int32_t(acquire_result));
      Abandon(request);
      return false;
    }
    if (acquire_result == VK_ERROR_DEVICE_LOST) {
      device_->SetLost();
      Fail(Failure::kDeviceLost, int32_t(acquire_result));
    } else {
      Fail(Failure::kAcquire, int32_t(acquire_result));
    }
    Abandon(request);
    return false;
  }
  const uint64_t acquire_block_ns =
      EgressMonotonicTimeNs() - acquire_begin_ns;
  if (image_index >= images_.size()) {
    context.acquire_dirty = true;
    Fail(Failure::kAcquire, 0);
    Abandon(request);
    return false;
  }

  // FinalOutput -> swapchain image. The FinalOutput is already in
  // TRANSFER_SRC_OPTIMAL (the Post's last barrier); the Post timeline wait
  // below makes its writes visible to this copy.
  const auto record_failed = [&]() {
    context.acquire_dirty = true;
    Fail(Failure::kRecord, 0);
    Abandon(request);
    return false;
  };
  if (dfn.vkResetCommandPool(device, context.command_pool, 0) != VK_SUCCESS) {
    return record_failed();
  }
  VkCommandBufferBeginInfo begin_info = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (dfn.vkBeginCommandBuffer(context.command_buffer, &begin_info) !=
      VK_SUCCESS) {
    return record_failed();
  }
  VkImageMemoryBarrier2 image_barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
  image_barrier.srcAccessMask = VK_ACCESS_2_NONE;
  image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
  image_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  // The whole image is overwritten, so its previous contents are discarded.
  image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  image_barrier.image = images_[image_index];
  image_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  image_barrier.subresourceRange.levelCount = 1;
  image_barrier.subresourceRange.layerCount = 1;
  VkDependencyInfo dependency = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &image_barrier;
  device_->vkCmdPipelineBarrier2()(context.command_buffer, &dependency);
  VkImageCopy region = {};
  region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.srcSubresource.layerCount = 1;
  region.dstSubresource = region.srcSubresource;
  region.extent = {extent_.width, extent_.height, 1};
  dfn.vkCmdCopyImage(context.command_buffer, request.image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     images_[image_index],
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  image_barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
  image_barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  image_barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
  image_barrier.dstAccessMask = VK_ACCESS_2_NONE;
  image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  image_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  device_->vkCmdPipelineBarrier2()(context.command_buffer, &dependency);
  if (dfn.vkEndCommandBuffer(context.command_buffer) != VK_SUCCESS) {
    return record_failed();
  }

  VkSemaphore wait_semaphores[2] = {};
  VkPipelineStageFlags wait_stages[2] = {};
  uint64_t wait_values[2] = {};
  uint32_t wait_count = 0;
  if (wait_acquire) {
    wait_semaphores[wait_count] = context.acquire_semaphore;
    wait_stages[wait_count] = VK_PIPELINE_STAGE_TRANSFER_BIT;
    wait_values[wait_count] = 0;
    ++wait_count;
  }
  if (request.post_semaphore != VK_NULL_HANDLE) {
    wait_semaphores[wait_count] = request.post_semaphore;
    wait_stages[wait_count] = VK_PIPELINE_STAGE_TRANSFER_BIT;
    wait_values[wait_count] = request.post_value;
    ++wait_count;
  }
  const VkSemaphore signal_semaphores[2] = {render_done_[image_index],
                                            context.copy_done_semaphore};
  VkTimelineSemaphoreSubmitInfo timeline_info = {
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline_info.waitSemaphoreValueCount = wait_count;
  timeline_info.pWaitSemaphoreValues = wait_values;
  VkSubmitInfo submit_info = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit_info.pNext = &timeline_info;
  submit_info.waitSemaphoreCount = wait_count;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &context.command_buffer;
  submit_info.signalSemaphoreCount = 2;
  submit_info.pSignalSemaphores = signal_semaphores;

  bool presented = false;
  uint64_t present_call_ns = 0;
  uint64_t present_ns = 0;
  bool keep_serving = true;
  int copy_done_fd = -1;
  uint64_t submit_ns = 0;
  {
    const VulkanDevice::Queue::Acquisition queue =
        device_->AcquireQueue(queue_family_, queue_index_);
    if (device_->zerofg_teardown_idle()) {
      // Device B is already idle for teardown: no new work may enter it. The
      // acquired image stays with the swapchain until its destruction.
      context.acquire_dirty = true;
      Abandon(request);
      return false;
    }
    const VkResult submit_result =
        dfn.vkQueueSubmit(queue.queue(), 1, &submit_info, VK_NULL_HANDLE);
    if (submit_result != VK_SUCCESS) {
      context.acquire_dirty = true;
      if (submit_result == VK_ERROR_DEVICE_LOST) {
        device_->SetLost();
        Fail(Failure::kDeviceLost, int32_t(submit_result));
      } else {
        Fail(Failure::kSubmit, int32_t(submit_result));
      }
      Abandon(request);
      return false;
    }
    submit_ns = EgressMonotonicTimeNs();
    VkSemaphoreGetFdInfoKHR fd_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    fd_info.semaphore = context.copy_done_semaphore;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    const VkResult export_result =
        device_->vkGetSemaphoreFdKHR()(device, &fd_info, &copy_done_fd);
    context.in_flight = true;
    context.sequence_id = request.sequence_id;
    context.submit_ns = submit_ns;
    if (export_result != VK_SUCCESS) {
      // The copy runs, but nothing can prove its end: the FinalOutput stays
      // owned until teardown idles the queue.
      copy_done_fd = -1;
      context.copy_done_fd = -2;
      Fail(Failure::kCopyFenceExport, int32_t(export_result));
      keep_serving = false;
    } else {
      // -1 is a valid already-signaled sync_file.
      context.copy_done_fd = copy_done_fd;
    }
    VkPresentTimeGOOGLE present_time = {};
    present_time.presentID = uint32_t(request.sequence_id);
    // The Apocalypse Guard may raise it; request.desired_present_ns stays the
    // target-vs-actual telemetry reference.
    present_time.desiredPresentTime =
        apocalypse_guard_ ? ShapeDesiredPresent(request, EgressMonotonicTimeNs())
                          : request.desired_present_ns;
    context.assigned_vsync_ns = 0;
    if (vsync_quantizer_) {
      present_time.desiredPresentTime =
          QuantizeToVsync(present_time.desiredPresentTime,
                          EgressMonotonicTimeNs(), context.assigned_vsync_ns);
    }
    VkPresentTimesInfoGOOGLE present_times = {
        VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE};
    present_times.swapchainCount = 1;
    present_times.pTimes = &present_time;
    VkPresentInfoKHR present_info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present_info.pNext = timing_available_ ? &present_times : nullptr;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_done_[image_index];
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &image_index;
    present_call_ns = EgressMonotonicTimeNs();
    const VkResult present_result =
        dfn.vkQueuePresentKHR(queue.queue(), &present_info);
    present_ns = EgressMonotonicTimeNs();
    if (present_result == VK_SUCCESS ||
        present_result == VK_SUBOPTIMAL_KHR) {
      presented = true;
      if (present_result == VK_SUBOPTIMAL_KHR) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++counters_.suboptimal;
      }
    } else if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
               present_result == VK_ERROR_SURFACE_LOST_KHR) {
      MarkSurfaceOutdated(int32_t(present_result));
      keep_serving = false;
    } else if (present_result == VK_ERROR_DEVICE_LOST) {
      device_->SetLost();
      Fail(Failure::kDeviceLost, int32_t(present_result));
      keep_serving = false;
    } else {
      Fail(Failure::kPresent, int32_t(present_result));
      keep_serving = false;
    }
  }
  context.presented = presented;
  if (free_output_ && present_call_ns && present_ns >= present_call_ns) {
    ObservePresentCall(present_call_ns, present_ns, probe);
  }
  if (presented) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    TimingRecord& record = timing_records_[timing_record_next_];
    if (timing_available_ && record.used && !record.resolved) {
      // Android keeps a short timing history; a record overwritten before
      // its timing came back is counted, never invented.
      ++counters_.timings_missing;
    }
    record.used = true;
    record.resolved = false;
    record.present_id = uint32_t(request.sequence_id);
    record.sequence_id = request.sequence_id;
    record.synthetic = request.synthetic;
    record.desired_ns = request.desired_present_ns;
    record.present_ns = present_ns;
    record.ready_ns = 0;
    record.assigned_vsync_ns = context.assigned_vsync_ns;
    if (request.desired_present_ns) {
      present_call_vs_desired_ns_.Add(BiasSigned(
          int64_t(present_call_ns) - int64_t(request.desired_present_ns)));
    }
    present_call_ns_.Add(present_ns - present_call_ns);
    timing_record_next_ = (timing_record_next_ + 1) % timing_records_.size();
    ++counters_.presents;
    if (request.synthetic) {
      ++counters_.presents_synthetic;
    }
    acquire_block_ns_.Add(acquire_block_ns);
  }
  return keep_serving;
#else
  return false;
#endif
}

// GPU guard: never show two outputs less than one 60 Hz
// period apart on the panel, while the panel itself stays at its maximum.
//
// When. Eligible only while the pacing plans at most 60 outputs per second
// (its output quantum is at least one 60 Hz period) and the panel runs above
// 60 Hz: above that the cap would only cut planned outputs. Eligible is not
// enough. At a healthy 60 planned outputs per second the cap has no spare
// capacity (bundle 144226: a burst's lag drained 0.05 ms per output), so the
// shaper starts off and latches on only once the egress observes the
// pathology it exists for (ObserveLatchArrival). Once latched it stays on
// until this egress stops: the shaper removes the very bursts and gaps the
// detector reads, so the detector has no authority after it acts. The pacing
// only authorizes, through the quantum; nothing feeds back to it.
//
// How. desiredPresentTime is a not-before in the display domain: the
// compositor holds the buffer until the first vertical blank after it. An
// output handed over now reaches the panel only about one natural latency
// later (FIFO plus compositor), so the spacing runs from each output's
// expected display, not from its desired time:
//   latest(n)      = max(desired(n) + refresh, call(n) + natural latency)
//   desired(n + 1) >= latest(n) + (vblanks - 1) * refresh
// with vblanks the vertical blanks in one 60 Hz period (2 at 120 Hz). The
// natural latency is fixed at 2.25 refresh cycles, its measured p90 at
// 120 Hz (bundle 141237), and is never learned: the only presents that
// sample it while the shaper binds are the ones it let through, and those
// queue behind the outputs it holds, so a learned value grows with the
// backlog the shaper itself created and runs away. An output that arrives
// after its floor keeps its own desired time and is never held, and the next
// one plans from it: no debt carries over.
//
// No actuator bound. While latched the shaper trades latency for spacing: in
// saturation it added 30-45 ms (bundle 152553). Bounded to one 60 Hz period
// from the present call it lost nearly all the smoothing there (bundle
// 161507: overtaken ~87 per window, against ~2 unbounded and ~110 without the
// shaper), because a saturated output already takes 31-37 ms to reach the
// panel. It acts only once the latch is on.
uint64_t ZeroFGMainSurfaceEgress::ShapeDesiredPresent(const Request& request,
                                                      uint64_t now_ns) {
  const uint64_t original_ns = request.desired_present_ns;
  std::lock_guard<std::mutex> lock(stats_mutex_);
  const uint64_t refresh_ns = counters_.refresh_cycle_now_ns;
  // 5% under one 60 Hz period absorbs the panel's real refresh (8.31 ms is
  // still 2 vertical blanks) and a Source period measured a little short.
  const uint64_t spacing_floor_ns = kShaperSpacingNs - kShaperSpacingNs / 20;
  const uint64_t vblanks =
      refresh_ns ? (spacing_floor_ns + refresh_ns - 1) / refresh_ns : 0;
  if (vblanks < 2 || request.output_quantum_ns < spacing_floor_ns) {
    shaper_latest_display_ns_ = 0;
    ResetLatchWindow();
    return original_ns;
  }
  if (!counters_.apocalypse_latched) {
    ObserveLatchArrival(request.apply_time_ns, refresh_ns);
    if (!counters_.apocalypse_latched) {
      shaper_latest_display_ns_ = 0;
      return original_ns;
    }
  }
  const uint64_t latency_ns = 2 * refresh_ns + refresh_ns / 4;
  uint64_t desired_ns = original_ns;
  if (shaper_latest_display_ns_) {
    const uint64_t floor_ns =
        shaper_latest_display_ns_ + (vblanks - 1) * refresh_ns;
    const uint64_t unshaped_ns = std::max(original_ns, now_ns);
    if (floor_ns > unshaped_ns) {
      desired_ns = floor_ns;
      ++counters_.shaper_raised;
      shaper_push_ns_.Add(floor_ns - unshaped_ns);
    }
  }
  shaper_latest_display_ns_ =
      std::max(desired_ns + refresh_ns, now_ns + latency_ns);
  return desired_ns;
}

void ZeroFGMainSurfaceEgress::ResetLatchWindow() {
  latch_window_begin_ns_ = 0;
  latch_window_arrivals_ = 0;
  latch_degraded_streak_ = 0;
  latch_interval_count_ = 0;
}

// The shaper's degradation detector, fed while it is eligible and not yet
// latched, so every interval it reads is the output as shown without the
// shaper. Request arrivals (the presenter's apply times) open consecutive
// windows of at least 1 s; the actual intervals come from the presentation
// timings seen meanwhile. A window is degraded when all of these hold:
//   - at least 30 arrivals per second: still running (a loading pause is not
//     this pathology). No ceiling: in bundle 155221 the display collapsed
//     (overtaken about a third of presents, interval p90 42-91 ms) for ~40 s
//     while arrivals held 56-61/s; the frames arrived and showed as gaps and
//     bursts;
//   - at least one interval of at most 1.5 vblanks: a burst on the panel;
//   - an interval p90 of at least 3.5 vblanks: 4-vblank gaps (33 ms at 120 Hz);
//   - at least 20 intervals to judge;
//   - the GPU at least 97% busy (VulkanDevice::QueryGpuBusyPermille; KGSL's
//     last ~1 s on Adreno): the plant really hit its ceiling. The latch is one
//     way and costs 30-45 ms while on, so a transient in healthy play must
//     not trip it. The temporal pathology alone latched ~33 s into healthy
//     play at GPU 76-80% (bundle 161507); no healthy phase reached 97% (0 of
//     40 s in each of bundles 152553, 155221, 161507 and 164403), and every
//     saturation latch had 97-99%. A device without the reading never
//     confirms. GPU load decides nothing else.
// kLatchWindows consecutive confirmed windows latch the shaper on. A single
// window would not do: bundle 144226 had isolated 1 s output dips (53.8, 50.2
// and 48.8/s at 20, 41 and 53 s) during its healthy phase.
void ZeroFGMainSurfaceEgress::ObserveLatchArrival(uint64_t arrival_ns,
                                                  uint64_t refresh_ns) {
  if (!latch_window_begin_ns_ || arrival_ns < latch_window_begin_ns_) {
    latch_window_begin_ns_ = arrival_ns;
    latch_window_arrivals_ = 1;
    latch_interval_count_ = 0;
    return;
  }
  const uint64_t duration_ns = arrival_ns - latch_window_begin_ns_;
  if (duration_ns < kLatchWindowNs) {
    ++latch_window_arrivals_;
    return;
  }
  const uint64_t rate_mfps =
      uint64_t(latch_window_arrivals_) * 1000000000000ull / duration_ns;
  const uint32_t count = latch_interval_count_;
  std::array<uint64_t, kLatchIntervalCapacity> sorted = latch_intervals_ns_;
  std::sort(sorted.begin(), sorted.begin() + count);
  uint32_t bursts = 0;
  while (bursts < count && 2 * sorted[bursts] <= 3 * refresh_ns) {
    ++bursts;
  }
  const uint64_t gap_p90_ns =
      count ? sorted[std::min<size_t>(count - 1,
                                      (size_t(count - 1) * 90 + 99) / 100)]
            : 0;
  const bool pathology =
      count >= kLatchMinIntervals && rate_mfps >= kLatchRateFloorMfps &&
      bursts && 2 * gap_p90_ns >= 7 * refresh_ns;
  const std::optional<uint32_t> gpu_busy_permille =
      device_->QueryGpuBusyPermille();
  counters_.gpu_busy_permille =
      gpu_busy_permille ? int32_t(*gpu_busy_permille) : -1;
  if (!gpu_busy_permille && !gpu_busy_unavailable_logged_) {
    gpu_busy_unavailable_logged_ = true;
    XELOGW(
        "ZeroFGMainSurface apocalypse_guard gpu_busy unavailable: saturation "
        "cannot be confirmed, so the guard never latches");
  }
  const bool confirmed = pathology && gpu_busy_permille &&
                         *gpu_busy_permille >= kLatchGpuBusyPermille;
  ++counters_.latch_windows_evaluated;
  if (pathology) {
    ++counters_.latch_windows_pathology;
  }
  if (confirmed) {
    ++counters_.latch_windows_confirmed;
    ++latch_degraded_streak_;
  } else {
    latch_degraded_streak_ = 0;
  }
  if (latch_degraded_streak_ >= kLatchWindows) {
    counters_.apocalypse_latched = true;
    XELOGI(
        "ZeroFGMainSurface apocalypse_latched windows={} arrivals_mfps={} "
        "bursts={} interval_p90_us={} intervals={} gpu_busy_pm={} "
        "refresh_us={}",
        latch_degraded_streak_, rate_mfps, bursts, gap_p90_ns / 1000, count,
        counters_.gpu_busy_permille, refresh_ns / 1000);
    ResetLatchWindow();
    return;
  }
  latch_window_begin_ns_ = arrival_ns;
  latch_window_arrivals_ = 1;
  latch_interval_count_ = 0;
}

ZeroFGMainSurfaceEgress::FreeImage ZeroFGMainSurfaceEgress::TryTakeFreeImage(
    uint32_t& image_index) {
  const VulkanDevice::Functions& dfn = device_->functions();
  const VkDevice device = device_->device();
  if (image_fence_ == VK_NULL_HANDLE) {
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (dfn.vkCreateFence(device, &fence_info, nullptr, &image_fence_) !=
        VK_SUCCESS) {
      image_fence_ = VK_NULL_HANDLE;
      Fail(Failure::kResourceCreate, 0);
      return FreeImage::kStop;
    }
  }
  if (held_image_ == UINT32_MAX) {
    // Android hands out the next buffer at once, with a release fence that
    // signals when the compositor lets it go; the fence, not this call, says
    // whether it is free.
    uint32_t index = 0;
    const VkResult acquire_result = dfn.vkAcquireNextImageKHR(
        device, swapchain_, kAcquireTimeoutNs, VK_NULL_HANDLE, image_fence_,
        &index);
    if (acquire_result == VK_SUCCESS || acquire_result == VK_SUBOPTIMAL_KHR) {
      if (acquire_result == VK_SUBOPTIMAL_KHR) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++counters_.suboptimal;
      }
      if (index >= images_.size()) {
        Fail(Failure::kAcquire, 0);
        return FreeImage::kStop;
      }
      held_image_ = index;
    } else if (acquire_result == VK_TIMEOUT ||
               acquire_result == VK_NOT_READY) {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      ++counters_.acquire_timeouts;
      return FreeImage::kBusy;
    } else if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR ||
               acquire_result == VK_ERROR_SURFACE_LOST_KHR) {
      MarkSurfaceOutdated(int32_t(acquire_result));
      return FreeImage::kStop;
    } else if (acquire_result == VK_ERROR_DEVICE_LOST) {
      device_->SetLost();
      Fail(Failure::kDeviceLost, int32_t(acquire_result));
      return FreeImage::kStop;
    } else {
      Fail(Failure::kAcquire, int32_t(acquire_result));
      return FreeImage::kStop;
    }
  }
  const VkResult fence_status = dfn.vkGetFenceStatus(device, image_fence_);
  if (fence_status == VK_NOT_READY) {
    return FreeImage::kBusy;
  }
  if (fence_status != VK_SUCCESS) {
    if (fence_status == VK_ERROR_DEVICE_LOST) {
      device_->SetLost();
      Fail(Failure::kDeviceLost, int32_t(fence_status));
    } else {
      Fail(Failure::kAcquire, int32_t(fence_status));
    }
    return FreeImage::kStop;
  }
  dfn.vkResetFences(device, 1, &image_fence_);
  image_index = held_image_;
  held_image_ = UINT32_MAX;
  return FreeImage::kFree;
}

bool ZeroFGMainSurfaceEgress::PaceFreePresent(Request& request,
                                              bool& probe) {
  probe = false;
  if (!limiter_period_ns_) {
    return true;
  }
  const uint64_t start_ns = EgressMonotonicTimeNs();
  if (start_ns >= limiter_last_sleep_ns_ + kLimiterProbeNs &&
      start_ns >= limiter_last_probe_ns_ + kLimiterProbeNs &&
      start_ns < last_present_return_ns_ + limiter_period_ns_ / 2) {
    // No present has slept for a while: this one skips the grid, less than
    // half a period after the last present, where a limiter still in place
    // must hold it. If it does not sleep, the limiter is gone.
    probe = true;
    limiter_last_probe_ns_ = start_ns;
    return true;
  }
  if (start_ns > limiter_anchor_ns_) {
    // A grid point that just passed unused (in its first half period): now.
    const uint64_t passed_ns =
        limiter_anchor_ns_ + (start_ns - limiter_anchor_ns_) /
                                 limiter_period_ns_ * limiter_period_ns_;
    if (passed_ns > limiter_last_slot_ns_ &&
        start_ns - passed_ns < limiter_period_ns_ / 2) {
      limiter_last_slot_ns_ = passed_ns;
      return true;
    }
  }
  // The next grid point after the anchor that is not yet used.
  const uint64_t base_ns = std::max(start_ns, limiter_last_slot_ns_ + 1);
  uint64_t slot_ns = limiter_anchor_ns_;
  if (base_ns > limiter_anchor_ns_) {
    slot_ns += (base_ns - limiter_anchor_ns_ + limiter_period_ns_ - 1) /
               limiter_period_ns_ * limiter_period_ns_;
  }
  limiter_last_slot_ns_ = slot_ns;
  const uint64_t due_ns = slot_ns + kLimiterMarginNs;
  for (;;) {
    const uint64_t now_ns = EgressMonotonicTimeNs();
    if (now_ns >= due_ns) {
      return true;
    }
    PollCopyCompletions();
    MaybePollTimings(false);
    std::array<Request, kMaxSlots> superseded = {};
    uint32_t superseded_count = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stop_requested_) {
        return false;
      }
      condition_.wait_for(
          lock,
          std::chrono::nanoseconds(std::min<uint64_t>(due_ns - now_ns,
                                                      1000000ull)),
          [this] { return stop_requested_ || ring_count_ != 0; });
      if (stop_requested_) {
        return false;
      }
      while (ring_count_ && superseded_count < superseded.size()) {
        superseded[superseded_count++] = request;
        request = ring_[ring_head_];
        ring_head_ = (ring_head_ + 1) % slot_count_;
        --ring_count_;
      }
    }
    for (uint32_t i = 0; i < superseded_count; ++i) {
      DropUnpresented(superseded[i], true);
    }
  }
}

void ZeroFGMainSurfaceEgress::ObservePresentCall(uint64_t call_ns,
                                                 uint64_t return_ns,
                                                 bool probe) {
  const uint64_t slept_ns = return_ns - call_ns;
  const uint64_t since_ns = last_present_return_ns_ &&
                                    return_ns > last_present_return_ns_
                                ? return_ns - last_present_return_ns_
                                : 0;
  last_present_return_ns_ = return_ns;
  uint64_t refresh_ns = 0;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    refresh_ns = counters_.refresh_cycle_now_ns;
  }
  if (!limiter_period_ns_) {
    if (slept_ns >= kLimiterBlockedNs && since_ns && since_ns < 100000000ull) {
      limiter_samples_ns_[limiter_block_streak_++] = since_ns;
      if (limiter_block_streak_ >= kLimiterEngageStreak) {
        limiter_block_streak_ = 0;
        // The shortest interval: the others are whole multiples of it.
        const uint64_t period_ns = *std::min_element(
            limiter_samples_ns_.begin(), limiter_samples_ns_.end());
        if (refresh_ns && period_ns * 10 < refresh_ns * 11) {
          return;  // The panel's own rate: no limiter.
        }
        // A limiter releases on its grid: every interval is a whole number of
        // periods. A present held by GPU back-pressure returns anywhere.
        bool on_grid = period_ns != 0;
        for (const uint64_t sample_ns : limiter_samples_ns_) {
          const uint64_t whole = (sample_ns + period_ns / 2) / period_ns;
          const uint64_t grid_ns = whole * period_ns;
          const uint64_t error_ns =
              sample_ns > grid_ns ? sample_ns - grid_ns : grid_ns - sample_ns;
          if (error_ns * kLimiterGridTolerance > period_ns) {
            on_grid = false;
          }
        }
        static_assert(kLimiterEngageStreak == 4, "the logs print four");
        if (!on_grid) {
          {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++counters_.present_limiter_rejected;
          }
          XELOGI(
              "ZeroFGMainSurface present_limiter rejected (off grid) "
              "intervals_us={}/{}/{}/{} blocked_present_us={} refresh_us={}",
              limiter_samples_ns_[0] / 1000, limiter_samples_ns_[1] / 1000,
              limiter_samples_ns_[2] / 1000, limiter_samples_ns_[3] / 1000,
              slept_ns / 1000, refresh_ns / 1000);
          return;
        }
        limiter_period_ns_ = period_ns;
        limiter_anchor_ns_ = return_ns;
        limiter_last_slot_ns_ = return_ns;
        limiter_last_sleep_ns_ = return_ns;
        limiter_last_probe_ns_ = return_ns;
        limiter_sleep_streak_ = 0;
        {
          std::lock_guard<std::mutex> lock(stats_mutex_);
          counters_.present_limiter_ns = limiter_period_ns_;
          ++counters_.present_limiter_engaged;
        }
        XELOGI(
            "ZeroFGMainSurface present_limiter engaged period_us={} "
            "intervals_us={}/{}/{}/{} blocked_present_us={} refresh_us={}",
            limiter_period_ns_ / 1000, limiter_samples_ns_[0] / 1000,
            limiter_samples_ns_[1] / 1000, limiter_samples_ns_[2] / 1000,
            limiter_samples_ns_[3] / 1000, slept_ns / 1000,
            refresh_ns / 1000);
      }
    } else {
      limiter_block_streak_ = 0;
    }
    return;
  }
  bool cleared = false;
  if (slept_ns >= kLimiterEarlyNs) {
    // Released on a grid point: re-anchor there (never refine the period).
    limiter_anchor_ns_ = return_ns;
    limiter_last_slot_ns_ = return_ns;
    limiter_last_sleep_ns_ = return_ns;
    if (++limiter_sleep_streak_ >= kLimiterResleepStreak) {
      // Every paced present sleeps: the grid changed. Learn it again.
      cleared = true;
    }
  } else {
    limiter_sleep_streak_ = 0;
    if (probe) {
      // Off the grid and not held: the limiter is gone.
      cleared = true;
    }
  }
  if (refresh_ns && limiter_period_ns_ * 10 < refresh_ns * 11) {
    cleared = true;
  }
  if (cleared) {
    limiter_period_ns_ = 0;
    limiter_block_streak_ = 0;
    limiter_sleep_streak_ = 0;
  }
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    counters_.present_limiter_ns = limiter_period_ns_;
    if (cleared) {
      ++counters_.present_limiter_cleared;
    }
  }
  if (cleared) {
    XELOGI("ZeroFGMainSurface present_limiter cleared probe={} refresh_us={}",
           probe, refresh_ns / 1000);
  }
}

uint64_t ZeroFGMainSurfaceEgress::QuantizeToVsync(uint64_t target_ns,
                                                 uint64_t now_ns,
                                                 uint64_t& assigned_ns) {
  assigned_ns = 0;
  uint64_t period_ns = 0;
  uint64_t copy_ns = 0;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    period_ns = counters_.refresh_cycle_now_ns;
    if (copy_residence_ns_.count()) {
      copy_ns = copy_residence_ns_.Quantile(50, 100);
    }
  }
  if (!period_ns || !vsync_anchor_ns_) {
    return target_ns;
  }
  const uint64_t anchor_ns = vsync_anchor_ns_;
  // The first vsync of the lattice at or after t.
  const auto vsync_at_or_after = [anchor_ns, period_ns](uint64_t t) {
    if (t <= anchor_ns) {
      return anchor_ns - (anchor_ns - t) / period_ns * period_ns;
    }
    return anchor_ns + (t - anchor_ns + period_ns - 1) / period_ns * period_ns;
  };
  // The first vsync this output's copy can still make.
  const uint64_t reach_ns =
      vsync_at_or_after(now_ns + copy_ns + period_ns + kVsyncLatchMarginNs);
  uint64_t slot_ns =
      vsync_at_or_after(target_ns) + vsync_offset_periods_ * period_ns;
  bool reanchored = false;
  bool refused = false;
  bool absorbed = false;
  if (vsync_last_slot_ns_ && slot_ns < vsync_last_slot_ns_ + period_ns / 2) {
    // The previous present has this vsync: the next one, in order.
    slot_ns = vsync_last_slot_ns_ + period_ns;
    refused = true;
  }
  if (slot_ns + period_ns / 2 < reach_ns) {
    // Not reachable: the phase moves forward (future-only) to the first
    // vsync the copy can make, and stays there.
    const uint64_t periods =
        (reach_ns - slot_ns + period_ns / 2) / period_ns;
    vsync_offset_periods_ += periods;
    slot_ns += periods * period_ns;
    reanchored = true;
  } else if (vsync_offset_periods_ && !refused &&
             slot_ns >= period_ns + reach_ns &&
             (!vsync_last_slot_ns_ ||
              slot_ns - period_ns >= vsync_last_slot_ns_ + period_ns)) {
    // The vsync before is free and reachable (the slow beat of the two clocks
    // left a gap): shed one refresh of phase into it.
    --vsync_offset_periods_;
    slot_ns -= period_ns;
    absorbed = true;
  }
  if (slot_ns > now_ns + 8 * period_ns) {
    // A lattice far ahead of now is not a phase any more: start over.
    vsync_offset_periods_ = 0;
    vsync_last_slot_ns_ = 0;
    return target_ns;
  }
  vsync_last_slot_ns_ = slot_ns;
  assigned_ns = slot_ns;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++counters_.vsync_quantized;
    if (reanchored) {
      ++counters_.vsync_phase_reanchor;
    }
    if (refused) {
      ++counters_.vsync_duplicate_refused;
    }
    if (absorbed) {
      ++counters_.vsync_gap_absorbed;
    }
  }
  // Half a refresh before the vsync: the compositor targets exactly this one.
  return slot_ns - period_ns / 2;
}

void ZeroFGMainSurfaceEgress::DropUnpresented(const Request& request,
                                              bool superseded) {
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (superseded) {
      ++counters_.free_superseded;
      if (request.synthetic) {
        ++counters_.free_superseded_synthetic;
      }
    } else {
      ++counters_.free_dropped;
      if (request.synthetic) {
        ++counters_.free_dropped_synthetic;
      }
    }
  }
  Completion completion;
  completion.slot = request.slot;
  completion.sequence_id = request.sequence_id;
  completion.copy_done_ns = EgressMonotonicTimeNs();
  completion.presented = false;
  PushCompletion(completion);
}

void ZeroFGMainSurfaceEgress::Abandon(const Request& request) {
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++counters_.abandoned;
  }
  Completion completion;
  completion.slot = request.slot;
  completion.sequence_id = request.sequence_id;
  completion.copy_done_ns = 0;
  completion.presented = false;
  // An abandoned request never had a copy submitted, unless the copy context
  // is still in flight, in which case its completion reports it.
  if (request.slot < slot_count_ && contexts_[request.slot].in_flight &&
      contexts_[request.slot].sequence_id == request.sequence_id) {
    return;
  }
  PushCompletion(completion);
}

void ZeroFGMainSurfaceEgress::PollCopyCompletions() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  for (uint32_t i = 0; i < slot_count_; ++i) {
    CopyContext& context = contexts_[i];
    if (!context.in_flight || context.copy_done_fd == -2) {
      continue;
    }
    if (context.copy_done_fd >= 0) {
      pollfd descriptor = {context.copy_done_fd, POLLIN, 0};
      const int result = poll(&descriptor, 1, 0);
      if (result == 0 || (result < 0 && errno == EINTR)) {
        continue;
      }
      const bool signaled = result > 0 && (descriptor.revents & POLLIN) &&
                            !(descriptor.revents & (POLLERR | POLLNVAL));
      if (!signaled) {
        // No proof that the copy stopped reading the FinalOutput: keep it
        // owned and fail open rather than recycle it under a live read.
        {
          std::lock_guard<std::mutex> lock(stats_mutex_);
          ++counters_.copy_fence_errors;
        }
        context.copy_done_fd = -2;
        Fail(Failure::kCopyFenceExport, result);
        continue;
      }
      close(context.copy_done_fd);
      context.copy_done_fd = -1;
    }
    context.in_flight = false;
    const uint64_t now_ns = EgressMonotonicTimeNs();
    {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      ++counters_.copies_completed;
      if (context.submit_ns && now_ns >= context.submit_ns) {
        copy_residence_ns_.Add(now_ns - context.submit_ns);
      }
      if (context.assigned_vsync_ns) {
        ready_before_vsync_ns_.Add(BiasSigned(
            int64_t(context.assigned_vsync_ns) - int64_t(now_ns)));
      }
      if (context.presented) {
        // The image is ready for the compositor from here on.
        for (TimingRecord& record : timing_records_) {
          if (record.used && !record.resolved && !record.ready_ns &&
              record.sequence_id == context.sequence_id) {
            record.ready_ns = now_ns;
            if (record.desired_ns) {
              ready_vs_desired_ns_.Add(BiasSigned(
                  int64_t(now_ns) - int64_t(record.desired_ns)));
            }
            break;
          }
        }
      }
    }
    Completion completion;
    completion.slot = i;
    completion.sequence_id = context.sequence_id;
    completion.copy_done_ns = now_ns;
    completion.presented = context.presented;
    PushCompletion(completion);
  }
#endif
}

void ZeroFGMainSurfaceEgress::MaybePollTimings(bool force) {
  if (!timing_available_ || swapchain_ == VK_NULL_HANDLE) {
    return;
  }
  const uint64_t now_ns = EgressMonotonicTimeNs();
  if (!force && now_ns < next_timing_poll_ns_) {
    return;
  }
  next_timing_poll_ns_ = now_ns + 4000000ull;
  if (now_ns >= next_refresh_poll_ns_ &&
      device_->vkGetRefreshCycleDurationGOOGLE()) {
    // The display mode can change under us (60 <-> 120): sample it twice a
    // second so the intervals can be read against the live refresh.
    next_refresh_poll_ns_ = now_ns + 500000000ull;
    VkRefreshCycleDurationGOOGLE refresh = {};
    if (device_->vkGetRefreshCycleDurationGOOGLE()(
            device_->device(), swapchain_, &refresh) == VK_SUCCESS &&
        refresh.refreshDuration) {
      std::lock_guard<std::mutex> lock(stats_mutex_);
      const uint64_t previous_ns = counters_.refresh_cycle_now_ns;
      const uint64_t current_ns = refresh.refreshDuration;
      const uint64_t change_ns = current_ns > previous_ns
                                     ? current_ns - previous_ns
                                     : previous_ns - current_ns;
      if (previous_ns && change_ns * 8 > previous_ns) {
        ++counters_.refresh_mode_changes;
      }
      counters_.refresh_cycle_now_ns = current_ns;
    }
  }
  std::array<VkPastPresentationTimingGOOGLE, 16> timings = {};
  uint32_t count = uint32_t(timings.size());
  const VkResult result = device_->vkGetPastPresentationTimingGOOGLE()(
      device_->device(), swapchain_, &count, timings.data());
  std::lock_guard<std::mutex> lock(stats_mutex_);
  if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
    ++counters_.timing_query_errors;
    if (result == VK_ERROR_DEVICE_LOST) {
      device_->SetLost();
    }
    return;
  }
  count = std::min(count, uint32_t(timings.size()));
  for (uint32_t i = 0; i < count; ++i) {
    const VkPastPresentationTimingGOOGLE& timing = timings[i];
    TimingRecord* record = nullptr;
    for (size_t offset = 1; offset <= timing_records_.size(); ++offset) {
      TimingRecord& candidate =
          timing_records_[(timing_record_next_ + timing_records_.size() -
                           offset) %
                          timing_records_.size()];
      if (candidate.used && !candidate.resolved &&
          candidate.present_id == timing.presentID) {
        record = &candidate;
        break;
      }
    }
    if (!record) {
      ++counters_.timings_unmatched;
      continue;
    }
    record->resolved = true;
    ++counters_.timings_returned;
    // Android reports timings in present order and discards older entries
    // that are still pending once a later one is ready: those will never be
    // reported, most likely dropped by the compositor.
    for (TimingRecord& older : timing_records_) {
      if (older.used && !older.resolved &&
          int32_t(older.present_id - timing.presentID) < 0) {
        older.resolved = true;
        ++counters_.overtaken;
      }
    }
    // Observation only: the actual time never moves a target, a floor or
    // the next desired time.
    const uint64_t actual_ns = timing.actualPresentTime;
    present_margin_ns_.Add(BiasSigned(int64_t(timing.presentMargin)));
    if (timing.earliestPresentTime && actual_ns >= timing.earliestPresentTime) {
      const uint64_t earliest_gap_ns = actual_ns - timing.earliestPresentTime;
      earliest_gap_ns_.Add(earliest_gap_ns);
      if (counters_.refresh_cycle_now_ns &&
          earliest_gap_ns * 2 >= counters_.refresh_cycle_now_ns) {
        ++counters_.missed_earliest;
      }
    }
    if (record->ready_ns && actual_ns >= record->ready_ns) {
      ready_to_actual_ns_.Add(actual_ns - record->ready_ns);
    }
    if (record->assigned_vsync_ns && counters_.refresh_cycle_now_ns) {
      const int64_t assigned_ns = int64_t(record->assigned_vsync_ns);
      const bool late = int64_t(actual_ns) >=
                        assigned_ns + int64_t(counters_.refresh_cycle_now_ns / 2);
      const uint64_t queue_before_ns =
          BiasSigned(assigned_ns - int64_t(record->present_ns));
      if (late) {
        ++counters_.vsync_late;
        late_queue_before_vsync_ns_.Add(queue_before_ns);
      } else {
        ++counters_.vsync_hit;
        hit_queue_before_vsync_ns_.Add(queue_before_ns);
      }
      if (record->ready_ns) {
        const uint64_t ready_before_ns =
            BiasSigned(assigned_ns - int64_t(record->ready_ns));
        (late ? late_ready_before_vsync_ns_ : hit_ready_before_vsync_ns_)
            .Add(ready_before_ns);
      }
    }
    if (actual_ns > last_actual_ns_) {
      if (last_actual_ns_) {
        actual_interval_ns_.Add(actual_ns - last_actual_ns_);
        // The shaper's degradation detector reads the output only while its
        // window is open, i.e. eligible and not latched.
        if (latch_window_begin_ns_ &&
            latch_interval_count_ < kLatchIntervalCapacity) {
          latch_intervals_ns_[latch_interval_count_++] =
              actual_ns - last_actual_ns_;
        }
      }
      last_actual_ns_ = actual_ns;
      vsync_anchor_ns_ = actual_ns;
    }
    if (record->desired_ns) {
      if (actual_ns >= record->desired_ns) {
        desired_to_actual_ns_.Add(actual_ns - record->desired_ns);
      } else {
        ++counters_.actual_early;
      }
    }
  }
}

void ZeroFGMainSurfaceEgress::PushCompletion(const Completion& completion) {
  bool overflow = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (completion_count_ >= completions_.size()) {
      overflow = true;
    } else {
      completions_[(completion_head_ + completion_count_) %
                   completions_.size()] = completion;
      ++completion_count_;
    }
  }
  if (overflow) {
    Fail(Failure::kCompletionOverflow, 0);
    return;
  }
  if (wake_owner_) {
    wake_owner_();
  }
}

void ZeroFGMainSurfaceEgress::Fail(Failure failure, int32_t vk_result) {
  Failure expected = Failure::kNone;
  if (failure_.compare_exchange_strong(expected, failure,
                                       std::memory_order_acq_rel)) {
    XELOGE("ZeroFGMainSurface egress_failure reason={} vk_result={}",
           FailureName(failure), vk_result);
  }
  if (wake_owner_) {
    wake_owner_();
  }
}

void ZeroFGMainSurfaceEgress::MarkSurfaceOutdated(int32_t vk_result) {
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++counters_.outdated;
  }
  if (!surface_outdated_.exchange(true, std::memory_order_acq_rel)) {
    XELOGW("ZeroFGMainSurface surface_outdated vk_result={}", vk_result);
  }
  if (wake_owner_) {
    wake_owner_();
  }
}

bool ZeroFGMainSurfaceEgress::StopRequested() {
  std::lock_guard<std::mutex> lock(mutex_);
  return stop_requested_;
}

bool ZeroFGMainSurfaceEgress::AnyCopyInFlight() const {
  for (uint32_t i = 0; i < slot_count_; ++i) {
    if (contexts_[i].in_flight) {
      return true;
    }
  }
  return false;
}

void ZeroFGMainSurfaceEgress::ApplyThreadPriority() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  // Same ladder as the presenter thread: FIFO acquire and present must not
  // lose a vertical blank to the Source's threads.
  static constexpr int kElevatedNiceLadder[] = {-20, -19, -16, -12,
                                                -8,  -4,  -2};
  int requested_nice = 0;
  bool set_ok = !elevated_priority_;
  if (elevated_priority_) {
    for (const int nice_value : kElevatedNiceLadder) {
      requested_nice = nice_value;
      if (setpriority(PRIO_PROCESS, gettid(), nice_value) == 0) {
        set_ok = true;
        break;
      }
    }
  }
  errno = 0;
  const int effective_nice = getpriority(PRIO_PROCESS, gettid());
  XELOGI(
      "ZeroFGMainSurface egress thread priority requested_nice={} "
      "effective_nice={} set_ok={}",
      requested_nice, effective_nice, set_ok);
#endif
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe
