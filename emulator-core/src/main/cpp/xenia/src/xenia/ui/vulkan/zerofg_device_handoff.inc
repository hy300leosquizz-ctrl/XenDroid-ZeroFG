// Included by vulkan_presenter.cc inside xe::ui::vulkan. No core/WSI ownership.
namespace {
uint64_t HandoffNowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
class HandoffValidationWatch {
 public:
  explicit HandoffValidationWatch(const VulkanInstance* instance)
      : instance_(instance) {
    if (!instance->extensions().ext_EXT_debug_utils) return;
    const auto get = instance->functions().vkGetInstanceProcAddr;
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        get(instance->instance(), "vkCreateDebugUtilsMessengerEXT"));
    destroy_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        get(instance->instance(), "vkDestroyDebugUtilsMessengerEXT"));
    if (!create || !destroy_) return;
    VkDebugUtilsMessengerCreateInfoEXT info = {
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    info.pUserData = &errors_;
    info.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT,
                              VkDebugUtilsMessageTypeFlagsEXT,
                              const VkDebugUtilsMessengerCallbackDataEXT*,
                              void* data) -> VkBool32 {
      static_cast<std::atomic<uint32_t>*>(data)->fetch_add(
          1, std::memory_order_relaxed);
      return VK_FALSE;
    };
    create(instance->instance(), &info, nullptr, &messenger_);
  }
  ~HandoffValidationWatch() {
    if (messenger_) destroy_(instance_->instance(), messenger_, nullptr);
  }
  bool available() const { return messenger_ != VK_NULL_HANDLE; }
  uint32_t errors() const { return errors_.load(std::memory_order_relaxed); }

 private:
  const VulkanInstance* instance_;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  PFN_vkDestroyDebugUtilsMessengerEXT destroy_ = nullptr;
  std::atomic<uint32_t> errors_{0};
};

void HandoffClose(int& fd) {
  if (fd >= 0) close(fd);
  fd = -1;
}
// Only sync_file polling, timeout zero. Never queries a pending Vulkan point.
// 1=ready, 0=pending, -1=failed observation. An error is never readiness.
int HandoffReady(int fd) {
  if (fd < 0) return 1;
  pollfd p = {fd, POLLIN, 0};
  const int result = poll(&p, 1, 0);
  if (result < 0 && errno == EINTR) return 0;
  if (result < 0 || (p.revents & (POLLERR | POLLNVAL | POLLHUP))) return -1;
  return result == 1 && (p.revents & POLLIN) ? 1 : 0;
}
void HandoffBarrier(VulkanDevice* d, VkCommandBuffer cmd, VkImage image,
                    VkImageLayout old_layout, VkImageLayout new_layout,
                    VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access,
                    uint32_t src_family = VK_QUEUE_FAMILY_IGNORED,
                    uint32_t dst_family = VK_QUEUE_FAMILY_IGNORED) {
  VkImageMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  barrier.srcStageMask = src_stage;
  barrier.srcAccessMask = src_access;
  barrier.dstStageMask = dst_stage;
  barrier.dstAccessMask = dst_access;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = src_family;
  barrier.dstQueueFamilyIndex = dst_family;
  barrier.image = image;
  barrier.subresourceRange = util::InitializeSubresourceRange();
  VkDependencyInfo dependency = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dependency.imageMemoryBarrierCount = 1;
  dependency.pImageMemoryBarriers = &barrier;
  d->vkCmdPipelineBarrier2()(cmd, &dependency);
}
bool HandoffSemaphore(VulkanDevice* d, bool exportable,
                      VkSemaphore& semaphore) {
  VkExportSemaphoreCreateInfo external = {
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
  external.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
  VkSemaphoreCreateInfo info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  info.pNext = exportable ? &external : nullptr;
  return d->functions().vkCreateSemaphore(d->device(), &info, nullptr,
                                          &semaphore) == VK_SUCCESS;
}
VkResult HandoffExport(VulkanDevice* d, VkSemaphore semaphore, int& fd) {
  VkSemaphoreGetFdInfoKHR info = {VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  info.semaphore = semaphore;
  info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
  return d->vkGetSemaphoreFdKHR()(d->device(), &info, &fd);
}
VkResult HandoffImport(VulkanDevice* d, VkSemaphore semaphore, int& fd) {
  VkImportSemaphoreFdInfoKHR info = {
      VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
  info.semaphore = semaphore;
  info.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
  info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
  info.fd = fd;  // -1 is an already-signaled sync_fd payload.
  VkResult result = d->vkImportSemaphoreFdKHR()(d->device(), &info);
  if (result == VK_SUCCESS) fd = -1;  // Vulkan owns successful imports.
  return result;
}
// Self-test content: a different value in every texel (an odd multiplier is a
// bijection modulo 2^32), so any tiling, swizzle, compression or row-pitch
// disagreement between the devices changes the readback. A uniform color
// reads back the same under every such disagreement.
uint32_t HandoffPattern(uint32_t index) {
  return index * 2654435761u + 0x9E3779B9u;
}
// The blit path converts through the transfer format, so it may differ by one
// unit per 10-bit channel. A layout disagreement moves whole texels instead.
bool HandoffTexelClose(uint32_t a, uint32_t b) {
  for (uint32_t shift = 0; shift < 30; shift += 10) {
    const int32_t ca = int32_t((a >> shift) & 0x3FFu);
    const int32_t cb = int32_t((b >> shift) & 0x3FFu);
    if (ca - cb > 1 || cb - ca > 1) return false;
  }
  return (a >> 30) == (b >> 30);
}
#endif
}  // namespace

uint64_t ZeroFGDeviceHandoff::TimingWindow::Quantile(
    uint32_t percentile) const {
  if (!count) return 0;
  std::array<uint64_t, 64> ordered = samples;
  std::sort(ordered.begin(), ordered.begin() + count);
  const uint32_t rank = std::min<uint32_t>(
      count - 1, (count - 1) * std::min<uint32_t>(percentile, 100) / 100);
  return ordered[rank];
}

uint64_t ZeroFGDeviceHandoff::TimingWindow::Maximum() const {
  if (!count) return 0;
  return *std::max_element(samples.begin(), samples.begin() + count);
}

struct ZeroFGDeviceHandoff::Pool {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  enum class State {
    kFree,
    kWriting,
    kPublished,
    kAcquired,
    kReleased,
    kPoisoned
  };
  struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
  };
  struct Slot {
    State state = State::kFree;
    uint64_t generation = 0;
    AHardwareBuffer* ahb = nullptr;
    Image a, b;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkSemaphore a_signal = VK_NULL_HANDLE;
    VkSemaphore b_wait = VK_NULL_HANDLE;
    VkSemaphore b_release = VK_NULL_HANDLE;
    int import_fd = -1, source_done_fd = -1, release_fd = -1;
    bool initialized = false;
    bool recreate_wait = false;
    ZeroFGIndependentPresenter::IngressSourcePublication metadata;
  };
  std::array<Slot, kSlotCount> slots;
#endif
  VkExtent2D extent = {};
};

struct ZeroFGDeviceHandoff::Lease {
  ZeroFGDeviceHandoff* owner;
  uint32_t index;
  uint64_t generation;
  bool submitted = false;
  ~Lease() {
    if (!submitted) owner->AbortLease(index, generation);
  }
  bool Submit() {
    submitted = true;
    return owner->ReleaseLease(index, generation);
  }
};

ZeroFGDeviceHandoff::ZeroFGDeviceHandoff(VulkanDevice* a, VulkanDevice* b)
    : a_(a), b_(b) {}
ZeroFGDeviceHandoff::~ZeroFGDeviceHandoff() { DestroyAfterIdle(); }
void ZeroFGDeviceHandoff::Arm() {
  if (!failed()) armed_.store(true, std::memory_order_release);
}
void ZeroFGDeviceHandoff::Disarm() {
  armed_.store(false, std::memory_order_release);
}
void ZeroFGDeviceHandoff::DisarmAndQuiesceSource() {
  Disarm();
  // PrepareSource holds this Source-only guard until SourceSubmitted finished
  // exporting. The hot Source path never waits on the presenter; teardown may
  // wait for this host-side critical section, including an already-started
  // driver submit/export duration. It never waits for GPU completion and does
  // not take the B pool mutex.
  std::lock_guard<std::mutex> lock(source_marker_mutex_);
}
void ZeroFGDeviceHandoff::Fail(const char* op, VkResult result) {
  failed_.store(true, std::memory_order_release);
  Disarm();
  XELOGE(
      "ZeroFGDeviceSplit failure operation={} result={} B_lost={} A_lost={} "
      "split_armed=false shared_device_fallback=false",
      op, int32_t(result), b_->IsLost(), a_->IsLost());
}
void ZeroFGDeviceHandoff::FinalizePendingSourceLocked() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (!pending_finalize_.load(std::memory_order_acquire)) return;
  const PendingSourceFinalize pending = pending_source_finalize_;
  if (!pool_ || pending.slot >= kSlotCount ||
      pool_->slots[pending.slot].generation != pending.generation ||
      pool_->slots[pending.slot].state != Pool::State::kWriting) {
    ++aba_failures_;
    Fail("source_submission_identity", VK_ERROR_UNKNOWN);
  } else {
    auto& slot = pool_->slots[pending.slot];
    source_queue_ns_.Add(pending.queue_ns);
    source_submit_ns_.Add(pending.submit_ns);
    source_export_ns_.Add(pending.export_ns);
    source_total_ns_.Add(pending.total_ns);
    source_mailbox_ = pending.mailbox;
    if (!pending.submitted || pending.submit_result != VK_SUCCESS) {
      // A recording/submit failure means no accepted A work owns the slot.
      // Release the reservation so a later Source frame can try again.
      slot.state = Pool::State::kFree;
      if (pending.submitted) {
        Fail("A_submit", pending.submit_result);
      }
    } else if (pending.export_result == VK_SUCCESS) {
      publication_slot_ = pending.slot;
    } else {
      ++export_failures_;
      slot.state = Pool::State::kPoisoned;
      Fail("A_export", pending.export_result);
    }
  }
  pending_finalize_.store(false, std::memory_order_release);
#endif
}
void ZeroFGDeviceHandoff::Log() {
  TimingWindow source_prepare;
  TimingWindow source_queue;
  TimingWindow source_submit;
  TimingWindow source_export;
  TimingWindow source_total;
  uint64_t high_water = 0, published = 0, acquired = 0, released = 0;
  uint64_t skip_no_slot = 0, export_failures = 0, import_failures = 0;
  uint64_t release_failures = 0, aba_failures = 0;
  uint64_t reuse_before_release = 0, replaced_unaccepted = 0;
  uint64_t acquire_deferred = 0, import_ns = 0, release_ns = 0;
  const uint64_t source_mutex_attempts =
      source_pool_mutex_acquire_attempts_.load(std::memory_order_relaxed);
  const uint64_t source_mutex_contended =
      source_pool_mutex_contended_total_.load(std::memory_order_relaxed);
  const uint64_t source_mutex_attributed_ns =
      source_pool_mutex_attributed_wait_ns_.load(std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    FinalizePendingSourceLocked();
    high_water = high_water_;
    published = published_;
    acquired = acquired_;
    released = released_;
    skip_no_slot = skip_no_slot_;
    export_failures = export_failures_;
    import_failures = import_failures_;
    release_failures = release_failures_;
    aba_failures = aba_failures_;
    reuse_before_release = reuse_before_release_;
    replaced_unaccepted = replaced_unaccepted_;
    acquire_deferred = acquire_deferred_;
    source_prepare = source_prepare_ns_;
    source_queue = source_queue_ns_;
    source_submit = source_submit_ns_;
    source_export = source_export_ns_;
    source_total = source_total_ns_;
    import_ns = import_ns_;
    release_ns = release_ns_;
  }
  XELOGI(
      "ZeroFGHandoff domain=A_to_B ownership=EXTERNAL pool=6 high_water={} "
      "published/acquired/released={}/{}/{} skip_no_slot={} skip_busy={} "
      "source_telemetry_sample_drop={} "
      "export/import/release_failures={}/{}/{} aba_failure={} "
      "reuse_before_release={} replaced_unaccepted={} acquire_deferred={} "
      "source_bp_waits/ms={}/{} "
      "source_bp_ms slot/depth/mutex={}/{}/{} mutex_waits={} "
      "source_pool_mutex_attempts/contended/attributed_ms={}/{}/{} "
      "A_prepare_us_p50/p90/max={}/{}/{} "
      "A_queue_us_p50/p90/max={}/{}/{} "
      "A_submit_us_p50/p90/max={}/{}/{} "
      "A_export_us_p50/p90/max={}/{}/{} "
      "A_total_us_p50/p90/max={}/{}/{} B_import/release_host_ns={}/{}",
      high_water, published, acquired, released, skip_no_slot,
      skip_busy_.load(std::memory_order_relaxed),
      source_telemetry_sample_drop_.load(std::memory_order_relaxed),
      export_failures,
      import_failures, release_failures, aba_failures, reuse_before_release,
      replaced_unaccepted, acquire_deferred,
      source_bp_waits_.load(std::memory_order_relaxed),
      source_bp_wait_ns_.load(std::memory_order_relaxed) / 1000000,
      source_bp_slot_wait_ns_.load(std::memory_order_relaxed) / 1000000,
      source_bp_depth_wait_ns_.load(std::memory_order_relaxed) / 1000000,
      source_bp_mutex_wait_ns_.load(std::memory_order_relaxed) / 1000000,
      source_bp_mutex_waits_.load(std::memory_order_relaxed),
      source_mutex_attempts, source_mutex_contended,
      source_mutex_attributed_ns / 1000000,
      source_prepare.Quantile(50) / 1000,
      source_prepare.Quantile(90) / 1000,
      source_prepare.Maximum() / 1000,
      source_queue.Quantile(50) / 1000,
      source_queue.Quantile(90) / 1000,
      source_queue.Maximum() / 1000,
      source_submit.Quantile(50) / 1000,
      source_submit.Quantile(90) / 1000,
      source_submit.Maximum() / 1000,
      source_export.Quantile(50) / 1000,
      source_export.Quantile(90) / 1000,
      source_export.Maximum() / 1000,
      source_total.Quantile(50) / 1000,
      source_total.Quantile(90) / 1000,
      source_total.Maximum() / 1000, import_ns, release_ns);
}

void ZeroFGDeviceHandoff::MaybeLog() {
  if (log_pending_.exchange(false, std::memory_order_acq_rel)) {
    Log();
  }
}

void ZeroFGDeviceHandoff::DestroyAfterIdle() {
  Disarm();
  DisarmAndQuiesceSource();
  bool had_pool = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    FinalizePendingSourceLocked();
    had_pool = pool_ != nullptr;
    DestroyPoolLocked();
  }
  if (had_pool) Log();
}
void ZeroFGDeviceHandoff::DestroyPoolLocked() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (pool_) {
    for (auto& s : pool_->slots) {
      HandoffClose(s.import_fd);
      HandoffClose(s.source_done_fd);
      HandoffClose(s.release_fd);
      // B imports first; A resources and allocation last.
      for (int side = 1; side >= 0; --side) {
        auto* d = side ? b_ : a_;
        auto& image = side ? s.b : s.a;
        const auto& f = d->functions();
        if (image.view) f.vkDestroyImageView(d->device(), image.view, nullptr);
        if (image.image) f.vkDestroyImage(d->device(), image.image, nullptr);
        if (image.memory) f.vkFreeMemory(d->device(), image.memory, nullptr);
      }
      const auto& af = a_->functions();
      const auto& bf = b_->functions();
      if (s.b_wait) bf.vkDestroySemaphore(b_->device(), s.b_wait, nullptr);
      if (s.b_release)
        bf.vkDestroySemaphore(b_->device(), s.b_release, nullptr);
      if (s.a_signal) af.vkDestroySemaphore(a_->device(), s.a_signal, nullptr);
      if (s.commands)
        af.vkDestroyCommandPool(a_->device(), s.commands, nullptr);
      if (s.ahb) AHardwareBuffer_release(s.ahb);
    }
    pool_.reset();
  }
#endif
  publication_slot_ = recording_ = UINT32_MAX;
}

bool ZeroFGDeviceHandoff::CreatePool(VkExtent2D extent, uint32_t slot_count) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (pool_) return false;
  pool_ = std::make_unique<Pool>();
  pool_->extent = extent;
  const VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT;
  uint64_t ahb_usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
                       AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER;
  for (auto* d : {a_, b_}) {
    VkPhysicalDeviceExternalImageFormatInfo external = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    external.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    VkPhysicalDeviceImageFormatInfo2 info = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    info.pNext = &external;
    info.format = VulkanPresenter::kGuestOutputFormat;
    info.type = VK_IMAGE_TYPE_2D;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    VkAndroidHardwareBufferUsageANDROID android_usage = {
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID};
    VkExternalImageFormatProperties ext_properties = {
        VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    ext_properties.pNext = &android_usage;
    VkImageFormatProperties2 properties = {
        VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    properties.pNext = &ext_properties;
    if (d->vkGetPhysicalDeviceImageFormatProperties2()(
            d->physical_device(), &info, &properties) != VK_SUCCESS ||
        !(ext_properties.externalMemoryProperties.externalMemoryFeatures &
          VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
      return false;
    ahb_usage |= android_usage.androidHardwareBufferUsage;
  }
  for (uint32_t slot = 0; slot < slot_count && slot < kSlotCount; ++slot) {
    auto& s = pool_->slots[slot];
    AHardwareBuffer_Desc desc = {};
    desc.width = extent.width;
    desc.height = extent.height;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM;
    desc.usage = ahb_usage;
    if (AHardwareBuffer_allocate(&desc, &s.ahb) || !s.ahb) return false;
    // Exactly the same create info on A and B; no mutable/storage flags.
    for (int side = 0; side < 2; ++side) {
      auto* d = side ? b_ : a_;
      auto& target = side ? s.b : s.a;
      const auto& f = d->functions();
      VkAndroidHardwareBufferFormatPropertiesANDROID format = {
          VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
      VkAndroidHardwareBufferPropertiesANDROID properties = {
          VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
      properties.pNext = &format;
      if (d->vkGetAndroidHardwareBufferPropertiesANDROID()(
              d->device(), s.ahb, &properties) != VK_SUCCESS ||
          format.format != VulkanPresenter::kGuestOutputFormat)
        return false;
      VkExternalMemoryImageCreateInfo external = {
          VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
      external.handleTypes =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
      VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      info.pNext = &external;
      info.imageType = VK_IMAGE_TYPE_2D;
      info.format = VulkanPresenter::kGuestOutputFormat;
      info.extent = {extent.width, extent.height, 1};
      info.mipLevels = info.arrayLayers = 1;
      info.samples = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = usage;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      if (f.vkCreateImage(d->device(), &info, nullptr, &target.image) !=
          VK_SUCCESS)
        return false;
      // AHB dedicated imports use the AHB's allocation size/type bits. Do not
      // query image memory requirements on an unbound external AHB image.
      uint32_t memory_type = 0;
      while (memory_type < 32 &&
             !(properties.memoryTypeBits & (1u << memory_type)))
        ++memory_type;
      if (memory_type == 32) return false;
      VkMemoryDedicatedAllocateInfo dedicated = {
          VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
      dedicated.image = target.image;
      VkImportAndroidHardwareBufferInfoANDROID import = {
          VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
      import.pNext = &dedicated;
      import.buffer = s.ahb;
      VkMemoryAllocateInfo allocation = {
          VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.pNext = &import;
      allocation.allocationSize = properties.allocationSize;
      allocation.memoryTypeIndex = memory_type;
      if (f.vkAllocateMemory(d->device(), &allocation, nullptr,
                             &target.memory) != VK_SUCCESS ||
          f.vkBindImageMemory(d->device(), target.image, target.memory, 0) !=
              VK_SUCCESS)
        return false;
      VkImageViewCreateInfo view = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = target.image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = info.format;
      view.subresourceRange = util::InitializeSubresourceRange();
      if (f.vkCreateImageView(d->device(), &view, nullptr, &target.view) !=
          VK_SUCCESS)
        return false;
    }
    VkCommandPoolCreateInfo commands = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    commands.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commands.queueFamilyIndex = a_->queue_family_graphics_compute();
    if (a_->functions().vkCreateCommandPool(a_->device(), &commands, nullptr,
                                            &s.commands) != VK_SUCCESS)
      return false;
    VkCommandBufferAllocateInfo allocate = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = s.commands;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (a_->functions().vkAllocateCommandBuffers(a_->device(), &allocate,
                                                 &s.command) != VK_SUCCESS ||
        !HandoffSemaphore(a_, true, s.a_signal) ||
        !HandoffSemaphore(b_, false, s.b_wait) ||
        !HandoffSemaphore(b_, true, s.b_release))
      return false;
  }
  return true;
#else
  return false;
#endif
}

bool ZeroFGDeviceHandoff::SelfTest() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  HandoffValidationWatch validation(a_->vulkan_instance());
  if (!a_->properties().synchronization2 || !a_->vkCmdPipelineBarrier2() ||
      !a_->vkDeviceWaitIdle() || !a_->vkGetSemaphoreFdKHR() ||
      !a_->vkGetAndroidHardwareBufferPropertiesANDROID() ||
      !a_->vkGetPhysicalDeviceImageFormatProperties2() ||
      !b_->vkImportSemaphoreFdKHR() || !b_->vkGetSemaphoreFdKHR() ||
      !b_->vkCmdPipelineBarrier2() || !b_->vkDeviceWaitIdle() ||
      !b_->vkGetAndroidHardwareBufferPropertiesANDROID() ||
      !b_->vkGetPhysicalDeviceImageFormatProperties2()) {
    Fail("self_test_capabilities", VK_ERROR_FEATURE_NOT_PRESENT);
    return false;
  }
  auto get_ids = [](VulkanDevice* d, VkPhysicalDeviceIDProperties& ids) {
    const auto* instance = d->vulkan_instance();
    auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        instance->functions().vkGetInstanceProcAddr(
            instance->instance(), "vkGetPhysicalDeviceProperties2"));
    if (!query) return false;
    VkPhysicalDeviceProperties2 properties = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &ids;
    query(d->physical_device(), &properties);
    auto external_query =
        reinterpret_cast<PFN_vkGetPhysicalDeviceExternalSemaphoreProperties>(
            instance->functions().vkGetInstanceProcAddr(
                instance->instance(),
                "vkGetPhysicalDeviceExternalSemaphoreProperties"));
    if (!external_query) return false;
    VkPhysicalDeviceExternalSemaphoreInfo external = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
    external.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    VkExternalSemaphoreProperties support = {
        VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    external_query(d->physical_device(), &external, &support);
    const auto required = VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT |
                          VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT;
    return (support.externalSemaphoreFeatures & required) == required;
  };
  VkPhysicalDeviceIDProperties ids_a = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceIDProperties ids_b = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  if (!get_ids(a_, ids_a) || !get_ids(b_, ids_b) ||
      std::memcmp(ids_a.deviceUUID, ids_b.deviceUUID, VK_UUID_SIZE) ||
      std::memcmp(ids_a.driverUUID, ids_b.driverUUID, VK_UUID_SIZE)) {
    Fail("self_test_UUID_or_external_semaphore",
         VK_ERROR_INVALID_EXTERNAL_HANDLE);
    return false;
  }
  auto uuid = [](const uint8_t* bytes) {
    std::string s;
    for (uint32_t i = 0; i < VK_UUID_SIZE; ++i)
      s += fmt::format("{:02x}", bytes[i]);
    return s;
  };
  XELOGI(
      "ZeroFGDeviceSplit A_deviceUUID={} A_driverUUID={} B_deviceUUID={} "
      "B_driverUUID={} ownership=EXTERNAL",
      uuid(ids_a.deviceUUID), uuid(ids_a.driverUUID), uuid(ids_b.deviceUUID),
      uuid(ids_b.driverUUID));

  // Startup qualification, never reached by a game publication. Two real
  // sizes: the common guest output, and an odd one for partial tiles and row
  // padding.
  static constexpr VkExtent2D kSelfTestExtents[] = {{1280, 720},
                                                    {1283, 723}};
  bool pass = true;
  for (const VkExtent2D& extent : kSelfTestExtents) {
    if (!SelfTestExtent(extent)) {
      pass = false;
      break;
    }
  }
  pass = pass && validation.errors() == 0;
  XELOGI(
      "ZeroFGDeviceSplit self_test_validation_observer={} errors={} "
      "layer_availability_is_separate=true",
      validation.available(), validation.errors());
  XELOGI(
      "ZeroFGDeviceSplit self_test={} pattern=unique_per_texel "
      "extents=1280x720,1283x723 reads=copy_exact,blit_within_1 "
      "GPU_wait=sync_fd ownership=EXTERNAL game_publications_armed=false",
      pass ? "PASS" : "FAIL");
  if (!pass)
    Fail("self_test_content_or_submit", VK_ERROR_INITIALIZATION_FAILED);
  return pass;
#else
  Fail("self_test_platform", VK_ERROR_EXTENSION_NOT_PRESENT);
  return false;
#endif
}

bool ZeroFGDeviceHandoff::SelfTestExtent(VkExtent2D extent) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  const auto& af = a_->functions();
  const auto& f = b_->functions();
  const uint32_t texel_count = extent.width * extent.height;
  const VkDeviceSize size = VkDeviceSize(texel_count) * sizeof(uint32_t);
  VkBuffer upload = VK_NULL_HANDLE, readback = VK_NULL_HANDLE;
  VkDeviceMemory upload_memory = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory = VK_NULL_HANDLE;
  uint32_t upload_type = 0, readback_type = 0;
  VkImage target = VK_NULL_HANDLE;
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkCommandPool commands = VK_NULL_HANDLE;
  VkCommandBuffer command = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  bool read = false;
  bool a_submitted = false, b_submitted = false;
  uint32_t copy_mismatches = 0, blit_mismatches = 0;
  uint32_t first_mismatch = UINT32_MAX;
  uint32_t first_expected = 0, first_actual = 0;
  do {
    if (!CreatePool(extent, 1)) break;
    auto& s = pool_->slots[0];
    // A writes the pattern into its handoff image from a Source-device buffer.
    void* mapping = nullptr;
    if (!util::CreateDedicatedAllocationBuffer(
            a_, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            util::MemoryPurpose::kUpload, upload, upload_memory,
            &upload_type) ||
        af.vkMapMemory(a_->device(), upload_memory, 0, VK_WHOLE_SIZE, 0,
                       &mapping) != VK_SUCCESS)
      break;
    auto* pattern = static_cast<uint32_t*>(mapping);
    for (uint32_t i = 0; i < texel_count; ++i) pattern[i] = HandoffPattern(i);
    util::FlushMappedMemoryRange(a_, upload_memory, upload_type);
    af.vkUnmapMemory(a_->device(), upload_memory);
    if (!RecordSource(0, VK_NULL_HANDLE, upload)) break;
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &s.command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &s.a_signal;
    {
      auto queue = a_->AcquireQueue(a_->queue_family_graphics_compute(), 0);
      VkResult result =
          a_->SubmitAndUpdateLost(queue.queue(), 1, &submit, VK_NULL_HANDLE);
      a_submitted = result == VK_SUCCESS;
      if (!a_submitted) break;
    }
    // The Source queue lock is released before the fd export.
    if (HandoffExport(a_, s.a_signal, s.import_fd) != VK_SUCCESS ||
        HandoffImport(b_, s.b_wait, s.import_fd) != VK_SUCCESS)
      break;
    // B reads twice: an exact copy, and the blit the game ingress uses into a
    // B-local image, which is then copied out as well.
    VkImageCreateInfo target_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    target_info.imageType = VK_IMAGE_TYPE_2D;
    target_info.format = VulkanPresenter::kGuestOutputFormat;
    target_info.extent = {extent.width, extent.height, 1};
    target_info.mipLevels = target_info.arrayLayers = 1;
    target_info.samples = VK_SAMPLE_COUNT_1_BIT;
    target_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    target_info.usage =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    target_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    target_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!util::CreateDedicatedAllocationBuffer(
            b_, size * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            util::MemoryPurpose::kReadback, readback, readback_memory,
            &readback_type) ||
        !util::CreateDedicatedAllocationImage(
            b_, target_info, util::MemoryPurpose::kDeviceLocal, target,
            target_memory))
      break;
    VkCommandPoolCreateInfo pool = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.queueFamilyIndex = b_->queue_family_graphics_compute();
    if (f.vkCreateCommandPool(b_->device(), &pool, nullptr, &commands) !=
        VK_SUCCESS)
      break;
    VkCommandBufferAllocateInfo command_allocate = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_allocate.commandPool = commands;
    command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocate.commandBufferCount = 1;
    if (f.vkAllocateCommandBuffers(b_->device(), &command_allocate, &command) !=
        VK_SUCCESS)
      break;
    VkCommandBufferBeginInfo begin = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (f.vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) break;
    HandoffBarrier(
        b_, command, s.b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_QUEUE_FAMILY_EXTERNAL, b_->queue_family_graphics_compute());
    HandoffBarrier(b_, command, target, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_PIPELINE_STAGE_2_NONE, 0,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy = {};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {extent.width, extent.height, 1};
    f.vkCmdCopyImageToBuffer(command, s.b.image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                             &copy);
    VkImageBlit blit = {};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[1] = {int32_t(extent.width), int32_t(extent.height), 1};
    blit.dstOffsets[1] = blit.srcOffsets[1];
    f.vkCmdBlitImage(command, s.b.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_NEAREST);
    HandoffBarrier(b_, command, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT);
    copy.bufferOffset = size;
    f.vkCmdCopyImageToBuffer(command, target,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                             &copy);
    VkMemoryBarrier2 visibility = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    visibility.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    visibility.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    visibility.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    visibility.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo dependency = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &visibility;
    b_->vkCmdPipelineBarrier2()(command, &dependency);
    if (f.vkEndCommandBuffer(command) != VK_SUCCESS) break;
    VkFenceCreateInfo fence_info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (f.vkCreateFence(b_->device(), &fence_info, nullptr, &fence) !=
        VK_SUCCESS)
      break;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &s.b_wait;
    submit.pWaitDstStageMask = &wait_stage;
    {
      auto queue = b_->AcquireQueue(b_->queue_family_graphics_compute(), 0);
      VkResult result =
          b_->SubmitAndUpdateLost(queue.queue(), 1, &submit, fence);
      b_submitted = result == VK_SUCCESS;
      if (!b_submitted) break;
    }
    // Explicit startup readback, not a production wait or observation poll.
    if (f.vkWaitForFences(b_->device(), 1, &fence, VK_TRUE, 5000000000ull) !=
        VK_SUCCESS)
      break;
    void* readback_mapping = nullptr;
    if (f.vkMapMemory(b_->device(), readback_memory, 0, VK_WHOLE_SIZE, 0,
                      &readback_mapping) != VK_SUCCESS)
      break;
    util::InvalidateMappedMemoryRange(b_, readback_memory, readback_type);
    const auto* texels = static_cast<const uint32_t*>(readback_mapping);
    for (uint32_t i = 0; i < texel_count; ++i) {
      const uint32_t expected = HandoffPattern(i);
      const bool copy_ok = texels[i] == expected;
      const bool blit_ok = HandoffTexelClose(texels[texel_count + i], expected);
      copy_mismatches += !copy_ok;
      blit_mismatches += !blit_ok;
      if ((!copy_ok || !blit_ok) && first_mismatch == UINT32_MAX) {
        first_mismatch = i;
        first_expected = expected;
        first_actual = copy_ok ? texels[texel_count + i] : texels[i];
      }
    }
    f.vkUnmapMemory(b_->device(), readback_memory);
    read = true;
  } while (false);
  // Even failed imports/submits must not destroy A's pending write/release.
  // These waits exist only at startup; all gameplay paths are non-waiting.
  if (b_submitted) b_->vkDeviceWaitIdle()(b_->device());
  if (a_submitted) {
    auto queue = a_->AcquireQueue(a_->queue_family_graphics_compute(), 0);
    af.vkQueueWaitIdle(queue.queue());
  }
  if (fence) f.vkDestroyFence(b_->device(), fence, nullptr);
  if (commands) f.vkDestroyCommandPool(b_->device(), commands, nullptr);
  if (target) f.vkDestroyImage(b_->device(), target, nullptr);
  if (target_memory) f.vkFreeMemory(b_->device(), target_memory, nullptr);
  if (readback) f.vkDestroyBuffer(b_->device(), readback, nullptr);
  if (readback_memory) f.vkFreeMemory(b_->device(), readback_memory, nullptr);
  if (upload) af.vkDestroyBuffer(a_->device(), upload, nullptr);
  if (upload_memory) af.vkFreeMemory(a_->device(), upload_memory, nullptr);
  DestroyAfterIdle();
  const bool pass = read && !copy_mismatches && !blit_mismatches;
  XELOGI(
      "ZeroFGDeviceSplit self_test_extent={}x{} texels={} read={} "
      "copy_mismatches={} blit_mismatches={} first_mismatch_texel={} "
      "expected/actual={:08X}/{:08X} result={}",
      extent.width, extent.height, texel_count, read, copy_mismatches,
      blit_mismatches,
      first_mismatch == UINT32_MAX ? int64_t(-1) : int64_t(first_mismatch),
      first_expected, first_actual, pass ? "PASS" : "FAIL");
  return pass;
#else
  return false;
#endif
}

bool ZeroFGDeviceHandoff::RecordSource(uint32_t index, VkImage source,
                                       VkBuffer test_pattern) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  auto& s = pool_->slots[index];
  const auto& f = a_->functions();
  if (f.vkResetCommandPool(a_->device(), s.commands, 0) != VK_SUCCESS)
    return false;
  VkCommandBufferBeginInfo begin = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (f.vkBeginCommandBuffer(s.command, &begin) != VK_SUCCESS) return false;
  if (s.initialized) {
    HandoffBarrier(
        a_, s.command, s.a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_QUEUE_FAMILY_EXTERNAL, a_->queue_family_graphics_compute());
  }
  HandoffBarrier(
      a_, s.command, s.a.image,
      s.initialized ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                    : VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  if (test_pattern != VK_NULL_HANDLE) {
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {pool_->extent.width, pool_->extent.height, 1};
    f.vkCmdCopyBufferToImage(s.command, test_pattern, s.a.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  } else {
    HandoffBarrier(
        a_, s.command, source, VulkanPresenter::kGuestOutputInternalLayout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VulkanPresenter::kGuestOutputInternalStageMask,
        VulkanPresenter::kGuestOutputInternalAccessMask,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkImageCopy copy = {};
    copy.srcSubresource.aspectMask = copy.dstSubresource.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;
    copy.srcSubresource.layerCount = copy.dstSubresource.layerCount = 1;
    copy.extent = {pool_->extent.width, pool_->extent.height, 1};
    f.vkCmdCopyImage(s.command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     s.a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    HandoffBarrier(a_, s.command, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VulkanPresenter::kGuestOutputInternalLayout,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT,
                   VulkanPresenter::kGuestOutputInternalStageMask,
                   VulkanPresenter::kGuestOutputInternalAccessMask);
  }
  HandoffBarrier(
      a_, s.command, s.a.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_ACCESS_2_TRANSFER_READ_BIT);
  HandoffBarrier(a_, s.command, s.a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_NONE, 0,
                 a_->queue_family_graphics_compute(), VK_QUEUE_FAMILY_EXTERNAL);
  return f.vkEndCommandBuffer(s.command) == VK_SUCCESS;
#else
  return false;
#endif
}

bool ZeroFGDeviceHandoff::Reclaim(uint32_t index) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  auto& s = pool_->slots[index];
  if (s.state == Pool::State::kFree) return true;
  if ((s.state != Pool::State::kPublished && s.state != Pool::State::kReleased))
    return false;
  const int source_ready = HandoffReady(s.source_done_fd);
  const int release_ready = HandoffReady(s.release_fd);
  if (source_ready < 0 || release_ready < 0) {
    ++release_failures_;
    s.state = Pool::State::kPoisoned;
    Fail("reclaim_fd_observation", VK_ERROR_UNKNOWN);
    return false;
  }
  if (!source_ready || !release_ready) return false;
  HandoffClose(s.source_done_fd);
  HandoffClose(s.import_fd);
  HandoffClose(s.release_fd);
  if (s.state == Pool::State::kPublished) ++replaced_unaccepted_;
  s.metadata = {};
  s.state = Pool::State::kFree;
  return true;
#else
  return false;
#endif
}

const VkSubmitInfo* ZeroFGDeviceHandoff::PrepareSource(uint32_t mailbox,
                                                       VkImage image,
                                                       VkExtent2D extent) {
  source_bp_frame_ns_ = 0;  // A new Source publication begins.
  if (!armed_.load(std::memory_order_acquire) || failed()) return nullptr;
  desired_extent_.store((uint64_t(extent.width) << 32) | extent.height,
                        std::memory_order_release);
  // Keep only the Source marker identity across A recording/submit/export.
  // The pool mutex is acquired in short scopes below and is never held while
  // the driver records or submits the Source command buffer.
  if (source_record_lock_.owns_lock()) {
    ++skip_busy_;
    return nullptr;
  }
  source_record_lock_ =
      std::unique_lock<std::mutex>(source_marker_mutex_, std::try_to_lock);
  if (!source_record_lock_.owns_lock()) {
    ++skip_busy_;
    return nullptr;
  }
  // Close the race with teardown after the optimistic check above.
  if (!armed_.load(std::memory_order_acquire) || failed()) {
    source_record_lock_.unlock();
    return nullptr;
  }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  const uint64_t prepare_begin_ns = HandoffNowNs();
  uint32_t selected = UINT32_MAX;
  uint64_t selected_generation = 0;
  // Lossless backpressure: with a matching pool the Source waits for the pool
  // mutex and for a reclaimable slot instead of skipping or replacing.
  for (bool slot_wait = true; slot_wait;) {
    slot_wait = false;
    // Phase 0 sensor 5: the blocking acquisition itself is backpressure and
    // was never measured as any (#5). Attribution only; the totals the
    // presenter reads are unchanged.
    std::unique_lock<std::mutex> pool_lock;
    AcquireSourcePoolLock(pool_lock);
    FinalizePendingSourceLocked();
    // A previous copy may have missed publication (busy/disarmed mailbox).
    // It was never acquired by B. Retain A's completion fd before reclaiming.
    if (pool_ && publication_slot_ != UINT32_MAX) {
      pool_->slots[publication_slot_].state = Pool::State::kReleased;
    }
    publication_slot_ = UINT32_MAX;
    if (pool_ && pool_->extent.width == extent.width &&
        pool_->extent.height == extent.height) {
      // Prefer genuinely free/released slots before replacing an unacquired
      // publication. Never take an acquired slot, even if its A copy is done.
      for (uint32_t i = 0; i < kSlotCount; ++i) {
        if (pool_->slots[i].state != Pool::State::kPublished && Reclaim(i)) {
          selected = i;
          break;
        }
      }
      if (selected == UINT32_MAX && armed_.load(std::memory_order_acquire) &&
          !failed()) {
        // Never replace an unaccepted publication: retry once a slot frees.
        slot_wait = true;
        pool_lock.unlock();
        const uint64_t wait_begin_ns = HandoffNowNs();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        source_bp_slot_wait_ns_.fetch_add(HandoffNowNs() - wait_begin_ns,
                                          std::memory_order_relaxed);
        AddSourceBackpressureWait(HandoffNowNs() - wait_begin_ns);
        continue;
      }
      if (failed()) {
        source_record_lock_.unlock();
        return nullptr;
      }
      if (selected != UINT32_MAX) {
        const uint32_t i = selected;
        auto& s = pool_->slots[i];
        if (s.state != Pool::State::kFree) {
          ++reuse_before_release_;
          Fail("reuse_before_release", VK_ERROR_UNKNOWN);
          source_record_lock_.unlock();
          return nullptr;
        }
        s.generation = next_generation_++;
        s.state = Pool::State::kWriting;
        selected_generation = s.generation;
        recording_ = i;
        recording_generation_ = s.generation;
        source_mailbox_ = mailbox;
        source_submit_ = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        source_submit_.commandBufferCount = 1;
        source_submit_.pCommandBuffers = &s.command;
        source_submit_.signalSemaphoreCount = 1;
        source_submit_.pSignalSemaphores = &s.a_signal;
        uint64_t occupied = 0;
        for (const auto& slot : pool_->slots)
          occupied += slot.state != Pool::State::kFree;
        high_water_ = std::max(high_water_, occupied);
      }
    }
  }
  if (selected == UINT32_MAX) {
    // Only pool creation/resize (structural) or disarm/failure reach here.
    source_record_lock_.unlock();
    return nullptr;
  }
  // The Writing state prevents B from reclaiming this slot while Source is
  // recording/submitting it, even though the pool mutex is not held here.
  if (!RecordSource(selected, image, VK_NULL_HANDLE)) {
    {
      std::unique_lock<std::mutex> pool_lock(mutex_, std::try_to_lock);
      PendingSourceFinalize pending;
      pending.slot = selected;
      pending.generation = selected_generation;
      pending.submitted = false;
      pending.submit_result = VK_ERROR_INITIALIZATION_FAILED;
      pending.total_ns = HandoffNowNs() - prepare_begin_ns;
      pending_source_finalize_ = pending;
      pending_finalize_.store(true, std::memory_order_release);
      if (pool_lock.owns_lock()) {
        FinalizePendingSourceLocked();
      }
    }
    recording_ = UINT32_MAX;
    recording_generation_ = 0;
    Fail("A_record", VK_ERROR_INITIALIZATION_FAILED);
    source_record_lock_.unlock();
    return nullptr;
  }
  source_prepare_begin_ns_ = prepare_begin_ns;
  {
    std::unique_lock<std::mutex> pool_lock(mutex_, std::try_to_lock);
    if (pool_lock.owns_lock()) {
      source_prepare_ns_.Add(HandoffNowNs() - prepare_begin_ns);
    } else {
      ++source_telemetry_sample_drop_;
    }
  }
  return &source_submit_;
#endif
  ++skip_no_slot_;
  source_record_lock_.unlock();
  return nullptr;
}

void ZeroFGDeviceHandoff::SourceSubmitted(VkResult result, uint64_t queue_ns,
                                          uint64_t submit_ns) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (recording_ == UINT32_MAX || !source_record_lock_.owns_lock()) return;
  const uint32_t recording = recording_;
  const uint64_t generation = recording_generation_;
  const uint32_t mailbox = source_mailbox_;
  const VkResult submit_result = result;
  uint64_t export_ns = 0;
  if (submit_result == VK_SUCCESS) {
    if (!pool_ || recording >= kSlotCount) {
      ++aba_failures_;
      recording_ = UINT32_MAX;
      recording_generation_ = 0;
      source_record_lock_.unlock();
      Fail("source_submission_identity", VK_ERROR_UNKNOWN);
      return;
    }
    auto& s = pool_->slots[recording];
    s.initialized = true;
    uint64_t begin = HandoffNowNs();
    result = HandoffExport(a_, s.a_signal, s.import_fd);
    export_ns = HandoffNowNs() - begin;
    if (result == VK_SUCCESS) {
      // Retain our own fd; the exported original is consumed by B's import.
      s.source_done_fd = s.import_fd >= 0 ? dup(s.import_fd) : -1;
      if (s.import_fd >= 0 && s.source_done_fd < 0)
        result = VK_ERROR_TOO_MANY_OBJECTS;
    }
  }
  {
    // Lossless backpressure: the Source waits for the pool mutex.
    std::unique_lock<std::mutex> pool_lock;
    AcquireSourcePoolLock(pool_lock);
    if (pool_lock.owns_lock()) {
      FinalizePendingSourceLocked();
      PendingSourceFinalize pending;
      pending.slot = recording;
      pending.generation = generation;
      pending.mailbox = mailbox;
      pending.submitted = true;
      pending.submit_result = submit_result;
      pending.export_result = result;
      pending.queue_ns = queue_ns;
      pending.submit_ns = submit_ns;
      pending.export_ns = export_ns;
      pending.total_ns = source_prepare_begin_ns_
                             ? HandoffNowNs() - source_prepare_begin_ns_
                             : 0;
      pending_source_finalize_ = pending;
      pending_finalize_.store(true, std::memory_order_release);
      FinalizePendingSourceLocked();
    } else {
      pending_source_finalize_.slot = recording;
      pending_source_finalize_.generation = generation;
      pending_source_finalize_.mailbox = mailbox;
      pending_source_finalize_.submitted = true;
      pending_source_finalize_.submit_result = submit_result;
      pending_source_finalize_.export_result = result;
      pending_source_finalize_.queue_ns = queue_ns;
      pending_source_finalize_.submit_ns = submit_ns;
      pending_source_finalize_.export_ns = export_ns;
      pending_source_finalize_.total_ns =
          source_prepare_begin_ns_ ? HandoffNowNs() - source_prepare_begin_ns_
                                   : 0;
      pending_finalize_.store(true, std::memory_order_release);
    }
  }
  recording_ = UINT32_MAX;
  recording_generation_ = 0;
  source_prepare_begin_ns_ = 0;
  source_record_lock_.unlock();
#endif
}

bool ZeroFGDeviceHandoff::Publish(
    uint32_t mailbox,
    const ZeroFGIndependentPresenter::IngressSourcePublication& metadata,
    const ZeroFGIndependentPresenter* consumer) {
  // Lossless backpressure: the Source waits for the pool mutex.
  std::unique_lock<std::mutex> lock;
  AcquireSourcePoolLock(lock);
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  FinalizePendingSourceLocked();
  // Bounded depth 1: publish only once B has acquired the Published head.
  // Acquire wakes this; the 1 ms recheck only observes lifecycle exits.
  while (armed_.load(std::memory_order_acquire) && !failed() && pool_ &&
         consumer && consumer->accepting()) {
    bool head_pending = false;
    for (const auto& slot : pool_->slots) {
      head_pending |= slot.state == Pool::State::kPublished;
    }
    if (!head_pending) break;
    const uint64_t wait_begin_ns = HandoffNowNs();
    publish_cv_.wait_for(lock, std::chrono::milliseconds(1));
    source_bp_depth_wait_ns_.fetch_add(HandoffNowNs() - wait_begin_ns,
                                       std::memory_order_relaxed);
    AddSourceBackpressureWait(HandoffNowNs() - wait_begin_ns);
  }
  if (publication_slot_ == UINT32_MAX || mailbox != source_mailbox_ || !pool_)
    return false;
  auto& s = pool_->slots[publication_slot_];
  publication_slot_ = UINT32_MAX;
  if (s.state != Pool::State::kWriting) {
    ++aba_failures_;
    Fail("publication_state", VK_ERROR_UNKNOWN);
    return false;
  }
  if (!armed_.load(std::memory_order_acquire) || failed()) {
    s.state = Pool::State::kReleased;
    return false;
  }
  s.metadata = metadata;
  // B subtracts these known ZeroFG waits from its pacing measurements.
  s.metadata.source_bp_wait_ns = source_bp_frame_ns_;
  s.metadata.handoff_published_ns = HandoffNowNs();
  source_bp_frame_ns_ = 0;
  s.state = Pool::State::kPublished;
  ++published_;
  return true;
#else
  return false;
#endif
}

void ZeroFGDeviceHandoff::AcquireSourcePoolLock(
    std::unique_lock<std::mutex>& lock) {
  source_pool_mutex_acquire_attempts_.fetch_add(1, std::memory_order_relaxed);
  lock = std::unique_lock<std::mutex>(mutex_, std::defer_lock);
  if (lock.try_lock()) {
    return;
  }
  source_pool_mutex_contended_total_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t blocked_begin_ns = HandoffNowNs();
  lock.lock();
  const uint64_t now_ns = HandoffNowNs();
  if (now_ns <= blocked_begin_ns) {
    return;
  }
  const uint64_t blocked_ns = now_ns - blocked_begin_ns;
  AddSourceMutexWait(blocked_ns);
  AddSourceMutexBackpressureWait(blocked_ns);
}

void ZeroFGDeviceHandoff::AddSourceMutexWait(uint64_t wait_ns) {
  // Phase 0 sensor 5: only a failed try_lock is a contention episode. The
  // blocked duration also counts as Source backpressure and P censoring (C0).
  if (!wait_ns) {
    return;
  }
  source_bp_mutex_wait_ns_.fetch_add(wait_ns,
                                     std::memory_order_relaxed);
  source_bp_mutex_waits_.fetch_add(1, std::memory_order_relaxed);
}

void ZeroFGDeviceHandoff::AddSourceMutexBackpressureWait(uint64_t wait_ns) {
  if (!wait_ns) {
    return;
  }
  // Deliberately do not increment source_bp_waits_: that counter continues
  // to represent the existing slot/depth wait episodes. The duration is
  // nevertheless carried with this publication and attributed to frame stats.
  source_bp_frame_ns_ += wait_ns;
  source_bp_wait_ns_.fetch_add(wait_ns, std::memory_order_relaxed);
  source_pool_mutex_attributed_wait_ns_.fetch_add(
      wait_ns, std::memory_order_relaxed);
  xe::AddSourceBackpressureWaitNs(wait_ns);
}

void ZeroFGDeviceHandoff::AddSourceBackpressureWait(uint64_t wait_ns) {
  // Source thread. Also attributed to the Source's issue interval in
  // progress (frame_stats), which the presenter subtracts for P_intrinsic.
  source_bp_frame_ns_ += wait_ns;
  source_bp_wait_ns_.fetch_add(wait_ns, std::memory_order_relaxed);
  source_bp_waits_.fetch_add(1, std::memory_order_relaxed);
  xe::AddSourceBackpressureWaitNs(wait_ns);
}

void ZeroFGDeviceHandoff::AbortLease(uint32_t index, uint64_t generation) {
  std::lock_guard<std::mutex> lock(mutex_);  // B only; Source always try-locks.
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (!pool_) return;  // The pool, and this slot's state, are already gone.
  auto& s = pool_->slots[index];
  if (s.generation != generation || s.state != Pool::State::kAcquired) {
    ++aba_failures_;
    return;
  }
  s.recreate_wait = true;  // imported payload was never submitted to a wait.
  s.state = Pool::State::kReleased;
#endif
}

bool ZeroFGDeviceHandoff::ReleaseLease(uint32_t index, uint64_t generation) {
  std::lock_guard<std::mutex> lock(mutex_);
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (!pool_) {
    ++aba_failures_;
    return false;
  }
  auto& s = pool_->slots[index];
  if (s.generation != generation || s.state != Pool::State::kAcquired) {
    ++aba_failures_;
    return false;
  }
  uint64_t begin = HandoffNowNs();
  VkResult result = HandoffExport(b_, s.b_release, s.release_fd);
  release_ns_ += HandoffNowNs() - begin;
  if (result != VK_SUCCESS) {
    ++release_failures_;
    s.state = Pool::State::kPoisoned;
    Fail("B_release_export", result);
    return false;
  }
  s.state = Pool::State::kReleased;
  ++released_;
  return true;
#else
  return false;
#endif
}

bool ZeroFGDeviceHandoff::Acquire(
    ZeroFGIndependentPresenter::IngressSourcePublication& output) {
  output = {};
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || failed() || !armed_.load(std::memory_order_acquire))
    return false;
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  FinalizePendingSourceLocked();
  uint64_t desired = desired_extent_.load(std::memory_order_acquire);
  if (!desired) return false;
  VkExtent2D extent = {uint32_t(desired >> 32), uint32_t(desired)};
  if (!pool_) {
    if (!CreatePool(extent))
      Fail("handoff_allocation", VK_ERROR_INITIALIZATION_FAILED);
    return false;
  }
  if (pool_->extent.width != extent.width ||
      pool_->extent.height != extent.height) {
    bool idle = true;
    for (uint32_t i = 0; i < kSlotCount; ++i) idle &= Reclaim(i);
    if (idle) {
      // Both devices finished these slots (sync_fd), no queue idle or Source
      // wait. Disarm before replacing; B alone performs driver allocation.
      DestroyPoolLocked();
      if (!CreatePool(extent))
        Fail("handoff_resize", VK_ERROR_INITIALIZATION_FAILED);
    }
    return false;
  }
  // Bounded FIFO: consume the oldest publication in order and leave the rest
  // Published. A copy still queued on A is deferred, never skipped: waiting on
  // it would put a pending cross-device dependency into B's only queue (KGSL
  // runs a context's queue in order).
  uint32_t selected = UINT32_MAX;
  for (uint32_t i = 0; i < kSlotCount; ++i) {
    const auto& s = pool_->slots[i];
    if (s.state != Pool::State::kPublished) continue;
    if (selected == UINT32_MAX ||
        s.metadata.publication.publication_id <
            pool_->slots[selected].metadata.publication.publication_id)
      selected = i;
  }
  // Nothing published: no retry, the next publication wakes the presenter.
  if (selected == UINT32_MAX) return true;
  const int head_ready = HandoffReady(pool_->slots[selected].source_done_fd);
  if (head_ready < 0) {
    ++release_failures_;
    Fail("acquire_fd_observation", VK_ERROR_UNKNOWN);
    return false;
  }
  if (!head_ready) {
    ++acquire_deferred_;
    output.external_pending = true;
    return false;
  }
  const uint64_t ready_seen_ns = HandoffNowNs();
  auto& s = pool_->slots[selected];
  if (s.recreate_wait) {
    b_->functions().vkDestroySemaphore(b_->device(), s.b_wait, nullptr);
    s.b_wait = VK_NULL_HANDLE;
    if (!HandoffSemaphore(b_, false, s.b_wait)) {
      Fail("import_semaphore_recreate", VK_ERROR_INITIALIZATION_FAILED);
      return false;
    }
    s.recreate_wait = false;
  }
  uint64_t begin = HandoffNowNs();
  VkResult result = HandoffImport(b_, s.b_wait, s.import_fd);
  import_ns_ += HandoffNowNs() - begin;
  if (result != VK_SUCCESS) {
    ++import_failures_;
    s.state = Pool::State::kPoisoned;
    Fail("B_import", result);
    return false;
  }
  // This acquisition's poll is the first sight of the completed Source copy:
  // a host upper bound on its completion, at most one retry interval late when
  // an earlier attempt saw it pending. No fd is kept for the observation.
  auto observation =
      std::make_shared<ZeroFGIndependentPresenter::HandoffReadyObservation>();
  observation->source_id = s.metadata.publication.source_id;
  observation->publish_ns = s.metadata.publication.publish_time_ns;
  observation->first_seen_ready_ns = ready_seen_ns;
  s.state = Pool::State::kAcquired;
  publish_cv_.notify_all();
  ++acquired_;
  auto lease = std::make_shared<Lease>();
  lease->owner = this;
  lease->index = selected;
  lease->generation = s.generation;
  output = s.metadata;
  output.image = s.b.image;
  output.view = s.b.view;
  output.extent = extent;
  output.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  output.image_lifetime = lease;
  output.mailbox_acquisition = lease;
  output.external_wait = s.b_wait;
  output.external_release = s.b_release;
  output.external_submitted = [lease]() { return lease->Submit(); };
  output.handoff_ready = std::move(observation);
  // Defer the formatting/logging until the B caller has left this pool lock.
  if (acquired_ % 120 == 0) log_pending_.store(true, std::memory_order_release);
  return true;
#else
  return false;
#endif
}
