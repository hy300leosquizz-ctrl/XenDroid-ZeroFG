/**
 ******************************************************************************
 * XenDroid ZeroFG V2 completion/ownership layer                              *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_ZEROFG_COMPLETION_OWNER_H_
#define XENIA_UI_VULKAN_ZEROFG_COMPLETION_OWNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "xenia/base/platform.h"
#include "xenia/ui/vulkan/vulkan_device.h"

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
#include <errno.h>
#include <poll.h>
#include <unistd.h>
#endif

namespace xe {
namespace ui {
namespace vulkan {

// Completion/ownership layer v2.
//
// Every physical owner of ZeroFG GPU work owns its completion state: a
// FinalOutput slot (Post), a Generation context, a Residency slot (capture
// transfer) and an ingress context. The rules:
//
// 1. GPU->GPU dependencies use semaphores and timelines and never depend on
//    the CPU observing completion.
// 2. GPU->CPU observation uses an exportable binary semaphore signaled by the
//    existing submit, exported as a sync_fd and read with poll(fd, 0). A
//    timeline counter query on pending work is an exceptional fallback.
// 3. No shared stream creates a dependency between independent owners.
// 4. No submit exists only to observe completion.
// 5. An owner is signaled again only after its previous occupancy retired.
// 6. Export failure falls back safely and is counted.
// 7. Teardown may wait; the hot path may not.
//
// Why (Turnip/KGSL; measured in Builds B to D3, confirmed in the Mesa, Turnip
// and KGSL sources on 2026-09-14): Mesa emulates timelines on KGSL. Every
// signal allocates a time point, and the allocation first cleans up the
// timeline's pending points with a zero-timeout wait per point
// (vk_sync_timeline_gc_locked); a counter query cleans up the whole chain.
// For a point that holds a KGSL timestamp, Turnip passes the zero timeout to
// IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID as timeout 0, and the KGSL driver
// waits forever on 0, so the "poll" lasts until the point retires. It runs
// inside vkQueueSubmit, before Mesa's own queue lock, while XenDroid still
// holds the queue it acquired for the submit. Waiting on a point does no
// cleanup.
//
// A timeline is per owner when its previous point always belongs to the same
// owner's previous occupancy and that occupancy retired before the owner is
// signaled again. Its cleanup then only ever finds retired points and never
// waits.
//
// ZeroFGCompletionOwner holds the three parts of one owner's state. Each path
// uses the parts it needs:
// - the owner timeline (GPU->GPU), optional: without it the path signals its
//   shared timeline as before;
// - the occupancy, always tracked, whichever timeline was signaled;
// - the sync_fd observer (GPU->CPU), optional.
class ZeroFGCompletionOwner {
 public:
  // One owner per physical resource: its semaphores and its fd are never
  // shared, so the type can be neither copied nor moved.
  ZeroFGCompletionOwner() = default;
  ZeroFGCompletionOwner(const ZeroFGCompletionOwner&) = delete;
  ZeroFGCompletionOwner& operator=(const ZeroFGCompletionOwner&) = delete;
  ZeroFGCompletionOwner(ZeroFGCompletionOwner&&) = delete;
  ZeroFGCompletionOwner& operator=(ZeroFGCompletionOwner&&) = delete;

  // Owner timeline (GPU->GPU).

  VkResult CreateTimeline(const VulkanDevice* vulkan_device,
                          const std::string& name) {
    VkSemaphoreTypeCreateInfo timeline_type = {
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_type.initialValue = 0;
    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &timeline_type;
    const VkResult result = vulkan_device->functions().vkCreateSemaphore(
        vulkan_device->device(), &semaphore_info, nullptr, &timeline_);
    if (result != VK_SUCCESS) {
      timeline_ = VK_NULL_HANDLE;
      return result;
    }
    next_timeline_value_ = 1;
    vulkan_device->SetObjectName(VK_OBJECT_TYPE_SEMAPHORE, timeline_,
                                 name.c_str());
    return VK_SUCCESS;
  }

  // Teardown only: every submit that signaled the timeline has completed.
  void DestroyTimeline(const VulkanDevice* vulkan_device) {
    if (timeline_ != VK_NULL_HANDLE) {
      vulkan_device->functions().vkDestroySemaphore(vulkan_device->device(),
                                                    timeline_, nullptr);
      timeline_ = VK_NULL_HANDLE;
    }
    next_timeline_value_ = 1;
    signaled_timeline_ = VK_NULL_HANDLE;
    occupied_ = false;
  }

  bool owns_timeline() const { return timeline_ != VK_NULL_HANDLE; }
  VkSemaphore timeline() const { return timeline_; }

  // Values keep increasing for the semaphore's lifetime. A value claimed by a
  // submit that was then refused is skipped, which timelines allow.
  uint64_t ClaimTimelineValue() { return next_timeline_value_++; }

  // Occupancy.

  // Rule 5 check before a submit signals this owner again: true when the
  // previous occupancy has not been observed retired, so the cleanup of an
  // owner timeline could wait on its own previous point.
  bool occupied() const { return occupied_; }

  // After queue acceptance: the timeline (owner or shared) this occupancy
  // signaled. Its value stays with the path's own record (signal_value,
  // timeline_value, source_timeline_value), which GPU consumers and the
  // exceptional counter query pair with this timeline.
  void MarkSubmitted(VkSemaphore signaled_timeline) {
    signaled_timeline_ = signaled_timeline;
    occupied_ = true;
  }

  // The occupancy retired: observed through its sync_fd or a counter query,
  // or a GPU consumer that waited on it completed, or teardown waited on it.
  void MarkRetired() { occupied_ = false; }

  VkSemaphore signaled_timeline() const { return signaled_timeline_; }

  // sync_fd observer (GPU->CPU).

  VkResult CreateSyncFd(const VulkanDevice* vulkan_device,
                        const std::string& name) {
    VkExportSemaphoreCreateInfo export_info = {
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &export_info;
    const VkResult result = vulkan_device->functions().vkCreateSemaphore(
        vulkan_device->device(), &semaphore_info, nullptr, &sync_semaphore_);
    if (result != VK_SUCCESS) {
      sync_semaphore_ = VK_NULL_HANDLE;
      return result;
    }
    vulkan_device->SetObjectName(VK_OBJECT_TYPE_SEMAPHORE, sync_semaphore_,
                                 name.c_str());
    return VK_SUCCESS;
  }

  void DestroySyncFd(const VulkanDevice* vulkan_device) {
    CloseSyncFd();
    if (sync_semaphore_ != VK_NULL_HANDLE) {
      vulkan_device->functions().vkDestroySemaphore(vulkan_device->device(),
                                                    sync_semaphore_, nullptr);
      sync_semaphore_ = VK_NULL_HANDLE;
    }
    sync_fd_fallback_ = false;
    sync_fd_recreate_pending_ = false;
  }

  bool sync_fd_recreate_pending() const { return sync_fd_recreate_pending_; }

  // A semaphore whose export failed still holds its signal. Replace it before
  // it is signaled again; the owner retired, so that signal has completed.
  // False when the new semaphore could not be created: the owner then
  // observes through its timeline.
  bool RecreateSyncFd(const VulkanDevice* vulkan_device,
                      const std::string& name) {
    if (sync_semaphore_ != VK_NULL_HANDLE) {
      vulkan_device->functions().vkDestroySemaphore(vulkan_device->device(),
                                                    sync_semaphore_, nullptr);
      sync_semaphore_ = VK_NULL_HANDLE;
    }
    sync_fd_recreate_pending_ = false;
    sync_fd_fallback_ = CreateSyncFd(vulkan_device, name) != VK_SUCCESS;
    return !sync_fd_fallback_;
  }

  bool can_signal_sync_fd() const { return sync_semaphore_ != VK_NULL_HANDLE; }
  VkSemaphore sync_semaphore() const { return sync_semaphore_; }

  // After queue acceptance, outside the queue lock. The export has the side
  // effects of a wait: the binary semaphore is unsignaled afterwards and can
  // be signaled again. On failure the semaphore keeps its signal, so it is
  // recreated before its next use, and this occupancy falls back to the
  // timeline. An fd of -1 also takes the fallback, as C2 and D3 do.
  bool ExportSyncFd(const VulkanDevice* vulkan_device) {
    CloseSyncFd();
    VkSemaphoreGetFdInfoKHR fd_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    fd_info.semaphore = sync_semaphore_;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    int exported_fd = -1;
    const VkResult result = vulkan_device->vkGetSemaphoreFdKHR()(
        vulkan_device->device(), &fd_info, &exported_fd);
    if (result == VK_SUCCESS && exported_fd >= 0) {
      sync_fd_ = exported_fd;
      sync_fd_fallback_ = false;
      return true;
    }
    sync_fd_fallback_ = true;
    sync_fd_recreate_pending_ = true;
    return false;
  }

  // No sync_fd for this occupancy: it is observed through the timeline.
  void SetSyncFdFallback() { sync_fd_fallback_ = true; }
  bool sync_fd_fallback() const { return sync_fd_fallback_; }
  bool sync_fd_observable() const {
    return sync_fd_ >= 0 && !sync_fd_fallback_;
  }

  enum class SyncFdPoll : uint8_t {
    kPending,
    kSignaled,
    // The kernel reported the fd invalid or in error. It proves nothing any
    // more: the occupancy falls back to its timeline (AbandonSyncFd).
    kError,
  };

  // Non-waiting. Only meaningful when sync_fd_observable(). A sync_file
  // reports POLLIN once its fence signaled; POLLNVAL (a closed fd), POLLERR
  // and POLLHUP are errors, never readiness.
  SyncFdPoll PollSyncFd() const {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
    if (sync_fd_ < 0) {
      return SyncFdPoll::kError;
    }
    pollfd descriptor = {sync_fd_, POLLIN, 0};
    const int result = poll(&descriptor, 1, 0);
    if (result == 0) {
      return SyncFdPoll::kPending;
    }
    if (result < 0) {
      return errno == EINTR || errno == EAGAIN ? SyncFdPoll::kPending
                                               : SyncFdPoll::kError;
    }
    if (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      return SyncFdPoll::kError;
    }
    return (descriptor.revents & POLLIN) ? SyncFdPoll::kSignaled
                                         : SyncFdPoll::kPending;
#else
    return SyncFdPoll::kError;
#endif
  }

  // After kError: this occupancy is observed through its timeline from now
  // on. The fd is forgotten, not closed: if the kernel reported it invalid,
  // its number may already belong to another file.
  void AbandonSyncFd() {
    sync_fd_ = -1;
    sync_fd_fallback_ = true;
  }

  void CloseSyncFd() {
    if (sync_fd_ < 0) {
      return;
    }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
    close(sync_fd_);
#endif
    sync_fd_ = -1;
  }

  // The occupancy retired: its fd and fallback end with it. A pending
  // recreation is kept for the next submit.
  void RetireSyncFd() {
    CloseSyncFd();
    sync_fd_fallback_ = false;
  }

 private:
  VkSemaphore timeline_ = VK_NULL_HANDLE;
  uint64_t next_timeline_value_ = 1;
  VkSemaphore signaled_timeline_ = VK_NULL_HANDLE;
  bool occupied_ = false;
  VkSemaphore sync_semaphore_ = VK_NULL_HANDLE;
  int sync_fd_ = -1;
  bool sync_fd_fallback_ = false;
  bool sync_fd_recreate_pending_ = false;
};

// Falsifier counters of one path, cumulative since its owners were created.
struct ZeroFGCompletionOwnerStats {
  // Signals by timeline: [0] the path's shared timeline, [1] an owner
  // timeline.
  std::array<uint64_t, 2> signals = {};
  // Submits whose host time reached 1 ms and 4 ms, split the same way.
  // Hypothesis H1: the submit stalls come from the shared-timeline cleanup,
  // so owner submits do not reach 1 ms.
  std::array<uint64_t, 2> submit_over_1ms = {};
  std::array<uint64_t, 2> submit_over_4ms = {};
  // Rule 5, bookkeeping: an owner signaled again before its previous
  // occupancy was observed retired. 0 by construction.
  uint64_t reuse_before_retire = 0;
  // Rule 5, GPU truth: releases checked through the owner's own sync_fd, and
  // those that found the previous occupancy still pending (expected 0). Only
  // the Generation is released without observing its own completion, so it
  // is the only path that checks; elsewhere 0 checks means not measured.
  uint64_t release_checks = 0;
  uint64_t release_unretired = 0;
  // Rule 6: sync_fd polls the kernel answered with an error; the occupancy
  // fell back to its timeline.
  uint64_t sync_fd_errors = 0;
  // Rule 2: timeline counter queries, which can wait on pending work.
  uint64_t timeline_queries = 0;
  // Owner timelines that could not be created; those owners stay shared.
  uint64_t timeline_create_failures = 0;

  void CountSubmit(bool owner_timeline, uint64_t submit_host_ns) {
    const size_t index = owner_timeline ? 1 : 0;
    ++signals[index];
    submit_over_1ms[index] += submit_host_ns >= 1000000ull;
    submit_over_4ms[index] += submit_host_ns >= 4000000ull;
  }

  const char* mode() const {
    if (signals[0] && signals[1]) {
      return "mixed";
    }
    if (signals[1]) {
      return "owner";
    }
    return signals[0] ? "shared" : "idle";
  }
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_ZEROFG_COMPLETION_OWNER_H_
