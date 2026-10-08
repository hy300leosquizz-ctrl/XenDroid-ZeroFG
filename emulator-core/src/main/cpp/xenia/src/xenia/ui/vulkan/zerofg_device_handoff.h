// Host-only AHB/sync_fd bridge between Source A and presenter B.
#ifndef XENIA_UI_VULKAN_ZEROFG_DEVICE_HANDOFF_H_
#define XENIA_UI_VULKAN_ZEROFG_DEVICE_HANDOFF_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "xenia/ui/vulkan/zerofg_independent_presenter.h"

namespace xe::ui::vulkan {

class ZeroFGDeviceHandoff {
 public:
  static constexpr uint32_t kSlotCount = 6;
  ZeroFGDeviceHandoff(VulkanDevice* source, VulkanDevice* presenter);
  ~ZeroFGDeviceHandoff();
  ZeroFGDeviceHandoff(const ZeroFGDeviceHandoff&) = delete;
  ZeroFGDeviceHandoff& operator=(const ZeroFGDeviceHandoff&) = delete;

  // Startup only. No game publication is armed before PASS.
  bool SelfTest();
  void Arm();
  void Disarm();
  // Teardown only: synchronize with a marker already being recorded/exported.
  // This is a Source-marker guard, deliberately separate from the pool mutex.
  void DisarmAndQuiesceSource();
  // Called only after A refresher completion and B teardown drain/idle.
  void DestroyAfterIdle();

  // Source uses try_lock only. nullptr retains the original empty marker.
  const VkSubmitInfo* PrepareSource(uint32_t mailbox, VkImage image,
                                    VkExtent2D extent);
  void SourceSubmitted(VkResult result, uint64_t queue_ns, uint64_t submit_ns);
  bool Publish(
      uint32_t mailbox,
      const ZeroFGIndependentPresenter::IngressSourcePublication& properties,
      const ZeroFGIndependentPresenter* consumer);
  // B/presenter thread only. Emits a bounded summary when Acquire marked a
  // reporting window complete; never called from the Source path.
  void MaybeLog();
  // B only. Acquires the newest publication whose Source copy has completed.
  // Returns true with an empty output when nothing is published, and false
  // with output.external_pending while a newer copy is still in flight on A.
  bool Acquire(ZeroFGIndependentPresenter::IngressSourcePublication& output);
  bool failed() const { return failed_.load(std::memory_order_acquire); }

 private:
  struct Pool;
  struct Lease;
  bool CreatePool(VkExtent2D extent, uint32_t slot_count = kSlotCount);
  void DestroyPoolLocked();
  // test_pattern: a Source-device buffer copied instead of the guest output.
  bool RecordSource(uint32_t index, VkImage image, VkBuffer test_pattern);
  bool SelfTestExtent(VkExtent2D extent);
  bool Reclaim(uint32_t index);
  void FinalizePendingSourceLocked();
  void AbortLease(uint32_t index, uint64_t generation);
  void AddSourceBackpressureWait(uint64_t wait_ns);
  void AcquireSourcePoolLock(std::unique_lock<std::mutex>& lock);
  void AddSourceMutexBackpressureWait(uint64_t wait_ns);
  // Phase 0 sensor 5: duration of an acquisition proven contended by a failed
  // try_lock. C0 independently controls whether it feeds Source BP totals.
  void AddSourceMutexWait(uint64_t wait_ns);
  bool ReleaseLease(uint32_t index, uint64_t generation);
  void Fail(const char* operation, VkResult result);
  void Log();

  VulkanDevice* a_;
  VulkanDevice* b_;
  // Allocation/reconfiguration runs on B. Source NEVER blocks on this mutex.
  std::mutex mutex_;
  // Wakes a Source publish waiting for B to acquire the Published head.
  std::condition_variable publish_cv_;
  std::unique_ptr<Pool> pool_;
  std::atomic<uint64_t> desired_extent_{0};
  std::atomic<bool> armed_{false};
  std::atomic<bool> failed_{false};
  uint64_t next_generation_ = 1;
  uint32_t recording_ = UINT32_MAX;
  uint64_t recording_generation_ = 0;
  uint32_t publication_slot_ = UINT32_MAX;
  uint32_t source_mailbox_ = UINT32_MAX;
  VkSubmitInfo source_submit_ = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  uint64_t published_ = 0, acquired_ = 0, released_ = 0;
  uint64_t skip_no_slot_ = 0, high_water_ = 0;
  std::atomic<uint64_t> skip_busy_{0};
  // Source waits imposed by lossless backpressure (slot + depth-1 Publish).
  // source_bp_frame_ns_ is Source-thread only, for the current publication.
  uint64_t source_bp_frame_ns_ = 0;
  std::atomic<uint64_t> source_bp_wait_ns_{0};
  std::atomic<uint64_t> source_bp_waits_{0};
  // Phase 0 sensor 5: attempts and true contention are separate; contention
  // time also feeds the BP/P totals (C0).
  std::atomic<uint64_t> source_bp_mutex_wait_ns_{0};
  std::atomic<uint64_t> source_bp_mutex_waits_{0};
  std::atomic<uint64_t> source_pool_mutex_acquire_attempts_{0};
  std::atomic<uint64_t> source_pool_mutex_contended_total_{0};
  std::atomic<uint64_t> source_pool_mutex_attributed_wait_ns_{0};
  std::atomic<uint64_t> source_bp_slot_wait_ns_{0};
  std::atomic<uint64_t> source_bp_depth_wait_ns_{0};
  uint64_t export_failures_ = 0, import_failures_ = 0;
  uint64_t release_failures_ = 0, aba_failures_ = 0;
  uint64_t reuse_before_release_ = 0, replaced_unaccepted_ = 0;
  uint64_t acquire_deferred_ = 0;
  struct TimingWindow {
    std::array<uint64_t, 64> samples = {};
    uint32_t count = 0;
    uint32_t next = 0;
    void Add(uint64_t value) {
      samples[next] = value;
      next = (next + 1) % uint32_t(samples.size());
      count = std::min<uint32_t>(count + 1, uint32_t(samples.size()));
    }
    uint64_t Quantile(uint32_t percentile) const;
    uint64_t Maximum() const;
  };
  TimingWindow source_prepare_ns_, source_queue_ns_, source_submit_ns_,
      source_export_ns_, source_total_ns_;
  std::atomic<uint64_t> source_telemetry_sample_drop_{0};
  uint64_t import_ns_ = 0, release_ns_ = 0;
  struct PendingSourceFinalize {
    uint32_t slot = UINT32_MAX;
    uint64_t generation = 0;
    uint32_t mailbox = UINT32_MAX;
    bool submitted = true;
    VkResult submit_result = VK_SUCCESS;
    VkResult export_result = VK_SUCCESS;
    uint64_t queue_ns = 0;
    uint64_t submit_ns = 0;
    uint64_t export_ns = 0;
    uint64_t total_ns = 0;
  };
  std::atomic<bool> pending_finalize_{false};
  PendingSourceFinalize pending_source_finalize_;
  std::atomic<bool> log_pending_{false};
  std::mutex source_marker_mutex_;
  std::unique_lock<std::mutex> source_record_lock_;
  uint64_t source_prepare_begin_ns_ = 0;
};

}  // namespace xe::ui::vulkan
#endif
