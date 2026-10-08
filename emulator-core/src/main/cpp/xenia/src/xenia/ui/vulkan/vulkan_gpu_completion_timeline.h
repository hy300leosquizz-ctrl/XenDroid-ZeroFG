/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_VULKAN_GPU_COMPLETION_TIMELINE_H_
#define XENIA_UI_VULKAN_VULKAN_GPU_COMPLETION_TIMELINE_H_

#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "xenia/base/assert.h"
#include "xenia/ui/gpu_completion_timeline.h"
#include "xenia/ui/vulkan/vulkan_device.h"

// Host timing of completion waits, queue locks and submits: a development
// diagnostic, compiled out of the release.
namespace cvars {
inline constexpr bool vulkan_completion_wait_telemetry = false;
}  // namespace cvars

namespace xe {
namespace ui {
namespace vulkan {

class VulkanGPUCompletionTimeline : public GPUCompletionTimeline {
 public:
  explicit VulkanGPUCompletionTimeline(VulkanDevice* const vulkan_device,
                                       const char* const name = "")
      : vulkan_device_(vulkan_device), name_(name) {}

  const char* name() const { return name_; }
  size_t pending_submission_count() const {
    return pending_submission_fences_.size();
  }
  uint64_t front_pending_submission() const {
    return pending_submission_fences_.empty()
               ? 0
               : pending_submission_fences_.front().first;
  }

  VulkanGPUCompletionTimeline(const VulkanGPUCompletionTimeline&) = delete;
  VulkanGPUCompletionTimeline& operator=(const VulkanGPUCompletionTimeline&) =
      delete;
  VulkanGPUCompletionTimeline(VulkanGPUCompletionTimeline&&) = delete;
  VulkanGPUCompletionTimeline& operator=(VulkanGPUCompletionTimeline&&) =
      delete;

  ~VulkanGPUCompletionTimeline();

  class FenceAcquisition {
   public:
    explicit FenceAcquisition(
        VulkanGPUCompletionTimeline* const completion_timeline,
        const VkFence fence)
        : completion_timeline_(completion_timeline), fence_(fence) {
      assert_not_null(completion_timeline);
      assert_true(fence != VK_NULL_HANDLE);
    }

    FenceAcquisition(const FenceAcquisition&) = delete;
    FenceAcquisition& operator=(const FenceAcquisition&) = delete;

    FenceAcquisition(FenceAcquisition&& other)
        : completion_timeline_(other.completion_timeline_),
          fence_(other.fence_),
          submission_successful_(other.submission_successful_) {
      other.completion_timeline_ = nullptr;
      other.fence_ = VK_NULL_HANDLE;
      other.submission_successful_.reset();
    }

    ~FenceAcquisition() {
      if (completion_timeline_ && fence_) {
#ifndef NDEBUG
        assert_not_zero(completion_timeline_->fences_acquired_);
        --completion_timeline_->fences_acquired_;
#endif
        if (submission_successful_.value_or(false)) {
          assert_true(
              completion_timeline_->pending_submission_fences_.empty() ||
              completion_timeline_->pending_submission_fences_.front().first <
                  completion_timeline_->GetUpcomingSubmission());
          completion_timeline_->pending_submission_fences_.emplace_back(
              completion_timeline_->GetUpcomingSubmission(), fence_);
          completion_timeline_->IncrementUpcomingSubmission();
        } else {
          completion_timeline_->free_fences_.push_back(fence_);
        }
        completion_timeline_->RecordFenceCounts();
      }
    }

    VkFence GetFenceForSubmitting() {
      // Don't mark the fence as used in a submission if already tried to
      // submit, but failed.
      assert_true(!submission_successful_.has_value() ||
                  *submission_successful_);
      submission_successful_ = true;
      return fence_;
    }

    void SetSubmissionFailedOrAborted() { submission_successful_ = false; }

   private:
    VulkanGPUCompletionTimeline* completion_timeline_ = nullptr;

    VkFence fence_ = VK_NULL_HANDLE;

    std::optional<bool> submission_successful_;
  };

  // If the submission has succeeded (`GetFenceForSubmitting` was called, but
  // `SetSubmissionFailedOrAborted` was not), will advance to the next
  // submission once the acquisition is released.
  //
  // It's possible to acquire a fence not right before submitting, but also well
  // in advance, such as before recording the command buffer, for instance, to
  // skip recording it if fence acquisition has failed.
  //
  // Acquiring a fence also updates the completed submission in order to reuse
  // fences if this completion timeline is used without regular checks or waits
  // (for instance, if it's supplementary to another completion timeline, and
  // awaited only before destroying something).
  [[nodiscard]] std::optional<FenceAcquisition> AcquireFenceForSubmission(
      VkResult* result_out_opt = nullptr);

  VkResult AcquireFenceAndSubmit(uint32_t queue_family_index,
                                 uint32_t queue_index, uint32_t submit_count,
                                 const VkSubmitInfo* submits,
                                 uint64_t* queue_lock_wait_ns_out = nullptr,
                                 uint64_t* submit_host_ns_out = nullptr);

  // Some owners (currently the command processor) have an explicit
  // frame-in-flight wait that is the authority for fence reuse. They may opt
  // in to skipping opportunistic pending-fence status polls, using the
  // explicit oldest-fence wait only at the storage safety cap.
  void SetNoPendingReclaimPoll(bool enabled) {
    no_pending_reclaim_poll_ = enabled;
  }

  void UpdateCompletedSubmission() override;

 protected:
  void AwaitSubmissionImpl(uint64_t awaited_submission) override;
  bool CompletionPollMayBlock() const override;

 private:
  struct Telemetry {
    uint64_t interval_start_ns = 0;

    uint64_t update_calls = 0;
    uint64_t update_total_ns = 0;
    uint64_t update_max_ns = 0;
    size_t last_update_pending_before = 0;
    uint64_t status_calls = 0;
    uint64_t status_total_ns = 0;
    uint64_t status_max_ns = 0;
    uint64_t status_over_1ms = 0;
    uint64_t status_over_4ms = 0;
    uint64_t retired_fences = 0;

    uint64_t wait_calls = 0;
    uint64_t wait_total_ns = 0;
    uint64_t wait_max_ns = 0;
    uint64_t wait_over_1ms = 0;
    uint64_t wait_over_4ms = 0;
    uint64_t reclaim_wait_calls = 0;
    uint64_t last_awaited_submission = 0;
    uint64_t last_wait_front_submission = 0;
    uint64_t last_wait_tail_submission = 0;
    size_t last_wait_pending = 0;

    uint64_t acquire_calls = 0;
    uint64_t acquire_total_ns = 0;
    uint64_t acquire_max_ns = 0;
    uint64_t acquire_over_1ms = 0;
    uint64_t acquire_over_4ms = 0;
    size_t last_acquire_pending_before = 0;
    size_t last_acquire_free_before = 0;

    uint64_t queue_lock_calls = 0;
    uint64_t queue_lock_total_ns = 0;
    uint64_t queue_lock_max_ns = 0;
    uint64_t queue_lock_over_1ms = 0;
    uint64_t queue_lock_over_4ms = 0;
    uint64_t submit_calls = 0;
    uint64_t submit_total_ns = 0;
    uint64_t submit_max_ns = 0;
    uint64_t submit_over_1ms = 0;
    uint64_t submit_over_4ms = 0;

    size_t pending_high_water = 0;
    size_t free_high_water = 0;
  };

  static uint64_t TelemetryNowNs();
  static void AddTimedOperation(uint64_t duration_ns, uint64_t& calls,
                                uint64_t& total_ns, uint64_t& max_ns,
                                uint64_t& over_1ms, uint64_t& over_4ms);
  void RecordFenceCounts();
  void MaybeLogTelemetry();
  bool WaitForOldestPendingFence();

  VulkanDevice* const vulkan_device_;
  const char* const name_;
  bool no_pending_reclaim_poll_ = false;

  std::vector<VkFence> free_fences_;

  // <Submission index, fence>, in submission index order.
  std::deque<std::pair<uint64_t, VkFence>> pending_submission_fences_;

  Telemetry telemetry_;

#ifndef NDEBUG
  size_t fences_acquired_ = 0;
#endif
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_VULKAN_GPU_COMPLETION_TIMELINE_H_
