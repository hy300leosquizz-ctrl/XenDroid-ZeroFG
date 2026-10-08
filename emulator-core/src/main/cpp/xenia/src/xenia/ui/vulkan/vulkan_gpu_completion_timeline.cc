/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/vulkan_gpu_completion_timeline.h"

#include <algorithm>
#include <chrono>
#include <iterator>

#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/frame_stats.h"
#include "xenia/base/logging.h"

namespace xe {
namespace ui {
namespace vulkan {

namespace {

constexpr uint64_t kTelemetryReportIntervalNs = 2500000000ull;
constexpr uint64_t kOneMillisecondNs = 1000000ull;
constexpr uint64_t kFourMillisecondsNs = 4000000ull;
constexpr size_t kNewestFencesExcludedFromBoundedPoll = 2;

}  // namespace

uint64_t VulkanGPUCompletionTimeline::TelemetryNowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

void VulkanGPUCompletionTimeline::AddTimedOperation(
    const uint64_t duration_ns, uint64_t& calls, uint64_t& total_ns,
    uint64_t& max_ns, uint64_t& over_1ms, uint64_t& over_4ms) {
  ++calls;
  total_ns += duration_ns;
  max_ns = std::max(max_ns, duration_ns);
  over_1ms += duration_ns >= kOneMillisecondNs;
  over_4ms += duration_ns >= kFourMillisecondsNs;
}

void VulkanGPUCompletionTimeline::RecordFenceCounts() {
  telemetry_.pending_high_water =
      std::max(telemetry_.pending_high_water, pending_submission_fences_.size());
  telemetry_.free_high_water =
      std::max(telemetry_.free_high_water, free_fences_.size());
}

void VulkanGPUCompletionTimeline::MaybeLogTelemetry() {
  if (!cvars::vulkan_completion_wait_telemetry) {
    return;
  }
  const uint64_t now_ns = TelemetryNowNs();
  if (!telemetry_.interval_start_ns) {
    telemetry_.interval_start_ns = now_ns;
    return;
  }
  if (now_ns - telemetry_.interval_start_ns < kTelemetryReportIntervalNs) {
    return;
  }

  const uint64_t front_submission =
      pending_submission_fences_.empty()
          ? 0
          : pending_submission_fences_.front().first;
  const uint64_t tail_submission =
      pending_submission_fences_.empty()
          ? 0
          : pending_submission_fences_.back().first;
  XELOGI(
      "VulkanCompletionTimeline name={} bounded_poll=true "
      "update_calls/status_calls/retired={}/{}/{} "
      "update_total/max_us={}/{} update_pending_before={} "
      "status_total/max_us={}/{} "
      "status_over_1ms/4ms={}/{} wait_calls={} wait_total/max_us={}/{} "
      "wait_over_1ms/4ms={}/{} reclaim_wait_calls={} "
      "last_wait requested/front/tail/pending="
      "{}/{}/{}/{} acquire_calls={} acquire_total/max_us={}/{} "
      "acquire_pending/free_before={}/{} "
      "acquire_over_1ms/4ms={}/{} queue_lock_calls={} "
      "queue_lock_total/max_us={}/{} queue_lock_over_1ms/4ms={}/{} "
      "submit_calls={} submit_total/max_us={}/{} submit_over_1ms/4ms={}/{} "
      "fences pending/current_high/free/current_high={}/{}/{}/{} "
      "pending_front/tail={}/{}",
      name_,
      telemetry_.update_calls,
      telemetry_.status_calls, telemetry_.retired_fences,
      telemetry_.update_total_ns / 1000, telemetry_.update_max_ns / 1000,
      telemetry_.last_update_pending_before,
      telemetry_.status_total_ns / 1000, telemetry_.status_max_ns / 1000,
      telemetry_.status_over_1ms, telemetry_.status_over_4ms,
      telemetry_.wait_calls, telemetry_.wait_total_ns / 1000,
      telemetry_.wait_max_ns / 1000, telemetry_.wait_over_1ms,
      telemetry_.wait_over_4ms, telemetry_.reclaim_wait_calls,
      telemetry_.last_awaited_submission,
      telemetry_.last_wait_front_submission,
      telemetry_.last_wait_tail_submission, telemetry_.last_wait_pending,
      telemetry_.acquire_calls, telemetry_.acquire_total_ns / 1000,
      telemetry_.acquire_max_ns / 1000,
      telemetry_.last_acquire_pending_before,
      telemetry_.last_acquire_free_before, telemetry_.acquire_over_1ms,
      telemetry_.acquire_over_4ms, telemetry_.queue_lock_calls,
      telemetry_.queue_lock_total_ns / 1000,
      telemetry_.queue_lock_max_ns / 1000,
      telemetry_.queue_lock_over_1ms, telemetry_.queue_lock_over_4ms,
      telemetry_.submit_calls, telemetry_.submit_total_ns / 1000,
      telemetry_.submit_max_ns / 1000, telemetry_.submit_over_1ms,
      telemetry_.submit_over_4ms, pending_submission_fences_.size(),
      telemetry_.pending_high_water, free_fences_.size(),
      telemetry_.free_high_water, front_submission, tail_submission);

  telemetry_ = Telemetry{};
  telemetry_.interval_start_ns = now_ns;
  RecordFenceCounts();
}

VulkanGPUCompletionTimeline::~VulkanGPUCompletionTimeline() {
#ifndef NDEBUG
  assert_zero(fences_acquired_);
#endif

  if (!pending_submission_fences_.empty()) {
    if (vulkan_device_->functions().vkWaitForFences(
            vulkan_device_->device(), 1,
            &pending_submission_fences_.back().second, VK_TRUE,
            UINT64_MAX) == VK_ERROR_DEVICE_LOST) {
      XELOGE(
          "VulkanGPUCompletionTimeline[{}]: DEVICE_LOST waiting on final "
          "submission {} during destruction ({} fences pending)",
          name_, pending_submission_fences_.back().first,
          pending_submission_fences_.size());
      if (vulkan_device_->SetLost()) {
        vulkan_device_->LogFaultInfo();
      }
    }

    while (!pending_submission_fences_.empty()) {
      vulkan_device_->functions().vkDestroyFence(
          vulkan_device_->device(), pending_submission_fences_.back().second,
          nullptr);
      pending_submission_fences_.pop_back();
    }
  }

  while (!free_fences_.empty()) {
    vulkan_device_->functions().vkDestroyFence(vulkan_device_->device(),
                                               free_fences_.back(), nullptr);
    free_fences_.pop_back();
  }
}

std::optional<VulkanGPUCompletionTimeline::FenceAcquisition>
VulkanGPUCompletionTimeline::AcquireFenceForSubmission(
    VkResult* const result_out_opt) {
  const bool telemetry_enabled = cvars::vulkan_completion_wait_telemetry;
  const uint64_t acquire_begin_ns =
      telemetry_enabled ? TelemetryNowNs() : 0;
  if (telemetry_enabled) {
    telemetry_.last_acquire_pending_before =
        pending_submission_fences_.size();
    telemetry_.last_acquire_free_before = free_fences_.size();
  }
  // Prefer reusing fences already reclaimed by awaits without polling the
  // driver: the status query of a still-pending fence blocks until it retires
  // on Turnip/kgsl (vkGetFenceStatus is not non-blocking there), which would
  // serialize the CPU with the GPU on every submit. Only poll when fences
  // would otherwise accumulate (users like the presenter's guest output
  // refresher never await in steady state) - by the time this many
  // submissions are pending, the oldest one has long retired, so the poll
  // reclaims it without a meaningful wait.
  constexpr size_t kMaxPendingBeforeReclaimPoll = 8;
  constexpr size_t kMaxPendingBeforeExplicitReclaimWait = 32;
  if (free_fences_.empty() &&
      pending_submission_fences_.size() >= kMaxPendingBeforeReclaimPoll) {
    if (no_pending_reclaim_poll_) {
      // Keep the bounded storage invariant without falling back to
      // vkGetFenceStatus on a pending CP fence. This is an explicit lifetime
      // wait at a generous safety cap, not an opportunistic status poll.
      if (pending_submission_fences_.size() >=
          kMaxPendingBeforeExplicitReclaimWait) {
        WaitForOldestPendingFence();
      }
    } else {
      UpdateCompletedSubmission();
    }
  }

  VkFence fence = VK_NULL_HANDLE;

  if (!free_fences_.empty()) {
    const VkResult fence_reset_result =
        vulkan_device_->functions().vkResetFences(vulkan_device_->device(), 1,
                                                  &free_fences_.back());
    if (fence_reset_result != VK_SUCCESS) {
      XELOGE("Failed to reset a Vulkan fence: {}",
             vk::to_string(vk::Result(fence_reset_result)));
    } else {
      fence = free_fences_.back();
      free_fences_.pop_back();
    }
  }

  if (fence == VK_NULL_HANDLE) {
    const VkFenceCreateInfo fence_create_info = {
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    const VkResult fence_create_result =
        vulkan_device_->functions().vkCreateFence(
            vulkan_device_->device(), &fence_create_info, nullptr, &fence);
    if (fence_create_result != VK_SUCCESS) {
      XELOGE("Failed to create a Vulkan fence: {}",
             vk::to_string(vk::Result(fence_create_result)));
      if (result_out_opt != nullptr) {
        *result_out_opt = fence_create_result;
      }
      if (telemetry_enabled) {
        const uint64_t acquire_ns = TelemetryNowNs() - acquire_begin_ns;
        AddTimedOperation(acquire_ns, telemetry_.acquire_calls,
                          telemetry_.acquire_total_ns,
                          telemetry_.acquire_max_ns,
                          telemetry_.acquire_over_1ms,
                          telemetry_.acquire_over_4ms);
        RecordFenceCounts();
        MaybeLogTelemetry();
      }
      return std::nullopt;
    }
  }

#ifndef NDEBUG
  ++fences_acquired_;
#endif

  if (result_out_opt != nullptr) {
    *result_out_opt = VK_SUCCESS;
  }
  if (telemetry_enabled) {
    const uint64_t acquire_ns = TelemetryNowNs() - acquire_begin_ns;
    AddTimedOperation(acquire_ns, telemetry_.acquire_calls,
                      telemetry_.acquire_total_ns, telemetry_.acquire_max_ns,
                      telemetry_.acquire_over_1ms,
                      telemetry_.acquire_over_4ms);
    RecordFenceCounts();
    MaybeLogTelemetry();
  }
  return FenceAcquisition(this, fence);
}

bool VulkanGPUCompletionTimeline::WaitForOldestPendingFence() {
  if (pending_submission_fences_.empty()) {
    return false;
  }
  const bool telemetry_enabled = cvars::vulkan_completion_wait_telemetry;
  const uint64_t wait_begin_ns = telemetry_enabled ? TelemetryNowNs() : 0;
  const VkResult wait_result = vulkan_device_->functions().vkWaitForFences(
      vulkan_device_->device(), 1, &pending_submission_fences_.front().second,
      VK_TRUE, UINT64_MAX);
  if (telemetry_enabled) {
    const uint64_t wait_ns = TelemetryNowNs() - wait_begin_ns;
    AddTimedOperation(wait_ns, telemetry_.wait_calls,
                      telemetry_.wait_total_ns, telemetry_.wait_max_ns,
                      telemetry_.wait_over_1ms, telemetry_.wait_over_4ms);
    ++telemetry_.reclaim_wait_calls;
  }
  if (wait_result != VK_SUCCESS) {
    XELOGE(
        "VulkanGPUCompletionTimeline[{}]: explicit reclaim wait -> {} "
        "(submission {}, {} fences pending)",
        name_, vk::to_string(vk::Result(wait_result)),
        pending_submission_fences_.front().first,
        pending_submission_fences_.size());
    if (wait_result == VK_ERROR_DEVICE_LOST) {
      if (vulkan_device_->SetLost()) {
        vulkan_device_->LogFaultInfo();
      }
    }
    return false;
  }
  SetCompletedSubmission(pending_submission_fences_.front().first);
  free_fences_.push_back(pending_submission_fences_.front().second);
  pending_submission_fences_.pop_front();
  if (telemetry_enabled) {
    ++telemetry_.retired_fences;
    RecordFenceCounts();
  }
  return true;
}

VkResult VulkanGPUCompletionTimeline::AcquireFenceAndSubmit(
    const uint32_t queue_family_index, const uint32_t queue_index,
    const uint32_t submit_count, const VkSubmitInfo* const submits,
    uint64_t* const queue_lock_wait_ns_out,
    uint64_t* const submit_host_ns_out) {
  VkResult fence_acquire_result;
  std::optional<FenceAcquisition> fence_acquisition =
      AcquireFenceForSubmission(&fence_acquire_result);
  if (!fence_acquisition.has_value()) {
    return fence_acquire_result;
  }

  VkResult submit_result;
  {
    const bool time_queue_lock =
        queue_lock_wait_ns_out || cvars::vulkan_completion_wait_telemetry;
    const uint64_t queue_lock_start_ns =
        time_queue_lock ? TelemetryNowNs() : 0;
    const VulkanDevice::Queue::Acquisition queue_acquisition =
        vulkan_device_->AcquireQueue(queue_family_index, queue_index);
    const uint64_t queue_lock_wait_ns =
        time_queue_lock ? TelemetryNowNs() - queue_lock_start_ns : 0;
    if (queue_lock_wait_ns_out) {
      *queue_lock_wait_ns_out = queue_lock_wait_ns;
    }
    const bool time_submit =
        submit_host_ns_out || cvars::vulkan_completion_wait_telemetry;
    const uint64_t submit_start_ns = time_submit ? TelemetryNowNs() : 0;
    submit_result = vulkan_device_->SubmitAndUpdateLost(
        queue_acquisition.queue(), submit_count, submits,
        fence_acquisition->GetFenceForSubmitting());
    const uint64_t submit_host_ns =
        time_submit ? TelemetryNowNs() - submit_start_ns : 0;
    if (submit_host_ns_out) {
      *submit_host_ns_out = submit_host_ns;
    }
    if (cvars::vulkan_completion_wait_telemetry) {
      AddTimedOperation(queue_lock_wait_ns, telemetry_.queue_lock_calls,
                        telemetry_.queue_lock_total_ns,
                        telemetry_.queue_lock_max_ns,
                        telemetry_.queue_lock_over_1ms,
                        telemetry_.queue_lock_over_4ms);
      AddTimedOperation(submit_host_ns, telemetry_.submit_calls,
                        telemetry_.submit_total_ns,
                        telemetry_.submit_max_ns,
                        telemetry_.submit_over_1ms,
                        telemetry_.submit_over_4ms);
    }
  }
  if (submit_result != VK_SUCCESS) {
    fence_acquisition->SetSubmissionFailedOrAborted();
  }
  MaybeLogTelemetry();
  return submit_result;
}

void VulkanGPUCompletionTimeline::UpdateCompletedSubmission() {
  const bool telemetry_enabled = cvars::vulkan_completion_wait_telemetry;
  const uint64_t update_begin_ns = telemetry_enabled ? TelemetryNowNs() : 0;
  if (telemetry_enabled) {
    telemetry_.last_update_pending_before =
        pending_submission_fences_.size();
  }
  // Bounded poll: the newest fences are still pending on the GPU, and a
  // status query of a pending fence blocks on Turnip/KGSL.
  size_t fences_allowed_to_poll = pending_submission_fences_.size();
  fences_allowed_to_poll =
      fences_allowed_to_poll > kNewestFencesExcludedFromBoundedPoll
          ? fences_allowed_to_poll - kNewestFencesExcludedFromBoundedPoll
          : 0;
  while (fences_allowed_to_poll && !pending_submission_fences_.empty()) {
    const uint64_t status_begin_ns = telemetry_enabled ? TelemetryNowNs() : 0;
    const VkResult fence_status = vulkan_device_->functions().vkGetFenceStatus(
        vulkan_device_->device(), pending_submission_fences_.front().second);
    if (telemetry_enabled) {
      const uint64_t status_ns = TelemetryNowNs() - status_begin_ns;
      AddTimedOperation(status_ns, telemetry_.status_calls,
                        telemetry_.status_total_ns,
                        telemetry_.status_max_ns,
                        telemetry_.status_over_1ms,
                        telemetry_.status_over_4ms);
    }
    if (fence_status != VK_SUCCESS) {
      // Not ready, or an error.
      if (fence_status == VK_ERROR_DEVICE_LOST) {
        XELOGE(
            "VulkanGPUCompletionTimeline[{}]: DEVICE_LOST polling fence for "
            "submission {} ({} fences pending)",
            name_, pending_submission_fences_.front().first,
            pending_submission_fences_.size());
        if (vulkan_device_->SetLost()) {
          vulkan_device_->LogFaultInfo();
        }
      }
      break;
    }
    SetCompletedSubmission(pending_submission_fences_.front().first);
    free_fences_.push_back(pending_submission_fences_.front().second);
    pending_submission_fences_.pop_front();
    --fences_allowed_to_poll;
    if (telemetry_enabled) {
      ++telemetry_.retired_fences;
    }
  }
  if (telemetry_enabled) {
    const uint64_t update_ns = TelemetryNowNs() - update_begin_ns;
    ++telemetry_.update_calls;
    telemetry_.update_total_ns += update_ns;
    telemetry_.update_max_ns = std::max(telemetry_.update_max_ns, update_ns);
    RecordFenceCounts();
    MaybeLogTelemetry();
  }
}

void VulkanGPUCompletionTimeline::AwaitSubmissionImpl(
    const uint64_t awaited_submission) {
  const bool telemetry_enabled = cvars::vulkan_completion_wait_telemetry;
  if (telemetry_enabled) {
    telemetry_.last_awaited_submission = awaited_submission;
    telemetry_.last_wait_pending = pending_submission_fences_.size();
    telemetry_.last_wait_front_submission =
        pending_submission_fences_.empty()
            ? 0
            : pending_submission_fences_.front().first;
    telemetry_.last_wait_tail_submission =
        pending_submission_fences_.empty()
            ? 0
            : pending_submission_fences_.back().first;
  }
  // According to the Vulkan 1.4.335 specification:
  // "The first synchronization scope includes every batch submitted in the same
  // queue submission command. Fence signal operations that are defined by
  // vkQueueSubmit or vkQueueSubmit2 additionally include in the first
  // synchronization scope all commands that occur earlier in submission order.
  // Fence signal operations that are defined by vkQueueSubmit or vkQueueSubmit2
  // or vkQueueBindSparse additionally include in the first synchronization
  // scope any semaphore and fence signal operations that occur earlier in
  // signal operation order."
  auto submission_end_iterator = pending_submission_fences_.cbegin();
  while (submission_end_iterator != pending_submission_fences_.cend() &&
         submission_end_iterator->first <= awaited_submission) {
    submission_end_iterator = std::next(submission_end_iterator);
  }
  if (submission_end_iterator != pending_submission_fences_.cbegin()) {
    const uint64_t wait_begin_ns = telemetry_enabled ? TelemetryNowNs() : 0;
    const VkResult fence_wait_result =
        vulkan_device_->functions().vkWaitForFences(
            vulkan_device_->device(), 1,
            &std::prev(submission_end_iterator)->second, VK_TRUE, UINT64_MAX);
    if (telemetry_enabled) {
      const uint64_t wait_ns = TelemetryNowNs() - wait_begin_ns;
      AddTimedOperation(wait_ns, telemetry_.wait_calls,
                        telemetry_.wait_total_ns, telemetry_.wait_max_ns,
                        telemetry_.wait_over_1ms,
                        telemetry_.wait_over_4ms);
    }
    if (fence_wait_result != VK_SUCCESS) {
      XELOGE(
          "VulkanGPUCompletionTimeline[{}]: vkWaitForFences -> {} (awaited "
          "submission {}, last-pending {}, {} fences pending)",
          name_, vk::to_string(vk::Result(fence_wait_result)),
          awaited_submission, std::prev(submission_end_iterator)->first,
          pending_submission_fences_.size());
      if (fence_wait_result == VK_ERROR_DEVICE_LOST) {
        if (vulkan_device_->SetLost()) {
          vulkan_device_->LogFaultInfo();
        }
      }
      return;
    }
    for (auto free_fence_iterator = pending_submission_fences_.cbegin();
         free_fence_iterator != submission_end_iterator;
         free_fence_iterator = std::next(free_fence_iterator)) {
      free_fences_.push_back(free_fence_iterator->second);
    }
    pending_submission_fences_.erase(pending_submission_fences_.cbegin(),
                                     submission_end_iterator);
  }
  if (GetCompletedSubmissionFromLastUpdate() < awaited_submission) {
    SetCompletedSubmission(awaited_submission);
  }
  if (telemetry_enabled) {
    RecordFenceCounts();
    MaybeLogTelemetry();
  }
}

bool VulkanGPUCompletionTimeline::CompletionPollMayBlock() const {
  return true;
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe
