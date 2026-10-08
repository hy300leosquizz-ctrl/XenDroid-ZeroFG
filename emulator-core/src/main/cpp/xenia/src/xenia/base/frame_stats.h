/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 */

#ifndef XENIA_BASE_FRAME_STATS_H_
#define XENIA_BASE_FRAME_STATS_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>

namespace xe {

// Guest-frame present timing for the debug overlay.
//
// RecordGuestPresent() is called exactly once per presented GUEST frame (from
// the GPU command processor's IssueSwap), so the reported FPS / frame time
// reflect the actual game frame rate -- NOT the host UI repaint cadence (which
// keeps running at panel refresh even when the guest stalls). The producer is
// single-threaded (the command-processor thread); the published values are read
// from the UI thread via GetFrameStats(). A torn read across the three values
// is harmless for a debug readout.
namespace internal {
inline std::atomic<float>& frame_instant_ms() {
  static std::atomic<float> v{0.0f};
  return v;
}
inline std::atomic<float>& frame_avg_ms() {
  static std::atomic<float> v{0.0f};
  return v;
}
inline std::atomic<float>& frame_fps() {
  static std::atomic<float> v{0.0f};
  return v;
}

// Source / output telemetry is separate from the raw guest-frame stats above.
// Resets may be requested from lifecycle threads. Source's non-atomic window is
// producer-owned; final Output uses the bounded atomic ring below because the
// active presentation authority may move between backend threads.
constexpr uint64_t kFrameTelemetryWindowNs = 1000000000ull;
constexpr size_t kFrameTelemetrySampleCapacity = 1024;

inline uint64_t FrameTelemetryNowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

inline std::atomic<uint64_t>& frame_telemetry_reset_generation() {
  // Zero is reserved for a thread that has not produced a Source event.
  static std::atomic<uint64_t> v{1};
  return v;
}
inline std::atomic<uint64_t>& source_published_generation() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_last_event_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<float>& source_fps() {
  static std::atomic<float> v{0.0f};
  return v;
}
// Passive producer / mailbox / WSI boundary telemetry. Totals are cumulative
// for the lifetime of the process (including while ZeroFG is disabled). The
// interval rings use atomic sequence publication so the presentation thread
// can take a bounded, nonblocking snapshot.
constexpr size_t kSourceBoundaryPeriodCapacity = 64;

struct SourceBoundaryPeriodSlot {
  std::atomic<uint64_t> sequence{0};
  std::atomic<uint64_t> interval_ns{0};
  // ZeroFG handoff backpressure the Source waited inside this interval.
  std::atomic<uint64_t> bp_wait_ns{0};
};

struct SourceBoundaryPeriodRing {
  std::atomic<uint64_t> sequence{0};
  std::array<SourceBoundaryPeriodSlot, kSourceBoundaryPeriodCapacity> slots;
};

inline SourceBoundaryPeriodRing& source_issue_period_ring() {
  static SourceBoundaryPeriodRing ring;
  return ring;
}
// ZeroFG handoff backpressure the Source waited since its last IssueSwap.
inline std::atomic<uint64_t>& source_issue_bp_wait_pending_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline SourceBoundaryPeriodRing& source_publish_period_ring() {
  static SourceBoundaryPeriodRing ring;
  return ring;
}
inline std::atomic<uint64_t>& source_issue_period_last_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publish_period_last_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_issue_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_published_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_sequence() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_order_regression_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_last_publication_id() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_last_published_source_id() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_last_publish_thread_tag() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint32_t>& source_issue_inflight() {
  static std::atomic<uint32_t> v{0};
  return v;
}
inline std::atomic<uint32_t>& source_issue_inflight_high_water() {
  static std::atomic<uint32_t> v{0};
  return v;
}
inline std::atomic<uint32_t>& guest_refresh_inflight() {
  static std::atomic<uint32_t> v{0};
  return v;
}
inline std::atomic<uint32_t>& guest_refresh_inflight_high_water() {
  static std::atomic<uint32_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_issue_first_thread_tag() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_issue_last_thread_tag() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_issue_thread_change_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publish_first_thread_tag() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publish_previous_thread_tag() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint32_t>& source_publication_order_fault_state() {
  static std::atomic<uint32_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_order_fault_previous_source() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_order_fault_previous_publication() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_order_fault_current_source() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_order_fault_current_publication() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_order_fault_issue_thread() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publication_order_fault_publish_thread() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publish_thread_change_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_replaced_before_consume_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_consumed_new_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& real_present_accepted_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_publish_error_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& real_present_error_total() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_queue_lock_wait_total_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_queue_lock_wait_max_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_queue_lock_wait_count() {
  static std::atomic<uint64_t> v{0};
  return v;
}
// Host time of the Source's own vkQueueSubmit call, inside the q0 lock.
inline std::atomic<uint64_t>& source_submit_host_total_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_submit_host_max_ns() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_submit_host_count() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t>& source_submit_host_blocked_count() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline uint64_t& source_current_issue_id() {
  static thread_local uint64_t v = 0;
  return v;
}
inline uint64_t& source_current_issue_time_ns() {
  static thread_local uint64_t v = 0;
  return v;
}
inline uint64_t& source_current_issue_thread_tag() {
  static thread_local uint64_t v = 0;
  return v;
}

inline void RecordSourceBoundaryPeriod(SourceBoundaryPeriodRing& ring,
                                       std::atomic<uint64_t>& last_ns,
                                       uint64_t bp_wait_ns = 0) {
  const uint64_t now_ns = FrameTelemetryNowNs();
  const uint64_t previous_ns =
      last_ns.exchange(now_ns, std::memory_order_acq_rel);
  if (!previous_ns || now_ns <= previous_ns ||
      now_ns - previous_ns > kFrameTelemetryWindowNs) {
    return;
  }
  const uint64_t sequence =
      ring.sequence.fetch_add(1, std::memory_order_acq_rel) + 1;
  SourceBoundaryPeriodSlot& slot =
      ring.slots[(sequence - 1) % kSourceBoundaryPeriodCapacity];
  slot.interval_ns.store(now_ns - previous_ns, std::memory_order_relaxed);
  slot.bp_wait_ns.store(bp_wait_ns, std::memory_order_relaxed);
  slot.sequence.store(sequence, std::memory_order_release);
}

struct SourceFrameTelemetrySample {
  uint64_t corrected_time_ns = 0;
};

struct SourceFrameTelemetryState {
  uint64_t generation = 0;
  uint64_t last_wall_ns = 0;
  uint64_t pending_pacing_wait_ns = 0;
  uint64_t corrected_time_ns = 0;
  SourceFrameTelemetrySample samples[kFrameTelemetrySampleCapacity] = {};
  size_t head = 0;
  size_t count = 0;
};

inline SourceFrameTelemetryState& source_frame_telemetry_state() {
  // Owned by the serialized GPU command-processor thread.
  static SourceFrameTelemetryState state;
  return state;
}

inline uint64_t& source_producer_thread_generation() {
  // A pacing wait is attributable only when it runs on the same thread that
  // produced the Source event for the current reset generation.
  static thread_local uint64_t generation = 0;
  return generation;
}

struct FinalOutputTelemetrySlot {
  // Zero marks a slot while a producer is publishing it. Sequence is written
  // last with release ordering so bounded readers never accept torn metadata.
  std::atomic<uint64_t> sequence{0};
  std::atomic<uint64_t> generation{0};
  std::atomic<uint64_t> timestamp_ns{0};
};

struct FinalOutputTelemetryRing {
  std::atomic<uint64_t> sequence{0};
  std::array<FinalOutputTelemetrySlot, kFrameTelemetrySampleCapacity> slots;
};

inline FinalOutputTelemetryRing& final_output_telemetry_ring() {
  // Multiple presentation backends may briefly coexist during a fail-open
  // drain. Their call sites select exactly one visible authority, while this
  // ring makes the common delivery event itself lock-free and MPSC-safe.
  static FinalOutputTelemetryRing ring;
  return ring;
}

inline void PublishSourceFps(float fps, uint64_t event_ns,
                             uint64_t generation) {
  if (frame_telemetry_reset_generation().load(std::memory_order_acquire) !=
      generation) {
    return;
  }
  source_fps().store(fps, std::memory_order_relaxed);
  source_last_event_ns().store(event_ns, std::memory_order_relaxed);
  source_published_generation().store(generation, std::memory_order_release);
}

inline float GetUnexpiredTelemetryFps(
    std::atomic<float>& published_fps,
    std::atomic<uint64_t>& published_last_event_ns,
    std::atomic<uint64_t>& published_generation) {
  const uint64_t generation =
      frame_telemetry_reset_generation().load(std::memory_order_acquire);
  if (published_generation.load(std::memory_order_acquire) != generation) {
    return 0.0f;
  }
  const uint64_t last_event_ns =
      published_last_event_ns.load(std::memory_order_relaxed);
  const uint64_t now_ns = FrameTelemetryNowNs();
  if (!last_event_ns || now_ns < last_event_ns ||
      now_ns - last_event_ns > kFrameTelemetryWindowNs) {
    return 0.0f;
  }
  const float fps = published_fps.load(std::memory_order_relaxed);
  if (frame_telemetry_reset_generation().load(std::memory_order_acquire) !=
          generation ||
      published_generation.load(std::memory_order_acquire) != generation) {
    return 0.0f;
  }
  return fps;
}
}  // namespace internal

struct SourceBoundaryTotals {
  uint64_t source_issue_total = 0;
  uint64_t source_published_total = 0;
  uint64_t source_replaced_before_consume_total = 0;
  uint64_t source_consumed_new_total = 0;
  uint64_t real_present_accepted_total = 0;
  uint64_t source_publish_error_total = 0;
  uint64_t real_present_error_total = 0;
  uint64_t source_queue_lock_wait_total_ns = 0;
  uint64_t source_queue_lock_wait_max_ns = 0;
  uint64_t source_queue_lock_wait_count = 0;
  uint64_t source_submit_host_total_ns = 0;
  uint64_t source_submit_host_max_ns = 0;
  uint64_t source_submit_host_count = 0;
  uint64_t source_submit_host_blocked_count = 0;
  uint64_t source_publication_order_regression_total = 0;
  uint64_t source_last_publication_id = 0;
  uint64_t source_last_published_source_id = 0;
  uint64_t source_last_publish_thread_tag = 0;
  uint64_t source_issue_first_thread_tag = 0;
  uint64_t source_issue_thread_change_total = 0;
  uint64_t source_publish_first_thread_tag = 0;
  uint64_t source_publish_thread_change_total = 0;
  uint32_t source_issue_inflight_high_water = 0;
  uint32_t guest_refresh_inflight_high_water = 0;
  uint64_t source_publication_order_fault_previous_source = 0;
  uint64_t source_publication_order_fault_previous_publication = 0;
  uint64_t source_publication_order_fault_current_source = 0;
  uint64_t source_publication_order_fault_current_publication = 0;
  uint64_t source_publication_order_fault_issue_thread = 0;
  uint64_t source_publication_order_fault_publish_thread = 0;
};

struct SourcePublicationTelemetry {
  uint64_t source_id = 0;
  uint64_t publication_id = 0;
  uint64_t issue_time_ns = 0;
  uint64_t publish_time_ns = 0;
  uint64_t issue_thread_tag = 0;
  uint64_t publish_thread_tag = 0;
  bool issue_identity_valid = false;
};

inline uint64_t SourceTelemetryThreadTag() {
  return uint64_t(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

enum class SourceContractFault : size_t {
  kPublicationCommitRegression,
  kIssuePublishFallback,
  kIssuePublishThreadMismatch,
  kIssueConcurrent,
  kRefreshConcurrent,
  kCount
};

struct SourceContractEvidence {
  std::array<std::atomic<uint64_t>, size_t(SourceContractFault::kCount)> counts{};
  std::atomic<uint32_t> first_state{0};
  SourceContractFault first_kind = SourceContractFault::kCount;
  SourcePublicationTelemetry first_publication;
  uint64_t first_previous_publication = 0;
};

inline SourceContractEvidence& GetSourceContractEvidence() {
  static SourceContractEvidence evidence;
  return evidence;
}

inline void RecordSourceContractFault(
    SourceContractFault kind, const SourcePublicationTelemetry& publication,
    uint64_t previous_publication = 0) {
  auto& evidence = GetSourceContractEvidence();
  uint32_t empty = 0;
  if (evidence.first_state.compare_exchange_strong(
          empty, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
    evidence.first_kind = kind;
    evidence.first_publication = publication;
    evidence.first_previous_publication = previous_publication;
    evidence.first_state.store(2, std::memory_order_release);
  }
  evidence.counts[size_t(kind)].fetch_add(1, std::memory_order_release);
}

class SourceBoundaryInflightScope {
 public:
  enum class Kind { kIssue, kRefresh };

  explicit SourceBoundaryInflightScope(Kind kind) : kind_(kind) {
    std::atomic<uint32_t>& inflight =
        kind == Kind::kIssue ? internal::source_issue_inflight()
                             : internal::guest_refresh_inflight();
    std::atomic<uint32_t>& high_water =
        kind == Kind::kIssue ? internal::source_issue_inflight_high_water()
                             : internal::guest_refresh_inflight_high_water();
    const uint32_t current =
        inflight.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (current > 1) {
      SourcePublicationTelemetry evidence;
      evidence.source_id = internal::source_current_issue_id();
      evidence.issue_thread_tag = internal::source_current_issue_thread_tag();
      evidence.publish_thread_tag = SourceTelemetryThreadTag();
      RecordSourceContractFault(
          kind == Kind::kIssue ? SourceContractFault::kIssueConcurrent
                              : SourceContractFault::kRefreshConcurrent,
          evidence);
    }
    uint32_t previous = high_water.load(std::memory_order_relaxed);
    while (previous < current &&
           !high_water.compare_exchange_weak(
               previous, current, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }
  ~SourceBoundaryInflightScope() {
    (kind_ == Kind::kIssue ? internal::source_issue_inflight()
                           : internal::guest_refresh_inflight())
        .fetch_sub(1, std::memory_order_acq_rel);
  }
  SourceBoundaryInflightScope(const SourceBoundaryInflightScope&) = delete;
  SourceBoundaryInflightScope& operator=(const SourceBoundaryInflightScope&) =
      delete;

 private:
  Kind kind_;
};

struct SourceBoundaryPeriodSnapshot {
  bool consistent = false;
  uint64_t ending_sequence = 0;
  size_t count = 0;
  std::array<uint64_t, internal::kSourceBoundaryPeriodCapacity> intervals_ns =
      {};
  // ZeroFG handoff backpressure attributed to each interval (issue ring).
  std::array<uint64_t, internal::kSourceBoundaryPeriodCapacity> bp_wait_ns =
      {};
};

struct SourceBoundarySnapshot {
  SourceBoundaryTotals totals;
  SourceBoundaryPeriodSnapshot issue_periods;
  SourceBoundaryPeriodSnapshot publish_periods;
};

inline bool SnapshotSourceBoundaryPeriods(
    internal::SourceBoundaryPeriodRing& ring,
    SourceBoundaryPeriodSnapshot& snapshot_out) {
  snapshot_out = {};
  const uint64_t ending_sequence =
      ring.sequence.load(std::memory_order_acquire);
  snapshot_out.ending_sequence = ending_sequence;
  const size_t count = size_t(std::min<uint64_t>(
      ending_sequence, internal::kSourceBoundaryPeriodCapacity));
  const uint64_t first_sequence = ending_sequence - count + 1;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t sequence = first_sequence + i;
    internal::SourceBoundaryPeriodSlot& slot =
        ring.slots[(sequence - 1) % internal::kSourceBoundaryPeriodCapacity];
    if (slot.sequence.load(std::memory_order_acquire) != sequence) {
      return false;
    }
    snapshot_out.intervals_ns[i] =
        slot.interval_ns.load(std::memory_order_relaxed);
    snapshot_out.bp_wait_ns[i] =
        slot.bp_wait_ns.load(std::memory_order_relaxed);
    if (slot.sequence.load(std::memory_order_acquire) != sequence) {
      return false;
    }
  }
  if (ring.sequence.load(std::memory_order_acquire) != ending_sequence) {
    return false;
  }
  snapshot_out.count = count;
  snapshot_out.consistent = true;
  return true;
}

inline SourceBoundarySnapshot GetSourceBoundarySnapshot() {
  SourceBoundarySnapshot snapshot;
  SnapshotSourceBoundaryPeriods(internal::source_issue_period_ring(),
                                snapshot.issue_periods);
  SnapshotSourceBoundaryPeriods(internal::source_publish_period_ring(),
                                snapshot.publish_periods);
  snapshot.totals.source_issue_total =
      internal::source_issue_total().load(std::memory_order_relaxed);
  snapshot.totals.source_published_total =
      internal::source_published_total().load(std::memory_order_relaxed);
  snapshot.totals.source_replaced_before_consume_total =
      internal::source_replaced_before_consume_total().load(
          std::memory_order_relaxed);
  snapshot.totals.source_consumed_new_total =
      internal::source_consumed_new_total().load(std::memory_order_relaxed);
  snapshot.totals.real_present_accepted_total =
      internal::real_present_accepted_total().load(std::memory_order_relaxed);
  snapshot.totals.source_publish_error_total =
      internal::source_publish_error_total().load(std::memory_order_relaxed);
  snapshot.totals.real_present_error_total =
      internal::real_present_error_total().load(std::memory_order_relaxed);
  snapshot.totals.source_queue_lock_wait_total_ns =
      internal::source_queue_lock_wait_total_ns().load(
          std::memory_order_relaxed);
  snapshot.totals.source_queue_lock_wait_max_ns =
      internal::source_queue_lock_wait_max_ns().load(
          std::memory_order_relaxed);
  snapshot.totals.source_queue_lock_wait_count =
      internal::source_queue_lock_wait_count().load(
          std::memory_order_relaxed);
  snapshot.totals.source_submit_host_total_ns =
      internal::source_submit_host_total_ns().load(std::memory_order_relaxed);
  snapshot.totals.source_submit_host_max_ns =
      internal::source_submit_host_max_ns().load(std::memory_order_relaxed);
  snapshot.totals.source_submit_host_count =
      internal::source_submit_host_count().load(std::memory_order_relaxed);
  snapshot.totals.source_submit_host_blocked_count =
      internal::source_submit_host_blocked_count().load(
          std::memory_order_relaxed);
  snapshot.totals.source_publication_order_regression_total =
      internal::source_publication_order_regression_total().load(
          std::memory_order_relaxed);
  snapshot.totals.source_last_publication_id =
      internal::source_last_publication_id().load(std::memory_order_relaxed);
  snapshot.totals.source_last_published_source_id =
      internal::source_last_published_source_id().load(
          std::memory_order_relaxed);
  snapshot.totals.source_last_publish_thread_tag =
      internal::source_last_publish_thread_tag().load(
          std::memory_order_relaxed);
  snapshot.totals.source_issue_first_thread_tag =
      internal::source_issue_first_thread_tag().load(
          std::memory_order_relaxed);
  snapshot.totals.source_issue_thread_change_total =
      internal::source_issue_thread_change_total().load(
          std::memory_order_relaxed);
  snapshot.totals.source_publish_first_thread_tag =
      internal::source_publish_first_thread_tag().load(
          std::memory_order_relaxed);
  snapshot.totals.source_publish_thread_change_total =
      internal::source_publish_thread_change_total().load(
          std::memory_order_relaxed);
  snapshot.totals.source_issue_inflight_high_water =
      internal::source_issue_inflight_high_water().load(
          std::memory_order_relaxed);
  snapshot.totals.guest_refresh_inflight_high_water =
      internal::guest_refresh_inflight_high_water().load(
          std::memory_order_relaxed);
  if (internal::source_publication_order_fault_state().load(
          std::memory_order_acquire) == 2) {
    snapshot.totals.source_publication_order_fault_previous_source =
        internal::source_publication_order_fault_previous_source().load(
            std::memory_order_relaxed);
    snapshot.totals.source_publication_order_fault_previous_publication =
        internal::source_publication_order_fault_previous_publication().load(
            std::memory_order_relaxed);
    snapshot.totals.source_publication_order_fault_current_source =
        internal::source_publication_order_fault_current_source().load(
            std::memory_order_relaxed);
    snapshot.totals.source_publication_order_fault_current_publication =
        internal::source_publication_order_fault_current_publication().load(
            std::memory_order_relaxed);
    snapshot.totals.source_publication_order_fault_issue_thread =
        internal::source_publication_order_fault_issue_thread().load(
            std::memory_order_relaxed);
    snapshot.totals.source_publication_order_fault_publish_thread =
        internal::source_publication_order_fault_publish_thread().load(
            std::memory_order_relaxed);
  }
  return snapshot;
}

inline SourcePublicationTelemetry PrepareSourcePublicationTelemetry() {
  SourcePublicationTelemetry telemetry;
  telemetry.publish_time_ns = internal::FrameTelemetryNowNs();
  telemetry.publication_id =
      internal::source_publication_sequence().fetch_add(
          1, std::memory_order_relaxed) +
      1;
  telemetry.source_id = internal::source_current_issue_id();
  telemetry.issue_time_ns = internal::source_current_issue_time_ns();
  telemetry.issue_thread_tag = internal::source_current_issue_thread_tag();
  telemetry.publish_thread_tag = SourceTelemetryThreadTag();
  telemetry.issue_identity_valid =
      telemetry.source_id && telemetry.issue_time_ns &&
      telemetry.issue_thread_tag == telemetry.publish_thread_tag;
  if (!telemetry.source_id || !telemetry.issue_time_ns) {
    RecordSourceContractFault(SourceContractFault::kIssuePublishFallback,
                              telemetry);
  }
  if (!telemetry.source_id) {
    telemetry.source_id = telemetry.publication_id;
  }
  if (telemetry.issue_thread_tag != telemetry.publish_thread_tag) {
    RecordSourceContractFault(SourceContractFault::kIssuePublishThreadMismatch,
                              telemetry);
  }
  return telemetry;
}

inline void CommitSourcePublicationTelemetry(
    const SourcePublicationTelemetry& telemetry,
    bool replaced_before_consume) {
  internal::source_published_total().fetch_add(1, std::memory_order_relaxed);
  if (replaced_before_consume) {
    internal::source_replaced_before_consume_total().fetch_add(
        1, std::memory_order_relaxed);
  }
  internal::RecordSourceBoundaryPeriod(
      internal::source_publish_period_ring(),
      internal::source_publish_period_last_ns());

  const uint64_t previous_publication_id =
      internal::source_last_publication_id().exchange(
          telemetry.publication_id, std::memory_order_acq_rel);
  if (previous_publication_id &&
      telemetry.publication_id <= previous_publication_id) {
    RecordSourceContractFault(SourceContractFault::kPublicationCommitRegression,
                              telemetry, previous_publication_id);
  }

  const uint64_t previous_source_id =
      internal::source_last_published_source_id().exchange(
          telemetry.source_id, std::memory_order_acq_rel);
  if (previous_source_id && telemetry.source_id <= previous_source_id) {
    internal::source_publication_order_regression_total().fetch_add(
        1, std::memory_order_relaxed);
    uint32_t empty = 0;
    if (internal::source_publication_order_fault_state()
            .compare_exchange_strong(empty, 1, std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
      internal::source_publication_order_fault_previous_publication().store(
          previous_publication_id,
          std::memory_order_relaxed);
      internal::source_publication_order_fault_previous_source().store(
          previous_source_id, std::memory_order_relaxed);
      internal::source_publication_order_fault_current_source().store(
          telemetry.source_id, std::memory_order_relaxed);
      internal::source_publication_order_fault_current_publication().store(
          telemetry.publication_id, std::memory_order_relaxed);
      internal::source_publication_order_fault_issue_thread().store(
          telemetry.issue_thread_tag, std::memory_order_relaxed);
      internal::source_publication_order_fault_publish_thread().store(
          telemetry.publish_thread_tag, std::memory_order_relaxed);
      internal::source_publication_order_fault_state().store(
          2, std::memory_order_release);
    }
  }
  internal::source_last_publish_thread_tag().store(
      telemetry.publish_thread_tag, std::memory_order_relaxed);

  uint64_t first_publish_thread = 0;
  internal::source_publish_first_thread_tag().compare_exchange_strong(
      first_publish_thread, telemetry.publish_thread_tag,
      std::memory_order_relaxed, std::memory_order_relaxed);
  const uint64_t previous_publish_thread =
      internal::source_publish_previous_thread_tag().exchange(
          telemetry.publish_thread_tag, std::memory_order_relaxed);
  if (previous_publish_thread &&
      previous_publish_thread != telemetry.publish_thread_tag) {
    internal::source_publish_thread_change_total().fetch_add(
        1, std::memory_order_relaxed);
  }
}

inline void RecordSourceQueueLockWait(uint64_t wait_ns) {
  internal::source_queue_lock_wait_total_ns().fetch_add(
      wait_ns, std::memory_order_relaxed);
  internal::source_queue_lock_wait_count().fetch_add(
      1, std::memory_order_relaxed);
  uint64_t previous =
      internal::source_queue_lock_wait_max_ns().load(
          std::memory_order_relaxed);
  while (previous < wait_ns &&
         !internal::source_queue_lock_wait_max_ns().compare_exchange_weak(
             previous, wait_ns, std::memory_order_relaxed,
             std::memory_order_relaxed)) {
  }
}

// Telemetry only: the Source's own vkQueueSubmit host time. Calls of at least
// kSourceSubmitBlockedNs are counted separately.
inline void RecordSourceSubmitHost(uint64_t submit_ns) {
  constexpr uint64_t kSourceSubmitBlockedNs = 4000000;
  internal::source_submit_host_total_ns().fetch_add(
      submit_ns, std::memory_order_relaxed);
  internal::source_submit_host_count().fetch_add(1,
                                                  std::memory_order_relaxed);
  if (submit_ns >= kSourceSubmitBlockedNs) {
    internal::source_submit_host_blocked_count().fetch_add(
        1, std::memory_order_relaxed);
  }
  uint64_t previous =
      internal::source_submit_host_max_ns().load(std::memory_order_relaxed);
  while (previous < submit_ns &&
         !internal::source_submit_host_max_ns().compare_exchange_weak(
             previous, submit_ns, std::memory_order_relaxed,
             std::memory_order_relaxed)) {
  }
}

inline void RecordSourcePublishError() {
  internal::source_publish_error_total().fetch_add(1,
                                                    std::memory_order_relaxed);
}

inline void RecordSourceConsumedNew() {
  internal::source_consumed_new_total().fetch_add(1,
                                                  std::memory_order_relaxed);
}

inline void RecordRealPresentAccepted() {
  internal::real_present_accepted_total().fetch_add(1,
                                                    std::memory_order_relaxed);
}

inline void RecordRealPresentError() {
  internal::real_present_error_total().fetch_add(1,
                                                 std::memory_order_relaxed);
}

// Call once per presented guest frame (single producer thread).
// FPS is a RenderDoc-style average over a ~1s sliding time window (framerate
// independent); instant_ms stays the raw present-to-present delta.
inline void RecordGuestPresent() {
  using clock = std::chrono::steady_clock;
  static constexpr double kWindowMs = 1000.0;
  static constexpr size_t kCap = 1024;  // covers >1000 fps within the window
  static clock::time_point ts[kCap];
  static size_t head = 0;   // oldest sample
  static size_t count = 0;  // samples in the window
  static clock::time_point last{};

  clock::time_point now = clock::now();
  const bool have_last = last != clock::time_point{};
  double instant_ms =
      have_last ? std::chrono::duration<double, std::milli>(now - last).count()
                : 0.0;
  last = now;

  // A long gap (first frame / pause / load) restarts the window so the average
  // recovers within a second instead of being dragged by a stale outlier.
  if (!have_last || instant_ms > kWindowMs) {
    head = 0;
    count = 1;
    ts[0] = now;
    internal::frame_instant_ms().store(0.0f, std::memory_order_relaxed);
    internal::frame_avg_ms().store(0.0f, std::memory_order_relaxed);
    internal::frame_fps().store(0.0f, std::memory_order_relaxed);
    return;
  }

  const size_t tail = (head + count) % kCap;
  ts[tail] = now;
  if (count < kCap) {
    ++count;
  } else {
    head = (head + 1) % kCap;  // ring full: drop oldest
  }

  // Evict samples older than the window.
  while (count > 1 &&
         std::chrono::duration<double, std::milli>(now - ts[head]).count() >
             kWindowMs) {
    head = (head + 1) % kCap;
    --count;
  }

  const double span_ms =
      std::chrono::duration<double, std::milli>(now - ts[head]).count();
  const double fps =
      (count > 1 && span_ms > 0.0) ? double(count - 1) * 1000.0 / span_ms : 0.0;
  const double avg_ms = fps > 0.0 ? 1000.0 / fps : 0.0;

  internal::frame_instant_ms().store(float(instant_ms),
                                     std::memory_order_relaxed);
  internal::frame_avg_ms().store(float(avg_ms), std::memory_order_relaxed);
  internal::frame_fps().store(float(fps), std::memory_order_relaxed);
}

// Read the latest published stats (any thread).
inline void GetFrameStats(float& instant_ms, float& avg_ms, float& fps) {
  instant_ms = internal::frame_instant_ms().load(std::memory_order_relaxed);
  avg_ms = internal::frame_avg_ms().load(std::memory_order_relaxed);
  fps = internal::frame_fps().load(std::memory_order_relaxed);
}

// Request a thread-safe reset without touching producer-owned window state.
// Producers consume the generation on their next event; getters return zero
// immediately and refuse publications from an older generation.
inline void RequestFrameTelemetryReset() {
  internal::frame_telemetry_reset_generation().fetch_add(
      1, std::memory_order_acq_rel);
  internal::source_published_generation().store(0,
                                                 std::memory_order_release);
  internal::source_last_event_ns().store(0, std::memory_order_relaxed);
  internal::source_fps().store(0.0f, std::memory_order_relaxed);
}

// ZeroFG lossless handoff backpressure: time the Source waited on presenter
// capacity. It is attributed to the issue interval in progress so the
// presenter can recover the Source's intrinsic cadence; raw intervals are
// never altered.
inline void AddSourceBackpressureWaitNs(uint64_t wait_ns) {
  internal::source_issue_bp_wait_pending_ns().fetch_add(
      wait_ns, std::memory_order_relaxed);
}

// Call exactly once per guest IssueSwap, adjacent to RecordGuestPresent().
// The official Source window advances on corrected time rather than wall time.
inline void RecordSourcePresent() {
  const uint64_t issue_time_ns = internal::FrameTelemetryNowNs();
  internal::source_current_issue_id() =
      internal::source_issue_total().fetch_add(
          1, std::memory_order_relaxed) +
      1;
  internal::source_current_issue_time_ns() = issue_time_ns;
  const uint64_t issue_thread_tag = SourceTelemetryThreadTag();
  internal::source_current_issue_thread_tag() = issue_thread_tag;
  uint64_t first_issue_thread = 0;
  internal::source_issue_first_thread_tag().compare_exchange_strong(
      first_issue_thread, issue_thread_tag, std::memory_order_relaxed,
      std::memory_order_relaxed);
  const uint64_t previous_issue_thread =
      internal::source_issue_last_thread_tag().exchange(
          issue_thread_tag, std::memory_order_relaxed);
  if (previous_issue_thread && previous_issue_thread != issue_thread_tag) {
    internal::source_issue_thread_change_total().fetch_add(
        1, std::memory_order_relaxed);
  }
  // ZeroFG backpressure waited since the previous IssueSwap belongs to the
  // interval ending here.
  internal::RecordSourceBoundaryPeriod(
      internal::source_issue_period_ring(),
      internal::source_issue_period_last_ns(),
      internal::source_issue_bp_wait_pending_ns().exchange(
          0, std::memory_order_relaxed));

  const uint64_t generation =
      internal::frame_telemetry_reset_generation().load(
          std::memory_order_acquire);
  internal::source_producer_thread_generation() = generation;

  internal::SourceFrameTelemetryState& state =
      internal::source_frame_telemetry_state();
  const uint64_t now_ns = internal::FrameTelemetryNowNs();

  const auto restart_window = [&]() {
    state = {};
    state.generation = generation;
    state.last_wall_ns = now_ns;
    state.count = 1;
    state.samples[0].corrected_time_ns = 0;
    internal::PublishSourceFps(0.0f, now_ns, generation);
  };

  if (state.generation != generation || !state.last_wall_ns) {
    restart_window();
    return;
  }

  if (now_ns < state.last_wall_ns) {
    restart_window();
    return;
  }
  const uint64_t observed_interval_ns = now_ns - state.last_wall_ns;
  const uint64_t pacing_wait_ns = state.pending_pacing_wait_ns;
  state.pending_pacing_wait_ns = 0;
  state.last_wall_ns = now_ns;

  // A long real-time gap is a lifecycle/stall boundary even if subtracting a
  // wait could make the corrected interval appear short.
  if (!observed_interval_ns ||
      observed_interval_ns > internal::kFrameTelemetryWindowNs ||
      pacing_wait_ns >= observed_interval_ns) {
    restart_window();
    return;
  }

  const uint64_t corrected_interval_ns =
      observed_interval_ns - pacing_wait_ns;
  if (corrected_interval_ns >
      UINT64_MAX - state.corrected_time_ns) {
    restart_window();
    return;
  }
  state.corrected_time_ns += corrected_interval_ns;

  const size_t tail = (state.head + state.count) %
                      internal::kFrameTelemetrySampleCapacity;
  state.samples[tail].corrected_time_ns = state.corrected_time_ns;
  if (state.count < internal::kFrameTelemetrySampleCapacity) {
    ++state.count;
  } else {
    state.head =
        (state.head + 1) % internal::kFrameTelemetrySampleCapacity;
  }

  while (state.count > 1 &&
         state.corrected_time_ns -
                 state.samples[state.head].corrected_time_ns >
             internal::kFrameTelemetryWindowNs) {
    state.head =
        (state.head + 1) % internal::kFrameTelemetrySampleCapacity;
    --state.count;
  }

  const uint64_t span_ns =
      state.corrected_time_ns -
      state.samples[state.head].corrected_time_ns;
  const double fps =
      state.count > 1 && span_ns
          ? double(state.count - 1) * 1000000000.0 / double(span_ns)
          : 0.0;
  internal::PublishSourceFps(float(fps), now_ns, generation);
}

// Read-only probe for whether this thread produced a Source event in the
// current telemetry generation. This doesn't modify Source window state.
inline bool IsCurrentThreadSourceProducer() {
  const uint64_t generation =
      internal::frame_telemetry_reset_generation().load(
          std::memory_order_acquire);
  return internal::source_producer_thread_generation() == generation;
}

// Copy an already-measured ZeroFG pacing wait into Source telemetry only when
// the wait ran synchronously on the current Source producer. In Android/FIFO,
// pacing normally runs on the UI thread, so it is deliberately not subtracted
// from IssueSwap cadence.
inline void RecordZeroFGPacingWait(uint64_t actual_wait_ns) {
  if (!actual_wait_ns) {
    return;
  }
  const uint64_t generation =
      internal::frame_telemetry_reset_generation().load(
          std::memory_order_acquire);
  if (internal::source_producer_thread_generation() != generation) {
    return;
  }

  internal::SourceFrameTelemetryState& state =
      internal::source_frame_telemetry_state();
  if (state.generation != generation || !state.last_wall_ns) {
    return;
  }
  if (actual_wait_ns > UINT64_MAX - state.pending_pacing_wait_ns) {
    state.pending_pacing_wait_ns = UINT64_MAX;
  } else {
    state.pending_pacing_wait_ns += actual_wait_ns;
  }
}

namespace internal {
inline std::atomic<bool>& zerofg_main_surface_active() {
  static std::atomic<bool> v{false};
  return v;
}
}  // namespace internal

// Main Surface Authority: true while ZeroFG's device B produces into the main
// Surface. The Android host polls it and holds its display frame-rate vote only
// then, so a ZeroFG that is off or failed to qualify leaves the normal path
// without the vote.
inline void SetZeroFGMainSurfaceActive(bool active) {
  internal::zerofg_main_surface_active().store(active,
                                               std::memory_order_release);
}

inline bool IsZeroFGMainSurfaceActive() {
  return internal::zerofg_main_surface_active().load(
      std::memory_order_acquire);
}

// Call once at the active presentation backend's final delivery event. For the
// normal / Phase-B Vulkan backend this is accepted-by-WSI (VK_SUCCESS or
// VK_SUBOPTIMAL_KHR); for Phase C it is the SurfaceControl apply handoff.
// This measures output submitted by the active presentation authority, not
// Synthetic production, callback completion or physical scanout. Publication
// is lock-free and safe for distinct backend threads during a bounded authority
// transition.
inline void RecordOutputPresent() {
  const uint64_t generation =
      internal::frame_telemetry_reset_generation().load(
          std::memory_order_acquire);
  const uint64_t now_ns = internal::FrameTelemetryNowNs();
  internal::FinalOutputTelemetryRing& ring =
      internal::final_output_telemetry_ring();
  const uint64_t sequence =
      ring.sequence.fetch_add(1, std::memory_order_relaxed) + 1;
  internal::FinalOutputTelemetrySlot& slot =
      ring.slots[(sequence - 1) % internal::kFrameTelemetrySampleCapacity];
  slot.sequence.store(0, std::memory_order_release);
  slot.generation.store(generation, std::memory_order_relaxed);
  slot.timestamp_ns.store(now_ns, std::memory_order_relaxed);
  slot.sequence.store(sequence, std::memory_order_release);
}

inline float GetSourceFps() {
  return internal::GetUnexpiredTelemetryFps(
      internal::source_fps(), internal::source_last_event_ns(),
      internal::source_published_generation());
}

inline float GetOutputFps() {
  const uint64_t generation =
      internal::frame_telemetry_reset_generation().load(
          std::memory_order_acquire);
  const uint64_t now_ns = internal::FrameTelemetryNowNs();

  // Output-rate readers are low-frequency UI / resource telemetry consumers.
  // Scan the fixed ring here so every delivery producer stays bounded, with no
  // sorting or allocation. Generation filtering makes reset immediate without
  // a separate last-writer-wins publication atomic that distinct backends
  // could update out of chronological order during an authority transition.
  uint64_t oldest_ns = UINT64_MAX;
  uint64_t newest_ns = 0;
  size_t count = 0;
  for (const internal::FinalOutputTelemetrySlot& slot :
       internal::final_output_telemetry_ring().slots) {
    const uint64_t sequence_before =
        slot.sequence.load(std::memory_order_acquire);
    if (!sequence_before) {
      continue;
    }
    const uint64_t slot_generation =
        slot.generation.load(std::memory_order_relaxed);
    const uint64_t timestamp_ns =
        slot.timestamp_ns.load(std::memory_order_relaxed);
    if (slot.sequence.load(std::memory_order_acquire) != sequence_before ||
        slot_generation != generation || !timestamp_ns ||
        timestamp_ns > now_ns ||
        now_ns - timestamp_ns > internal::kFrameTelemetryWindowNs) {
      continue;
    }
    oldest_ns = std::min(oldest_ns, timestamp_ns);
    newest_ns = std::max(newest_ns, timestamp_ns);
    ++count;
  }
  if (internal::frame_telemetry_reset_generation().load(
          std::memory_order_acquire) != generation ||
      count <= 1 || newest_ns <= oldest_ns) {
    return 0.0f;
  }
  return float(double(count - 1) * 1000000000.0 /
               double(newest_ns - oldest_ns));
}

}  // namespace xe

#endif  // XENIA_BASE_FRAME_STATS_H_
