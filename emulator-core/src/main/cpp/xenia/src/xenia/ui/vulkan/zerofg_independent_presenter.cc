/**
 ******************************************************************************
* XenDroid ZeroFG V2 independent Android presenter                           *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/zerofg_independent_presenter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>

#include "xenia/base/frame_stats.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/ui/presenter.h"
#include "xenia/ui/vulkan/vulkan_presenter.h"
#include "xenia/ui/vulkan/zerofg_config.h"
#include "xenia/ui/vulkan/zerofg_main_surface_egress.h"

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
#include <errno.h>
#include <poll.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "emulator.h"
#include "xenia/ui/vulkan/vulkan_util.h"
#endif

namespace xe {
namespace ui {
namespace vulkan {

ZeroFGIndependentPresenter::HandoffReadyObservation::
    ~HandoffReadyObservation() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (fd >= 0) close(fd);
#endif
}
uint64_t ZeroFGIndependentPresenter::HandoffReadyObservation::Observe() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (first_seen_ready_ns || error) return first_seen_ready_ns;
  if (fd >= 0) {
    pollfd descriptor = {fd, POLLIN, 0};
    int result = poll(&descriptor, 1, 0);
    if (result < 0 && errno == EINTR) return 0;
    if (result < 0 || (descriptor.revents & (POLLERR | POLLNVAL))) {
      error = true;
      XELOGW("ZeroFGDeviceB N_first_seen_ready observation_error source={}",
             source_id);
      return 0;
    }
    if (!result || !(descriptor.revents & POLLIN)) return 0;
    close(fd);
    fd = -1;
  }
  first_seen_ready_ns =
      uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
#endif
  return first_seen_ready_ns;
}

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid

namespace {

uint64_t PresenterMonotonicTimeNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

int64_t PresenterSignedDeltaNs(uint64_t value_ns, uint64_t reference_ns) {
  if (value_ns >= reference_ns) {
    return int64_t(std::min<uint64_t>(
        value_ns - reference_ns,
        uint64_t(std::numeric_limits<int64_t>::max())));
  }
  return -int64_t(std::min<uint64_t>(
      reference_ns - value_ns,
      uint64_t(std::numeric_limits<int64_t>::max())));
}

template <size_t Capacity>
class PresenterSampleWindow {
 public:
  void Add(uint64_t value) {
    values_[next_] = value;
    next_ = (next_ + 1) % Capacity;
    count_ = std::min(count_ + 1, Capacity);
    maximum_ = std::max(maximum_, value);
  }

  uint64_t maximum() const { return maximum_; }
  uint64_t WindowMaximum() const {
    uint64_t maximum = 0;
    for (size_t i = 0; i < count_; ++i) {
      maximum = std::max(maximum, values_[i]);
    }
    return maximum;
  }
  size_t count() const { return count_; }

  uint64_t Sum() const {
    uint64_t sum = 0;
    for (size_t i = 0; i < count_; ++i) {
      sum = values_[i] > std::numeric_limits<uint64_t>::max() - sum
                ? std::numeric_limits<uint64_t>::max()
                : sum + values_[i];
    }
    return sum;
  }

  uint64_t Mean() const {
    if (!count_) {
      return 0;
    }
    uint64_t sum = 0;
    for (size_t i = 0; i < count_; ++i) {
      sum = values_[i] > std::numeric_limits<uint64_t>::max() - sum
                ? std::numeric_limits<uint64_t>::max()
                : sum + values_[i];
    }
    return sum / count_;
  }

  uint64_t Quantile(uint32_t numerator, uint32_t denominator) const {
    if (!count_) {
      return 0;
    }
    std::array<uint64_t, Capacity> sorted = values_;
    std::sort(sorted.begin(), sorted.begin() + count_);
    const size_t index = std::min(
        count_ - 1,
        size_t((uint64_t(count_ - 1) * numerator + denominator - 1) /
               denominator));
    return sorted[index];
  }

 private:
  std::array<uint64_t, Capacity> values_ = {};
  size_t count_ = 0;
  size_t next_ = 0;
  uint64_t maximum_ = 0;
};

// Window statistics of paired L0 mean-luma samples. Keep the means at double
// precision until reporting: a mean of per-pair A/B ratios is not a ratio of
// the window means, and quantizing to milli-luma before the black gate biases it.
template <size_t Capacity>
class PresenterLumaSampleWindow {
 public:
  void Add(uint32_t sum_a, uint32_t sum_b, uint32_t texels) {
    if (!texels) {
      return;
    }
    const double denominator = double(texels) * 65535.0;
    const double mean_a = double(sum_a) / denominator;
    const double mean_b = double(sum_b) / denominator;
    means_a_[next_] = mean_a;
    means_b_[next_] = mean_b;
    next_ = (next_ + 1) % Capacity;
    count_ = std::min(count_ + 1, Capacity);
    near_black_permille_.Add(mean_a < 0.02 || mean_b < 0.02 ? 1000 : 0);
    const double relative_delta =
        std::abs(mean_a - mean_b) / (mean_a + mean_b + 1e-6);
    relative_delta_nano_.Add(uint64_t(std::llround(relative_delta * 1e9)));
  }

  uint64_t MeanAMilli() const { return MeanMilli(means_a_); }
  uint64_t MeanBMilli() const { return MeanMilli(means_b_); }
  int64_t RatioOfWindowMeansMilli() const {
    const double sum_a = Sum(means_a_);
    const double sum_b = Sum(means_b_);
    if (!count_ || sum_a < 0.02 * double(count_) ||
        sum_b < 0.02 * double(count_)) {
      return -1;
    }
    return int64_t(std::llround(1000.0 * sum_a / sum_b));
  }
  uint64_t NearBlackPairsPermille() const {
    return near_black_permille_.Mean();
  }
  uint64_t RelativeDeltaMeanMilli() const {
    return (relative_delta_nano_.Mean() + 500000) / 1000000;
  }
  uint64_t RelativeDeltaP90Milli() const {
    return (relative_delta_nano_.Quantile(90, 100) + 500000) / 1000000;
  }
  uint64_t RelativeDeltaMaxMilli() const {
    return (relative_delta_nano_.WindowMaximum() + 500000) / 1000000;
  }

 private:
  double Sum(const std::array<double, Capacity>& values) const {
    double sum = 0.0;
    for (size_t i = 0; i < count_; ++i) {
      sum += values[i];
    }
    return sum;
  }
  uint64_t MeanMilli(const std::array<double, Capacity>& values) const {
    return count_ ? uint64_t(std::llround(1000.0 * Sum(values) / double(count_)))
                  : 0;
  }

  std::array<double, Capacity> means_a_ = {};
  std::array<double, Capacity> means_b_ = {};
  size_t count_ = 0;
  size_t next_ = 0;
  PresenterSampleWindow<Capacity> near_black_permille_;
  PresenterSampleWindow<Capacity> relative_delta_nano_;
};

template <size_t Capacity>
class PresenterTimestampWindow {
 public:
  void Add(uint64_t timestamp_ns) {
    timestamps_[next_] = timestamp_ns;
    next_ = (next_ + 1) % Capacity;
    count_ = std::min(count_ + 1, Capacity);
  }

  double RateHz() const {
    if (count_ < 2) {
      return 0.0;
    }
    const size_t oldest = (next_ + Capacity - count_) % Capacity;
    const size_t newest = (next_ + Capacity - 1) % Capacity;
    const uint64_t span_ns = timestamps_[newest] - timestamps_[oldest];
    return span_ns ? double(count_ - 1) * 1000000000.0 / double(span_ns)
                   : 0.0;
  }

  size_t count() const { return count_; }

  uint64_t AveragePeriodNs() const {
    if (count_ < 2) {
      return 0;
    }
    const uint64_t oldest = OrderedTimestamp(0);
    const uint64_t newest = OrderedTimestamp(count_ - 1);
    return newest > oldest ? (newest - oldest) / (count_ - 1) : 0;
  }

  bool EffectiveRateStable(uint64_t tolerance_divisor) const {
    if (count_ < Capacity || count_ < 4 || !tolerance_divisor) {
      return false;
    }
    const size_t half_interval_count = (count_ - 1) / 2;
    if (!half_interval_count) {
      return false;
    }
    const uint64_t first_begin = OrderedTimestamp(0);
    const uint64_t first_end = OrderedTimestamp(half_interval_count);
    const uint64_t second_begin =
        OrderedTimestamp(count_ - 1 - half_interval_count);
    const uint64_t second_end = OrderedTimestamp(count_ - 1);
    if (first_end <= first_begin || second_end <= second_begin) {
      return false;
    }
    const uint64_t full_period = AveragePeriodNs();
    const uint64_t first_period =
        (first_end - first_begin) / half_interval_count;
    const uint64_t second_period =
        (second_end - second_begin) / half_interval_count;
    const uint64_t minimum =
        std::min({full_period, first_period, second_period});
    const uint64_t maximum =
        std::max({full_period, first_period, second_period});
    return full_period && maximum - minimum <= full_period / tolerance_divisor;
  }

 private:
  uint64_t OrderedTimestamp(size_t index) const {
    const size_t oldest = (next_ + Capacity - count_) % Capacity;
    return timestamps_[(oldest + index) % Capacity];
  }

  std::array<uint64_t, Capacity> timestamps_ = {};
  size_t count_ = 0;
  size_t next_ = 0;
};

uint64_t PresenterAveragePeriodFromIntervals(const uint64_t* intervals_ns,
                                      size_t count) {
  if (!intervals_ns || !count) {
    return 0;
  }
  uint64_t sum_ns = 0;
  for (size_t i = 0; i < count; ++i) {
    if (intervals_ns[i] < 5000000ull || intervals_ns[i] > 100000000ull ||
        sum_ns > std::numeric_limits<uint64_t>::max() - intervals_ns[i]) {
      return 0;
    }
    sum_ns += intervals_ns[i];
  }
  return sum_ns / count;
}

bool PresenterEffectiveRateIntervalsStable(const uint64_t* intervals_ns,
                                    size_t count,
                                    uint64_t tolerance_divisor) {
  if (!intervals_ns || count < 8 || !tolerance_divisor) {
    return false;
  }
  const size_t half_count = count / 2;
  const uint64_t full_period =
      PresenterAveragePeriodFromIntervals(intervals_ns, count);
  const uint64_t first_period =
      PresenterAveragePeriodFromIntervals(intervals_ns, half_count);
  const uint64_t second_period = PresenterAveragePeriodFromIntervals(
      intervals_ns + count - half_count, half_count);
  if (!full_period || !first_period || !second_period) {
    return false;
  }
  const uint64_t minimum =
      std::min({full_period, first_period, second_period});
  const uint64_t maximum =
      std::max({full_period, first_period, second_period});
  return maximum - minimum <= full_period / tolerance_divisor;
}

}  // namespace

struct ZeroFGIndependentPresenter::Impl {
  static constexpr uint32_t kGenerationBootstrapSamples = 6;
  static constexpr uint32_t kSyntheticChainEstimatorArmSamples = 8;
  static constexpr uint32_t kDispatchLeadArmSamples = 16;
  static constexpr uint64_t kDispatchLeadFloorNs = 500000ull;
  static constexpr uint64_t kDispatchLeadBootstrapNs = 2000000ull;
  static constexpr uint64_t kDispatchLeadCeilingNs = 8000000ull;
  static constexpr uint64_t kDispatchLeadMarginNs = 500000ull;
  static constexpr uint32_t kBlockingHazardArmSamples = 8;
  static constexpr uint64_t kBlockingHazardFloorNs = 1000000ull;
  static constexpr uint64_t kBlockingHazardBootstrapNs = 4000000ull;
  // Split pre-wake: the guard wake leads the dispatch by the measured
  // cold-wake latency (p90) plus this pad, never more than the maximum.
  static constexpr uint64_t kSplitColdWakePadNs = 500000ull;
  static constexpr uint64_t kSplitColdWakeMarginMaxNs = 6000000ull;
  static constexpr uint64_t kBlockingHazardMarginNs = 500000ull;
  // A causal 2x pair needs one Source period of nominal runway. Measured
  // service may raise D above this floor, but queue residence is not a license
  // to pre-commit multiple Source periods into the semantic future.
  static constexpr uint32_t kLatencyDepthBootstrapQuanta = 2;
  // Thirty intervals cover the common 2/3/5-frame cap patterns evenly in
  // both half-windows (30/40/45/50/60 on a 60-base producer) while still
  // rejecting a single doubled hitch through the half-span comparison.
  static constexpr uint32_t kNativeSourceQualificationSamples = 30;
  static constexpr uint32_t kRealOnlyValidationSamples = 16;
  // 31 timestamps contain 30 intervals, split into two exact 15-interval
  // half-spans. This avoids modal bias for common 2/3/5-frame cap patterns.
  static constexpr uint32_t kSourceEffectiveRateTimestampSamples = 31;
  static constexpr uint64_t kSourceEffectiveRateToleranceDivisor = 20;
  static constexpr uint32_t kSourcePhaseStableSamples = 16;
  static constexpr uint64_t kSourcePhaseMinimumFrequencyErrorNs = 50000ull;
  static constexpr uint32_t kSourcePhaseMinimumConfirmPeriods = 4;
  static constexpr uint32_t kSourcePhaseMaximumConfirmPeriods = 16;
  static constexpr uint32_t kSourcePhaseUrgentConfirmSamples = 3;
  static constexpr uint32_t kPhaseDebtBlockIntervals = 30;
  // Consecutive backpressure-free intervals a rate transition needs from the
  // normal path: one censored as a possible catch-up, then four clustered
  // effective-rate samples.
  static constexpr uint32_t kSourceTransitionCleanRunIntervals = 5;
  // Source rate probe. Under a lossless handoff the Source's own rate becomes
  // unobservable once we pace it: every interval carries the backpressure we
  // imposed, and P censors those (plus the first clean one after them, which
  // may be a catch-up). A Source that got faster can then never be seen. The
  // probe opens a short window in which the handoff is drained eagerly, so the
  // Source runs free and its intervals are clean by construction.
  static constexpr uint32_t kSourceRateProbeSamples = 6;
  static constexpr uint32_t kSourceRateProbeIntervalBudget = 24;
  static constexpr uint64_t kSourceRateProbeWallBudgetNs = 1000000000ull;
  // Shared with the regular Source-period estimator's physical plausibility
  // guard. A bounded probe freezes these absolute limits at episode start;
  // P/2 would reject a legitimate 33.4 ms -> 16.6 ms transition.
  static constexpr uint64_t kSourcePeriodMinimumPlausibleNs = 5000000ull;
  static constexpr uint64_t kSourcePeriodMaximumPlausibleNs = 100000000ull;
  static constexpr uint64_t kSourceRateProbeCooldownConfirmNs = 3000000000ull;
  static constexpr uint64_t kSourceRateProbeCooldownAbortNs = 10000000000ull;
  static constexpr uint64_t kSourceRateProbeCooldownNoChangeNs =
      30000000000ull;
  static constexpr uint32_t kSourceRateProbeBpWindow = 16;
  static constexpr uint32_t kSourceRateProbePacedIntervals = 4;
  // A probe interval is the Source's own as long as any wait we still imposed
  // is negligible against its cadence: at most this, and subtracted from the
  // sample. Larger than that is our pacing and restarts the measurement.
  static constexpr uint64_t kSourceRateProbeCleanToleranceNs = 250000ull;
  // §7 observer: a rolling mixed-evidence window must show censorship,
  // downstream pressure and a slower raw cadence together. The dwell prevents
  // a self-draining transient from becoming a latched operating point.
  static constexpr uint32_t kPhysicalOperatingPointWindow = 32;
  static constexpr uint64_t kPhysicalOperatingPointConfirmDwellNs =
      2500000000ull;
  static constexpr uint64_t kPhysicalOperatingPointConfirmMinimumIssues = 48;
  static constexpr uint64_t kPhysicalOperatingPointConfirmIssueBudget = 128;
  // §7 raw-cadence hysteresis against the delivered P. A candidate (initial
  // or refinement) is born at the entry ratio and stays alive down to the
  // maintenance ratio. §7's own distinction rule is the maintenance floor: it
  // adopts a factual P at least that far from the delivered one, so a
  // candidate that legitimately stayed inside the band can confirm. The
  // global one-fifth gate of the normal authority is never relaxed by it.
  static constexpr uint64_t kPhysicalOperatingPointEntryPercent = 110;
  static constexpr uint64_t kPhysicalOperatingPointDistinctPercent = 5;
  static constexpr uint64_t kPhysicalOperatingPointMaintenancePercent =
      100 + kPhysicalOperatingPointDistinctPercent;
  static_assert(kPhysicalOperatingPointEntryPercent >
                    kPhysicalOperatingPointMaintenancePercent,
                "§7 entry must sit above its maintenance floor");
  static constexpr uint32_t kPhysicalOperatingPointSlackIntervals = 32;
  static constexpr uint64_t kPhysicalOperatingPointCooldownNs =
      5000000000ull;
  static constexpr uint64_t kPhysicalOperatingPointProbeArmTimeoutNs =
      5000000000ull;
  static constexpr uint64_t kPhysicalOperatingPointProbeArmIssueBudget = 128;
  // H1c is bounded to the D1 evidence episode. A stalled/abandoned episode
  // always returns to normal admission without catch-up.
  static constexpr uint64_t kTransitionPlannedSpaceTimeoutNs =
      2500000000ull;
  static constexpr uint64_t kTransitionPlannedSpaceIssueBudget = 128;
  // H1c gate A: the direction of the raw Source cadence over the last few
  // consumed intervals, clean, BP and post-BP alike. A median slower than the
  // delivered P is a plant that cannot keep up, not an accelerating Source;
  // censorship cannot tell the two apart. Observation only, never P.
  static constexpr uint32_t kH1cShortRawWindow = 8;
  static constexpr uint64_t kH1cRawSlowPercent = 105;
  // H1c gate B: qualified clean-fast samples of one evidence sequence before
  // its first TransitionRealOnly pair. One alone is jitter or a catch-up.
  static constexpr uint32_t kH1cArmCleanFastSamples = 2;
  // Phase 5 Better D parameters. §8.1 left them open to be chosen with
  // runtime numbers; these are the first choice, logged on every summary so
  // the next run can move them.
  //   min samples  before the window's median replaces the component prior
  //   F max        the temporal ceiling: reaching it is information, not a
  //                clamp that hides a problem (architecture §1.8)
  //   step         F decay per Source issue, O/64: about 7.8 ms/s at any P
  //   arm          hand the actuator a change only once D moved O/4
  //   margin       floor for the fully-qualified F target (also stale decay)
  static constexpr uint32_t kBetterDMinSamples = 32;
  static constexpr size_t kBetterDWindowCapacity = 128;
  static constexpr uint64_t kBetterDFMaxNs = 32000000ull;
  static constexpr uint32_t kBetterDStepDivisor = 64;
  static constexpr uint32_t kBetterDArmDivisor = 4;
  static constexpr uint64_t kBetterDMarginNs = 1000000ull;
  // B4: Source issues without a qualified sample before the tail is stale,
  // about 1 s at 60 fps and 2 s at 30.
  static constexpr uint32_t kBetterDStaleIssues = 64;
  // B4: every F rise is logged individually, up to this many per session, so
  // the next run shows what a rising sample actually looks like.
  // Capture completion and release-fence observation restore physical
  // ownership, so they retain a short liveness retry. Optional Generation/Post
  // completion is scheduled separately from service evidence and semantic
  // deadlines rather than waking the presenter at this cadence.
  static constexpr uint64_t kOwnershipPollIntervalNs = 250000ull;
  static constexpr uint64_t kReleaseFencePollIntervalNs = 1000000ull;
  // Split presenter device: a publication whose copy is still queued on the
  // Source device. It completes behind the Source's queued frames (tens of ms
  // at saturation), so it is polled at the release-fence cadence.
  static constexpr uint64_t kHandoffReadyPollIntervalNs = 1000000ull;
  static constexpr uint64_t kObservationDueRetryIntervalNs = 1000000ull;
  // Main Surface Authority: how often the UI thread is asked to act while the
  // Surface changes hands, and how long a transfer may take before MSA fails
  // open to the normal presenter.
  static constexpr uint64_t kMainSurfaceUIRequestIntervalNs = 16000000ull;
  static constexpr uint64_t kMainSurfaceTransferTimeoutNs = 1000000000ull;
  // Residency keeps one turnover slot outside optional Synthetic retention.
  // Source ingress has its own physical capture pool and is never represented
  // by projected residency-release credit.
  static constexpr uint32_t kCandidateReserveTarget = 1;
  // Split sidecar alarm: an accepted Real still waiting for materialization
  // capacity after this long is counted and logged once. Alarm only: a
  // sustained physical deficit keeps it held and backpressure does the rest.
  static constexpr uint64_t kSplitSidecarHoldAlarmNs = 1000000000ull;
  // The split presenter owns three physical Synthetic chains plus one
  // bounded logical obligation that may wait for a chain to retire. Shared
  // device keeps the historical three-S logical budget.
  static constexpr uint32_t kLogicalSyntheticCapacity = kSyntheticPoolSize + 1;
  static constexpr uint32_t kPlannedSpaceTraceMaxEvents = 64;
  static constexpr uint64_t kPlannedSpaceTraceWindowNs = 10000000000ull;
  static constexpr uint64_t kPlannedSpaceTraceIssueWindow = 512;
  // Accepted Real may still be waiting in Capture, resident in the Real store,
  // or represented by a FinalOutput. Reserve logical identity for all three
  // bounded domains; Synthetic has its own independent capacity.
  static constexpr uint32_t kLogicalRealReserve =
      kCapturePoolSize + kPoolSize + kFinalOutputPoolSize;
  static constexpr uint32_t kLogicalOutputCapacity =
      kLogicalRealReserve + kLogicalSyntheticCapacity;
  static_assert(kFinalOutputPoolSize > 1);
  static_assert(kRealResidencyCapacity > 2);
  static_assert(kCapturePoolSize >= kMailboxCount);
  static_assert(kPoolSize > kCandidateReserveTarget);

  enum class SlotState : uint8_t {
    kFree,
    kTransferSubmitted,
    kReady,
    kCount,
  };

  enum class CaptureState : uint8_t {
    kFree,
    kIngressReserved,
    kReadyWaitingResidency,
    kCount,
  };

  enum class TerminalReason : uint8_t {
    kNone,
    kUnsupportedApi,
    kVulkanCapability,
    kHardwareBufferAllocation,
    kPoolExhausted,
    kSourceSubmitFailure,
    kInvalidHandoff,
    kTimelineQueryFailure,
    kOrderingFailure,
    kLifecycle,
    kInvalidGeometry,
    kInputExtentUnsupported,
    kPostProcessCapability,
    kPostProcessFailure,
    kGenerationFailure,
    kFinalOutputPoolExhausted,
    kLogicalOutputCapacityExhausted,
    kCaptureTransferCapability,
    kCaptureTransferFailure,
    kSourceSovereignty,
    kMainSurfaceAuthority,
  };

  enum class FinalOutputState : uint8_t {
    kFree,
    kPostProcessing,
    kReady,
    kTransactionApplied,
    kCount,
  };

  enum class CandidateKind : uint8_t { kReal, kSynthetic };
  enum class PairContract : uint8_t { kNormal, kTransitionRealOnly };

  // Phase 5 B2b. Who holds the FinalOutput pool at the instant a Post is
  // refused funding. The pacing gate in the ordered pump runs BEFORE the
  // completion gate, so a kReady slot whose logical has not yet reached
  // PlannedDispatchTimeNs is held by our own clock and by nothing else - we
  // deliberately never even ask whether its GPU work finished. That bucket is
  // the ONLY unequivocal policy hold. kReady with the time already passed is
  // either GPU-pending or blocked behind the ordered head; it is not claimed.
  struct FundingCensus {
    uint32_t free = 0;
    uint32_t postprocessing = 0;
    uint32_t ready_future_target = 0;
    uint32_t ready_time_passed = 0;
    uint32_t applied = 0;
  };

  // B2b asked "are ALL the holders ours?" and the answer in steady state is
  // structurally no - there is almost always a CompositorOwned and an Applied
  // in the pool - so policy was unreachable and everything fell into mixed.
  // That is the wrong question. The gate does not care who else is in the
  // pool; it cares whether THIS logical would still have been refused. So
  // B2c asks the but-for question instead: hand back ONLY the slots provably
  // held by our own dispatch clock, leave every other occupant exactly where
  // it is, and see whether the gate still refuses. Mixed disappears as a
  // class because a wait is now segmented in time rather than labelled once.
  enum class FundingWaitClass : uint8_t {
    kNone = 0,
    kPolicyButFor,      // returning our own pacing holds would have funded it
    kPhysicalResidual,  // it would have been refused anyway
    kCount,
  };

  // Phase 5 Better D. Why a C_pair sample was NOT allowed to teach D. The
  // decisive one is kBacklog: §5.14 showed the same 60 fps epoch sustains
  // ~30 ms when drained and ~100 ms when deep, with no cheaper service in
  // between, so the deep state is self-made backlog and not a causal
  // distribution. It is censored, not decomposed.
  enum class BetterDCensor : uint8_t {
    kUnarmed,           // no metronome or no output quantum yet
    kUnstable,          // not a stable pair
    kEpoch,             // pair admitted under another Source period epoch
    kGap,               // pair lattice distance, once RESOLVED, is not 1
    kSkippedOrRetired,  // forward-skipped, or presentation retired
    // B3b. The Source took more than half a period too long between A and B:
    // the pair measures the Source, not us. A throughput event, and F never
    // absorbs throughput deficit. The 1054 ms deficit that pinned F in the
    // first Arkham run was a pair spanning the game's own level load.
    kSourceLate,
    // Still in production while Synthetic production capacity was saturated.
    kBacklog,
    // B7: this S waited behind an earlier Post head whose FinalOutput
    // refusal was PolicyButFor. Its own funding wait is accounted separately.
    kFundingBacklog,
    // B4 removed kAnchorLag. It was built on a misread: in the deep Rayman
    // state the CENSORED backlog count read zero, but backlog_marked rose
    // 575 -> 1173 -> 2522 - the gap check simply ran first and took those
    // samples. The marker was working. And the anchor wait does not separate
    // deep from healthy: healthy Arkham waits 19-25 ms for its anchor at
    // O = 16.7, so the censor threw away up to 60 % of good samples and
    // brought the fast-sample bias of B3a straight back.
    kCount,
  };


  enum class LatencyMissCause : uint8_t {
    kTrueFeasibility,
    kResidencyPressure,
    kFinalOutputPressure,
    kFunding,
    kOrdering,
    kScheduler,
    kTransition,
  };

  enum class LatencyMissProvenance : uint8_t {
    kPredictedResidenceRisk,
    kGenerationSoftLate,
    kGenerationHardLate,
    kPostSoftLate,
    kPostHardLate,
    kFinalReadySoftLate,
    kSemanticHardLate,
    kCount,
  };

  enum ResourceStarvationBits : uint32_t {
    kResourceStarvationNone = 0,
    kResourceStarvationResidency = 1u << 0,
    kResourceStarvationFinalOutput = 1u << 1,
    kResourceStarvationFunding = 1u << 2,
    kResourceStarvationOrdering = 1u << 3,
    kResourceStarvationActiveGpu = 1u << 4,
  };

  // Experimental observation cadence, not controller thresholds. No wake is
  // scheduled: sample opportunistically at most every 100 ms. The first eight
  // consecutive live-deficit samples arm one qualified proposal per episode.
  // Unqualified Source leaves qualification pending. Recovery to H_target or
  // an observed epoch/lifecycle change rearms the episode.
  static constexpr uint64_t kUnifiedRunwayShadowSampleNs = 100000000ull;
  static constexpr uint32_t kUnifiedRunwayShadowFireSamples = 8;

  enum class RunwayShadowKind : uint8_t {
    kUnqualified,
    kPhaseOnly,
    kSlower,
    kFaster,
    kCount,
  };

  struct RunwayShadowFuture {
    uint64_t source_id = 0;
    uint64_t sequence_id = 0;  // Zero: incoming accepted Real, not enqueued yet.
    uint64_t pair_a = 0;
    uint64_t pair_b = 0;  // Only an existing, unanchored S is identified.
  };

  struct RunwayShadowProposal {
    RunwayShadowKind kind = RunwayShadowKind::kUnqualified;
    uint64_t time_ns = 0;
    uint64_t period_epoch = 0;
    uint64_t epoch_origin_ns = 0;
    uint64_t quantum_ns = 0;
    uint64_t proposed_period_ns = 0;
    uint64_t proposed_quantum_ns = 0;
    uint64_t first_target_ns = 0;
    // Additional future latency is exactly this displacement, not D plus H.
    uint64_t impulse_ns = 0;
    RunwayShadowFuture future;
    uint32_t pressure_mask = 0;
    uint32_t free_final = 0;
    uint32_t free_residency = 0;
    uint32_t capture_used = 0;
    uint32_t generation_jobs = 0;
    uint32_t post_jobs = 0;
    bool candidate_pressure = false;
    bool scheduler_present = false;
    bool timing_only_candidate = false;
  };

  struct UnifiedRunwayShadow {
    uint64_t last_sample_ns = 0;
    uint64_t last_issue_sequence = 0;
    uint64_t sample_total = 0;
    int64_t h_live_ns = 0;
    int64_t h_live_min_ns = 0;
    int64_t ordered_gap_ns = 0;
    int64_t reform_gap_ns = 0;
    // H stays signed through storage, sorting, quantiles and logging.
    std::array<int64_t, 128> h_samples_ns = {};
    size_t h_sample_count = 0;
    size_t h_sample_next = 0;
    PresenterSampleWindow<128> impulse_ns;
    uint64_t h_target_ns = 0;
    uint64_t h_deficit_ns = 0;
    uint64_t epoch_period_ns = 0;
    uint64_t observed_period_ns = 0;
    bool source_fresh = false;
    bool source_clustered = false;
    bool episode_active = false;
    bool episode_fired = false;
    bool qualification_pending = false;
    bool epoch_observed = false;
    uint64_t observed_period_epoch = 0;
    uint64_t observed_quantum_ns = 0;
    uint64_t observed_epoch_origin_ns = 0;
    uint64_t epoch_reset_total = 0;
    uint64_t episode_start_ns = 0;
    uint64_t episode_duration_ns = 0;
    int64_t episode_h_min_ns = 0;
    uint64_t episode_deficit_max_ns = 0;
    uint64_t episode_start_period_ns = 0;
    uint64_t episode_current_period_ns = 0;
    uint64_t episode_period_min_ns = 0;
    uint64_t episode_period_max_ns = 0;
    uint64_t episode_generation_late_start = 0;
    uint64_t episode_generation_late_delta = 0;
    uint32_t deficit_streak = 0;
    uint64_t episode_total = 0;
    uint64_t qualified_proposal_total = 0;
    // kUnqualified counts first qualification-pending crossings, not proposals.
    std::array<uint64_t, size_t(RunwayShadowKind::kCount)> kind_total = {};
    uint64_t physical_pressure_total = 0;
    uint64_t no_affectable_future_total = 0;
    uint64_t commit_boundary_observed_total = 0;
    bool awaiting_commit_boundary = false;
    RunwayShadowProposal proposal;

    void AddHSample(int64_t value_ns) {
      h_samples_ns[h_sample_next] = value_ns;
      h_sample_next = (h_sample_next + 1) % h_samples_ns.size();
      h_sample_count = std::min(h_sample_count + 1, h_samples_ns.size());
    }

    int64_t HQuantile(uint32_t numerator) const {
      if (!h_sample_count) {
        return 0;
      }
      auto sorted = h_samples_ns;
      std::sort(sorted.begin(), sorted.begin() + h_sample_count);
      const size_t index = std::min(
          h_sample_count - 1, ((h_sample_count - 1) * numerator + 99) / 100);
      return sorted[index];
    }
  };

  // DEMOTED to shadow by H12: E2B-R proved its safety and robust-P mechanisms
  // but failed runtime efficacy, falsifying micro-frequency mismatch as a
  // sufficient cause. It retains no behavioral authority over the clock; the
  // witness/qualification fields below stay purely diagnostic evidence that
  // the robust Source period is sound.
  struct E2BClockCanary {
    bool pending = false;
    RunwayShadowProposal witness;
    uint64_t episode = 0;
    uint64_t issue_sequence = 0;
    size_t source_window_count = 0;
    size_t robust_sample_count = 0;
    bool source_qualified = false;
    uint64_t raw_period_witness_ns = 0;
    uint64_t robust_period_witness_ns = 0;
    int64_t robust_delta_witness_ns = 0;
    int64_t residual_ns = 0;
    int32_t residual_sign = 0;
    uint64_t residual_blocks = 0;
    uint64_t residual_period_epoch = 0;
    uint64_t residual_quantum_ns = 0;
    uint64_t residual_origin_ns = 0;
    uint64_t residual_resets = 0;
    uint64_t source_reanchors = 0;
    uint64_t latency_reanchors = 0;
    uint64_t generation_late_start = 0;
    uint64_t generation_late_snapshot = 0;
    uint64_t request_total = 0;
    // Counterfactual: an armed witness that reached the commit boundary and
    // would have reformed the clock under the old behavioral canary.
    uint64_t would_apply_total = 0;
    uint64_t blocked_source_unqualified = 0;
    uint64_t blocked_residual_invalid = 0;
    uint64_t blocked_residual_period_incoherent = 0;
  };

  struct ResidualPhaseShadow {
    bool epoch_observed = false;
    uint64_t period_epoch = 0;
    uint64_t quantum_ns = 0;
    uint64_t origin_ns = 0;
    int64_t residual_ns = 0;
    int64_t minimum_ns = 0;
    int64_t maximum_ns = 0;
    uint64_t block_period_ns = 0;
    uint64_t block_span_ns = 0;
    uint32_t block_intervals = 0;
    bool paused = false;
    uint64_t pause_begin_ns = 0;
    uint64_t pause_duration_ns = 0;
    uint64_t pause_total = 0;
    uint64_t resume_total = 0;
    uint64_t paused_interval_total = 0;
    bool fired = false;
    bool previous_h_valid = false;
    int64_t previous_h_ns = 0;
    uint64_t blocks_total = 0;
    uint64_t positive_block_total = 0;
    uint64_t negative_block_total = 0;
    uint64_t structural_reset_total = 0;
    uint64_t would_fire_positive_total = 0;
    uint64_t would_fire_negative_total = 0;
  };

  enum class SourceReanchorRequestReason : uint8_t {
    kNone,
    kPhaseDebt,
    kFrequencyConfirmation,
    kConfirmedTransition,
    kOther,
    kCount,
  };

  struct SourceReanchorProvenance {
    SourceReanchorRequestReason pending = SourceReanchorRequestReason::kNone;
    SourceReanchorRequestReason last_applied = SourceReanchorRequestReason::kNone;
    SourceReanchorRequestReason last_overwritten =
        SourceReanchorRequestReason::kNone;
    std::array<uint64_t, size_t(SourceReanchorRequestReason::kCount)> requests = {};
    std::array<uint64_t, size_t(SourceReanchorRequestReason::kCount)> applied = {};
    uint64_t overwritten_total = 0;
    uint64_t equal_quantum_noop_total = 0;
  };

  enum class SyntheticDropReason : uint8_t {
    kNoHistory,
    kPeriodNotArmed,
    kDiscontinuity,
    kNoSyntheticSlot,
    kNoLogicalCapacity,
    kPredictedRealRisk,
    kGenerationFailure,
    kGenerationLate,
    kNoFinalOutput,
    kPostLate,
    kMetronomeLate,
    kLifecycleEpoch,
    kSourceProtection,
    kServiceBudget,
    kResidenceCutoff,
    kResidencyPressure,
    kRealBacklog,
    // H12 lifecycle reclamation: the pair's A can no longer become or find a
    // valid semantic commitment, so this S can never be anchored. Causal and
    // identity-derived; never an elapsed-time timeout.
    kStructuralOrphan,
    // H13 successful outcome, NOT a failure. The S was admitted, produced
    // correctly, its presentation position was already resolved as a semantic
    // HOLD, Production then completed, and this is the final resource
    // retirement. It is deliberately distinct from kMetronomeLate /
    // kGenerationLate / kPostLate so the drop line cannot read a correct H13
    // result as a scheduler, Generation, Post or metronome failure.
    kPresentationRetired,
    kCount,
  };

  struct Slot {
    std::atomic<SlotState> state{SlotState::kFree};
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    bool ever_written = false;
    std::shared_ptr<HandoffReadyObservation> handoff_ready;
    std::atomic<uint64_t> source_id{0};
    std::atomic<uint64_t> issue_time_ns{0};
    // Phase 5 commit B1: the sidecar PROMOTION HOLD of this Real - the
    // interval from split_sidecar_hold_begin_ns to promotion, which is the
    // part that actually blocked it. Not full sidecar residence, which runs
    // from armed to exit and is a wider interval. Stamped at promotion and
    // read by the S that pairs with it, travelling the same route as
    // issue_time_ns so the subtraction stays sample-aligned.
    std::atomic<uint64_t> policy_wait_ns{0};
    std::atomic<uint64_t> publish_time_ns{0};
    std::atomic<uint64_t> timeline_value{0};
    std::atomic<uint64_t> generation{0};
    VkExtent2D source_content_extent = {};
    VkExtent2D storage_content_extent = {};
    uint32_t display_aspect_ratio_x = 0;
    uint32_t display_aspect_ratio_y = 0;
    bool is_8bpc = false;
    uint64_t ready_time_ns = 0;
    uint64_t transfer_signal_value = 0;
    uint64_t transfer_submit_time_ns = 0;
    uint64_t transfer_first_poll_begin_time_ns = 0;
    // Capture observation: when a non-waiting sync_fd poll last found this
    // transfer pending and first found it signaled. Observation only; no
    // state follows from them.
    uint64_t observed_pending_time_ns = 0;
    uint64_t observed_ready_time_ns = 0;
    uint64_t apply_time_ns = 0;
    uint32_t owner_refs = 0;
    bool runtime_admitted = false;
    uint32_t ingress_context_index = UINT32_MAX;
    // Timeline the current content's capture transfer signaled (the shared
    // Capture-to-Residency timeline, or this slot's own with V2-2). GPU
    // consumers wait on (ready_timeline, timeline_value); both are set and
    // cleared together.
    VkSemaphore ready_timeline = VK_NULL_HANDLE;
    // Completion owner of this Residency slot.
    // - Timeline (V2-2): created with the
    //   presenter and kept across Residency reallocation. Its previous point
    //   belongs to the slot's previous capture transfer, which was observed
    //   complete before the slot could become free again.
    // - sync_fd (C2): signaled by the
    //   existing capture-transfer submit and exported only after queue
    //   acceptance; it never owns lifecycle or readiness authority.
    ZeroFGCompletionOwner completion;
  };

  struct CaptureSlot {
    std::atomic<CaptureState> state{CaptureState::kFree};
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    bool ever_written = false;
    uint64_t source_id = 0;
    uint64_t reserve_time_ns = 0;
    uint64_t issue_time_ns = 0;
    uint64_t publish_time_ns = 0;
    // Timeline and value the ingress copy into this capture signaled: the
    // shared Source-copy timeline, or its ingress context's own (C3).
    VkSemaphore source_timeline = VK_NULL_HANDLE;
    uint64_t source_timeline_value = 0;
    uint64_t generation = 0;
    VkExtent2D source_content_extent = {};
    VkExtent2D storage_content_extent = {};
    uint32_t display_aspect_ratio_x = 0;
    uint32_t display_aspect_ratio_y = 0;
    bool is_8bpc = false;
    uint32_t ingress_context_index = UINT32_MAX;
  };

  struct IngressCopyContext {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    bool inflight = false;
    uint32_t capture_index = UINT32_MAX;
    uint64_t source_timeline_value = 0;
    uint64_t capture_transfer_retirement_value = 0;
    SourcePublicationTelemetry publication;
    std::shared_ptr<void> source_image_lifetime;
    std::shared_ptr<HandoffReadyObservation> handoff_ready;
    uint64_t submit_begin_ns = 0;
    // Timeline the dependent capture transfer signaled with
    // capture_transfer_retirement_value (the shared one, or a Residency
    // slot's own with V2-2).
    VkSemaphore capture_transfer_retirement_timeline = VK_NULL_HANDLE;
    // Completion owner of this ingress context. Its timeline exists only on
    // device B (C3, per-context ingress). The context is recycled only
    // through RetireIngressContext, after the capture transfer that waited on
    // it was observed complete.
    ZeroFGCompletionOwner completion;
  };

  // Cadence of the telemetry-only capture observation poll.
  static constexpr uint64_t kCaptureObservationIntervalNs = 1000000;

  struct IncomingRealAuthority {
    uint64_t source_id = 0;
    uint64_t publication_id = 0;
    // Phase 0 sensor 1: armed -> exit, the accepted Real's true end-to-end
    // residence. The pre-existing split_sidecar_hold_ns starts only once the
    // Real is found in residency, so it measures a subinterval.
    uint64_t armed_ns = 0;

    bool occupied() const { return source_id != 0; }
    void Reset() {
      source_id = 0;
      publication_id = 0;
      armed_ns = 0;
    }
  };

  // Phase 0 sensor 7: why an occupied sidecar is not being promoted. The
  // decisive split is kPresentationResidence against kProductionFrontier:
  // both hold the same logical Synthetic tickets, but only the first is a
  // clock-derived hold, and it is blocker #20's signature. The success
  // criterion after phase 2 is that its accrued time is zero.
  enum class SidecarHoldCause : uint8_t {
    kNone,
    // The accepted Real's capture has not reached residency yet. Physical.
    kAwaitingResidency,
    // Materialization capacity is taken by S still in Generation/Post.
    kProductionFrontier,
    // Materialization capacity is taken by S that finished Post and are only
    // waiting for their tick. Policy: presentation residence as production
    // pressure.
    kPresentationResidence,
    // Logical output identity exhausted, independent of Synthetic capacity.
    kLogicalIdentity,
    kCount,
  };

  // Phase 0 sensor 1: how an occupied sidecar stopped being occupied.
  enum class SidecarExitReason : uint8_t {
    kPromoted,
    // Lifecycle/fail-open drain discarded it. Blocker #23: this must reach
    // zero, and until it does it is the accepted-Real destruction counter.
    kLifecycleDrain,
    kReinit,
    kCount,
  };

  // Phase 0 sensor 6: PresenterWakeReason::kGpuPoll is one bucket for six
  // distinct mechanisms (D8-1). Any cold-wake margin sized from it averages
  // all six. The reason enum is left alone; this rides alongside it.
  enum class GpuPollSource : uint8_t {
    kNone,
    kIngressRetry,
    kCaptureReclaim,
    kGenerationObservation,
    kPostFence,
    kCount,
  };

  static const char* SidecarHoldCauseName(SidecarHoldCause cause) {
    switch (cause) {
      case SidecarHoldCause::kAwaitingResidency:
        return "awaiting_residency";
      case SidecarHoldCause::kProductionFrontier:
        return "production_frontier";
      case SidecarHoldCause::kPresentationResidence:
        return "presentation_residence";
      case SidecarHoldCause::kLogicalIdentity:
        return "logical_identity";
      default:
        return "none";
    }
  }

  enum class IngressSubmitResult : uint8_t {
    kSubmitted,
    kQueueBusy,
    kInvalid,
    kVulkanError,
  };

  struct FinalOutputSlot {
    std::atomic<FinalOutputState> state{FinalOutputState::kFree};
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint64_t source_id = 0;
    uint64_t sequence_id = 0;
    uint64_t pair_a_source_id = 0;
    uint64_t latency_epoch = 0;
    CandidateKind candidate_kind = CandidateKind::kReal;
    uint64_t semantic_target_time_ns = 0;
    uint64_t target_time_ns = 0;
    uint64_t semantic_epoch_origin_ns = 0;
    uint64_t semantic_output_quantum_ns = 0;
    uint64_t semantic_tick_index = 0;
    uint64_t apply_time_ns = 0;
    // Main Surface Authority: the Post's completion timeline point, which the
    // egress copy waits on. Valid from Post submit until the slot is free.
    VkSemaphore post_completion_semaphore = VK_NULL_HANDLE;
    uint64_t post_completion_value = 0;
    bool semantic_forward_skipped = false;
    uint64_t operating_latency_ns = 0;
    uint64_t final_ready_time_ns = 0;
    uint64_t final_ready_soft_deadline_ns = 0;
    uint64_t dispatch_lead_ns = 0;
    uint64_t actual_apply_lead_ns = 0;
    uint64_t ordered_ready_wait_ns = 0;
    // An early Real post may evacuate Residency to unblock a Capture that is
    // physically ready but has nowhere to land. Each physical FinalOutput
    // carries its own charge until it enters the transaction stream; another
    // free FinalOutput may independently fund another required evacuation.
    std::atomic<bool> residency_recovery_debt{false};
    // Telemetry only: when the presenter last observed this slot free again
    // (its release fence ready), who occupied it, and whether the compositor
    // returned a real release fence. Zero free time means never recycled.
    uint64_t free_time_ns = 0;
    CandidateKind last_released_kind = CandidateKind::kReal;
    bool last_release_fenced = false;
  };

  struct OutputCandidate {
    CandidateKind kind = CandidateKind::kReal;
    // Pair contract is fixed when the pair is created. TransitionRealOnly is
    // not a compressed normal pair: no Synthetic obligation exists and its B
    // boundary is explicitly A + ld*O.
    PairContract pair_contract = PairContract::kNormal;
    // Set only when an uncommitted A later proves that another eligibility
    // floor prevented its explicit TransitionRealOnly boundary from moving
    // earlier. Counted only when this B receives its actual commitment.
    bool transition_planned_floor_was_blocked = false;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    VkExtent2D storage_extent = {};
    VkExtent2D content_extent = {};
    uint32_t display_aspect_ratio_x = 0;
    uint32_t display_aspect_ratio_y = 0;
    bool is_8bpc = false;
    uint32_t real_slot = UINT32_MAX;
    uint32_t synthetic_index = UINT32_MAX;
    uint64_t source_id = 0;
    uint64_t pair_a_source_id = 0;
    // Phase 0 sensor 8: A's issue time, carried so the causal sample stays a
    // single end-to-end measurement instead of two summed terms.
    uint64_t pair_a_issue_time_ns = 0;
    // Commit B1: the policy wait inside THIS pair's causal interval.
    uint64_t pair_a_policy_wait_ns = 0;
    // Lattice distance of this candidate's A/B pair (S and B carry the same
    // value), decided once when the pair is first placed from A's commitment;
    // 0 means not yet placed. 1 <= value <= B.source_id - A.source_id. See
    // PhaseBoundedPairDistance().
    uint64_t pair_lattice_distance = 0;
    uint64_t issue_time_ns = 0;
    uint64_t publish_time_ns = 0;
    uint64_t ready_time_ns = 0;
    // Synthetic is committed to a semantic opportunity at admission. A Real
    // is accepted with only an absolute, Source-derived eligibility floor;
    // its epoch, nominal target and executive target are all assigned together
    // only when it becomes the ordered head.
    uint64_t earliest_eligible_time_ns = 0;
    uint64_t semantic_target_time_ns = 0;
    uint64_t target_time_ns = 0;
    uint64_t semantic_epoch_origin_ns = 0;
    uint64_t semantic_output_quantum_ns = 0;
    uint64_t semantic_tick_index = 0;
    uint64_t sequence_id = 0;
  };

  struct EventBridge {
    std::mutex mutex;
    std::condition_variable condition;
    // Counting wakeups remain latched even when a producer signals between the
    // presenter's last pump and its wait. Source only calls release(): it never
    // acquires or waits on the presenter mutex.
    std::counting_semaphore<> presenter_wake{0};
    bool alive = true;
  };

  static void SignalPresenterWake(
      const std::shared_ptr<EventBridge>& event_bridge, bool wake_all = false) {
    if (!event_bridge) {
      return;
    }
    event_bridge->presenter_wake.release();
    if (wake_all) {
      event_bridge->condition.notify_all();
    }
  }

  // §5.20 diagnostic-only trace for one faster Source transition. It records
  // event-local placement/frontier evidence; it has no scheduling authority.
  struct PlannedSpaceTrace {
    bool fast_evidence_valid = false;
    bool fast_evidence_delay_exact = true;
    bool started = false;
    bool active = false;
    bool summary_logged = false;
    bool ordered_refusal_seen = false;
    bool new_quantum_issue_seen = false;
    uint64_t fast_evidence_time_ns = 0;
    uint64_t fast_evidence_issue_sequence = 0;
    uint64_t fast_evidence_last_issue_sequence = 0;
    uint64_t fast_evidence_interval_ns = 0;
    uint64_t fast_evidence_bp_wait_ns = 0;
    uint64_t fast_evidence_old_period_ns = 0;
    uint64_t fast_evidence_delay_ns = 0;
    uint64_t fast_evidence_delay_issues = 0;
    uint64_t fast_evidence_missing_issues = 0;
    uint64_t fast_evidence_censored_issues = 0;
    uint32_t fast_evidence_clean_count = 0;
    uint64_t fast_evidence_frontier = 0;
    uint64_t fast_reset_observed_ns = 0;
    uint64_t fast_reset_issue_sequence = 0;
    uint64_t fast_reset_interval_ns = 0;
    uint32_t fast_sequence_event_log_total = 0;
    uint64_t start_time_ns = 0;
    uint64_t start_issue_sequence = 0;
    uint64_t confirmation_time_ns = 0;
    uint64_t confirmation_issue_sequence = 0;
    uint64_t confirmation_source_id = 0;
    uint64_t confirmation_period_epoch = 0;
    uint32_t frontier_at_confirmation = 0;
    uint32_t live_chains_at_confirmation = 0;
    uint64_t actual_compacted_at_confirmation = 0;
    uint64_t old_period_ns = 0;
    uint64_t new_period_ns = 0;
    uint64_t old_quantum_ns = 0;
    uint64_t new_quantum_ns = 0;
    uint64_t new_quantum_issue_source_id = 0;
    uint64_t new_quantum_issue_ns = 0;
    uint64_t first_ordered_refusal_time_ns = 0;
    uint32_t live_chains_at_first_ordered_refusal = 0;
    uint32_t frontier_at_first_ordered_refusal = UINT32_MAX;
    uint64_t first_ordered_refusal_source_id = 0;
    uint64_t first_ordered_refusal_target_ns = 0;
    uint32_t frontier_at_finish = 0;
    uint32_t live_chains_at_finish = 0;
    uint32_t later_s_refill_3_to_4_total = 0;
    uint64_t natural_source_gap_max_ns = 0;
    uint64_t natural_source_gap_max_issue_sequence = 0;
    uint32_t episode_clean_source_interval_total = 0;
    uint32_t episode_censored_source_interval_total = 0;
    uint32_t event_count = 0;
    uint32_t last_frontier = UINT32_MAX;
    uint32_t frontier_levels_seen = 0;
    std::array<uint64_t, 5> frontier_first_time_ns = {};
    std::array<uint64_t, 5> frontier_first_issue_sequence = {};
    std::array<uint64_t, 5> frontier_first_source_id = {};
    std::array<uint64_t, 5> frontier_first_logical_sequence = {};
    uint32_t prospective_s_opportunities = 0;
    uint32_t prospective_s_logged = 0;
    std::array<uint64_t, 8> prospective_s_source_ids = {};
  };

  struct PhysicalOperatingPointObservation {
    uint64_t raw_period_ns = 0;
    uint32_t frontier = 0;
    uint32_t live_chains = 0;
    uint32_t free_final_outputs = 0;
    bool bp = false;
    bool post_bp = false;
    bool clean = false;
    bool hold = false;
    bool real_final_ready_before_target = false;
    bool downstream_pressure = false;
  };

  enum class PhysicalOperatingPointState : uint8_t {
    kIdle,
    kCandidate,
    kConfirmed,
  };

  enum class SourceRateProbeReason : uint8_t {
    kNormal,
    kPhysicalOperatingPointRecovery,
  };

  enum class TransitionPlannedSpaceState : uint8_t {
    kIdle,
    kFastSequence,
    kPostConfirmDrain,
  };

  // Who commands a Source-period change. kNormal keeps the global regime gate
  // (more than a fifth of P). The §7 authorities use §7's own distinction rule
  // and only with their own state prerequisites; see
  // SourcePeriodTransitionAuthorized().
  enum class SourcePeriodAuthority : uint8_t {
    // Clean transition window or a normal rate probe. Learned P follows.
    kNormal,
    // §7: first physical operating point, from a maintained candidate.
    kPhysicalInitial,
    // §7: a slower physical operating point while one is confirmed.
    kPhysicalRefinement,
    // §7 recovery probe: faster than the delivered physical P but still
    // physically distinct from learned P. §7 stays confirmed.
    kPhysicalRecoveryPartial,
    // §7 recovery probe back inside the learned regime. §7 exits.
    kPhysicalRecoveryFull,
  };

  enum class H1cResetReason : uint8_t {
    kRawSlow,
    kCleanNonFast,
    kIssueGap,
    kTimeout,
    kRecoveryProbe,
    kPeriodTransition,
    kCount,
  };

  enum class PhysicalOperatingPointAbortReason : uint8_t {
    // Raw median fell below the maintenance ratio.
    kEvidenceRecovered,
    // Censorship or downstream-pressure floor lost while raw is still slow.
    kPressureReleased,
    // Raw median outside the physical plausibility range.
    kImplausible,
    kIssueBudget,
    kPeriodNotDistinct,
    kRateProbeActive,
    kRecoveryProbe,
    // Another authority changed the delivered P under a live candidate.
    kReferenceChanged,
    kCount,
  };

  enum class LogicalOutputState : uint8_t {
    kFree,
    // RETIRED by H12 and no longer reachable: admission now always enters
    // kWaitingGeneration because production does not wait for the semantic
    // anchor. The value is preserved so state-indexed telemetry arrays keep
    // their historical indices; it must never regain production authority.
    kWaitingPairAnchor,
    kWaitingGeneration,
    kWaitingPost,
    kPostSubmitted,
    kFinalReady,
    kDropped,
    kCount,
  };

  // The Post lifecycle may retain its historical readiness behavior when no
  // CPU-visible fence is available, but only explicit completion evidence is
  // allowed to teach the optional B1 latency controller.
  enum class PostCompletionEvidence : uint8_t {
    kUnprovenFallback,
    kObservationError,
    kSyncFile,
    kTimeline,
    kTeardownIdle,
  };

  enum class ApplyFinalOutputResult : uint8_t {
    kApplied,
    kFailure,
    // The egress is not ready (B not producing yet, or the ring is full):
    // retry the same ordered head later.
    kNotReady,
  };

  enum class BlockingOperation : uint8_t {
    kNone,
    kCapturePoll,
    kGenerationPoll,
    kPostPoll,
    kIngressSubmit,
    kCaptureSubmit,
    kRecoveryPost,
    kNormalPost,
    kGeneration,
    kCount,
  };

  enum class PresenterWakeReason : uint8_t {
    kGuardStart,
    kPlannedDispatch,
    kGpuPoll,
    kSemanticDeadline,
    kProducerEvent,
    kCount,
  };

  struct PresenterWakePlan {
    uint64_t deadline_ns = std::numeric_limits<uint64_t>::max();
    PresenterWakeReason reason = PresenterWakeReason::kProducerEvent;
  };

  struct DriverPollResult {
    bool invoked = false;
    bool progress = false;
    bool deferred_not_due = false;
  };

  struct LogicalOutput {
    LogicalOutputState state = LogicalOutputState::kFree;
    uint64_t token = 0;
    uint64_t sequence_id = 0;
    // Soft staging deadlines make work urgent and provide internal D/A
    // feedback. Synthetic target_time_ns is its soft nominal midpoint. The
    // historical pair-safe deadline is telemetry-only under H14; only a
    // factually due and FinalReady B may consume an unfinished S position.
    // Real target_time_ns remains the sovereign desired presentation target.
    uint64_t post_submit_soft_deadline_ns = 0;
    uint64_t final_ready_soft_deadline_ns = 0;
    uint64_t generation_submit_not_before_ns = 0;
    uint64_t generation_submit_soft_deadline_ns = 0;
    uint64_t post_submit_not_before_ns = 0;
    uint64_t dispatch_lead_ns = 0;
    uint64_t synthetic_pair_safe_deadline_ns = 0;
    // Frozen when the logical output is admitted. Later host-cost samples may
    // not rewrite its already-assigned semantic tick or deadlines. The sole
    // exception is a causal hard-ready miss: the sovereign Real is necessarily
    // deferred, and carries the newly proven lead into that future tick.
    uint64_t post_residence_budget_ns = 0;
    uint64_t generation_residence_budget_ns = 0;
    uint64_t staging_cushion_ns = 0;
    // First instant at which the accepted Post entered the ordered
    // FinalReady production state. GPU completion is tracked separately
    // below; this is not an Apply permission by itself.
    uint64_t final_ready_time_ns = 0;
    // Better D diagnostics: host-side boundaries of this logical S. In the
    // normal GPU chain, entering WaitingPost does not mean Generation's GPU
    // work completed; Post waits on its semaphore without a CPU observation.
    uint64_t generation_submit_time_ns = 0;
    uint64_t waiting_post_enter_time_ns = 0;
    uint64_t final_output_acquire_time_ns = 0;
    uint64_t post_submit_time_ns = 0;
    // Phase 5 B2b. Funding wait attributed to THIS logical: it opens only when
    // this logical concretely hit a funding gate, and closes the instant it
    // acquires a FinalOutput. A logical that was merely never selected because
    // it sits behind the head accrues nothing here - that wait is ordering,
    // and ordering must never be laundered into funding.
    uint64_t funding_wait_begin_ns = 0;
    uint64_t funding_wait_total_ns = 0;
    uint64_t funding_policy_wait_ns = 0;
    uint64_t funding_physical_wait_ns = 0;
    uint8_t funding_wait_segment_class = 0;
    // B7 provenance is sticky for this logical S, independent of its own
    // funding segments and of which Better D censor eventually wins.
    bool funding_backlog_marked = false;
    bool behind_physical = false;
    // Phase 5 Better D. This S was still in production - not yet physically
    // complete - at a moment the arbiter found Synthetic production capacity
    // saturated. Its C_pair carries queue residence behind earlier chains and
    // may not teach D. B4 widened it from "waited in kWaitingGeneration":
    // every entry into the deep state leaked S that were already past
    // Generation when the pool filled. Set by ScheduleGpuWork, read only by
    // Better D.
    bool in_production_during_saturation = false;
    // Phase 5 B3a. A C_pair sample taken while the pair was still unplaced
    // (pair_lattice_distance 0) is held here until the anchor resolves it,
    // instead of being called a gap. Scored exactly once.
    bool better_d_sample_pending = false;
    bool better_d_sample_scored = false;
    uint64_t better_d_pending_clean_ns = 0;
    uint64_t better_d_pending_completion_ns = 0;
    // Latency feedback judged for this S, exactly once: at FinalReady if it
    // was anchored by then, otherwise when its anchor arrives.
    bool latency_feedback_scored = false;
    // Host-side first lifecycle observation that Post is ready to retire/apply.
    // This may be a legacy fallback observation, so B1's physical proof and
    // score timestamp are stored separately. final_ready_time_ns is earlier:
    // it marks when the accepted Post entered ordered production state.
    uint64_t post_completion_ready_time_ns = 0;
    bool post_completion_sensor_recorded = false;
    // B1 uses only a physical completion proof for latency feedback. The
    // lifecycle timestamp above remains independent and preserves the normal
    // Post/output bookkeeping contract even for legacy fallback/error paths.
  uint64_t post_completion_proven_time_ns = 0;
    bool post_completion_evidence_observed = false;
    bool post_completion_physical_proven = false;
    bool post_completion_fallback_observed = false;
    bool post_completion_observation_error = false;
    bool latency_judgement_skipped_no_proof_recorded = false;
    uint64_t source_period_epoch = 0;
    uint64_t latency_epoch = 0;
    uint64_t operating_latency_ns = 0;
    bool stable_pair = false;
    bool stable_drop_recorded = false;
    bool semantic_forward_skipped = false;
    bool generation_jit_deferred_recorded = false;
    bool generation_residency_recovery_recorded = false;
    bool post_jit_deferred_recorded = false;
    bool synthetic_chain_warmup_submission = false;
    bool synthetic_nominal_late_recorded = false;
    bool synthetic_late_salvage_started = false;
    bool synthetic_safe_window_expired_recorded = false;
    bool real_early_transfer_recorded = false;
    bool real_early_transfer_block_recorded = false;
    // H12: production and presentation are orthogonal. A Synthetic admitted
    // before its pair A received a shallow commitment has no semantic contract
    // yet, so it must neither carry a provisional hard deadline nor teach any
    // temporal controller. These record that provenance for telemetry only.
    bool synthetic_unanchored_at_admit = false;
    bool synthetic_anchor_stage_recorded = false;
    uint64_t synthetic_admit_time_ns = 0;
    // H13: presentation expiry must not cancel admitted production. This flag
    // is orthogonal to the production state machine (WaitingGeneration ->
    // WaitingPost -> PostSubmitted -> FinalReady) and answers exactly one
    // question: may this S still be PRESENTED? Once true the ordered executive
    // has already committed its semantic HOLD and advanced past this position,
    // so the logical survives only as a production ticket.
    bool synthetic_presentation_retired = false;
    bool synthetic_generation_attempted = false;
    // Temporary V2 first-commit diagnostic. This stays with the logical
    // sequence so a repeated pass (if one ever becomes possible) is visible
    // without an unbounded sequence map.
    uint32_t previous_real_slot = UINT32_MAX;
    uint32_t current_real_slot = UINT32_MAX;
    OutputCandidate candidate;
    uint32_t final_output_index = UINT32_MAX;
  };

  struct GenerationJob {
    bool active = false;
    uint32_t synthetic_index = UINT32_MAX;
    uint32_t logical_index = UINT32_MAX;
    uint64_t logical_token = 0;
    uint32_t previous_real_slot = UINT32_MAX;
    uint32_t current_real_slot = UINT32_MAX;
    uint64_t next_observation_time_ns = 0;
    uint64_t chain_submit_time_ns = 0;
    bool chained_to_post = false;
    bool synthetic_chain_warmup_submission = false;
    GenerationResult submit;
  };

  struct PostJob {
    bool active = false;
    uint32_t final_output_index = UINT32_MAX;
    uint32_t logical_index = UINT32_MAX;
    uint64_t logical_token = 0;
    uint64_t next_observation_time_ns = 0;
    int completion_fence_fd = -1;
    bool timeline_fallback = false;
    bool release_output_on_completion = false;
    OutputCandidate candidate;
    PostProcessResult submit;
    bool residency_recovery_early_transfer = false;
    bool residency_recovery_compound_transfer = false;
    bool residency_recovery_post = false;
  };

  explicit Impl(VulkanDevice* device, bool elevated_priority,
                uint64_t cost_seed_ns,
                IngressSourceAcquireCallback acquire_ingress_source_callback,
                PaintConfigProvider config_provider,
                GenerationCallback generate_callback,
                GenerationPollCallback poll_generation_callback,
                SyntheticReleaseCallback release_synthetic_callback,
                GenerationShutdownCallback shutdown_generation_callback,
                PostProcessCallback process_callback,
                PostProcessReleaseCallback release_post_callback,
                PostProcessShutdownCallback shutdown_callback,
                PresenterDeviceDrainCallback drain_device_callback)
      : vulkan_device(device),
        elevated_presenter_priority(elevated_priority),
        synthetic_cost_seed_ns(cost_seed_ns),
        ingress_source_acquire_callback(
            std::move(acquire_ingress_source_callback)),
        paint_config_provider(std::move(config_provider)),
        generation_callback(std::move(generate_callback)),
        generation_poll_callback(std::move(poll_generation_callback)),
        synthetic_release_callback(std::move(release_synthetic_callback)),
        generation_shutdown_callback(
            std::move(shutdown_generation_callback)),
        post_process_callback(std::move(process_callback)),
        post_process_release_callback(std::move(release_post_callback)),
        post_process_shutdown_callback(std::move(shutdown_callback)),
        presenter_device_drain_callback(std::move(drain_device_callback)) {}

  ~Impl() {
    BeginSurfaceDisconnect();
    DestroySurfaceResourcesAfterSourceIdle();
    DestroyIngressContexts();
    if (timeline != VK_NULL_HANDLE) {
      vulkan_device->functions().vkDestroySemaphore(
          vulkan_device->device(), timeline, nullptr);
      timeline = VK_NULL_HANDLE;
    }
    if (capture_transfer_timeline != VK_NULL_HANDLE) {
      vulkan_device->functions().vkDestroySemaphore(
          vulkan_device->device(), capture_transfer_timeline, nullptr);
      capture_transfer_timeline = VK_NULL_HANDLE;
    }
    // V2-2 readiness timelines live as long as the shared one; teardown
    // waited for every transfer that signaled them.
    for (Slot& slot : slots) {
      slot.completion.DestroyTimeline(vulkan_device);
    }
  }

  static const char* TerminalReasonName(TerminalReason reason) {
    switch (reason) {
      case TerminalReason::kNone:
        return "none";
      case TerminalReason::kUnsupportedApi:
        return "unsupported_api";
      case TerminalReason::kVulkanCapability:
        return "vulkan_capability";
      case TerminalReason::kHardwareBufferAllocation:
        return "ahb_allocation";
      case TerminalReason::kPoolExhausted:
        return "capture_pool_exhausted";
      case TerminalReason::kSourceSubmitFailure:
        return "source_submit_failure";
      case TerminalReason::kInvalidHandoff:
        return "invalid_handoff";
      case TerminalReason::kTimelineQueryFailure:
        return "timeline_query_failure";
      case TerminalReason::kOrderingFailure:
        return "ordering_failure";
      case TerminalReason::kLifecycle:
        return "lifecycle";
      case TerminalReason::kInvalidGeometry:
        return "invalid_geometry";
      case TerminalReason::kInputExtentUnsupported:
        return "input_extent_unsupported";
      case TerminalReason::kPostProcessCapability:
        return "post_process_capability";
      case TerminalReason::kPostProcessFailure:
        return "post_process_failure";
      case TerminalReason::kGenerationFailure:
        return "generation_failure";
      case TerminalReason::kFinalOutputPoolExhausted:
        return "final_output_pool_exhausted";
      case TerminalReason::kLogicalOutputCapacityExhausted:
        return "logical_output_capacity_exhausted";
      case TerminalReason::kCaptureTransferCapability:
        return "capture_transfer_capability";
      case TerminalReason::kCaptureTransferFailure:
        return "capture_transfer_failure";
      case TerminalReason::kSourceSovereignty:
        return "source_sovereignty";
      case TerminalReason::kMainSurfaceAuthority:
        return "main_surface_authority";
    }
    return "unknown";
  }

  static const char* LogicalOutputStateName(LogicalOutputState state) {
    switch (state) {
      case LogicalOutputState::kFree:
        return "free";
      case LogicalOutputState::kWaitingPairAnchor:
        return "waiting_pair_anchor";
      case LogicalOutputState::kWaitingGeneration:
        return "waiting_generation";
      case LogicalOutputState::kWaitingPost:
        return "waiting_post";
      case LogicalOutputState::kPostSubmitted:
        return "post_submitted";
      case LogicalOutputState::kFinalReady:
        return "final_ready";
      case LogicalOutputState::kDropped:
        return "dropped";
      case LogicalOutputState::kCount:
        break;
    }
    return "unknown";
  }

  static const char* PostFailureStageName(PostFailureStage stage) {
    switch (stage) {
      case PostFailureStage::kNone:
        return "none";
      case PostFailureStage::kRequestValidation:
        return "request_validation";
      case PostFailureStage::kContextCommandPool:
        return "context_command_pool";
      case PostFailureStage::kContextCommandBuffer:
        return "context_command_buffer";
      case PostFailureStage::kContextTimeline:
        return "context_timeline";
      case PostFailureStage::kDescriptorPool:
        return "descriptor_pool";
      case PostFailureStage::kDescriptorSets:
        return "descriptor_sets";
      case PostFailureStage::kFinalRenderPass:
        return "final_render_pass";
      case PostFailureStage::kFinalFormat:
        return "final_format";
      case PostFailureStage::kLogicalInput:
        return "logical_input";
      case PostFailureStage::kPaintFlow:
        return "paint_flow";
      case PostFailureStage::kIntermediateResource:
        return "intermediate_resource";
      case PostFailureStage::kIntermediateFramebuffer:
        return "intermediate_framebuffer";
      case PostFailureStage::kIntermediatePipeline:
        return "intermediate_pipeline";
      case PostFailureStage::kFinalPipeline:
        return "final_pipeline";
      case PostFailureStage::kFinalFramebuffer:
        return "final_framebuffer";
      case PostFailureStage::kCommandReset:
        return "command_reset";
      case PostFailureStage::kCommandBegin:
        return "command_begin";
      case PostFailureStage::kCommandEnd:
        return "command_end";
      case PostFailureStage::kQueueSubmit:
        return "queue_submit";
      case PostFailureStage::kAcquireFenceExport:
        return "acquire_fence_export";
    }
    return "unknown";
  }

  static const char* GenerationFailureStageName(
      GenerationFailureStage stage) {
    switch (stage) {
      case GenerationFailureStage::kNone:
        return "none";
      case GenerationFailureStage::kRequestValidation:
        return "request_validation";
      case GenerationFailureStage::kAdapter:
        return "adapter";
      case GenerationFailureStage::kResize:
        return "resize";
      case GenerationFailureStage::kSyntheticPool:
        return "synthetic_pool";
      case GenerationFailureStage::kCommandReset:
        return "command_reset";
      case GenerationFailureStage::kCommandBegin:
        return "command_begin";
      case GenerationFailureStage::kRecord:
        return "record";
      case GenerationFailureStage::kCommandEnd:
        return "command_end";
      case GenerationFailureStage::kPreparationYield:
        return "preparation_yield";
      case GenerationFailureStage::kQueueBusy:
        return "queue_busy";
      case GenerationFailureStage::kQueueSubmit:
        return "queue_submit";
      case GenerationFailureStage::kTimelineQuery:
        return "timeline_query";
    }
    return "unknown";
  }

  bool Initialize() {
    if (!ingress_source_acquire_callback || !generation_callback ||
        !generation_poll_callback ||
        !synthetic_release_callback || !post_process_callback ||
        !post_process_release_callback) {
      return false;
    }
    // The physical completion profile is fixed: bounded polls, sync_file
    // observation, per-slot and per-context completion timelines, three
    // Synthetic chains, all on device B.
    const VulkanDevice::Extensions& extensions = vulkan_device->extensions();
    if (!extensions.ext_ANDROID_external_memory_android_hardware_buffer ||
        !extensions.ext_KHR_external_semaphore_fd ||
        !vulkan_device->properties().timelineSemaphore ||
        !vulkan_device->vkGetAndroidHardwareBufferPropertiesANDROID() ||
        !vulkan_device->vkGetSemaphoreCounterValue() ||
        !vulkan_device->vkGetSemaphoreFdKHR()) {
      XELOGW(
          "ZeroFGC0: unavailable Vulkan capabilities ahb={} sync_fd={} "
          "timeline={} ahb_query={} timeline_query={} fd_export={}",
          extensions.ext_ANDROID_external_memory_android_hardware_buffer,
          extensions.ext_KHR_external_semaphore_fd,
          vulkan_device->properties().timelineSemaphore,
          bool(vulkan_device->vkGetAndroidHardwareBufferPropertiesANDROID()),
          bool(vulkan_device->vkGetSemaphoreCounterValue()),
          bool(vulkan_device->vkGetSemaphoreFdKHR()));
      return false;
    }
    const VulkanInstance* instance = vulkan_device->vulkan_instance();
    auto get_external_semaphore_properties =
        PFN_vkGetPhysicalDeviceExternalSemaphoreProperties(
            instance->functions().vkGetInstanceProcAddr(
                instance->instance(),
                "vkGetPhysicalDeviceExternalSemaphoreProperties"));
    if (!get_external_semaphore_properties) {
      get_external_semaphore_properties =
          PFN_vkGetPhysicalDeviceExternalSemaphoreProperties(
              instance->functions().vkGetInstanceProcAddr(
                  instance->instance(),
                  "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR"));
    }
    if (!get_external_semaphore_properties) {
      XELOGW("ZeroFGC0: external semaphore capability query unavailable");
      return false;
    }
    VkPhysicalDeviceExternalSemaphoreInfo external_info = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
    external_info.handleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    VkExternalSemaphoreProperties external_properties = {
        VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    get_external_semaphore_properties(vulkan_device->physical_device(),
                                      &external_info,
                                      &external_properties);
    const VkExternalSemaphoreFeatureFlags required_external_features =
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT;
    if ((external_properties.externalSemaphoreFeatures &
         required_external_features) != required_external_features ||
        !(external_properties.compatibleHandleTypes &
          VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT)) {
      XELOGW("ZeroFGC0: Android sync_file semaphore export is unsupported");
      return false;
    }

    VkSemaphoreTypeCreateInfo timeline_type = {
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_type.initialValue = 0;
    VkSemaphoreCreateInfo semaphore_info = {
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphore_info.pNext = &timeline_type;
    if (vulkan_device->functions().vkCreateSemaphore(
            vulkan_device->device(), &semaphore_info, nullptr,
            &timeline) != VK_SUCCESS) {
      XELOGW("ZeroFGC0: failed to create Source-copy readiness timeline");
      return false;
    }
    vulkan_device->SetObjectName(VK_OBJECT_TYPE_SEMAPHORE, timeline,
"ZeroFG Source-copy readiness");
    if (vulkan_device->functions().vkCreateSemaphore(
            vulkan_device->device(), &semaphore_info, nullptr,
            &capture_transfer_timeline) != VK_SUCCESS) {
      XELOGW("ZeroFGC0: failed to create Capture-to-Residency timeline");
      return false;
    }
    vulkan_device->SetObjectName(VK_OBJECT_TYPE_SEMAPHORE,
                                 capture_transfer_timeline,
"ZeroFG Capture-to-Residency readiness");
    // V2-2: one readiness timeline per Residency slot; ZeroFG needs them all.
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      const VkResult result = slots[i].completion.CreateTimeline(
          vulkan_device, "ZeroFG Residency readiness " + std::to_string(i));
      if (result != VK_SUCCESS) {
        ++capture_owner_stats.timeline_create_failures;
        return false;
      }
    }
    // Main Surface Authority is ZeroFG's only output: device B must be the WSI
    // profile. Without it ZeroFG stays off (normal XenDroid presenter).
    if (!vulkan_device->is_zerofg_presenter_device()) {
      XELOGE(
          "ZeroFGMainSurface refused reason=no_main_surface_device_B "
          "fallback=native_A");
      return false;
    }
    XELOGI("ZeroFGMainSurface egress_queue=B:{} timing_available={}",
           vulkan_device->queue_index_zerofg_main_surface_present(),
           vulkan_device->vkGetPastPresentationTimingGOOGLE() != nullptr);
    // ZeroFG pacing section 7 is off since 1.0.1. It was built for ZeroFG's
    // work stuck behind the game's on the GPU, which GPU priority solved;
    // what was left of it was a 20 fps lock at the capacity of the pools
    // (~540 ms) under ZeroFG stalls (2026-10-08 DEV runs).
    split_physical_operating_point_enabled = false;

    if (!InitializeIngressContexts()) {
      XELOGW("ZeroFGC0: failed to initialize dormant D6 ingress contexts");
      return false;
    }

    return true;
  }

  bool AllocateCaptureSlot(CaptureSlot& slot, uint32_t slot_index,
                           VkExtent2D slot_extent) {
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    image_info.extent = {slot_extent.width, slot_extent.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!util::CreateDedicatedAllocationImage(
            vulkan_device, image_info, util::MemoryPurpose::kDeviceLocal,
            slot.image, slot.memory)) {
      return false;
    }

    VkImageViewCreateInfo view_info = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = slot.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    view_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY};
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    if (dfn.vkCreateImageView(device, &view_info, nullptr, &slot.view) !=
        VK_SUCCESS) {
      dfn.vkDestroyImage(device, slot.image, nullptr);
      dfn.vkFreeMemory(device, slot.memory, nullptr);
      slot.image = VK_NULL_HANDLE;
      slot.memory = VK_NULL_HANDLE;
      return false;
    }

    slot.ever_written = false;
    slot.state.store(CaptureState::kFree, std::memory_order_release);
    vulkan_device->SetObjectName(
        VK_OBJECT_TYPE_IMAGE, slot.image,
("ZeroFG Capture device-local " + std::to_string(slot_index))
            .c_str());
    return true;
  }

  void DestroyCaptureSlot(CaptureSlot& slot) {
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    if (slot.view != VK_NULL_HANDLE) {
      dfn.vkDestroyImageView(device, slot.view, nullptr);
      slot.view = VK_NULL_HANDLE;
    }
    if (slot.image != VK_NULL_HANDLE) {
      dfn.vkDestroyImage(device, slot.image, nullptr);
      slot.image = VK_NULL_HANDLE;
    }
    if (slot.memory != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, slot.memory, nullptr);
      slot.memory = VK_NULL_HANDLE;
    }
    slot.source_id = 0;
    slot.reserve_time_ns = 0;
    slot.issue_time_ns = 0;
    slot.publish_time_ns = 0;
    slot.source_timeline = VK_NULL_HANDLE;
    slot.source_timeline_value = 0;
    slot.source_content_extent = {};
    slot.storage_content_extent = {};
    slot.display_aspect_ratio_x = 0;
    slot.display_aspect_ratio_y = 0;
    slot.is_8bpc = false;
    slot.ingress_context_index = UINT32_MAX;
    slot.state.store(CaptureState::kFree, std::memory_order_release);
  }

  bool InitializeIngressContexts() {
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    for (uint32_t i = 0; i < kCapturePoolSize; ++i) {
      IngressCopyContext& context = ingress_copy_contexts[i];
      VkCommandPoolCreateInfo pool_info = {
          VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
      pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      pool_info.queueFamilyIndex =
          vulkan_device->queue_family_graphics_compute();
      if (dfn.vkCreateCommandPool(device, &pool_info, nullptr,
                                  &context.command_pool) != VK_SUCCESS) {
        return false;
      }
      VkCommandBufferAllocateInfo allocate_info = {
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      allocate_info.commandPool = context.command_pool;
      allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocate_info.commandBufferCount = 1;
      if (dfn.vkAllocateCommandBuffers(device, &allocate_info,
                                       &context.command_buffer) != VK_SUCCESS) {
        return false;
      }
      if (context.completion.CreateTimeline(
              vulkan_device, "ZeroFG D6 ingress timeline " +
                                 std::to_string(i)) != VK_SUCCESS) {
        return false;
      }
      vulkan_device->SetObjectName(
          VK_OBJECT_TYPE_COMMAND_POOL, context.command_pool,
          ("ZeroFG D6 ingress pool " + std::to_string(i)).c_str());
      vulkan_device->SetObjectName(
          VK_OBJECT_TYPE_COMMAND_BUFFER, context.command_buffer,
          ("ZeroFG D6 ingress commands " + std::to_string(i))
              .c_str());
    }
    return true;
  }

  void DestroyIngressContexts() {
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    for (IngressCopyContext& context : ingress_copy_contexts) {
      context.source_image_lifetime.reset();
      context.handoff_ready.reset();
      context.publication = {};
      context.inflight = false;
      context.capture_index = UINT32_MAX;
      context.source_timeline_value = 0;
      context.capture_transfer_retirement_value = 0;
      context.capture_transfer_retirement_timeline = VK_NULL_HANDLE;
      context.completion.DestroyTimeline(vulkan_device);
      if (context.command_pool != VK_NULL_HANDLE) {
        dfn.vkDestroyCommandPool(device, context.command_pool, nullptr);
        context.command_pool = VK_NULL_HANDLE;
        context.command_buffer = VK_NULL_HANDLE;
      }
    }
  }

  void WaitForIngressGpuCompletionForTeardown() {
    // Teardown may wait (rule 7). Every ingress copy and capture transfer the
    // presenter still owns is waited on the timeline it signaled: a shared
    // timeline once with its highest value, an owner timeline (C3 ingress
    // context, V2-2 Residency slot) on its own.
    std::array<VkSemaphore, kCapturePoolSize + kPoolSize + 2> semaphores = {};
    std::array<uint64_t, kCapturePoolSize + kPoolSize + 2> values = {};
    uint32_t count = 0;
    uint64_t shared_source_value = 0;
    uint64_t shared_transfer_value = 0;
    for (const IngressCopyContext& context : ingress_copy_contexts) {
      if (!context.inflight) {
        continue;
      }
      if (context.source_timeline_value) {
        const VkSemaphore source_timeline =
            context.completion.signaled_timeline();
        if (source_timeline == VK_NULL_HANDLE || source_timeline == timeline) {
          shared_source_value =
              std::max(shared_source_value, context.source_timeline_value);
        } else {
          semaphores[count] = source_timeline;
          values[count++] = context.source_timeline_value;
        }
      }
      // A transfer on a Residency slot's own timeline is waited through the
      // slot below.
      if (context.capture_transfer_retirement_value &&
          (context.capture_transfer_retirement_timeline == VK_NULL_HANDLE ||
           context.capture_transfer_retirement_timeline ==
               capture_transfer_timeline)) {
        shared_transfer_value = std::max(
            shared_transfer_value, context.capture_transfer_retirement_value);
      }
    }
    for (const Slot& residency : slots) {
      if (residency.state.load(std::memory_order_acquire) !=
              SlotState::kTransferSubmitted ||
          !residency.transfer_signal_value ||
          residency.ready_timeline == VK_NULL_HANDLE ||
          residency.ready_timeline == capture_transfer_timeline) {
        continue;
      }
      semaphores[count] = residency.ready_timeline;
      values[count++] = residency.transfer_signal_value;
    }
    if (shared_source_value) {
      semaphores[count] = timeline;
      values[count++] = shared_source_value;
    }
    if (shared_transfer_value) {
      semaphores[count] = capture_transfer_timeline;
      values[count++] = shared_transfer_value;
    }
    if (count && !vulkan_device->zerofg_teardown_idle()) {
      VkSemaphoreWaitInfo wait_info = {
          VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
      wait_info.semaphoreCount = count;
      wait_info.pSemaphores = semaphores.data();
      wait_info.pValues = values.data();
      const VkResult result = vulkan_device->vkWaitSemaphores()(
          vulkan_device->device(), &wait_info, UINT64_MAX);
      if (result != VK_SUCCESS) {
        XELOGW("ZeroFGC0: ingress teardown timeline wait failed: {}",
               int32_t(result));
      }
    }
    for (uint32_t i = 0; i < ingress_copy_contexts.size(); ++i) {
      RetireIngressContext(i, std::numeric_limits<uint64_t>::max());
    }
    for (Slot& residency : slots) {
      if (residency.state.load(std::memory_order_acquire) ==
          SlotState::kTransferSubmitted) {
        residency.completion.MarkRetired();
      }
    }
  }

  uint32_t FindFreeIngressContext() const {
    for (uint32_t i = 0; i < ingress_copy_contexts.size(); ++i) {
      if (!ingress_copy_contexts[i].inflight) {
        return i;
      }
    }
    return UINT32_MAX;
  }

  uint32_t CountInflightIngressContexts() const {
    uint32_t count = 0;
    for (const IngressCopyContext& context : ingress_copy_contexts) {
      count += context.inflight ? 1u : 0u;
    }
    return count;
  }

  void RetireIngressContext(uint32_t context_index,
                            uint64_t retirement_value) {
    if (context_index >= ingress_copy_contexts.size()) {
      return;
    }
    IngressCopyContext& context = ingress_copy_contexts[context_index];
    if (!context.inflight ||
        (context.capture_transfer_retirement_value &&
         retirement_value < context.capture_transfer_retirement_value)) {
      return;
    }
    context.source_image_lifetime.reset();
    context.handoff_ready.reset();
    context.publication = {};
    context.inflight = false;
    context.capture_index = UINT32_MAX;
    context.source_timeline_value = 0;
    context.capture_transfer_retirement_value = 0;
    context.capture_transfer_retirement_timeline = VK_NULL_HANDLE;
    context.submit_begin_ns = 0;
    context.completion.MarkRetired();
  }

  bool RecordIngressCopy(
      IngressCopyContext& context,
      const IngressSourcePublication& source, CaptureSlot& capture) {
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    const bool external = source.external_wait != VK_NULL_HANDLE;
    if (!source ||
        (source.layout != VulkanPresenter::kGuestOutputInternalLayout &&
         !(external &&
           source.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)) ||
        !source.extent.width || !source.extent.height ||
        capture.image == VK_NULL_HANDLE ||
        !capture.storage_content_extent.width ||
        !capture.storage_content_extent.height) {
      return false;
    }
    if (dfn.vkResetCommandPool(device, context.command_pool, 0) != VK_SUCCESS) {
      return false;
    }
    VkCommandBufferBeginInfo begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (dfn.vkBeginCommandBuffer(context.command_buffer, &begin_info) !=
        VK_SUCCESS) {
      return false;
    }
    const VkImageSubresourceRange color_range =
        util::InitializeSubresourceRange();
    VkImageMemoryBarrier barriers[2] = {};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VulkanPresenter::kGuestOutputInternalAccessMask;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].oldLayout = source.layout;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = source.image;
    barriers[0].subresourceRange = color_range;
    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].srcAccessMask =
        capture.ever_written ? VulkanPresenter::kGuestOutputInternalAccessMask
                             : 0;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].oldLayout = capture.ever_written
                                ? VulkanPresenter::kGuestOutputInternalLayout
                                : VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = capture.image;
    barriers[1].subresourceRange = color_range;
    auto external_barriers = [&](bool release) {
      VkImageMemoryBarrier2 b[2] = {};
      for (uint32_t i = 0; i < 2; ++i) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b[i].srcStageMask = release ? VK_PIPELINE_STAGE_2_TRANSFER_BIT
                                    : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        b[i].dstStageMask = release ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                                    : VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b[i].srcAccessMask = barriers[i].srcAccessMask;
        b[i].dstAccessMask = barriers[i].dstAccessMask;
        b[i].oldLayout = barriers[i].oldLayout;
        b[i].newLayout = barriers[i].newLayout;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        b[i].image = barriers[i].image;
        b[i].subresourceRange = color_range;
      }
      b[0].oldLayout = b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      b[0].srcQueueFamilyIndex =
          release ? vulkan_device->queue_family_graphics_compute()
                  : VK_QUEUE_FAMILY_EXTERNAL;
      b[0].dstQueueFamilyIndex =
          release ? VK_QUEUE_FAMILY_EXTERNAL
                  : vulkan_device->queue_family_graphics_compute();
      b[0].srcStageMask =
          release ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : VK_PIPELINE_STAGE_2_NONE;
      b[0].srcAccessMask = release ? VK_ACCESS_2_TRANSFER_READ_BIT : 0;
      b[0].dstStageMask =
          release ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      b[0].dstAccessMask = release ? 0 : VK_ACCESS_2_TRANSFER_READ_BIT;
      VkDependencyInfo dependency = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      dependency.imageMemoryBarrierCount = 2;
      dependency.pImageMemoryBarriers = b;
      vulkan_device->vkCmdPipelineBarrier2()(context.command_buffer,
                                             &dependency);
    };
    if (external) {
      external_barriers(false);
    } else {
      dfn.vkCmdPipelineBarrier(context.command_buffer,
                               VulkanPresenter::kGuestOutputInternalStageMask,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                               nullptr, 2, barriers);
    }

    VkImageBlit blit = {};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1] = {int32_t(source.extent.width),
                          int32_t(source.extent.height), 1};
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[1] = {int32_t(capture.storage_content_extent.width),
                          int32_t(capture.storage_content_extent.height), 1};
    dfn.vkCmdBlitImage(
        context.command_buffer, source.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
        source.extent.width == capture.storage_content_extent.width &&
                source.extent.height == capture.storage_content_extent.height
            ? VK_FILTER_NEAREST
            : VK_FILTER_LINEAR);

    barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].dstAccessMask =
        VulkanPresenter::kGuestOutputInternalAccessMask;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].newLayout = VulkanPresenter::kGuestOutputInternalLayout;
    barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].dstAccessMask =
        VulkanPresenter::kGuestOutputInternalAccessMask;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].newLayout = VulkanPresenter::kGuestOutputInternalLayout;
    if (external) {
      external_barriers(true);
    } else {
      dfn.vkCmdPipelineBarrier(context.command_buffer,
                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VulkanPresenter::kGuestOutputInternalStageMask,
                               0, 0, nullptr, 0, nullptr, 2, barriers);
    }
    return dfn.vkEndCommandBuffer(context.command_buffer) == VK_SUCCESS;
  }

  IngressSubmitResult SubmitIngressCopy(
      IngressSourcePublication source, uint32_t capture_index,
      uint32_t context_index) {
    ++ingress_attempt_total;
    if (!source || capture_index >= kCapturePoolSize ||
        context_index >= ingress_copy_contexts.size()) {
      return IngressSubmitResult::kInvalid;
    }
    CaptureSlot& capture = capture_slots[capture_index];
    IngressCopyContext& context = ingress_copy_contexts[context_index];
    if (context.inflight ||
        capture.state.load(std::memory_order_acquire) !=
            CaptureState::kIngressReserved) {
      return IngressSubmitResult::kInvalid;
    }
    ++ingress_acquire_total;
    if (!RecordIngressCopy(context, source, capture)) {
      return IngressSubmitResult::kInvalid;
    }
    const uint64_t queue_begin_ns = PresenterMonotonicTimeNs();
    std::optional<VulkanDevice::Queue::Acquisition> queue =
        vulkan_device->TryAcquireQueue(
            vulkan_device->queue_family_graphics_compute(), 0);
    if (!queue) {
      ++device_b_ingress_queue_busy;  // retried, not a drop
      return IngressSubmitResult::kQueueBusy;
    }
    device_b_ingress_queue_host_ns.Add(PresenterMonotonicTimeNs() -
                                       queue_begin_ns);
    if (vulkan_device->RejectZeroFGSubmitAfterTeardownIdle()) {
      ++device_b_submit_after_teardown_idle;
      return IngressSubmitResult::kQueueBusy;
    }
    // C3 (V2-3): the context's own timeline, whose previous point belongs to
    // the context's previous ingress copy, retired through its capture.
    const bool per_context_timeline = context.completion.owns_timeline();
    const VkSemaphore source_timeline =
        per_context_timeline ? context.completion.timeline() : timeline;
    const uint64_t signal_value = per_context_timeline
                                      ? context.completion.ClaimTimelineValue()
                                      : next_timeline_value.fetch_add(
                                            1, std::memory_order_relaxed);
    if (context.completion.occupied()) {
      ++ingress_owner_stats.reuse_before_retire;
    }
    VkTimelineSemaphoreSubmitInfo timeline_submit = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    const bool external = source.external_wait != VK_NULL_HANDLE;
    const VkSemaphore signals[2] = {source_timeline, source.external_release};
    const uint64_t values[2] = {signal_value, 0};
    const uint64_t wait_value = 0;
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    timeline_submit.signalSemaphoreValueCount = external ? 2 : 1;
    timeline_submit.pSignalSemaphoreValues = values;
    timeline_submit.waitSemaphoreValueCount = external ? 1 : 0;
    timeline_submit.pWaitSemaphoreValues = &wait_value;
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline_submit;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &context.command_buffer;
    submit.signalSemaphoreCount = external ? 2 : 1;
    submit.pSignalSemaphores = signals;
    submit.waitSemaphoreCount = external ? 1 : 0;
    submit.pWaitSemaphores = &source.external_wait;
    submit.pWaitDstStageMask = &wait_stage;
    const uint64_t submit_begin_ns = PresenterMonotonicTimeNs();
    BeginHostDriverOperation(BlockingOperation::kIngressSubmit);
    const VkResult result = vulkan_device->SubmitAndUpdateLost(
        queue->queue(), 1, &submit, VK_NULL_HANDLE);
    const uint64_t submit_host_ns =
        PresenterMonotonicTimeNs() - submit_begin_ns;
    ingress_submit_host_ns.Add(submit_host_ns);
    EndHostDriverOperation();
    if (result != VK_SUCCESS) {
      return IngressSubmitResult::kVulkanError;
    }
    context.completion.MarkSubmitted(source_timeline);
    ingress_owner_stats.CountSubmit(per_context_timeline, submit_host_ns);

    context.inflight = true;
    context.capture_index = capture_index;
    context.source_timeline_value = signal_value;
    context.publication = source.publication;
    context.source_image_lifetime = std::move(source.image_lifetime);
    context.handoff_ready = std::move(source.handoff_ready);
    context.submit_begin_ns = submit_begin_ns;
    if (external) queue.reset();
    bool external_release_ok = true;
    if (source.external_submitted) {
      // The lease stays represented in the in-flight context even if export
      // fails. The bridge poisons it and reports activation failure on its
      // next acquisition, never prematurely recycling it.
      external_release_ok = source.external_submitted();
    }
    source.ReleaseMailboxAcquisition();
    capture.source_timeline = source_timeline;
    capture.source_timeline_value = signal_value;
    capture.ingress_context_index = context_index;
    capture.ever_written = true;
    ++ingress_submit_total;
    ++ingress_accept_ready_total;
    uint32_t active_contexts = 0;
    for (const IngressCopyContext& candidate : ingress_copy_contexts) {
      active_contexts += candidate.inflight ? 1u : 0u;
    }
    uint32_t previous_high =
        ingress_context_high_water.load(std::memory_order_relaxed);
    while (previous_high < active_contexts &&
           !ingress_context_high_water.compare_exchange_weak(
               previous_high, active_contexts, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    if (!external_release_ok) {
      // Keep the accepted submission and capture represented for shutdown.
      // Returning an error here would let the caller release its destination.
      Terminalize(TerminalReason::kInvalidHandoff,
                  context.publication.source_id);
    }
    return IngressSubmitResult::kSubmitted;
  }

  bool CheckSourceContract() {
    const auto& evidence = xe::GetSourceContractEvidence();
    bool failed = false;
    for (size_t i = 0; i < source_contract_baseline.size(); ++i) {
      failed |= evidence.counts[i].load(std::memory_order_acquire) >
                source_contract_baseline[i];
    }
    if (!failed) {
      return true;
    }
    if (evidence.first_state.load(std::memory_order_acquire) == 2) {
      const auto& first = evidence.first_publication;
      XELOGE(
          "ZeroFGSourceContractFault first_kind={} publication/source={}/{} "
          "previous_publication={} issue/publish_thread={}/{}",
          size_t(evidence.first_kind), first.publication_id, first.source_id,
          evidence.first_previous_publication, first.issue_thread_tag,
          first.publish_thread_tag);
    }
    Terminalize(TerminalReason::kOrderingFailure,
                last_accepted_source_id.load(std::memory_order_acquire));
    return false;
  }

  // Aborts the held lease: a pre-admitted publication was never accepted.
  void ClearPreadmission() {
    preadmitted_source = {};
    preadmitted = false;
    preadmitted_gate_held = false;
    preadmitted_gate_hold_ns = 0;
    preadmitted_acquire_ns = 0;
    preadmitted_queue_ns = 0;
  }

  enum class PreadmissionReplace : uint8_t {
    kFailed,
    kInFlight,
    kNoNewer,
    kReplaced,
  };

  // Replaces the held (unaccepted) publication with a newer Published one and
  // reports which case applied. It never waits for a copy still in flight on
  // A, and never touches an accepted Real: the released lease was never
  // accepted, so it carries no sovereignty.
  PreadmissionReplace TakeNewerPreadmission(uint64_t now_ns,
                                            uint64_t& skipped_source_id) {
    IngressSourcePublication newer;
    if (!ingress_source_acquire_callback(newer)) {
      return PreadmissionReplace::kInFlight;
    }
    if (newer.external_failure) {
      Terminalize(TerminalReason::kInvalidHandoff, 0);
      return PreadmissionReplace::kFailed;
    }
    if (!newer.publication.publication_id) {
      return PreadmissionReplace::kNoNewer;
    }
    skipped_source_id = preadmitted_source.publication.source_id;
    ClearPreadmission();  // Releases the held lease, never accepted.
    preadmitted_source = std::move(newer);
    preadmitted = true;
    preadmitted_acquire_ns = now_ns;
    const uint64_t published_ns = preadmitted_source.handoff_published_ns;
    // It sat Published only because the held slot was occupied.
    preadmitted_queue_ns =
        published_ns && now_ns > published_ns ? now_ns - published_ns : 0;
    return PreadmissionReplace::kReplaced;
  }

  // Rate probe drain. While a probe is open the handoff is emptied eagerly, so
  // the Source never waits on us and the intervals it reports are its own
  // cadence. It runs even while the sidecar is occupied: keeping the Source
  // free is the measurement. Returns false only on a
  // handoff failure (already terminalized).
  bool DrainPreadmissionForRateProbe(uint64_t now_ns) {
    uint64_t skipped_source = 0;
    const PreadmissionReplace result =
        TakeNewerPreadmission(now_ns, skipped_source);
    if (result == PreadmissionReplace::kFailed) {
      return false;
    }
    if (result == PreadmissionReplace::kReplaced) {
      ++source_rate_probe_drop_total;
    }
    return true;
  }

  DriverPollResult TryIngressLatestPublication() {
    DriverPollResult operation;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (!accepting.load(std::memory_order_acquire) ||
        detach_requested.load(std::memory_order_acquire) ||
        terminal_reason.load(std::memory_order_acquire) !=
            TerminalReason::kNone) {
      ClearPreadmission();
      return operation;
    }
    if (ingress_retry_due_ns && now_ns < ingress_retry_due_ns) {
      return operation;
    }
    const xe::SourceBoundarySnapshot boundary = xe::GetSourceBoundarySnapshot();
    if (!CheckSourceContract()) {
      operation.progress = true;
      return operation;
    }
    if (boundary.totals.source_publication_order_regression_total >
        publication_order_regression_seen) {
      publication_order_regression_seen =
          boundary.totals.source_publication_order_regression_total;
      XELOGE(
          "ZeroFGPhaseD6PublicationOrdering previous publication/source={}/{} "
          "current publication/source/issue_thread/publish_thread={}/{}/{}/{}",
          boundary.totals.source_publication_order_fault_previous_publication,
          boundary.totals.source_publication_order_fault_previous_source,
          boundary.totals.source_publication_order_fault_current_publication,
          boundary.totals.source_publication_order_fault_current_source,
          boundary.totals.source_publication_order_fault_issue_thread,
          boundary.totals.source_publication_order_fault_publish_thread);
      Terminalize(TerminalReason::kOrderingFailure,
                  boundary.totals.source_last_published_source_id);
      operation.progress = true;
      return operation;
    }
    // Split physical pre-admission, depth 1: take the Published head off the
    // handoff even while admission is closed, so a momentary Always-S gate
    // does not also stop the Source's next Publish. The held publication is
    // not accepted (no acceptance, S obligation, cursor or commitment); it
    // only keeps its handoff lease, in order, and is never skipped or
    // replaced. It stays flagged pending until the path below admits it.
    bool preadmitted_now = false;
    if (!preadmitted &&
        ingress_publication_pending.exchange(false,
                                             std::memory_order_acq_rel)) {
      IngressSourcePublication acquired;
      if (!ingress_source_acquire_callback(acquired)) {
        ingress_publication_pending.store(true, std::memory_order_release);
        ingress_retry_due_ns = SaturatingAddNs(
            now_ns, acquired.external_pending ? kHandoffReadyPollIntervalNs
                                              : kOwnershipPollIntervalNs);
        return operation;
      }
      if (acquired.external_failure) {
        Terminalize(TerminalReason::kInvalidHandoff, 0);
        operation.progress = true;
        return operation;
      }
      if (acquired.publication.publication_id <=
          last_accepted_publication_id) {
        // Nothing new Published: the next publication wakes the presenter.
        return operation;
      }
      preadmitted_source = std::move(acquired);
      preadmitted = true;
      preadmitted_now = true;
      preadmitted_acquire_ns = now_ns;
      // Time it sat Published only because the pending slot was occupied.
      const uint64_t published_ns = preadmitted_source.handoff_published_ns;
      preadmitted_queue_ns =
          published_ns && preadmission_free_since_ns > published_ns
              ? preadmission_free_since_ns - published_ns
              : 0;
      ++split_preadmission_total;
      ingress_publication_pending.store(true, std::memory_order_release);
      ingress_retry_due_ns = 0;
      operation.progress = true;
    }
    // A rate probe drains before the sidecar gate: while it is open the Source
    // must not be waiting on us, or the intervals it reports are ours and not
    // its own. Acceptance keeps its own cadence; only the unaccepted held
    // publication is replaced.
    if (SourceRateProbeActive()) {
      if (now_ns > SaturatingAddNs(source_rate_probe_begin_ns,
                                   kSourceRateProbeWallBudgetNs)) {
        ++source_rate_probe_episode_expired_total;
        ++source_rate_probe_abort_total;
        if (source_rate_probe_reason ==
            SourceRateProbeReason::kPhysicalOperatingPointRecovery) {
          ++source_rate_probe_recovery_expired_total;
          ++source_rate_probe_recovery_abort_total;
        }
        FinishSourceRateProbe(now_ns, kSourceRateProbeCooldownAbortNs);
      } else if (preadmitted && !DrainPreadmissionForRateProbe(now_ns)) {
        operation.progress = true;
        return operation;
      }
    }
    if (incoming_real.occupied()) {
      ++incoming_real_gate_total;
      return operation;
    }

    // Strict split: the sidecar (incoming_real, depth 1) is the
    // bounded-pressure boundary. The publication is accepted whenever the
    // sidecar is empty, even if the executive has no room yet to materialize
    // its S+B; the materialization-capacity gate now sits at promotion
    // (SplitSidecarPromotionHeld). The next publication stays unaccepted in
    // preadmission, so pressure rises upstream only once this one unit of
    // accepted elasticity is in use.

    uint32_t capture_index = UINT32_MAX;
    for (uint32_t i = 0; i < kCapturePoolSize; ++i) {
      if (capture_slots[i].state.load(std::memory_order_acquire) ==
          CaptureState::kFree) {
        capture_index = i;
        break;
      }
    }
    const uint32_t context_index = FindFreeIngressContext();
    if (capture_index == UINT32_MAX || context_index == UINT32_MAX) {
      return operation;
    }
    IngressSourcePublication source;
    if (!preadmitted) {
      return operation;
    }
    // A copy: the held lease survives a busy queue and is relinquished
    // only once the ingress copy has been submitted.
    source = preadmitted_source;
    if (source.external_failure) {
      Terminalize(TerminalReason::kInvalidHandoff, 0);
      operation.progress = true;
      return operation;
    }
    if (source.publication.publication_id <= last_accepted_publication_id) {
      // Never reached by the ordered handoff; do not strand the held lease.
      ClearPreadmission();
      return operation;
    }
    if (!source.publication.issue_identity_valid) {
      Terminalize(TerminalReason::kOrderingFailure,
                  source.publication.source_id);
      operation.progress = true;
      return operation;
    }
    if (!CheckSourceContract()) {
      operation.progress = true;
      return operation;
    }
    if (!source.publication.source_id || !source.publication.publication_id ||
        !source.content_width || !source.content_height ||
        source.content_width > extent.width ||
        source.content_height > extent.height || !source.extent.width ||
        !source.extent.height || source.extent.width > extent.width ||
        source.extent.height > extent.height ||
        !source.display_aspect_ratio_x || !source.display_aspect_ratio_y) {
      Terminalize(TerminalReason::kInvalidGeometry,
                  source.publication.source_id);
      operation.progress = true;
      return operation;
    }

    CaptureSlot& capture = capture_slots[capture_index];
    CaptureState expected = CaptureState::kFree;
    if (!capture.state.compare_exchange_strong(
            expected, CaptureState::kIngressReserved,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      ingress_publication_pending.store(true, std::memory_order_release);
      return operation;
    }
    const uint32_t capture_occupancy = CountSourceIngressSlots();
    uint32_t capture_high =
        normal_capture_high_water.load(std::memory_order_relaxed);
    while (capture_high < capture_occupancy &&
           !normal_capture_high_water.compare_exchange_weak(
               capture_high, capture_occupancy, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    capture.reserve_time_ns = PresenterMonotonicTimeNs();
    capture.source_id = source.publication.source_id;
    capture.issue_time_ns = source.publication.issue_time_ns;
    capture.publish_time_ns = source.publication.publish_time_ns;
    const uint64_t reserve_issue_ns = capture.issue_time_ns
                                          ? capture.issue_time_ns
                                          : capture.publish_time_ns;
    if (reserve_issue_ns && capture.reserve_time_ns >= reserve_issue_ns) {
      const uint64_t issue_to_reserve_ns =
          capture.reserve_time_ns - reserve_issue_ns;
      capture_issue_to_reserve_ns.Add(issue_to_reserve_ns);
      // D's dependency sample excludes the waits ZeroFG imposed on this
      // publication: the Source's handoff backpressure, the time it sat
      // Published only because the pre-admission slot was occupied, and its
      // hold from pre-admission to this reservation (Always-S gate,
      // resources, busy queue). Queue residence never sustains D.
      uint64_t excluded_ns = source.source_bp_wait_ns;
      if (preadmitted) {
        excluded_ns = SaturatingAddNs(
            SaturatingAddNs(excluded_ns, preadmitted_queue_ns),
            capture.reserve_time_ns > preadmitted_acquire_ns
                ? capture.reserve_time_ns - preadmitted_acquire_ns
                : 0);
      }
      if (excluded_ns >= issue_to_reserve_ns) {
        ++capture_dependency_clamped_total;
        excluded_ns = 0;
      }
      capture_issue_to_reserve_dependency_ns.Add(issue_to_reserve_ns -
                                                 excluded_ns);
    }
    capture.generation = generation;
    capture.source_content_extent = {source.content_width,
                                     source.content_height};
    capture.storage_content_extent = {source.content_width,
                                      source.content_height};
    capture.display_aspect_ratio_x = source.display_aspect_ratio_x;
    capture.display_aspect_ratio_y = source.display_aspect_ratio_y;
    capture.is_8bpc = source.is_8bpc;
    capture.ingress_context_index = context_index;

    const uint32_t mailbox_index = source.mailbox_index;
    const uint64_t ingress_source_id = source.publication.source_id;
    const IngressSubmitResult result =
        SubmitIngressCopy(std::move(source), capture_index, context_index);
    if (result == IngressSubmitResult::kQueueBusy) {
      // Split: the pre-admitted publication keeps its lease for the retry;
      // a busy queue never drops it.
      ReleaseCaptureSlot(capture_index);
      ingress_publication_pending.store(true, std::memory_order_release);
      ingress_retry_due_ns =
          SaturatingAddNs(PresenterMonotonicTimeNs(), kOwnershipPollIntervalNs);
      return operation;
    }
    if (result != IngressSubmitResult::kSubmitted) {
      ReleaseCaptureSlot(capture_index);
      Terminalize(result == IngressSubmitResult::kVulkanError
                      ? TerminalReason::kSourceSubmitFailure
                      : TerminalReason::kInvalidHandoff,
                  ingress_source_id);
      operation.progress = true;
      return operation;
    }

    operation.invoked = true;
    operation.progress = true;
    // Deferred admissions by cause: held by the Always-S gate at least
    // once, or only by resources / a busy queue (ingress_queue_busy_retry).
    if (preadmitted_gate_held) {
      ++split_preadmission_gate_deferred_total;
      split_preadmission_gate_hold_ns.Add(preadmitted_gate_hold_ns);
    } else if (!preadmitted_now) {
      ++split_preadmission_other_deferred_total;
    }
    // Submitted: the lease now belongs to the in-flight copy.
    ClearPreadmission();
    preadmission_free_since_ns = PresenterMonotonicTimeNs();
    const uint64_t source_id = capture.source_id;
    const uint64_t publication_id =
        ingress_copy_contexts[context_index].publication.publication_id;
    const uint64_t previous_source =
        last_accepted_source_id.load(std::memory_order_acquire);
    if (previous_source && source_id <= previous_source) {
      ++ordering_error_total;
      accepted_order_failure_current_publication = publication_id;
      accepted_order_failure_current_source = source_id;
      accepted_order_failure_current_mailbox = mailbox_index;
      accepted_order_failure_current_context = context_index;
      accepted_order_failure_current_capture = capture_index;
      accepted_order_failure_previous_publication =
          last_accepted_publication_id;
      accepted_order_failure_previous_source = previous_source;
      const SourcePublicationTelemetry& failed =
          ingress_copy_contexts[context_index].publication;
      XELOGE(
          "ZeroFGPhaseD6AcceptedOrdering current publication/source/mailbox/"
          "issue_thread/publish_thread/context/capture={}/{}/{}/{}/{}/{}/{} "
          "previous publication/source/publish_thread={}/{}/{}",
          publication_id, source_id, mailbox_index, failed.issue_thread_tag,
          failed.publish_thread_tag, context_index, capture_index,
          last_accepted_publication_id, previous_source,
          last_accepted_publish_thread_tag);
      Terminalize(TerminalReason::kOrderingFailure, source_id);
      return operation;
    }
    if (previous_source && source_id > previous_source + 1) {
      accepted_source_gap_total += source_id - previous_source - 1;
    }
    last_accepted_publication_id = publication_id;
    last_accepted_source_id.store(source_id, std::memory_order_release);
    last_accepted_publish_thread_tag =
        ingress_copy_contexts[context_index].publication.publish_thread_tag;
    capture.state.store(CaptureState::kReadyWaitingResidency,
                        std::memory_order_release);
    if (incoming_real.occupied()) {
      ++incoming_real_invariant_violation_total;
      Terminalize(TerminalReason::kOrderingFailure, source_id);
      return operation;
    }
    incoming_real.source_id = source_id;
    incoming_real.publication_id = publication_id;
    incoming_real.armed_ns = PresenterMonotonicTimeNs();
    ++incoming_real_arm_total;
    incoming_real_high_water = std::max(incoming_real_high_water, 1u);
    ++source_reserved_total;
    ++source_submitted_total;
    ++source_published_total;
    ++source_handoff_total;
    ingress_accepted_total.fetch_add(1, std::memory_order_relaxed);
    ingress_retry_due_ns = 0;
    // FIFO handoff: pre-admit the next Published head at once, releasing a
    // waiting Source Publish without a poll interval. Due now only keeps a
    // fallback wake in case this pump's driver turn goes elsewhere.
    ingress_publication_pending.store(true, std::memory_order_release);
    ingress_retry_due_ns = PresenterMonotonicTimeNs();
    UpdateCaptureWaitingResidencyHighWater();
    UpdateCaptureFreeMinimum(kCapturePoolSize - CountSourceIngressSlots());
    return operation;
  }

  bool AllocateResidencySlot(Slot& slot, uint32_t slot_index,
                             VkExtent2D slot_extent) {
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    image_info.extent = {slot_extent.width, slot_extent.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!util::CreateDedicatedAllocationImage(
            vulkan_device, image_info, util::MemoryPurpose::kDeviceLocal,
            slot.image, slot.memory)) {
      return false;
    }
    VkImageViewCreateInfo view_info = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = slot.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    view_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY};
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    if (dfn.vkCreateImageView(device, &view_info, nullptr, &slot.view) !=
        VK_SUCCESS) {
      dfn.vkDestroyImage(device, slot.image, nullptr);
      dfn.vkFreeMemory(device, slot.memory, nullptr);
      slot.image = VK_NULL_HANDLE;
      slot.memory = VK_NULL_HANDLE;
      return false;
    }
    // DestroyResidencySlot released the previous sync_fd state; this only
    // guards against a leak. The V2-2 timeline is kept.
    slot.completion.DestroySyncFd(vulkan_device);
    slot.ready_timeline = VK_NULL_HANDLE;
    if (!CreateCaptureSyncSemaphore(slot, slot_index)) {
      // ZeroFG needs the per-Residency sync_fd semaphore.
      slot.completion.SetSyncFdFallback();
      return false;
    }
    slot.ever_written = false;
    slot.state.store(SlotState::kFree, std::memory_order_release);
    vulkan_device->SetObjectName(
        VK_OBJECT_TYPE_IMAGE, slot.image,
("ZeroFG Real Residency " + std::to_string(slot_index)).c_str());
    return true;
  }

  bool CreateCaptureSyncSemaphore(Slot& slot, uint32_t slot_index) {
    const VkResult result = slot.completion.CreateSyncFd(
        vulkan_device, "ZeroFG Capture sync_fd " + std::to_string(slot_index));
    if (result != VK_SUCCESS) {
      XELOGW("ZeroFGC0: capture sync_fd semaphore unavailable slot={} result={}",
             slot_index, int32_t(result));
      return false;
    }
    return true;
  }

  void RecreateCaptureSyncSemaphoreIfNeeded(Slot& slot, uint32_t slot_index) {
    if (!slot.completion.sync_fd_recreate_pending()) {
      return;
    }
    if (!slot.completion.RecreateSyncFd(
            vulkan_device,
            "ZeroFG Capture sync_fd " + std::to_string(slot_index))) {
      XELOGW("ZeroFGC0: capture sync_fd semaphore unavailable slot={}",
             slot_index);
    }
  }

  void DestroyResidencySlot(Slot& slot) {
    slot.handoff_ready.reset();
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    // The sync_fd lives with the Residency images; the V2-2 timeline lives
    // with the presenter.
    slot.completion.DestroySyncFd(vulkan_device);
    if (slot.view != VK_NULL_HANDLE) {
      dfn.vkDestroyImageView(device, slot.view, nullptr);
      slot.view = VK_NULL_HANDLE;
    }
    if (slot.image != VK_NULL_HANDLE) {
      dfn.vkDestroyImage(device, slot.image, nullptr);
      slot.image = VK_NULL_HANDLE;
    }
    if (slot.memory != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, slot.memory, nullptr);
      slot.memory = VK_NULL_HANDLE;
    }
    slot.ever_written = false;
    slot.source_id.store(0, std::memory_order_relaxed);
    slot.issue_time_ns.store(0, std::memory_order_relaxed);
    slot.policy_wait_ns.store(0, std::memory_order_relaxed);
    slot.publish_time_ns.store(0, std::memory_order_relaxed);
    slot.timeline_value.store(0, std::memory_order_relaxed);
    slot.ready_timeline = VK_NULL_HANDLE;
    slot.generation.store(0, std::memory_order_relaxed);
    slot.source_content_extent = {};
    slot.storage_content_extent = {};
    slot.display_aspect_ratio_x = 0;
    slot.display_aspect_ratio_y = 0;
    slot.is_8bpc = false;
    slot.ready_time_ns = 0;
    slot.transfer_signal_value = 0;
    slot.transfer_submit_time_ns = 0;
    slot.transfer_first_poll_begin_time_ns = 0;
    slot.observed_pending_time_ns = 0;
    slot.observed_ready_time_ns = 0;
    slot.apply_time_ns = 0;
    slot.owner_refs = 0;
    slot.runtime_admitted = false;
    slot.state.store(SlotState::kFree, std::memory_order_release);
  }

  // Main Surface Authority: a FinalOutput is a plain B-local image that the
  // Post renders into and the egress copies from, so only those two
  // usages.
  bool LocalFinalOutputSupported(VkExtent2D requested_extent,
                                 VkFormat format) const {
    const VulkanInstance::Functions& ifn =
        vulkan_device->vulkan_instance()->functions();
    VkFormatProperties properties = {};
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device->physical_device(),
                                            format, &properties);
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    const VkExtent2D max_extent =
        util::GetMax2DFramebufferExtent(vulkan_device->properties());
    return (properties.optimalTilingFeatures & required) == required &&
           requested_extent.width && requested_extent.height &&
           requested_extent.width <= max_extent.width &&
           requested_extent.height <= max_extent.height;
  }

  bool AllocateLocalFinalOutputSlot(FinalOutputSlot& slot, uint32_t slot_index,
                                    VkExtent2D slot_extent, VkFormat format) {
    VkImageCreateInfo image_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = {slot_extent.width, slot_extent.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!util::CreateDedicatedAllocationImage(
            vulkan_device, image_info, util::MemoryPurpose::kDeviceLocal,
            slot.image, slot.memory)) {
      return false;
    }
    VkImageViewCreateInfo view_info = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = slot.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY,
                            VK_COMPONENT_SWIZZLE_IDENTITY};
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.layerCount = 1;
    if (vulkan_device->functions().vkCreateImageView(
            vulkan_device->device(), &view_info, nullptr, &slot.view) !=
        VK_SUCCESS) {
      DestroyFinalOutputSlot(slot);
      return false;
    }
    slot.state.store(FinalOutputState::kFree, std::memory_order_release);
    vulkan_device->SetObjectName(
        VK_OBJECT_TYPE_IMAGE, slot.image,
        ("ZeroFG Final Output B-local " + std::to_string(slot_index)).c_str());
    return true;
  }

  void DestroyFinalOutputSlot(FinalOutputSlot& slot) {
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    if (slot.view != VK_NULL_HANDLE) {
      dfn.vkDestroyImageView(device, slot.view, nullptr);
      slot.view = VK_NULL_HANDLE;
    }
    if (slot.image != VK_NULL_HANDLE) {
      dfn.vkDestroyImage(device, slot.image, nullptr);
      slot.image = VK_NULL_HANDLE;
    }
    if (slot.memory != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, slot.memory, nullptr);
      slot.memory = VK_NULL_HANDLE;
    }
    slot.post_completion_semaphore = VK_NULL_HANDLE;
    slot.post_completion_value = 0;
    slot.source_id = 0;
    slot.sequence_id = 0;
    slot.pair_a_source_id = 0;
    slot.latency_epoch = 0;
    slot.candidate_kind = CandidateKind::kReal;
    slot.semantic_target_time_ns = 0;
    slot.target_time_ns = 0;
    slot.semantic_epoch_origin_ns = 0;
    slot.semantic_output_quantum_ns = 0;
    slot.semantic_tick_index = 0;
    slot.apply_time_ns = 0;
    slot.semantic_forward_skipped = false;
    slot.operating_latency_ns = 0;
    slot.final_ready_time_ns = 0;
    slot.final_ready_soft_deadline_ns = 0;
    slot.dispatch_lead_ns = 0;
    slot.actual_apply_lead_ns = 0;
    slot.ordered_ready_wait_ns = 0;
    slot.residency_recovery_debt.store(false, std::memory_order_relaxed);
    slot.free_time_ns = 0;
    slot.last_released_kind = CandidateKind::kReal;
    slot.last_release_fenced = false;
    slot.state.store(FinalOutputState::kFree, std::memory_order_release);
  }


  bool Connect(void* native_window_pointer, VkExtent2D new_extent,
               VkFormat new_final_output_format,
               VkSurfaceKHR new_main_surface) {
    if (!native_window_pointer || !new_extent.width || !new_extent.height ||
        surface_connected) {
      return false;
    }
    // MSA is decided per session: after a terminal handback it stays refused,
    // and the normal presenter keeps the Surface.
    if (main_surface_refused.load(std::memory_order_acquire)) {
      XELOGW(
          "ZeroFGMainSurface refused reason={} session=true fallback=native_A",
          main_surface_refusal_reason.load(std::memory_order_acquire));
      return false;
    }
    if (new_main_surface == VK_NULL_HANDLE) {
      RefuseMainSurface("no_vk_surface");
      return false;
    }
    planned_space_trace = {};
    if (!post_process_callback ||
        !LocalFinalOutputSupported(new_extent, new_final_output_format)) {
      RefuseMainSurface("final_output_format");
      return false;
    }
    extent = new_extent;
    final_output_format = new_final_output_format;
    main_surface = new_main_surface;
    ++generation;
    if (!generation) {
      ++generation;
    }
    for (uint32_t i = 0; i < kCapturePoolSize; ++i) {
      if (!AllocateCaptureSlot(capture_slots[i], i, extent)) {
        for (uint32_t j = 0; j < i; ++j) {
          DestroyCaptureSlot(capture_slots[j]);
        }
        Terminalize(TerminalReason::kHardwareBufferAllocation, 0);
        return false;
      }
    }
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      if (!AllocateResidencySlot(slots[i], i, extent)) {
        for (uint32_t j = 0; j < i; ++j) {
          DestroyResidencySlot(slots[j]);
        }
        for (CaptureSlot& capture : capture_slots) {
          DestroyCaptureSlot(capture);
        }
        Terminalize(TerminalReason::kCaptureTransferCapability, 0);
        return false;
      }
    }
    for (uint32_t i = 0; i < kFinalOutputPoolSize; ++i) {
      // Bounded: FIFO does not buy a deeper FinalOutput pool.
      if (!AllocateLocalFinalOutputSlot(final_outputs[i], i, extent,
                                        final_output_format)) {
        for (uint32_t j = 0; j < i; ++j) {
          DestroyFinalOutputSlot(final_outputs[j]);
        }
        for (Slot& slot : slots) {
          DestroyResidencySlot(slot);
        }
        for (CaptureSlot& capture : capture_slots) {
          DestroyCaptureSlot(capture);
        }
        Terminalize(TerminalReason::kPostProcessCapability, 0);
        return false;
      }
    }

    // No child layer: B produces into A's own Surface once A retires its
    // swapchain at the authority claim.
    bridge = std::make_shared<EventBridge>();
    std::shared_ptr<EventBridge> egress_bridge = bridge;
    auto egress = std::make_unique<ZeroFGMainSurfaceEgress>(
        vulkan_device, &main_surface_producer, main_surface, extent,
        final_output_format, kFinalOutputPoolSize, elevated_presenter_priority,
        // The output shaper (the GPU guard) was tuned for the FIFO egress;
        // with free output and the vsync quantizer its push stacks on the
        // quantizer's lead (35-75 ms, Arkham 2026-10-07), so it stays off.
        /*apocalypse_guard=*/false, /*free_output=*/true,
        /*vsync_quantizer=*/true,
        [egress_bridge]() { SignalPresenterWake(egress_bridge); });
    const bool egress_started = egress->Start();
    if (egress_started) {
      std::lock_guard<std::mutex> lock(main_surface_egress_mutex);
      main_surface_egress = std::move(egress);
      main_surface_egress_live.store(true, std::memory_order_release);
    }
    if (!egress_started) {
      egress->Stop();
      egress.reset();
      bridge.reset();
      for (FinalOutputSlot& output : final_outputs) {
        DestroyFinalOutputSlot(output);
      }
      for (Slot& slot : slots) {
        DestroyResidencySlot(slot);
      }
      for (CaptureSlot& capture : capture_slots) {
        DestroyCaptureSlot(capture);
      }
      main_surface = VK_NULL_HANDLE;
      RefuseMainSurface("egress_start");
      return false;
    }
    ResetMainSurfaceConnectionState();

    capture_transfer_submitted_total.store(0, std::memory_order_relaxed);
    capture_transfer_completed_total.store(0, std::memory_order_relaxed);
    capture_committed_waiting_residency_total.store(
        0, std::memory_order_relaxed);
    capture_committed_waiting_residency_high_water.store(
        0, std::memory_order_relaxed);
    capture_pool_high_water.store(0, std::memory_order_relaxed);
    ingress_attempt_total.store(0, std::memory_order_relaxed);
    ingress_acquire_total.store(0, std::memory_order_relaxed);
    ingress_submit_total.store(0, std::memory_order_relaxed);
    ingress_accept_ready_total.store(0, std::memory_order_relaxed);
    ingress_accepted_total.store(0, std::memory_order_relaxed);
    ingress_context_high_water.store(0, std::memory_order_relaxed);
    ingress_submit_host_ns = {};
    incoming_real.Reset();
    incoming_real_arm_total = 0;
    incoming_real_admit_total = 0;
    incoming_real_release_total = 0;
    incoming_real_gate_total = 0;
    incoming_real_invariant_violation_total = 0;
    incoming_real_high_water = 0;
    split_sidecar_hold_begin_ns = 0;
    split_sidecar_hold_total = 0;
    split_sidecar_hold_s_capacity_total = 0;
    split_sidecar_hold_timeout_total = 0;
    split_sidecar_hold_alarm_raised = false;
    split_sidecar_hold_s_generation_total = 0;
    split_sidecar_hold_s_post_total = 0;
    split_sidecar_hold_s_submitted_total = 0;
    split_sidecar_hold_s_ready_total = 0;
    split_sidecar_hold_s_anchored_total = 0;
    split_sidecar_hold_predecessor_committed_total = 0;
    split_quiet_window_cycle_total = 0;
    split_quiet_window_progress_override_total = 0;
    arbitration_generation_after_blocked_post_total = 0;
    arbitration_progress_no_candidate_total = 0;
    latency_decay_floor_issue_age_ns = {};
    split_sidecar_hold_ns = {};
    split_sidecar_residence_ns = {};
    for (uint64_t& total : split_sidecar_exit_total) {
      total = 0;
    }
    for (uint64_t& accrued : split_sidecar_hold_cause_ns) {
      accrued = 0;
    }
    for (uint64_t& total : split_sidecar_hold_cause_total) {
      total = 0;
    }
    split_sidecar_hold_cause = SidecarHoldCause::kNone;
    split_sidecar_hold_cause_since_ns = 0;
    synthetic_backing_owned_mask = 0;
    synthetic_backing_owned_high_water = 0;
    synthetic_backing_false_free_total = 0;
    synthetic_backing_false_free_high_water = 0;
    synthetic_backing_refusal_avoided_total = 0;
    logical_synthetic_ticket_high_water = 0;
    frontier_saturated_backing_idle_total = 0;
    post_submit_to_completion_ns = {};
    final_ready_false_success_total = 0;
    final_ready_completion_judged_total = 0;
    post_completion_proven_total = 0;
    completion_legacy_unproven_total = 0;
    post_completion_observation_error_total = 0;
    final_ready_judgement_skipped_no_proof_total = 0;
    final_ready_judgement_duplicate_prevented_total = 0;
    for (uint64_t& total : gpu_poll_source_total) {
      total = 0;
    }
    pair_causal_raw_ns = {};
    pair_causal_clean_ns = {};
    pair_policy_wait_ns = {};
    pair_causal_clean_subtracted_total = 0;
    pair_chain_raw_ns = {};
    pair_causal_sample_total = 0;
    pool_high_water_capture = 0;
    pool_high_water_residency = 0;
    pool_high_water_final_output = 0;
    pool_high_water_final_output_synthetic = 0;
    pool_high_water_logical_synthetic = 0;
    final_output_synthetic_owns_all_but_one_total = 0;
    final_output_synthetic_owns_all_total = 0;
    split_sidecar_same_pump_promotion_total = 0;
    split_sidecar_promote_with_preadmitted_total = 0;
    split_sidecar_promote_reanchor_pending_total = 0;
    split_sidecar_bp_interval_total = 0;
    split_sidecar_drain_discard_total = 0;
    split_live_unanchored_synthetic_high_water = 0;
    source_reanchor_live_anchor_won_total = 0;
    source_reanchor_live_anchor_won_held_total = 0;
    normal_capture_high_water.store(0, std::memory_order_relaxed);
    capture_transfer_high_water.store(0, std::memory_order_relaxed);
    capture_free_min.store(kCapturePoolSize, std::memory_order_relaxed);
    source_timeline_cpu_poll_total.store(0, std::memory_order_relaxed);
    committed_capture_cpu_promotion_total.store(0,
                                                std::memory_order_relaxed);
    direct_capture_timeline_wait_submit_total.store(
        0, std::memory_order_relaxed);
    capture_proactive_turnover_attempt_total = 0;
    capture_proactive_turnover_submit_total = 0;
    capture_emergency_progress_attempt_total = 0;
    capture_emergency_progress_success_total = 0;
    capture_transfer_queue_wait_ns = {};
    capture_transfer_submit_host_ns = {};
    capture_transfer_residence_ns = {};
    capture_reserve_to_recycle_ns = {};
    capture_sync_fd_export_host_ns = {};
    capture_sync_fd_poll_host_ns = {};
    capture_sync_fd_poll_total.store(0, std::memory_order_relaxed);
    capture_sync_fd_ready_total.store(0, std::memory_order_relaxed);
    capture_sync_fd_export_total.store(0, std::memory_order_relaxed);
    capture_sync_fd_export_failure_total.store(0, std::memory_order_relaxed);
    capture_sync_fd_fallback_total.store(0, std::memory_order_relaxed);
    generation_sync_fd_export_host_ns = {};
    generation_sync_fd_poll_host_ns = {};
    generation_sync_fd_poll_total = 0;
    generation_sync_fd_ready_total = 0;
    generation_sync_fd_export_total = 0;
    generation_sync_fd_export_failure_total = 0;
    generation_sync_fd_fallback_total = 0;
    // The historical queue-wait counters are process-cumulative and are not
// reset by a Source/display epoch. Seed only the presenter's reporting interval here;
    // the producer-side measurement remains the single existing definition.
    const xe::SourceBoundarySnapshot source_snapshot =
        xe::GetSourceBoundarySnapshot();
    for (size_t i = 0; i < source_contract_baseline.size(); ++i) {
      source_contract_baseline[i] = xe::GetSourceContractEvidence()
          .counts[i].load(std::memory_order_acquire);
    }
    ingress_publication_pending.store(false, std::memory_order_relaxed);
    ingress_retry_due_ns = 0;
    publication_order_regression_seen =
        source_snapshot.totals.source_publication_order_regression_total;
    last_accepted_publication_id = 0;
    last_accepted_publish_thread_tag = 0;
    accepted_source_gap_total = 0;
    source_queue_wait_last_samples.store(
        source_snapshot.totals.source_queue_lock_wait_count,
        std::memory_order_relaxed);
    source_queue_wait_last_total_ns.store(
        source_snapshot.totals.source_queue_lock_wait_total_ns,
        std::memory_order_relaxed);
    source_qualification_start_issue_total.store(
        source_snapshot.totals.source_issue_total,
        std::memory_order_relaxed);
    qualified_native_source_period_ns.store(0, std::memory_order_relaxed);
    presenter_authority_activation_source_id.store(0,
                                                std::memory_order_relaxed);
    presenter_authority_activation_time_ns.store(0,
                                              std::memory_order_relaxed);
    source_activation_armed.store(true, std::memory_order_relaxed);
    source_bootstrap_real_only_remaining.store(0,
                                               std::memory_order_relaxed);
    source_native_qualification_attempt_total.store(
        0, std::memory_order_relaxed);
    source_native_qualification_success_total.store(
        0, std::memory_order_relaxed);
    source_real_only_validation_pass_total = 0;
    source_real_only_validation_fail_total = 0;
    source_real_only_baseline_period_ns = 0;
    source_real_only_c2_baseline_valid = false;
    source_real_only_c2_baseline_period_ns = 0;
    source_real_only_c2_baseline_epoch = 0;
    source_real_only_c2_native_reference_valid = false;
    source_real_only_c2_window_epoch = 0;
    source_real_only_c2_window_bp_wait_ns = 0;
    source_real_only_c2_window_remaining = 0;
    source_real_only_c2_rate_timestamps_ns = {};
    c2_bootstrap_completed_by_comparison_total = 0;
    c2_bootstrap_completed_without_comparison_total = 0;
    source_real_only_c2_baseline_acquired_total = 0;
    source_real_only_c2_baseline_invalidated_total = 0;
    source_real_only_c2_comparison_same_regime_total = 0;
    source_real_only_c2_comparison_skipped_regime_mismatch_total = 0;
    source_real_only_c2_faster_regime_evidence_total = 0;
    source_real_only_c2_slower_regime_evidence_total = 0;
    source_real_only_c2_unstable_window_total = 0;
    source_real_only_c2_window_overrun_total = 0;
    source_real_only_c2_bp_attributed_slowdown_total = 0;
    source_real_only_c2_last_bp_explained_ns = 0;
    source_validation_start_sequence.store(0, std::memory_order_relaxed);
    source_issue_cursor_initialized = false;
    last_processed_source_issue_period_sequence = 0;
    source_issue_interval_accumulator_ns = 1;
    source_issue_raw_accumulator_ns = 1;
    source_issue_previous_bp = false;
    source_issue_period_consumed_total = 0;
    source_issue_period_overrun_total = 0;
    source_issue_snapshot_inconsistent_total = 0;
    source_real_only_ratio_per_mille = 0;
    source_protection_pair_floor = 0;
    source_protection_after_issue_sequence = 0;
    source_phase_previous_source_id = 0;
    armed_real_without_eligibility = 0;
    armed_real_apply_without_target = 0;
    accepted_gap_epoch_reset = 0;
    accepted_gap_observation_total = 0;
    accepted_gap_pair_distance_total = {};
    accepted_gap_synthetic_admitted_total = 0;
    accepted_gap_synthetic_applied_total = 0;
    accepted_gap_synthetic_hold_total = 0;
    accepted_gap_real_scaled_commit_total = 0;
    accepted_gap_real_floor_lead_max_ns = 0;
    pair_distance_compressed_real_total = 0;
    accepted_gap_pair_phase_capped_total = 0;
    accepted_gap_pair_phase_capped_tick_total = 0;
    accepted_gap_issue_bunched_total = 0;
    accepted_gap_issue_bunch_max_ns = 0;
    accepted_gap_b_phase_excess_max_ns = 0;
    consecutive_pair_b_phase_excess_max_ns = 0;
    structural_timing_failure_total = 0;
    source_rate_transition_reanchor_pending = false;
    source_rate_transition_slower_pending = false;
    last_accepted_source_id.store(0, std::memory_order_relaxed);
    last_applied_source_id.store(0, std::memory_order_relaxed);
    last_applied_real_commitment = {};
    last_applied_synthetic_commitment = {};
    last_released_source_id.store(0, std::memory_order_relaxed);
    last_applied_logical_sequence.store(0, std::memory_order_relaxed);
    readiness_wait_ns = {};
    release_latency_ns = {};
    transaction_prepare_ns = {};
    surface_apply_host_ns = {};
    dispatch_start_late_ns = {};
    hard_ready_dispatch_late_ns = {};
    h14_b_apply_lateness_after_ready_s_ns = {};
    pump_logical_order_ns = {};
    e1f_shadow_impulse_ns = {};
    e1f_commit_boundary_impulse_ns = {};
    synthetic_anchor_latency_ns = {};
    pre_first_pump_ns = {};
    cycle_callback_ns = {};
    cycle_release_ns = {};
    active_capture_poll_host_ns = {};
    capture_issue_to_transfer_submit_ns = {};
    capture_issue_to_reserve_ns = {};
    capture_issue_to_reserve_dependency_ns = {};
    capture_dependency_clamped_total = 0;
    split_preadmission_gate_hold_ns = {};
    source_issue_raw_interval_ns = {};
    source_issue_learned_interval_ns = {};
    source_issue_bp_wait_ns = {};
    source_issue_bp_interval_total = 0;
    source_issue_bp_wait_ns_total = 0;
    source_issue_bp_clamped_total = 0;
    source_issue_p_censored_bp_total = 0;
    source_issue_p_censored_post_bp_total = 0;
    source_issue_p_censored_transition_kept_total = 0;
    source_issue_clean_run_length = 0;
    source_issue_clean_run_max = 0;
    source_issue_clean_run_learnable_total = 0;
    source_issue_clean_run_samples = {};
    source_issue_recent_bp_mask = 0;
    source_issue_recent_bp_count = 0;
    source_rate_probe_state = SourceRateProbeState::kIdle;
    source_rate_probe_reason = SourceRateProbeReason::kNormal;
    source_rate_probe_recovery_pending = false;
    source_rate_probe_recovery_requested_total = 0;
    source_rate_probe_recovery_requested_ns = 0;
    source_rate_probe_recovery_requested_issue = 0;
    source_rate_probe_recovery_attempt_total = 0;
    source_rate_probe_recovery_confirm_total = 0;
    source_rate_probe_recovery_no_change_total = 0;
    source_rate_probe_recovery_abort_total = 0;
    source_rate_probe_recovery_expired_total = 0;
    physical_operating_point_state = PhysicalOperatingPointState::kIdle;
    physical_op_observations = {};
    physical_op_observation_count = 0;
    physical_op_observation_next = 0;
    physical_op_candidate_windows = 0;
    physical_op_candidate_start_ns = 0;
    physical_op_candidate_start_issue = 0;
    physical_op_candidate_period_ns = 0;
    physical_op_refinement_active = false;
    physical_op_refinement_start_ns = 0;
    physical_op_refinement_start_issue = 0;
    physical_op_refinement_candidate_period_ns = 0;
    physical_op_refinement_windows = 0;
    physical_op_refinement_enter_total = 0;
    physical_op_refinement_confirm_total = 0;
    physical_op_refinement_abort_total = 0;
    physical_op_candidate_abort_total = 0;
    physical_op_candidate_enter_total = 0;
    physical_op_confirm_total = 0;
    physical_op_exit_total = 0;
    physical_op_reentry_total = 0;
    physical_op_last_entry_ns = 0;
    physical_op_last_exit_ns = 0;
    physical_op_idle_since_ns = 0;
    physical_op_last_raw_period_ns = 0;
    physical_op_last_bp_fraction_per_mille = 0;
    physical_op_last_post_bp_fraction_per_mille = 0;
    physical_op_last_censored_fraction_per_mille = 0;
    physical_op_last_clean_fraction_per_mille = 0;
    physical_op_last_hold_fraction_per_mille = 0;
    physical_op_last_pressure_fraction_per_mille = 0;
    physical_op_slack_streak = 0;
    physical_op_probe_cooldown_until_ns = 0;
    source_rate_probe_recovery_requested_ns = 0;
    source_rate_probe_recovery_requested_issue = 0;
    physical_op_confirmed = false;
    physical_op_hold_snapshot_initialized = false;
    physical_op_previous_hold_total = 0;
    physical_op_real_finalready_before_target_total.store(
        0, std::memory_order_relaxed);
    physical_op_real_finalready_observed_total = 0;
    physical_op_real_finalready_snapshot_initialized = false;
    physical_op_last_recovery_signal = false;
    physical_op_last_raw_ratio_per_mille = 0;
    physical_op_candidate_reference_ns = 0;
    physical_op_candidate_below_entry_windows = 0;
    physical_op_candidate_below_entry_logged = false;
    physical_op_candidate_below_entry_window_total = 0;
    physical_op_refinement_below_entry_windows = 0;
    physical_op_refinement_below_entry_logged = false;
    physical_op_refinement_below_entry_window_total = 0;
    physical_op_candidate_abort_reason_total = {};
    physical_op_refinement_abort_reason_total = {};
    physical_op_recovery_partial_total = 0;
    physical_op_recovery_full_total = 0;
    physical_op_recovery_normal_exit_total = 0;
    physical_op_recovery_last_candidate_ns = 0;
    physical_op_recovery_last_outcome = "none";
    source_period_learned_samples_ns = {};
    source_period_learned_ns = 0;
    transition_planned_space_state = TransitionPlannedSpaceState::kIdle;
    h1c_fast_sequence_start_ns = 0;
    h1c_fast_sequence_start_issue = 0;
    h1c_fast_evidence_start_ns = 0;
    h1c_fast_evidence_start_issue = 0;
    h1c_fast_evidence_last_issue = 0;
    h1c_fast_evidence_valid = false;
    h1c_fast_evidence_attempted = false;
    h1c_fast_evidence_inhibited = false;
    h1c_confirm_issue = 0;
    h1c_drain_start_ns = 0;
    h1c_drain_deadline_issue = 0;
    h1c_drain_synthetic_source_watermark = 0;
    h1c_episode_real_only_pairs = 0;
    h1c_episode_quarantined_pairs = 0;
    h1c_episode_s_at_start = 0;
    h1c_episode_s_at_drain_start = 0;
    h1c_episode_frontier_at_start = 0;
    h1c_episode_chains_at_start = 0;
    h1c_episode_frontier_at_drain_start = 0;
    h1c_episode_chains_at_drain_start = 0;
    h1c_fast_sequence_start_total = 0;
    h1c_transition_real_only_pair_total = 0;
    h1c_false_positive_pair_total = 0;
    h1c_post_confirm_quarantine_total = 0;
    h1c_drain_complete_total = 0;
    h1c_timeout_total = 0;
    h1c_reset_total = 0;
    h1c_recovery_probe_inhibit_total = 0;
    h1c_planned_ticks_saved_total = 0;
    h1c_short_boundary_commit_total = 0;
    h1c_s_retired_during_drain_total = 0;
    h1c_pair_compressed_normal_total = 0;
    h1c_pair_planned_floor_blocked_total = 0;
    h1c_commit_trace_total = 0;
    h1c_short_raw_interval_ns = {};
    h1c_short_raw_median_ns = 0;
    h1c_short_raw_mean_ns = 0;
    h1c_short_raw_ready = false;
    h1c_short_raw_slow = false;
    h1c_fast_evidence_clean_count = 0;
    h1c_evidence_sequence_total = 0;
    h1c_second_sample_total = 0;
    h1c_single_sample_sequence_total = 0;
    h1c_single_sample_end_reason_total = {};
    h1c_episode_end_reason_total = {};
    h1c_raw_slow_reset_total = 0;
    h1c_raw_slow_blocked_sample_total = 0;
    h1c_raw_slow_bimodal_suspect_total = 0;
    h1c_physical_op_block_total = 0;
    h1c_evidence_log_total = 0;
    source_rate_probe_begin_ns = 0;
    source_rate_probe_next_due_ns = 0;
    source_rate_probe_interval_budget = 0;
    source_rate_probe_samples_ns = {};
    source_rate_probe_reference_period_ns = 0;
    source_rate_probe_min_candidate_ns = 0;
    source_rate_probe_max_candidate_ns = 0;
    source_rate_probe_start_issue_sequence = 0;
    source_rate_probe_deadline_issue_sequence = 0;
    source_rate_probe_attempt_total = 0;
    source_rate_probe_episode_started_total = 0;
    source_rate_probe_candidate_in_range_total = 0;
    source_rate_probe_candidate_out_of_range_total = 0;
    source_rate_probe_episode_expired_total = 0;
    source_rate_probe_confirmed_transition_total = 0;
    source_rate_probe_confirm_total = 0;
    source_rate_probe_no_change_total = 0;
    source_rate_probe_abort_total = 0;
    source_rate_probe_drop_total = 0;
    source_rate_probe_last_candidate_ns = 0;
    preadmission_free_since_ns = 0;
    capture_submit_to_first_poll_begin_ns = {};
    capture_submit_to_ready_poll_begin_ns = {};
    capture_first_seen_ready_ns = {};
    capture_last_seen_pending_ns = {};
    capture_observation_uncertainty_ns = {};
    capture_reclaim_after_first_seen_ns = {};
    capture_observation_poll_total = 0;
    capture_observation_ready_total = 0;
    capture_observation_error_total = 0;
    capture_observation_last_poll_ns = 0;
    active_generation_poll_host_ns = {};
    active_post_poll_host_ns = {};
    for (auto& window : presenter_wait_overshoot_ns) {
      window = {};
    }
    apply_call_phase_error_ns = {};
    apply_call_phase_early_ns = {};
    apply_call_phase_late_ns = {};
    actual_apply_lead_ns = {};
    ordered_ready_wait_ns = {};
    hard_real_target_lateness_ns = {};
    hard_real_waiting_post_not_before_lateness_ns = {};
    hard_real_waiting_post_soft_deadline_lateness_ns = {};
    hard_real_waiting_post_free_final_outputs = {};
    hard_real_waiting_post_required_surplus = {};
    hard_real_waiting_post_earlier_unfunded = {};
    hard_real_final_ready_age_ns = {};
    hard_real_final_ready_runway_ns = {};
    hard_real_final_ready_late_at_ready_ns = {};
    generation_submit_ahead_of_latest_ns = {};
    generation_submit_late_ns = {};
    generation_submit_after_not_before_ns = {};
    post_submit_ahead_of_latest_ns = {};
    post_submit_late_ns = {};
    post_submit_after_not_before_ns = {};
    publish_to_apply_ns = {};
    ready_to_apply_ns = {};
    post_queue_wait_ns = {};
    post_submit_host_ns = {};
    post_host_block_ns = {};
    post_gpu_completion_ns = {};
    post_service_gpu_ns = {};
    post_submit_to_ready_ns = {};
    post_submit_host_input_complete_ns = {};
    post_submit_host_input_pending_ns = {};
    post_export_host_ns = {};
    post_submit_host_real_ns = {};
    post_submit_host_synthetic_ns = {};
    generation_queue_wait_ns = {};
    generation_submit_host_ns = {};
    generation_host_block_ns = {};
    generation_gpu_completion_ns = {};
    generation_service_gpu_ns = {};
    profile_readback_attempt_total = 0;
    profile_readback_success_total = 0;
    profile_readback_not_ready_total = 0;
    profile_readback_error_total = 0;
    generation_submit_to_ready_ns = {};
    generation_setup_ns = {};
    generation_total_ns = {};
    generation_bootstrap_total_ns = {};
    generation_steady_total_ns = {};
    synthetic_chain_residence_ns = {};
    synthetic_nominal_lateness_ns = {};
    synthetic_final_ready_lead_ns = {};
    synthetic_deferred_feedback_success_total = 0;
    synthetic_deferred_feedback_miss_total = 0;
    issue_to_candidate_ready_ns = {};
    ResetSourcePhaseTelemetry();
    unified_runway_shadow = {};
    e2b = {};
    residual_phase_shadow = {};
    source_period_samples_ns = {};
    source_transition_samples_ns = {};
    source_effective_rate_timestamps_ns = {};
    source_pair_interval_ns = {};
    source_effective_rate_last_ns = 0;
    source_effective_rate_observation_total = 0;
    source_effective_rate_unstable_total = 0;
    source_pair_shorter_than_rate_total = 0;
    source_pair_longer_than_rate_total = 0;
    source_protection_treatment_ns = {};
    source_period_ns = 0;
    source_period_learned_ns = 0;
    source_period_learned_samples_ns = {};
    ++period_epoch;
    period_sample_total = 0;
    source_rejected_sample_total = 0;
    source_transition_confirmed_total = 0;
    source_transition_slower_confirmed_total = 0;
    source_transition_sample_total = 0;
    split_apply_fence_wait_real_total = 0;
    split_apply_fence_wait_synthetic_total = 0;
    split_present_spacing_push_total = 0;
    source_transition_faster_refilled_total = 0;
    source_phase_debt_ours_reset_total = 0;
    latency_depth_decay_armed_reanchor_total = 0;
    latency_depth_reanchor_armed_by_decay = false;
    latency_depth_reanchor_previous_operating_ns = 0;
    latency_depth_decay_projection_compactable_ns = 0;
    latency_depth_decay_actual_compacted_total = 0;
    latency_depth_decay_already_tighter_total = 0;
    latency_depth_decay_blocked_immutable_total = 0;
    latency_depth_decay_blocked_reserve_only_total = 0;
    latency_depth_decay_forward_dilation_total = 0;
    reanchor_floor_term_observations_total = 0;
    for (uint64_t& total : reanchor_floor_winner_total) {
      total = 0;
    }
    reanchor_floor_reserve_only_block_total = 0;
    latency_depth_lattice_compacted_total = 0;
    latency_depth_lattice_compacted_ns = 0;
    split_present_spacing_push_max_ns = 0;
    split_preaccept_capacity_gate_total = 0;
    presentation_advance_ns = 0;
    presentation_advance_outputs_since = 0;
    presentation_advance_up_streak = 0;
    presentation_advance_late_streak = 0;
    presentation_advance_below_streak = 0;
    presentation_advance_gate_seen = 0;
    presentation_advance_capacity_down = 0;
    presentation_advance_steps_up = 0;
    presentation_advance_steps_down = 0;
    presentation_advance_max_ns = 0;
    presentation_advance_excess_max_ns = 0;
    split_preadmission_total = 0;
    split_preadmission_gate_deferred_total = 0;
    split_preadmission_other_deferred_total = 0;
    split_s_obligation_created_total = 0;
    split_s_waiting_for_physical_chain_total = 0;
    split_s_obligation_high_water = 0;
    split_b_ready_waiting_for_s_total = 0;
    split_forbidden_s_drop_total = 0;
    split_live_real_retarget_total = 0;
    stable_latency_metronome_armed = false;
    stable_output_quantum_ns = 0;
    operating_latency_ns = 0;
    latency_required_last_ns = 0;
    next_semantic_target_ns = 0;
    last_semantic_target_ns = 0;
    semantic_epoch_origin_ns = 0;
    next_semantic_tick_index = 0;
    last_accounted_semantic_target_ns = 0;
    last_applied_semantic_target_ns = 0;
    ordered_boundary_provenance = {};
    commit_route_pair_boundary_total = 0;
    commit_route_projection_total = 0;
    commit_projection_mask_total = {};
    commit_projection_mask_extra_tick_total = {};
    commit_boundary_kind_total = {};
    commit_ready_at_commit_total = 0;
    real_commit_late_total = 0;
    real_commit_observed_total = 0;
    real_ready_age_at_commit_ns = {};
    real_commit_runway_pair_ns = {};
    real_commit_runway_projection_ns = {};
    real_commit_lateness_ns = {};
    ++latency_epoch;
    latency_depth_reanchor_pending = false;
    latency_depth_reanchor_total = 0;
    latency_depth_reanchor_live_real_invalidation_total = 0;
    latency_depth_reanchor_future_anchor_ns = 0;
    latency_depth_reanchor_impulse_ns = 0;
    latency_metronome_arm_total = 0;
    latency_feasibility_miss_total = 0;
    latency_feedback_excluded_total = 0;
    latency_feedback_true_feasibility_total = 0;
    latency_feedback_excluded_residency_pressure_total = 0;
    latency_feedback_excluded_final_output_pressure_total = 0;
    latency_feedback_excluded_funding_total = 0;
    latency_feedback_excluded_ordering_total = 0;
    latency_feedback_excluded_scheduler_total = 0;
    latency_feedback_excluded_transition_total = 0;
    latency_miss_provenance_total = {};
    latency_miss_provenance_one_quantum_help_total = {};
    latency_success_total = 0;
    semantic_hold_total = 0;
    dispatch_lead_ns = kDispatchLeadBootstrapNs;
    dispatch_lead_controller_armed = false;
    dispatch_lead_attack_total = 0;
    dispatch_lead_last_attack_step_ns = 0;
    dispatch_lead_last_attack_time_ns = 0;
    dispatch_lead_hard_ready_miss_sample_total = 0;
    dispatch_lead_hard_ready_attack_total = 0;
    residency_recovery_post_physical_release_total = 0;
    presenter_arbiter_cycle_total = 0;
    presenter_zero_progress_cycle_total = 0;
    presenter_wait_call_total = 0;
    presenter_wait_blocked_ns_total = 0;
    arbiter_blocking_submits_current = 0;
    arbiter_blocking_submits_high_water = 0;
    arbiter_host_driver_ops_current = 0;
    arbiter_host_driver_ops_high_water = 0;
    arbiter_first_pump_completed = false;
    normal_driver_turn_prefers_poll = true;
    active_capture_poll_total = 0;
    active_generation_poll_total = 0;
    active_post_poll_total = 0;
    generation_poll_not_due_total = 0;
    post_poll_not_due_total = 0;
    generation_poll_retry_scheduled_total = 0;
    post_poll_retry_scheduled_total = 0;
    head_observation_not_due_total = 0;
    observation_bounded_retry_wake_total = 0;
    poll_allowed_head_critical_total = 0;
    poll_allowed_source_critical_total = 0;
    arbiter_last_blocking_operation = BlockingOperation::kNone;
    arbiter_last_blocking_begin_ns = 0;
    arbiter_last_blocking_end_ns = 0;
    previous_cycle_blocking_operation = BlockingOperation::kNone;
    previous_cycle_blocking_begin_ns = 0;
    previous_cycle_blocking_end_ns = 0;
    pump_logical_order_cpu_total_ns = 0;
    pump_logical_order_call_total = 0;
    hard_ready_miss_by_blocking_operation.fill(0);
    hard_ready_miss_crossed_by_blocking_operation.fill(0);
    hard_ready_miss_after_wake_reason.fill(0);
    hard_ready_miss_attribution_sample_total = 0;
    presenter_wait_requested_by_reason.fill(0);
    presenter_wait_deadline_return_by_reason.fill(0);
    presenter_wait_event_wake_total = 0;
    requested_wake_deadline_ns = 0;
    actual_wait_return_ns = 0;
    last_presenter_wake_reason = PresenterWakeReason::kProducerEvent;
    accepted_real_drop_violation_total = 0;
    apply_before_planned_violation_total = 0;
    multi_blocking_submit_quantum_violation_total = 0;
    multi_host_driver_op_quantum_violation_total = 0;
    first_pump_before_driver_violation_total = 0;
    semantic_real_tick_total = 0;
    semantic_synthetic_tick_total = 0;
    semantic_real_deferred_total = 0;
    semantic_real_deferred_tick_total = 0;
    future_cursor_advanced_by_real_defer_total = 0;
    future_cursor_advanced_by_real_defer_tick_total = 0;
    real_exec_debt_high_water = 0;
    synthetic_dropped_for_real_backlog_total = 0;
    real_exec_debt_discharge_tick_total = 0;
    real_exec_debt_zero_transition_total = 0;
    semantic_target_order_violation_total = 0;
    shallow_real_commit_total = 0;
    shallow_real_commit_nominal_total = 0;
    shallow_real_commit_backlog_total = 0;
    shallow_real_commit_backlog_tick_total = 0;
    shallow_real_commitment_violation_total = 0;
    pending_real_nominal_violation_total = 0;
    physical_real_acceptance_cursor_advance_total = 0;
    committed_real_high_water = 0;
    shallow_real_commitment_violation_latched = false;
    pending_real_nominal_violation_latched = false;
    soft_dispatch_miss_salvaged_real_total = 0;
    soft_dispatch_miss_salvaged_synthetic_total = 0;
    hard_target_miss_real_total = 0;
    hard_target_miss_synthetic_total = 0;
    real_late_eligible_total = 0;
    real_late_applied_total = 0;
    real_late_blocked_total = 0;
    real_late_blocked_order_total = 0;
    real_late_blocked_not_head_total = 0;
    real_late_blocked_structural_total = 0;
    real_old_policy_would_defer_total = 0;
    real_multi_late_apply_same_pump_total = 0;
    synthetic_admitted_anchored_total = 0;
    synthetic_admitted_unanchored_total = 0;
    synthetic_generation_submitted_before_anchor_total = 0;
    synthetic_generation_submitted_after_anchor_total = 0;
    synthetic_post_submitted_before_anchor_total = 0;
    synthetic_post_submitted_after_anchor_total = 0;
    synthetic_anchored_before_generation_total = 0;
    synthetic_anchored_after_generation_total = 0;
    synthetic_anchored_after_post_total = 0;
    synthetic_anchored_after_final_ready_total = 0;
    synthetic_final_ready_before_anchor_total = 0;
    synthetic_apply_blocked_unanchored_total = 0;
    synthetic_structural_orphan_reclaimed_total = 0;
    synthetic_unanchored_semantic_hard_drop_total = 0;
    anchored_s_reached_b_safe_without_generation_job_total = 0;
    synthetic_presentation_hold_total = 0;
    presentation_expired_before_generation_submit_total = 0;
    presentation_expired_generation_inflight_total = 0;
    presentation_expired_waiting_post_total = 0;
    presentation_expired_post_submitted_total = 0;
    presentation_expired_final_ready_total = 0;
    generation_submitted_after_presentation_hold_total = 0;
    generation_completed_after_presentation_hold_total = 0;
    post_submitted_after_presentation_hold_total = 0;
    production_retired_after_presentation_hold_total = 0;
    presentation_retired_reached_apply_total = 0;
    presentation_retired_blocked_real_total = 0;
    admitted_s_retired_without_generation_attempt_total = 0;
    h14_b_due_ready_s_not_ready_hold_total = 0;
    h14_b_due_ready_s_ready_total = 0;
    h14_s_applied_after_old_bsafe_total = 0;
    h14_b_applied_after_ready_s_total = 0;
    h14_b_head_pair_commit_total = 0;
    h14_b_head_current_epoch_commit_total = 0;
    h14_b_head_reanchor_cutover_total = 0;
    semantic_cursor_late_streak = 0;
    semantic_cursor_late_streak_max = 0;
    e1f_shadow_request_total = 0;
    e1f_shadow_would_move_total = 0;
    e1f_shadow_noop_total = 0;
    e1f_next_real_commit_same_pump_total = 0;
    e1f_same_pump_commit_would_use_old_cursor_total = 0;
    for (auto& count : hard_real_miss_by_state) {
      count.store(0, std::memory_order_relaxed);
    }
    hard_real_final_ready_target_slack_last_ns.store(
        0, std::memory_order_relaxed);
    hard_real_waiting_post_not_before_lateness_last_ns.store(
        0, std::memory_order_relaxed);
    hard_real_waiting_post_soft_deadline_lateness_last_ns.store(
        0, std::memory_order_relaxed);
    hard_real_waiting_post_free_final_outputs_last.store(
        0, std::memory_order_relaxed);
    hard_real_waiting_post_required_surplus_last.store(
        0, std::memory_order_relaxed);
    hard_real_waiting_post_earlier_unfunded_last.store(
        0, std::memory_order_relaxed);
    hard_real_final_ready_age_last_ns.store(0, std::memory_order_relaxed);
    hard_real_target_lateness_last_ns.store(0, std::memory_order_relaxed);
    hard_real_target_lateness_max_ns.store(0, std::memory_order_relaxed);
    semantic_stale_synthetic_drop_total = 0;
    semantic_apply_diagnostic_total = 0;
    final_ready_success_total = 0;
    final_ready_miss_total = 0;
    pair_opportunity_total = 0;
    stable_pair_opportunity_total = 0;
    stable_synthetic_applied_total = 0;
    stable_synthetic_dropped_total = 0;
    generation_attempt_total = 0;
    generation_to_post_chain_candidate_total = 0;
    generation_to_post_chain_submit_total = 0;
    generation_to_post_chain_retired_total = 0;
    synthetic_chain_warmup_attempt_total = 0;
    synthetic_chain_warmup_submitted_total = 0;
    synthetic_chain_warmup_completed_total = 0;
    synthetic_chain_predicted_bypassed_total = 0;
    synthetic_chain_warmup_applied_total = 0;
    synthetic_chain_warmup_hard_drop_total = 0;
    synthetic_chain_warmup_real_hard_miss_total = 0;
    synthetic_chain_warmup_source_protection_drop_total = 0;
    synthetic_chain_warmup_residency_pressure_drop_total = 0;
    synthetic_chain_warmup_capture_near_miss_baseline = 0;
    synthetic_chain_warmup_capture_near_miss_final = 0;
    synthetic_chain_warmup_started = false;
    synthetic_chain_estimator_armed = false;
    generation_bootstrap_attempt_total = 0;
    generation_bootstrap_success_total = 0;
    generation_bootstrap_estimate_ns = 0;
    generation_bootstrap_armed = false;
    generation_unsustainable = false;
    generation_unsustainable_total = 0;
    synthetic_residency_surplus_admission_total = 0;
    synthetic_residency_eager_admission_total = 0;
    synthetic_residency_active_pressure_drop_total = 0;
    synthetic_generation_eager_residency_total = 0;
    last_synthetic_slack_ns = 0;
    last_real_slack_ns = 0;
    synthetic_generated_total = 0;
    real_applied_total = 0;
    synthetic_applied_total = 0;
    synthetic_pool_high_water = 0;
    synthetic_pool_occupancy = 0;
    logical_outputs = {};
    generation_jobs = {};
    post_jobs = {};
    next_logical_token = 1;
    next_logical_sequence = 1;
    next_apply_sequence = 1;
    last_summary_transaction = 0;
    history_slot = UINT32_MAX;
    last_transaction_output_slot = UINT32_MAX;
    logical_output_high_water = 0;
    logical_synthetic_high_water = 0;
    future_real_commitment_high_water = 0;
    generation_in_flight_high_water = 0;
    post_in_flight_high_water = 0;
    post_acquire_fence_ready_total = 0;
    post_sync_fd_poll_total = 0;
    post_sync_fd_ready_total = 0;
    post_output_release_deferred_total = 0;
    transaction_in_flight_high_water = 0;
    arbitration_generation_selected_total = 0;
    arbitration_post_selected_total = 0;
    arbitration_post_urgent_total = 0;
    synthetic_post_submit_late_total = 0;
    synthetic_post_ready_late_total = 0;
    synthetic_nominal_late_total = 0;
    late_salvage_started_total = 0;
    late_salvage_applied_total = 0;
    late_salvage_safe_window_expired_total = 0;
    late_salvage_high_water = 0;
    new_s_blocked_no_logical_with_salvage_total = 0;
    new_s_blocked_no_s_slot_with_salvage_total = 0;
    real_hard_miss_while_late_salvage_total = 0;
    real_defer_while_late_salvage_total = 0;
    generation_jit_deferral_total = 0;
    post_jit_deferral_total = 0;
    synthetic_generation_would_jit_defer_total = 0;
    synthetic_post_would_jit_defer_total = 0;
    real_post_would_jit_defer_total = 0;
    synthetic_generation_eager_submit_total = 0;
    synthetic_post_eager_submit_total = 0;
    real_early_transfer_total = 0;
    early_real_transfer_attempt_total.store(0, std::memory_order_relaxed);
    early_real_transfer_success_total.store(0, std::memory_order_relaxed);
    early_real_transfer_blocked_funding_total.store(
        0, std::memory_order_relaxed);
    early_real_transfer_blocked_final_pool_total.store(
        0, std::memory_order_relaxed);
    early_real_transfer_blocked_ownership_total.store(
        0, std::memory_order_relaxed);
    early_real_transfer_actual_release_total.store(
        0, std::memory_order_relaxed);
    early_real_transfer_no_release_total.store(0,
                                               std::memory_order_relaxed);
    early_real_transfer_compound_pending_total.store(
        0, std::memory_order_relaxed);
    residency_recovery_debt_high_water.store(0,
                                             std::memory_order_relaxed);
    residency_recovery_with_prior_debt_total.store(
        0, std::memory_order_relaxed);
    residency_recovery_debt_discharged_total.store(
        0, std::memory_order_relaxed);
    residency_recovery_debt_recycled_before_apply_total.store(
        0, std::memory_order_relaxed);
    residency_recovery_generation_attempt_total = 0;
    residency_recovery_generation_submit_total = 0;
    residency_recovery_generation_poll_total = 0;
    residency_recovery_generation_drop_total = 0;
    residency_recovery_generation_physical_release_total = 0;
    synthetic_real_funding_yield_total.store(0,
                                             std::memory_order_relaxed);
    final_output_funding_block_total.store(0, std::memory_order_relaxed);
    funding_block_begin_ns = 0;
    funding_block_total_ns = 0;
    funding_block_ns = {};
    funding_policy_wait_ns = {};
    funding_physical_wait_ns = {};
    pair_funding_policy_wait_ns = {};
    for (size_t i = 0; i < size_t(FundingWaitClass::kCount); ++i) {
      funding_class_total_ns[i] = 0;
      funding_episode_open_total[i] = 0;
    }
    funding_segment_switch_total = 0;
    funding_closed_by_condition_total = 0;
    funding_closed_by_state_total = 0;
    funding_closed_by_acquire_total = 0;
    pair_causal_clean_funding_subtracted_total = 0;
    last_funding_census = {};
    better_d_need_ns = {};
    better_d_deficit_ns = {};
    better_d_window_epoch = 0;
    better_d_f_ns = 0;
    better_d_causal_last_ns = 0;
    better_d_armed_depth_ns = 0;
    better_d_qualified_total = 0;
    for (size_t i = 0; i < size_t(BetterDCensor::kCount); ++i) {
      better_d_censored_total[i] = 0;
    }
    better_d_backlog_marked_total = 0;
    funding_backlog_marked_total = 0;
    funding_backlog_censored_total = 0;
    behind_physical_total = 0;
    better_d_rise_total = 0;
    better_d_rise_ns_total = 0;
    better_d_decay_step_total = 0;
    better_d_f_at_ceiling = false;
    better_d_f_ceiling_enter_total = 0;
    better_d_f_ceiling_exit_total = 0;
    better_d_f_ceiling_excess_max_ns = 0;
    better_d_deferred_total = 0;
    better_d_deferred_resolved_total = 0;
    better_d_double_score_total = 0;
    better_d_issues_since_qualified = 0;
    better_d_tail_stale_active = false;
    better_d_stale_renewal_pending = false;
    better_d_qualified_since_stale = 0;
    better_d_stale_episode_total = 0;
    better_d_stale_renewal_recovered_total = 0;
    better_d_rise_unready_total = 0;
    better_d_anchor_lag_ns = {};
    better_d_source_late_ab_ns = {};
    better_d_epoch_reset_total = 0;
    better_d_arm_rise_total = 0;
    better_d_arm_decay_total = 0;
    better_d_rise_already_deep_total = 0;
    final_output_earlier_unfunded_high_water.store(
        0, std::memory_order_relaxed);
    residency_recovery_generation_advance_total.store(
        0, std::memory_order_relaxed);
    capture_reserve_near_miss_total.store(
        0, std::memory_order_relaxed);
    last_generation_failure_stage = GenerationFailureStage::kNone;
    last_generation_status = 0;
    last_generation_vk_result = VK_SUCCESS;
    last_generation_vk_result_valid = false;
    last_generation_submission_accepted = false;
    synthetic_drop_total = {};
    pipeline_snapshot_live_future_real_commitments.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_future_real_commitment_high_water.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_tail_real_target_minus_head_quanta.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_pending_uncommitted_real.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_committed_real.store(0, std::memory_order_relaxed);
    pipeline_snapshot_committed_real_high_water.store(
        0, std::memory_order_relaxed);
    for (auto& kind_states : pipeline_snapshot_logical_states) {
      for (auto& state : kind_states) {
        state.store(0, std::memory_order_relaxed);
      }
    }
    pipeline_snapshot_head_valid.store(false, std::memory_order_relaxed);
    pipeline_snapshot_head_sequence.store(0, std::memory_order_relaxed);
    pipeline_snapshot_head_kind.store(0, std::memory_order_relaxed);
    pipeline_snapshot_head_state.store(0, std::memory_order_relaxed);
    pipeline_snapshot_head_nominal_target_ns.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_head_assigned_target_ns.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_head_nominal_tick.store(0, std::memory_order_relaxed);
    pipeline_snapshot_head_assigned_tick.store(0, std::memory_order_relaxed);
    pipeline_snapshot_head_exec_debt.store(0, std::memory_order_relaxed);
    pipeline_snapshot_tail_valid.store(false, std::memory_order_relaxed);
    pipeline_snapshot_tail_kind.store(0, std::memory_order_relaxed);
    pipeline_snapshot_tail_source_id.store(0, std::memory_order_relaxed);
    pipeline_snapshot_tail_nominal_tick.store(0,
                                              std::memory_order_relaxed);
    pipeline_snapshot_tail_assigned_tick.store(0,
                                               std::memory_order_relaxed);
    pipeline_snapshot_tail_exec_debt.store(0, std::memory_order_relaxed);
    pipeline_snapshot_head_target_delta_ns.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_last_applied_semantic_target_ns.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_last_accounted_semantic_target_ns.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_next_semantic_target_delta_ns.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_real_final_ready_behind_head.store(
        0, std::memory_order_relaxed);
    for (auto& delta : pipeline_snapshot_real_final_ready_target_deltas_ns) {
      delta.store(0, std::memory_order_relaxed);
    }
    pipeline_snapshot_unfunded_total.store(0, std::memory_order_relaxed);
    pipeline_snapshot_real_unfunded.store(0, std::memory_order_relaxed);
    pipeline_snapshot_synthetic_unfunded.store(0,
                                               std::memory_order_relaxed);
    pipeline_snapshot_max_earlier_unfunded.store(0,
                                                 std::memory_order_relaxed);
    pipeline_snapshot_candidate_free.store(kPoolSize,
                                            std::memory_order_relaxed);
    pipeline_snapshot_candidate_ingress.store(0,
                                              std::memory_order_relaxed);
    pipeline_snapshot_candidate_residency.store(0,
                                                std::memory_order_relaxed);
    pipeline_snapshot_resource_starvation_mask.store(
        kResourceStarvationNone, std::memory_order_relaxed);
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      pipeline_snapshot_candidate_owner_refs[i].store(
          0, std::memory_order_relaxed);
      pipeline_snapshot_candidate_source_ids[i].store(
          0, std::memory_order_relaxed);
      pipeline_snapshot_candidate_states[i].store(
          uint32_t(SlotState::kFree), std::memory_order_relaxed);
    }
    pipeline_snapshot_residency_pressure_drops.store(
        0, std::memory_order_relaxed);
    pipeline_snapshot_orphan_ready_final_outputs.store(
        0, std::memory_order_relaxed);
    orphan_ready_final_output_total.store(0, std::memory_order_relaxed);
    final_output_owner_mismatch_total.store(0, std::memory_order_relaxed);
    source_protection_baseline_ns = 0;
    source_protection_strikes = 0;
    source_protection_would_strike_total = 0;
    source_protection_would_probe_total = 0;
    last_pair_synthetic_applied = false;
    post_effect_count = 0;
    pool_high_water.store(0, std::memory_order_relaxed);
    ingress_high_water.store(0, std::memory_order_relaxed);
    residency_high_water.store(0, std::memory_order_relaxed);
    residency_capacity_exceeded_total.store(0, std::memory_order_relaxed);
    residency_capacity_excess_high_water.store(0,
                                               std::memory_order_relaxed);
    final_output_high_water.store(0, std::memory_order_relaxed);
    applied_timestamps = {};
    terminal_logged.store(false, std::memory_order_relaxed);
    terminal_reason.store(TerminalReason::kNone, std::memory_order_relaxed);
    shutdown_requested.store(false, std::memory_order_release);
    detach_requested.store(false, std::memory_order_release);
    surface_connected = true;
    vulkan_device->BeginZeroFGSurface();
    XELOGI(
        "ZeroFGDeviceDomain source=A:0 presenter=B:0 "
        "owners=ingress_context,residency_slot,generation_context,post_slot "
        "completion=sync_fd pending_timeline_probes=false");
    // Stay dormant while the normal presenter establishes the uncontaminated
// native Source cadence. ZeroFG becomes final authority only on the frame
    // after that lock-free qualification succeeds.
    final_output_authority.store(false, std::memory_order_release);
    presenter_thread = std::thread(&Impl::PresenterThreadMain, this);
    accepting.store(false, std::memory_order_release);
    XELOGI(
        "ZeroFGC0: enabled generation={} capture_pool={} "
        "residency_pool={} final_pool={} "
        "extent={}x{} capture_storage=device_local_exchange final_format={} "
        "final_storage=B_local output=main_surface_fifo "
        "activation=native_source_qualification priority={}",
        generation, kCapturePoolSize, kRealResidencyCapacity,
        kFinalOutputPoolSize, extent.width,
        extent.height, int(final_output_format),
        elevated_presenter_priority ? "elevated" : "normal");
    return true;
  }

  void BeginSurfaceDisconnect() {
    // Close the acceptance boundary immediately, but keep the presenter alive
    // long enough to drain already-accepted work.
    source_activation_armed.store(false, std::memory_order_release);
    accepting.store(false, std::memory_order_release);
    std::shared_ptr<EventBridge> local_bridge = bridge;
    SignalPresenterWake(local_bridge, true);
  }

  void DestroySurfaceResourcesAfterSourceIdle() {
    if (!surface_connected) {
      return;
    }
    detach_requested.store(true, std::memory_order_release);
    std::shared_ptr<EventBridge> local_bridge = bridge;
    SignalPresenterWake(local_bridge, true);
    if (presenter_thread.joinable()) {
      // Lifecycle is outside the Source path. Give the output domain bounded
      // time to present already-accepted Reals and hand the Surface back. A
      // stalled compositor terminates output cleanup rather than extending an
      // unbounded wait into lifecycle teardown.
      if (local_bridge) {
        std::unique_lock<std::mutex> lock(local_bridge->mutex);
        local_bridge->condition.wait_for(
            lock, std::chrono::milliseconds(500), [&] {
              return !presenter_thread_started.load(
                  std::memory_order_acquire);
            });
      }
      if (presenter_thread_started.load(std::memory_order_acquire)) {
        shutdown_requested.store(true, std::memory_order_release);
        if (local_bridge) {
          {
            std::lock_guard<std::mutex> lock(local_bridge->mutex);
            local_bridge->alive = false;
          }
          SignalPresenterWake(local_bridge, true);
        }
      }
      presenter_thread.join();
    }
    // The presenter thread normally stopped the egress on its way out; this
    // covers a connection whose presenter thread never ran it. The egress must
    // be gone before the FinalOutputs its copies read are destroyed below.
    StopMainSurfaceEgress(false);
    WaitForIngressGpuCompletionForTeardown();
    if (local_bridge) {
      std::lock_guard<std::mutex> lock(local_bridge->mutex);
      local_bridge->alive = false;
    }
    uint64_t unresolved = 0;
    for (const CaptureSlot& capture : capture_slots) {
      if (capture.state.load(std::memory_order_acquire) !=
          CaptureState::kFree) {
        ++unresolved;
      }
    }
    for (const Slot& slot : slots) {
      if (slot.state.load(std::memory_order_acquire) != SlotState::kFree) {
        ++unresolved;
      }
    }
    if (unresolved) {
      lifecycle_unresolved_total.fetch_add(unresolved,
                                           std::memory_order_relaxed);
      XELOGW(
          "ZeroFGC0: lifecycle ending with {} non-free accepted/unaccepted "
          "slot states; storage will not be reused",
          unresolved);
    }
    bridge.reset();
    // Vulkan framebuffers/pipelines referencing FinalOutput views must be
    // destroyed after the output thread is joined, but before the images and
    // views below.
    if (presenter_device_drain_callback && !presenter_device_drain_callback()) {
      XELOGW(
          "ZeroFGC0: device-B teardown drain failed; retirement proof "
          "unavailable; continuing teardown under the explicit failure "
          "policy (Generation/Post shutdown callbacks still run)");
    }
    if (generation_shutdown_callback) {
      generation_shutdown_callback();
    }
    if (post_process_shutdown_callback) {
      post_process_shutdown_callback();
    }
    for (FinalOutputSlot& output : final_outputs) {
      DestroyFinalOutputSlot(output);
    }
    for (CaptureSlot& capture : capture_slots) {
      DestroyCaptureSlot(capture);
    }
    for (Slot& slot : slots) {
      DestroyResidencySlot(slot);
    }
    surface_connected = false;
    final_output_authority.store(false, std::memory_order_release);
    extent = {};
    final_output_format = VK_FORMAT_UNDEFINED;
    // The VkSurfaceKHR was only borrowed from A.
    main_surface = VK_NULL_HANDLE;
    main_surface_reconnect_requested.store(false, std::memory_order_release);
    shutdown_requested.store(false, std::memory_order_relaxed);
    detach_requested.store(false, std::memory_order_relaxed);
  }

  // ---------------------------------------------------------------------
  // Main Surface Authority (MSA)
  // ---------------------------------------------------------------------

  // Refusal outside a live connection: nothing to hand back, ZeroFG simply
  // stays off and A keeps the Surface.
  void RefuseMainSurface(const char* reason) {
    main_surface_refusal_reason.store(reason, std::memory_order_release);
    main_surface_refused.store(true, std::memory_order_release);
    XELOGE("ZeroFGMainSurface refused reason={} fallback=native_A", reason);
  }

  void ResetMainSurfaceConnectionState() {
    main_surface_handoff_request_begin_ns = 0;
    main_surface_handoff_next_request_ns = 0;
    main_surface_gap_begin_ns = 0;
    main_surface_outdated_begin_ns = 0;
    main_surface_outdated_next_request_ns = 0;
    main_surface_reconnect_requested.store(false, std::memory_order_release);
    main_surface_ring_full_total = 0;
    main_surface_waiting_for_b_total = 0;
    main_surface_waiting_for_b_last_sequence = 0;
    main_surface_waiting_for_b_begin_ns = 0;
    main_surface_waiting_for_b_max_ns = 0;
    main_surface_released_total = 0;
    main_surface_abandoned_total = 0;
    main_surface_completion_stale_total = 0;
    main_surface_applied_total = 0;
  }

  // Owner thread (presenter thread, or the UI thread in teardown). Joins the
  // egress thread (bounded acquire), idles its queue (not formally bounded)
  // and destroys B's swapchain; only then does the producer state leave B.
  void StopMainSurfaceEgress(bool log_summary) {
    // The UI thread may be notifying the egress (A retiring) right now: take
    // it out under the lock, then stop it outside.
    main_surface_egress_live.store(false, std::memory_order_release);
    std::unique_ptr<ZeroFGMainSurfaceEgress> egress;
    {
      std::lock_guard<std::mutex> lock(main_surface_egress_mutex);
      egress = std::move(main_surface_egress);
    }
    if (!egress) {
      return;
    }
    if (log_summary) {
      LogMainSurface("handback", egress.get());
    }
    egress->Stop();
  }

  // Presenter thread, once per cycle.
  void ObserveMainSurfaceAuthority() {
    if (!main_surface_egress ||
        terminal_reason.load(std::memory_order_acquire) !=
            TerminalReason::kNone ||
        detach_requested.load(std::memory_order_acquire)) {
      // A failing or detaching connection hands back; it never starts a new
      // handoff.
      return;
    }
    const ZeroFGMainSurfaceEgress::Failure failure =
        main_surface_egress->failure();
    if (failure != ZeroFGMainSurfaceEgress::Failure::kNone) {
      if (terminal_reason.load(std::memory_order_acquire) ==
          TerminalReason::kNone) {
        main_surface_refusal_reason.store(
            ZeroFGMainSurfaceEgress::FailureName(failure),
            std::memory_order_release);
        XELOGE(
            "ZeroFGMainSurface failure reason={} action=fail_open_native_A",
            main_surface_refusal_reason.load(std::memory_order_acquire));
        Terminalize(TerminalReason::kMainSurfaceAuthority,
                    last_accepted_source_id.load(std::memory_order_relaxed));
      }
      return;
    }
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (main_surface_egress->surface_outdated()) {
      // The Surface changed under B: not an MSA failure. The UI thread
      // reconnects, which tears ZeroFG down and lets A rebuild the swapchain.
      if (!main_surface_reconnect_requested.exchange(
              true, std::memory_order_acq_rel)) {
        main_surface_outdated_begin_ns = now_ns;
        XELOGW("ZeroFGMainSurface surface_outdated action=ui_reconnect");
      }
      if (now_ns >= main_surface_outdated_next_request_ns) {
        main_surface_outdated_next_request_ns =
            now_ns + kMainSurfaceUIRequestIntervalNs;
        RequestMainSurfaceUIPaint();
      }
      if (now_ns - main_surface_outdated_begin_ns >
          kMainSurfaceTransferTimeoutNs) {
        main_surface_refusal_reason.store("outdated_without_reconnect",
                                          std::memory_order_release);
        XELOGE(
            "ZeroFGMainSurface failure reason={} action=fail_open_native_A",
            main_surface_refusal_reason.load(std::memory_order_acquire));
        Terminalize(TerminalReason::kMainSurfaceAuthority,
                    last_accepted_source_id.load(std::memory_order_relaxed));
      }
      return;
    }
    const ZeroFGMainSurfaceProducer::State producer =
        main_surface_producer.state();
    if (producer == ZeroFGMainSurfaceProducer::State::kA &&
        final_output_authority.load(std::memory_order_acquire)) {
      // ZeroFG owns output but A still holds the Surface: ask the UI thread
      // to retire A's swapchain, and fail open if that never happens.
      if (!main_surface_handoff_request_begin_ns) {
        main_surface_handoff_request_begin_ns = now_ns;
        XELOGI("ZeroFGMainSurface handoff requested producer=A");
      }
      if (now_ns >= main_surface_handoff_next_request_ns) {
        main_surface_handoff_next_request_ns =
            now_ns + kMainSurfaceUIRequestIntervalNs;
        RequestMainSurfaceUIPaint();
      }
      if (now_ns - main_surface_handoff_request_begin_ns >
          kMainSurfaceTransferTimeoutNs) {
        main_surface_refusal_reason.store("handoff_timeout",
                                          std::memory_order_release);
        XELOGE(
            "ZeroFGMainSurface failure reason={} action=fail_open_native_A",
            main_surface_refusal_reason.load(std::memory_order_acquire));
        Terminalize(TerminalReason::kMainSurfaceAuthority,
                    last_accepted_source_id.load(std::memory_order_relaxed));
      }
      return;
    }
    main_surface_handoff_request_begin_ns = 0;
    main_surface_handoff_next_request_ns = 0;
    if (producer == ZeroFGMainSurfaceProducer::State::kHandoffGap) {
      if (!main_surface_gap_begin_ns) {
        main_surface_gap_begin_ns = now_ns;
      }
      if (now_ns - main_surface_gap_begin_ns > kMainSurfaceTransferTimeoutNs) {
        main_surface_refusal_reason.store("handoff_gap_timeout",
                                          std::memory_order_release);
        XELOGE(
            "ZeroFGMainSurface failure reason={} action=fail_open_native_A",
            main_surface_refusal_reason.load(std::memory_order_acquire));
        Terminalize(TerminalReason::kMainSurfaceAuthority,
                    last_accepted_source_id.load(std::memory_order_relaxed));
      }
      return;
    }
    main_surface_gap_begin_ns = 0;
  }

  // UI thread, normal presenter paint path under painting ownership.
  bool MainSurfaceHandoffPending() const {
    return main_surface_egress_live.load(std::memory_order_acquire) &&
           final_output_authority.load(std::memory_order_acquire) &&
           accepting.load(std::memory_order_acquire) &&
           !detach_requested.load(std::memory_order_acquire) &&
           terminal_reason.load(std::memory_order_acquire) ==
               TerminalReason::kNone &&
           main_surface_producer.state() ==
               ZeroFGMainSurfaceProducer::State::kA;
  }

  // UI thread: A has destroyed its swapchain and kept the VkSurfaceKHR. Only
  // now may the egress create B's swapchain on it.
  void MainSurfaceReleasedByA() {
    if (!main_surface_producer.Transition(
            ZeroFGMainSurfaceProducer::State::kA,
            ZeroFGMainSurfaceProducer::State::kHandoffGap,
            "A_retired_swapchain")) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(main_surface_egress_mutex);
      if (main_surface_egress) {
        main_surface_egress->NotifyProducerChanged();
      }
    }
    std::shared_ptr<EventBridge> local_bridge = bridge;
    SignalPresenterWake(local_bridge);
  }

  // UI thread: A owns the Surface lifecycle again, after recreating its
  // swapchain or disconnecting the Surface. The ZeroFG connection, and with it
  // B's swapchain, was torn down before either.
  void NoteMainSurfaceProducedByA() {
    const ZeroFGMainSurfaceProducer::State state =
        main_surface_producer.state();
    if (state == ZeroFGMainSurfaceProducer::State::kHandbackGap) {
      main_surface_producer.Transition(
          ZeroFGMainSurfaceProducer::State::kHandbackGap,
          ZeroFGMainSurfaceProducer::State::kA, "A_owns_surface");
    } else if (state == ZeroFGMainSurfaceProducer::State::kHandoffGap) {
      // A retired, but the connection ended before B ever produced.
      main_surface_producer.Transition(
          ZeroFGMainSurfaceProducer::State::kHandoffGap,
          ZeroFGMainSurfaceProducer::State::kA,
          "A_owns_surface_handoff_abandoned");
    } else if (state == ZeroFGMainSurfaceProducer::State::kB) {
      main_surface_producer.CountViolation("A_owns_surface_while_B_produces");
    }
  }

  void RequestMainSurfaceUIPaint() {
    std::function<void()> request;
    {
      std::lock_guard<std::mutex> lock(main_surface_ui_request_mutex);
      request = main_surface_ui_request;
    }
    if (request) {
      request();
    }
  }

  // Presenter thread: an ordered head found Presentation closed because B
  // does not produce yet. Counted once per head; the wait is timed until the
  // first output B accepts.
  void NoteMainSurfaceWaitingForB(uint64_t sequence_id) {
    if (!main_surface_waiting_for_b_begin_ns) {
      main_surface_waiting_for_b_begin_ns = PresenterMonotonicTimeNs();
    }
    if (sequence_id != main_surface_waiting_for_b_last_sequence) {
      main_surface_waiting_for_b_last_sequence = sequence_id;
      ++main_surface_waiting_for_b_total;
    }
  }

  // Presenter thread: a FinalOutput comes back once the copy that read it has
  // finished (or its request was abandoned). Presentation owns nothing after
  // that; the swapchain image holds the content.
  bool ProcessMainSurfaceCompletions() {
    if (!main_surface_egress) {
      return false;
    }
    bool progress = false;
    ZeroFGMainSurfaceEgress::Completion completion;
    while (main_surface_egress->PopCompletion(completion)) {
      progress = true;
      if (completion.slot >= kFinalOutputPoolSize) {
        ++main_surface_completion_stale_total;
        continue;
      }
      FinalOutputSlot& output = final_outputs[completion.slot];
      if (output.sequence_id != completion.sequence_id ||
          output.state.load(std::memory_order_acquire) !=
              FinalOutputState::kTransactionApplied) {
        ++main_surface_completion_stale_total;
        continue;
      }
      if (!completion.presented) {
        ++main_surface_abandoned_total;
      }
      ReleaseFinalOutputSlot(completion.slot);
      output.last_release_fenced = completion.presented;
      ++main_surface_released_total;
    }
    return progress;
  }

  void LogMainSurface(const char* event,
                      const ZeroFGMainSurfaceEgress* egress) {
    const ZeroFGMainSurfaceEgress::Stats stats =
        egress ? egress->SnapshotStats() : ZeroFGMainSurfaceEgress::Stats();
    const uint64_t timings_settled =
        stats.timings_returned + stats.overtaken + stats.timings_missing;
    const uint64_t outstanding_timing =
        stats.timing_available && stats.presents > timings_settled
            ? stats.presents - timings_settled
            : 0;
    // A wait still open (B never produced) counts up to now.
    const uint64_t log_now_ns = PresenterMonotonicTimeNs();
    const uint64_t waiting_for_b_max_ns = std::max(
        main_surface_waiting_for_b_max_ns,
        main_surface_waiting_for_b_begin_ns &&
                log_now_ns > main_surface_waiting_for_b_begin_ns
            ? log_now_ns - main_surface_waiting_for_b_begin_ns
            : uint64_t(0));
    XELOGI(
        "ZeroFGMainSurface event={} producer={} "
        "handoff begin/done={}/{} handback begin/done={}/{} "
        "dual_producer_violation={} active={} refused={} "
        "reason={} timing_available={} queue=B:{} images={} "
        "refresh_cycle_us created/now/mode_changes={}/{}/{} applied={} "
        "presents R/S={}/{} released={} "
        "abandoned={} ring_full={} waiting_for_B outputs/max_us={}/{} "
        "stale_completion={} "
        "acquire_timeouts={} suboptimal={} outdated={} copy_fence_errors={} "
        "timings returned/overtaken/missing/unmatched/query_errors="
        "{}/{}/{}/{}/{} "
        "outstanding_timing_records_estimate={} actual_early={} "
        "apocalypse_guard={} apocalypse_latched={} "
        "latch_windows evaluated/pathology/confirmed={}/{}/{} gpu_busy_pm={} "
        "shaper_raised={} shaper_push_us p50/p90/max={}/{}/{} "
        "actual_interval_us min/p10/p50/p90/p99/max={}/{}/{}/{}/{}/{} "
        "desired_to_actual_us p50/p90/p99/max={}/{}/{}/{} "
        "acquire_block_us p50/p90/max={}/{}/{} "
        "copy_residence_us p50/p90/max={}/{}/{} "
        "present_call_vs_desired_us p50/p90/max={}/{}/{} "
        "present_call_us p50/p90/max={}/{}/{} "
        "ready_vs_desired_us p50/p90/max={}/{}/{} "
        "ready_to_actual_us p50/p90/max={}/{}/{} "
        "present_margin_us p50/p90={}/{} "
        "earliest_gap_us p50/p90/max={}/{}/{} missed_earliest={} "
        "v1_ratchet=off inherited_tick=retired "
        "free_output={} free_dropped R/S={}/{} "
        "free_superseded R/S={}/{} present_limiter_us={} "
        "present_limiter engaged/cleared/rejected={}/{}/{} "
        "vsync_quantizer={} vsync quantized/reanchor/refused/absorbed="
        "{}/{}/{}/{} ready_before_vsync_us p10/p50/p90={}/{}/{} "
        "vsync hit/late={}/{} hit_min queue/ready_before_vsync_us={}/{} "
        "late_max queue/ready_before_vsync_us={}/{}",
        event,
        ZeroFGMainSurfaceProducer::Name(main_surface_producer.state()),
        main_surface_producer.handoff_begin_total(),
        main_surface_producer.handoff_done_total(),
        main_surface_producer.handback_begin_total(),
        main_surface_producer.handback_done_total(),
        main_surface_producer.violation_total(),
        stats.swapchain_ready,
        main_surface_refused.load(std::memory_order_acquire),
        main_surface_refusal_reason.load(std::memory_order_acquire),
        stats.timing_available,
        stats.queue_index, stats.image_count, stats.refresh_cycle_ns / 1000,
        stats.refresh_cycle_now_ns / 1000, stats.refresh_mode_changes,
        main_surface_applied_total,
        stats.presents - stats.presents_synthetic, stats.presents_synthetic,
        main_surface_released_total, stats.abandoned,
        main_surface_ring_full_total, main_surface_waiting_for_b_total,
        waiting_for_b_max_ns / 1000, main_surface_completion_stale_total,
        stats.acquire_timeouts, stats.suboptimal, stats.outdated,
        stats.copy_fence_errors, stats.timings_returned, stats.overtaken,
        stats.timings_missing, stats.timings_unmatched,
        stats.timing_query_errors, outstanding_timing, stats.actual_early,
        stats.apocalypse_guard, stats.apocalypse_latched,
        stats.latch_windows_evaluated, stats.latch_windows_pathology,
        stats.latch_windows_confirmed, stats.gpu_busy_permille,
        stats.shaper_raised, stats.shaper_push_p50_ns / 1000,
        stats.shaper_push_p90_ns / 1000, stats.shaper_push_max_ns / 1000,
        stats.actual_interval_min_ns / 1000,
        stats.actual_interval_p10_ns / 1000,
        stats.actual_interval_p50_ns / 1000,
        stats.actual_interval_p90_ns / 1000,
        stats.actual_interval_p99_ns / 1000,
        stats.actual_interval_max_ns / 1000,
        stats.desired_to_actual_p50_ns / 1000,
        stats.desired_to_actual_p90_ns / 1000,
        stats.desired_to_actual_p99_ns / 1000,
        stats.desired_to_actual_max_ns / 1000,
        stats.acquire_block_p50_ns / 1000, stats.acquire_block_p90_ns / 1000,
        stats.acquire_block_max_ns / 1000,
        stats.copy_residence_p50_ns / 1000,
        stats.copy_residence_p90_ns / 1000,
        stats.copy_residence_max_ns / 1000,
        stats.present_call_vs_desired_p50_ns / 1000,
        stats.present_call_vs_desired_p90_ns / 1000,
        stats.present_call_vs_desired_max_ns / 1000,
        stats.present_call_p50_ns / 1000, stats.present_call_p90_ns / 1000,
        stats.present_call_max_ns / 1000,
        stats.ready_vs_desired_p50_ns / 1000,
        stats.ready_vs_desired_p90_ns / 1000,
        stats.ready_vs_desired_max_ns / 1000,
        stats.ready_to_actual_p50_ns / 1000,
        stats.ready_to_actual_p90_ns / 1000,
        stats.ready_to_actual_max_ns / 1000,
        stats.present_margin_p50_ns / 1000,
        stats.present_margin_p90_ns / 1000,
        stats.earliest_gap_p50_ns / 1000, stats.earliest_gap_p90_ns / 1000,
        stats.earliest_gap_max_ns / 1000, stats.missed_earliest,
        stats.free_output, stats.free_dropped - stats.free_dropped_synthetic,
        stats.free_dropped_synthetic,
        stats.free_superseded - stats.free_superseded_synthetic,
        stats.free_superseded_synthetic, stats.present_limiter_ns / 1000,
        stats.present_limiter_engaged, stats.present_limiter_cleared,
        stats.present_limiter_rejected,
        stats.vsync_quantizer, stats.vsync_quantized,
        stats.vsync_phase_reanchor, stats.vsync_duplicate_refused,
        stats.vsync_gap_absorbed, stats.ready_before_vsync_p10_ns / 1000,
        stats.ready_before_vsync_p50_ns / 1000,
        stats.ready_before_vsync_p90_ns / 1000, stats.vsync_hit,
        stats.vsync_late, stats.hit_queue_before_vsync_min_ns / 1000,
        stats.hit_ready_before_vsync_min_ns / 1000,
        stats.late_queue_before_vsync_max_ns / 1000,
        stats.late_ready_before_vsync_max_ns / 1000);
  }

  bool TryActivateFromNativeSourceCadence() {
    if (accepting.load(std::memory_order_acquire) ||
        !source_activation_armed.load(std::memory_order_acquire) ||
        detach_requested.load(std::memory_order_acquire) ||
        terminal_reason.load(std::memory_order_acquire) !=
            TerminalReason::kNone) {
      return false;
    }
    source_native_qualification_attempt_total.fetch_add(
        1, std::memory_order_relaxed);
    const xe::SourceBoundarySnapshot snapshot =
        xe::GetSourceBoundarySnapshot();
    if (!snapshot.issue_periods.consistent) {
      return false;
    }
    const uint64_t start_issue = source_qualification_start_issue_total.load(
        std::memory_order_relaxed);
    const uint64_t current_issue = snapshot.totals.source_issue_total;
    const size_t new_interval_count = size_t(std::min<uint64_t>(
        current_issue > start_issue ? current_issue - start_issue : 0,
        snapshot.issue_periods.count));
    if (new_interval_count < kNativeSourceQualificationSamples) {
      return false;
    }
    const size_t qualification_interval_count =
        kNativeSourceQualificationSamples;
    std::array<uint64_t, xe::internal::kSourceBoundaryPeriodCapacity>
        intervals = {};
    const size_t first =
        snapshot.issue_periods.count - qualification_interval_count;
    std::copy_n(snapshot.issue_periods.intervals_ns.begin() + first,
                qualification_interval_count, intervals.begin());
    const uint64_t effective_period =
        PresenterAveragePeriodFromIntervals(intervals.data(),
                                     qualification_interval_count);
    if (!effective_period ||
        !PresenterEffectiveRateIntervalsStable(
            intervals.data(), qualification_interval_count,
            kSourceEffectiveRateToleranceDivisor)) {
      return false;
    }
    std::array<uint64_t, xe::internal::kSourceBoundaryPeriodCapacity> sorted =
        intervals;
    std::sort(sorted.begin(), sorted.begin() + qualification_interval_count);
    const uint64_t p10 = sorted[(qualification_interval_count - 1) / 10];
    const uint64_t median = sorted[(qualification_interval_count - 1) / 2];
    const uint64_t p90 =
        sorted[((qualification_interval_count - 1) * 9) / 10];
    if (median < 5000000ull || median > 100000000ull || p90 < p10) {
      return false;
    }

    bool expected = true;
    if (!source_activation_armed.compare_exchange_strong(
            expected, false, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      return false;
    }
    qualified_native_source_period_ns.store(effective_period,
                                             std::memory_order_release);
    presenter_authority_activation_source_id.store(
        current_issue, std::memory_order_release);
    presenter_authority_activation_time_ns.store(PresenterMonotonicTimeNs(),
                                              std::memory_order_release);
    source_validation_start_sequence.store(
        xe::internal::source_issue_period_ring().sequence.load(
            std::memory_order_acquire),
        std::memory_order_release);
    SetSourceBootstrapRealOnlyRemaining(kRealOnlyValidationSamples);
    source_real_only_baseline_period_ns = effective_period;
    source_real_only_c2_baseline_period_ns = effective_period;
    source_real_only_c2_baseline_epoch = period_epoch;
    source_real_only_c2_baseline_valid = effective_period != 0;
    source_real_only_c2_native_reference_valid =
        source_real_only_c2_baseline_valid;
    source_real_only_c2_window_epoch = period_epoch;
    source_real_only_c2_window_remaining = kRealOnlyValidationSamples;
    source_real_only_c2_rate_timestamps_ns = {};
    source_real_only_c2_window_bp_wait_ns = 0;
    if (source_real_only_c2_baseline_valid) {
      ++source_real_only_c2_baseline_acquired_total;
    }
    ResetPhaseDebt();
    source_native_qualification_success_total.fetch_add(
        1, std::memory_order_relaxed);
    accepting.store(true, std::memory_order_release);
    std::shared_ptr<EventBridge> local_bridge = bridge;
    SignalPresenterWake(local_bridge, true);
    XELOGI(
        "ZeroFGC0: native Source qualified P_rate_us={} pair_us "
        "p10/p50/p90={}/{}/{} samples={} activation_source={} "
        "Real-only validation={}",
        effective_period / 1000, p10 / 1000, median / 1000, p90 / 1000,
        qualification_interval_count, current_issue,
        kRealOnlyValidationSamples);
    return true;
  }

  uint32_t CountUsedSlots() const {
    uint32_t used = 0;
    for (const Slot& slot : slots) {
      used += slot.state.load(std::memory_order_acquire) != SlotState::kFree;
    }
    return used;
  }

  uint32_t CountSourceIngressSlots() const {
    uint32_t count = 0;
    for (const CaptureSlot& capture : capture_slots) {
      count += capture.state.load(std::memory_order_acquire) !=
               CaptureState::kFree;
    }
    return count;
  }

  uint32_t CountRealResidencySlots() const {
    return CountUsedSlots();
  }

  uint32_t CountCaptureTransfers() const {
    uint32_t count = 0;
    for (const Slot& slot : slots) {
      count += slot.state.load(std::memory_order_acquire) ==
               SlotState::kTransferSubmitted;
    }
    return count;
  }

  uint32_t CountCapturesWaitingForResidency() const {
    uint32_t count = 0;
    for (const CaptureSlot& capture : capture_slots) {
      count += capture.state.load(std::memory_order_acquire) ==
               CaptureState::kReadyWaitingResidency;
    }
    return count;
  }

  uint32_t CountResidencyRecoveryDebt() const {
    uint32_t count = 0;
    for (const FinalOutputSlot& output : final_outputs) {
      count += output.residency_recovery_debt.load(std::memory_order_acquire);
    }
    return count;
  }

  void UpdateCaptureWaitingResidencyHighWater() {
    const uint32_t waiting = CountCapturesWaitingForResidency();
    uint32_t old = capture_committed_waiting_residency_high_water.load(
        std::memory_order_relaxed);
    while (old < waiting &&
           !capture_committed_waiting_residency_high_water
                .compare_exchange_weak(
                old, waiting, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
    }
  }

  void UpdateCaptureFreeMinimum(uint32_t free_slots) {
    uint32_t old = capture_free_min.load(std::memory_order_relaxed);
    while (free_slots < old &&
           !capture_free_min.compare_exchange_weak(
               old, free_slots, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }

  void UpdateResidencyRecoveryDebtHighWater() {
    const uint32_t debt = CountResidencyRecoveryDebt();
    uint32_t old =
        residency_recovery_debt_high_water.load(std::memory_order_relaxed);
    while (old < debt &&
           !residency_recovery_debt_high_water.compare_exchange_weak(
               old, debt, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }

  void UpdateHighWater(uint32_t used) {
    uint64_t old = pool_high_water.load(std::memory_order_relaxed);
    while (old < used &&
           !pool_high_water.compare_exchange_weak(
               old, used, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    const uint64_t ingress = CountSourceIngressSlots();
    old = ingress_high_water.load(std::memory_order_relaxed);
    while (old < ingress &&
           !ingress_high_water.compare_exchange_weak(
               old, ingress, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    const uint64_t residency = CountRealResidencySlots();
    old = residency_high_water.load(std::memory_order_relaxed);
    while (old < residency &&
           !residency_high_water.compare_exchange_weak(
               old, residency, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    old = capture_pool_high_water.load(std::memory_order_relaxed);
    while (old < ingress &&
           !capture_pool_high_water.compare_exchange_weak(
               old, ingress, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    const uint64_t transfers = CountCaptureTransfers();
    old = capture_transfer_high_water.load(std::memory_order_relaxed);
    while (old < transfers &&
           !capture_transfer_high_water.compare_exchange_weak(
               old, transfers, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }

  static const char* BlockingOperationName(BlockingOperation operation) {
    switch (operation) {
      case BlockingOperation::kCapturePoll:
        return "capture_completion_poll";
      case BlockingOperation::kGenerationPoll:
        return "active_generation_poll";
      case BlockingOperation::kPostPoll:
        return "active_post_poll";
      case BlockingOperation::kIngressSubmit:
        return "ingress_submit";
      case BlockingOperation::kCaptureSubmit:
        return "capture_submit";
      case BlockingOperation::kRecoveryPost:
        return "recovery_post";
      case BlockingOperation::kNormalPost:
        return "normal_post";
      case BlockingOperation::kGeneration:
        return "generation";
      case BlockingOperation::kNone:
      case BlockingOperation::kCount:
        return "none";
    }
    return "none";
  }

  static bool IsBlockingSubmitOperation(BlockingOperation operation) {
    return operation == BlockingOperation::kIngressSubmit ||
           operation == BlockingOperation::kCaptureSubmit ||
           operation == BlockingOperation::kRecoveryPost ||
           operation == BlockingOperation::kNormalPost ||
           operation == BlockingOperation::kGeneration;
  }

  static const char* PresenterWakeReasonName(PresenterWakeReason reason) {
    switch (reason) {
      case PresenterWakeReason::kGuardStart:
        return "guard_start";
      case PresenterWakeReason::kPlannedDispatch:
        return "planned_dispatch";
      case PresenterWakeReason::kGpuPoll:
        return "gpu_poll";
      case PresenterWakeReason::kSemanticDeadline:
        return "semantic_deadline";
      case PresenterWakeReason::kProducerEvent:
        return "producer_event";
      case PresenterWakeReason::kCount:
        return "none";
    }
    return "none";
  }

  void BeginHostDriverOperation(BlockingOperation operation) {
    ++arbiter_host_driver_ops_current;
    arbiter_host_driver_ops_high_water = std::max(
        arbiter_host_driver_ops_high_water, arbiter_host_driver_ops_current);
    if (arbiter_host_driver_ops_current > 1) {
      ++multi_host_driver_op_quantum_violation_total;
    }
    if (!arbiter_first_pump_completed) {
      ++first_pump_before_driver_violation_total;
    }
    if (IsBlockingSubmitOperation(operation)) {
      ++arbiter_blocking_submits_current;
      arbiter_blocking_submits_high_water = std::max(
          arbiter_blocking_submits_high_water,
          arbiter_blocking_submits_current);
      if (arbiter_blocking_submits_current > 1) {
        ++multi_blocking_submit_quantum_violation_total;
      }
    }
    arbiter_last_blocking_operation = operation;
    arbiter_last_blocking_begin_ns = PresenterMonotonicTimeNs();
    arbiter_last_blocking_end_ns = 0;
  }

  void EndHostDriverOperation() {
    arbiter_last_blocking_end_ns = PresenterMonotonicTimeNs();
  }

  void BeginBlockingOperation(BlockingOperation operation) {
    BeginHostDriverOperation(operation);
  }

  void EndBlockingOperation() { EndHostDriverOperation(); }

  uint32_t CountUsedFinalOutputs() const {
    uint32_t used = 0;
    for (const FinalOutputSlot& output : final_outputs) {
      used += output.state.load(std::memory_order_acquire) !=
              FinalOutputState::kFree;
    }
    return used;
  }

  void UpdateFinalOutputHighWater() {
    const uint64_t used = CountUsedFinalOutputs();
    uint64_t old = final_output_high_water.load(std::memory_order_relaxed);
    while (old < used &&
           !final_output_high_water.compare_exchange_weak(
               old, used, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
  }

  void PublicationCommitted() {
    ingress_publication_pending.store(true, std::memory_order_release);
    std::shared_ptr<EventBridge> local_bridge = bridge;
    SignalPresenterWake(local_bridge);
  }

  void ClaimPublicationAuthority() {
    if (accepting.load(std::memory_order_acquire) &&
        !detach_requested.load(std::memory_order_acquire) &&
        terminal_reason.load(std::memory_order_acquire) ==
            TerminalReason::kNone) {
      final_output_authority.store(true, std::memory_order_release);
    }
  }

  void Terminalize(TerminalReason reason, uint64_t source_id,
                   PostFailureStage post_failure_stage =
                       PostFailureStage::kNone,
                   bool post_vk_result_valid = false,
                   VkResult post_vk_result = VK_SUCCESS,
                   bool post_submission_accepted = false) {
    source_activation_armed.store(false, std::memory_order_release);
    accepting.store(false, std::memory_order_release);
    detach_requested.store(true, std::memory_order_release);
    terminal_reason.store(reason, std::memory_order_release);
    ++fail_open_total;
    bool expected = false;
    if (terminal_logged.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel)) {
      std::array<uint32_t, size_t(SlotState::kCount)> state_counts = {};
      for (const Slot& slot : slots) {
        ++state_counts[size_t(slot.state.load(std::memory_order_acquire))];
      }
      std::array<uint32_t, size_t(CaptureState::kCount)> capture_counts = {};
      for (const CaptureSlot& capture : capture_slots) {
        ++capture_counts[
            size_t(capture.state.load(std::memory_order_acquire))];
      }
      const xe::SourceBoundarySnapshot source_snapshot =
          xe::GetSourceBoundarySnapshot();
      XELOGE(
          "ZeroFGC0TerminalSourceRate source_fps={:.2f} P_rate/O_us={}/{} "
          "pair_us p10/p50/p90={}/{}/{} window_rate_us={} "
          "observations/unstable={}/{} pair_short/long={}/{} "
          "boundary_rate_us issue/publish={}/{}",
          xe::GetSourceFps(), source_period_ns / 1000,
          stable_output_quantum_ns / 1000,
          source_pair_interval_ns.Quantile(10, 100) / 1000,
          source_pair_interval_ns.Quantile(50, 100) / 1000,
          source_pair_interval_ns.Quantile(90, 100) / 1000,
          source_effective_rate_last_ns / 1000,
          source_effective_rate_observation_total,
          source_effective_rate_unstable_total,
          source_pair_shorter_than_rate_total,
          source_pair_longer_than_rate_total,
          source_snapshot.issue_periods.consistent
              ? PresenterAveragePeriodFromIntervals(
                    source_snapshot.issue_periods.intervals_ns.data(),
                    source_snapshot.issue_periods.count) /
                    1000
              : 0,
          source_snapshot.publish_periods.consistent
              ? PresenterAveragePeriodFromIntervals(
                    source_snapshot.publish_periods.intervals_ns.data(),
                    source_snapshot.publish_periods.count) /
                    1000
              : 0);
      XELOGE(
          "ZeroFGC0Terminal generation={} reason={} source_id={} accepted={} applied={} "
          "released={} capture={}/{} capture_states="
          "free:{},ingress_reserved:{},wait_residency:{} "
          "residency={}/{} states=free:{},transfer:{},ready:{} "
          "source_queue_wait samples/total_us/max_us={}/{}/{} "
          "logical total/high/real/synthetic={}/{}/{}/{} "
          "jobs capture/generation/post={}/{}/{} "
          "final free/post/ready/applied={}/{}/{}/{} "
          "post_failure_stage={} post_vk_result_valid={} post_vk_result={} "
          "post_submission_accepted={} generation_failure_stage={} "
          "generation_status={} generation_vk_result_valid={} "
          "generation_vk_result={} generation_submission_accepted={}",
          generation, TerminalReasonName(reason), source_id,
          last_accepted_source_id.load(std::memory_order_relaxed),
          last_applied_source_id.load(std::memory_order_relaxed),
          last_released_source_id.load(std::memory_order_relaxed),
          CountSourceIngressSlots(), kCapturePoolSize,
          capture_counts[size_t(CaptureState::kFree)],
          capture_counts[size_t(CaptureState::kIngressReserved)],
          capture_counts[size_t(CaptureState::kReadyWaitingResidency)],
          CountUsedSlots(), kPoolSize,
          state_counts[size_t(SlotState::kFree)],
          state_counts[size_t(SlotState::kTransferSubmitted)],
          state_counts[size_t(SlotState::kReady)],
          source_snapshot.totals.source_queue_lock_wait_count,
          source_snapshot.totals.source_queue_lock_wait_total_ns / 1000,
          source_snapshot.totals.source_queue_lock_wait_max_ns / 1000,
          pipeline_snapshot_logical_total.load(std::memory_order_relaxed),
          pipeline_snapshot_logical_high_water.load(
              std::memory_order_relaxed),
          pipeline_snapshot_logical_real.load(std::memory_order_relaxed),
          pipeline_snapshot_logical_synthetic.load(
              std::memory_order_relaxed),
          CountCaptureTransfers(),
          pipeline_snapshot_generation_jobs.load(std::memory_order_relaxed),
          pipeline_snapshot_post_jobs.load(std::memory_order_relaxed),
          pipeline_snapshot_final_states[size_t(FinalOutputState::kFree)]
              .load(std::memory_order_relaxed),
          pipeline_snapshot_final_states[size_t(
              FinalOutputState::kPostProcessing)]
              .load(std::memory_order_relaxed),
          pipeline_snapshot_final_states[size_t(FinalOutputState::kReady)]
              .load(std::memory_order_relaxed),
          pipeline_snapshot_final_states[size_t(
              FinalOutputState::kTransactionApplied)]
              .load(std::memory_order_relaxed),
          PostFailureStageName(post_failure_stage), post_vk_result_valid,
          int32_t(post_vk_result), post_submission_accepted,
          GenerationFailureStageName(last_generation_failure_stage),
          last_generation_status, last_generation_vk_result_valid,
          int32_t(last_generation_vk_result),
          last_generation_submission_accepted);
      XELOGE(
          "ZeroFGC0TerminalCommitment future_real current/high={} / {} "
          "tail_minus_head_quanta={} pending_real_nominal_violation={} "
          "real_accept_cursor_violation={}",
          pipeline_snapshot_live_future_real_commitments.load(
              std::memory_order_relaxed),
          pipeline_snapshot_future_real_commitment_high_water.load(
              std::memory_order_relaxed),
          pipeline_snapshot_tail_real_target_minus_head_quanta.load(
              std::memory_order_relaxed),
          pending_real_nominal_violation_total,
          physical_real_acceptance_cursor_advance_total);
      const bool head_valid =
          pipeline_snapshot_head_valid.load(std::memory_order_relaxed);
      const uint32_t head_kind =
          pipeline_snapshot_head_kind.load(std::memory_order_relaxed);
      const uint32_t head_state_value =
          pipeline_snapshot_head_state.load(std::memory_order_relaxed);
      const LogicalOutputState head_state =
          head_state_value < uint32_t(LogicalOutputState::kCount)
              ? LogicalOutputState(head_state_value)
              : LogicalOutputState::kCount;
      XELOGE(
          "ZeroFGC0TerminalPipeline head valid/sequence/kind/state="
          "{}/{}/{}/{} unfunded total/max_earlier={} / {} "
          "logical_state R wait_gen/wait_post/post/ready/dropped="
          "{}/{}/{}/{}/{} S={}/{}/{}/{}/{} funding blocks/high={} / {} "
          "residency_pressure_drops={} "
          "ready_orphan current/recovered/owner_mismatch={}/{}/{}",
          head_valid,
          pipeline_snapshot_head_sequence.load(std::memory_order_relaxed),
          head_valid ? (head_kind == uint32_t(CandidateKind::kSynthetic) ? "S"
                                                                        : "R")
                     : "none",
          LogicalOutputStateName(head_state),
          pipeline_snapshot_unfunded_total.load(std::memory_order_relaxed),
          pipeline_snapshot_max_earlier_unfunded.load(
              std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kReal)]
                                          [size_t(LogicalOutputState::kWaitingGeneration)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kReal)]
                                          [size_t(LogicalOutputState::kWaitingPost)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kReal)]
                                          [size_t(LogicalOutputState::kPostSubmitted)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kReal)]
                                          [size_t(LogicalOutputState::kFinalReady)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kReal)]
                                          [size_t(LogicalOutputState::kDropped)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kSynthetic)]
                                          [size_t(LogicalOutputState::kWaitingGeneration)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kSynthetic)]
                                          [size_t(LogicalOutputState::kWaitingPost)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kSynthetic)]
                                          [size_t(LogicalOutputState::kPostSubmitted)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kSynthetic)]
                                          [size_t(LogicalOutputState::kFinalReady)]
                                              .load(std::memory_order_relaxed),
          pipeline_snapshot_logical_states[uint32_t(CandidateKind::kSynthetic)]
                                          [size_t(LogicalOutputState::kDropped)]
                                              .load(std::memory_order_relaxed),
          final_output_funding_block_total.load(std::memory_order_relaxed),
          final_output_earlier_unfunded_high_water.load(
              std::memory_order_relaxed),
          pipeline_snapshot_residency_pressure_drops.load(
              std::memory_order_relaxed),
          pipeline_snapshot_orphan_ready_final_outputs.load(
              std::memory_order_relaxed),
          orphan_ready_final_output_total.load(std::memory_order_relaxed),
          final_output_owner_mismatch_total.load(std::memory_order_relaxed));
      XELOGE(
          "ZeroFGC0TerminalSemantic head nominal/assigned/delta_us="
          "{}/{}/{} clock last_applied/last_accounted/next_delta_us="
          "{}/{}/{} FinalReady_R behind/deltas_us={}/[{},{},{}]",
          pipeline_snapshot_head_nominal_target_ns.load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_head_assigned_target_ns.load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_head_target_delta_ns.load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_last_applied_semantic_target_ns.load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_last_accounted_semantic_target_ns.load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_next_semantic_target_delta_ns.load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_real_final_ready_behind_head.load(
              std::memory_order_relaxed),
          pipeline_snapshot_real_final_ready_target_deltas_ns[0].load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_real_final_ready_target_deltas_ns[1].load(
              std::memory_order_relaxed) /
              1000,
          pipeline_snapshot_real_final_ready_target_deltas_ns[2].load(
              std::memory_order_relaxed) /
              1000);
      XELOGE(
          "ZeroFGC0TerminalExecutiveDebt head nominal/assigned/debt="
          "{}/{}/{} tail valid/kind/source/nominal/assigned/debt="
          "{}/{}/{}/{}/{}/{} current/high={} / {} pending/committed/high="
          "{}/{}/{} commits total/nominal/backlog/ticks={}/{}/{}/{} "
          "backlog_S={} discharge/zero={}/{} "
          "future_cursor_by_defer/ticks={}/{}",
          pipeline_snapshot_head_nominal_tick.load(
              std::memory_order_relaxed),
          pipeline_snapshot_head_assigned_tick.load(
              std::memory_order_relaxed),
          pipeline_snapshot_head_exec_debt.load(std::memory_order_relaxed),
          pipeline_snapshot_tail_valid.load(std::memory_order_relaxed),
          pipeline_snapshot_tail_valid.load(std::memory_order_relaxed)
              ? (pipeline_snapshot_tail_kind.load(
                         std::memory_order_relaxed) ==
                     uint32_t(CandidateKind::kSynthetic)
                     ? "S"
                     : "R")
              : "none",
          pipeline_snapshot_tail_source_id.load(std::memory_order_relaxed),
          pipeline_snapshot_tail_nominal_tick.load(
              std::memory_order_relaxed),
          pipeline_snapshot_tail_assigned_tick.load(
              std::memory_order_relaxed),
          pipeline_snapshot_tail_exec_debt.load(std::memory_order_relaxed),
          pipeline_snapshot_tail_exec_debt.load(std::memory_order_relaxed),
          real_exec_debt_high_water,
          pipeline_snapshot_pending_uncommitted_real.load(
              std::memory_order_relaxed),
          pipeline_snapshot_committed_real.load(std::memory_order_relaxed),
          pipeline_snapshot_committed_real_high_water.load(
              std::memory_order_relaxed),
          shallow_real_commit_total, shallow_real_commit_nominal_total,
          shallow_real_commit_backlog_total,
          shallow_real_commit_backlog_tick_total,
          synthetic_dropped_for_real_backlog_total,
          real_exec_debt_discharge_tick_total,
          real_exec_debt_zero_transition_total,
          future_cursor_advanced_by_real_defer_total,
          future_cursor_advanced_by_real_defer_tick_total);
      XELOGE(
          "ZeroFGC0TerminalHardRealMiss state wait_gen/wait_post/post/ready/"
          "dropped={}/{}/{}/{}/{} target_lateness_us last/max={}/{} "
          "WaitingPost last not_before_late/soft_late_us="
          "{}/{} free/surplus/earlier={}/{}/{} FinalReady last age/"
          "target_slack_us={}/{}",
          hard_real_miss_by_state[size_t(
              LogicalOutputState::kWaitingGeneration)]
              .load(std::memory_order_relaxed),
          hard_real_miss_by_state[size_t(LogicalOutputState::kWaitingPost)]
              .load(std::memory_order_relaxed),
          hard_real_miss_by_state[size_t(LogicalOutputState::kPostSubmitted)]
              .load(std::memory_order_relaxed),
          hard_real_miss_by_state[size_t(LogicalOutputState::kFinalReady)]
              .load(std::memory_order_relaxed),
          hard_real_miss_by_state[size_t(LogicalOutputState::kDropped)].load(
              std::memory_order_relaxed),
          hard_real_target_lateness_last_ns.load(std::memory_order_relaxed) /
              1000,
          hard_real_target_lateness_max_ns.load(std::memory_order_relaxed) /
              1000,
          hard_real_waiting_post_not_before_lateness_last_ns.load(
              std::memory_order_relaxed) /
              1000,
          hard_real_waiting_post_soft_deadline_lateness_last_ns.load(
              std::memory_order_relaxed) /
              1000,
          hard_real_waiting_post_free_final_outputs_last.load(
              std::memory_order_relaxed),
          hard_real_waiting_post_required_surplus_last.load(
              std::memory_order_relaxed),
          hard_real_waiting_post_earlier_unfunded_last.load(
              std::memory_order_relaxed),
          hard_real_final_ready_age_last_ns.load(std::memory_order_relaxed) /
              1000,
          hard_real_final_ready_target_slack_last_ns.load(
              std::memory_order_relaxed) /
              1000);
      XELOGE(
          "ZeroFGC0TerminalDispatchLead lead_us={} hard_ready samples/"
          "estimator_attacks_legacy={}/{} telemetry_only dispatch_late_us "
          "p50/p90/p99/max={}/{}/{}/{}",
          dispatch_lead_ns / 1000,
          dispatch_lead_hard_ready_miss_sample_total,
          dispatch_lead_hard_ready_attack_total,
          hard_ready_dispatch_late_ns.Quantile(50, 100) / 1000,
          hard_ready_dispatch_late_ns.Quantile(90, 100) / 1000,
          hard_ready_dispatch_late_ns.Quantile(99, 100) / 1000,
          hard_ready_dispatch_late_ns.maximum() / 1000);
      XELOGE("ZeroFGC0TerminalRecoveryPost physical_release={}",
             residency_recovery_post_physical_release_total);
      XELOGE(
          "ZeroFGC0TerminalArbiter B_us={} armed={} cycles={} "
          "blocking_high={} host_driver_high={} "
          "host_block_us post_p90/max={}/{} generation_p90/max={}/{} "
          "pump_us p50/p90/p99/max={}/{}/{}/{} pump_calls/cpu_total_us="
          "{}/{}",
          BlockingHazardLeadNs() / 1000,
          BlockingHazardEvidenceSamples() >= kBlockingHazardArmSamples,
          presenter_arbiter_cycle_total, arbiter_blocking_submits_high_water,
          arbiter_host_driver_ops_high_water,
          post_host_block_ns.Quantile(90, 100) / 1000,
          post_host_block_ns.maximum() / 1000,
          generation_host_block_ns.Quantile(90, 100) / 1000,
          generation_host_block_ns.maximum() / 1000,
          pump_logical_order_ns.Quantile(50, 100) / 1000,
          pump_logical_order_ns.Quantile(90, 100) / 1000,
          pump_logical_order_ns.Quantile(99, 100) / 1000,
          pump_logical_order_ns.maximum() / 1000,
          pump_logical_order_call_total,
          pump_logical_order_cpu_total_ns / 1000);
      XELOGE(
          "ZeroFGC0TerminalWake requested guard/planned/gpu/semantic/event="
          "{}/{}/{}/{}/{} deadline_returns={}/{}/{}/{} event_wakes={} "
          "overshoot_us p90/max guard={}/{} planned={}/{} gpu={}/{} "
          "semantic={}/{} last_reason={} requested/actual_us={}/{}",
          presenter_wait_requested_by_reason[
              size_t(PresenterWakeReason::kGuardStart)],
          presenter_wait_requested_by_reason[
              size_t(PresenterWakeReason::kPlannedDispatch)],
          presenter_wait_requested_by_reason[
              size_t(PresenterWakeReason::kGpuPoll)],
          presenter_wait_requested_by_reason[
              size_t(PresenterWakeReason::kSemanticDeadline)],
          presenter_wait_requested_by_reason[
              size_t(PresenterWakeReason::kProducerEvent)],
          presenter_wait_deadline_return_by_reason[
              size_t(PresenterWakeReason::kGuardStart)],
          presenter_wait_deadline_return_by_reason[
              size_t(PresenterWakeReason::kPlannedDispatch)],
          presenter_wait_deadline_return_by_reason[
              size_t(PresenterWakeReason::kGpuPoll)],
          presenter_wait_deadline_return_by_reason[
              size_t(PresenterWakeReason::kSemanticDeadline)],
          presenter_wait_event_wake_total,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kGuardStart)].Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kGuardStart)].maximum() / 1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kPlannedDispatch)]
                  .Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kPlannedDispatch)].maximum() /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kGpuPoll)].Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kGpuPoll)].maximum() / 1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kSemanticDeadline)]
                  .Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kSemanticDeadline)].maximum() /
              1000,
          PresenterWakeReasonName(last_presenter_wake_reason),
          requested_wake_deadline_ns / 1000, actual_wait_return_ns / 1000);
      XELOGE(
          "ZeroFGC0TerminalWakeDistribution overshoot_us p50/p90/p99/max "
          "guard={}/{}/{}/{} planned={}/{}/{}/{} gpu={}/{}/{}/{} "
          "semantic={}/{}/{}/{}",
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGuardStart)]
                  .Quantile(50, 100) /
              1000,
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGuardStart)]
                  .Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGuardStart)]
                  .Quantile(99, 100) /
              1000,
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGuardStart)]
                  .maximum() /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kPlannedDispatch)]
                  .Quantile(50, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kPlannedDispatch)]
                  .Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kPlannedDispatch)]
                  .Quantile(99, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kPlannedDispatch)]
                  .maximum() /
              1000,
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGpuPoll)]
                  .Quantile(50, 100) /
              1000,
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGpuPoll)]
                  .Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGpuPoll)]
                  .Quantile(99, 100) /
              1000,
          presenter_wait_overshoot_ns[size_t(PresenterWakeReason::kGpuPoll)]
                  .maximum() /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kSemanticDeadline)]
                  .Quantile(50, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kSemanticDeadline)]
                  .Quantile(90, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kSemanticDeadline)]
                  .Quantile(99, 100) /
              1000,
          presenter_wait_overshoot_ns[
              size_t(PresenterWakeReason::kSemanticDeadline)]
                  .maximum() /
              1000);
      XELOGE(
          "ZeroFGC0TerminalFirstPump total_us p50/p90/p99/max={}/{}/{}/{} "
          "bounded_stage_p90/max callback={}/{} release={}/{} "
          "hard_miss_after_wake "
          "guard/planned/gpu/semantic/event={}/{}/{}/{}/{}",
          pre_first_pump_ns.Quantile(50, 100) / 1000,
          pre_first_pump_ns.Quantile(90, 100) / 1000,
          pre_first_pump_ns.Quantile(99, 100) / 1000,
          pre_first_pump_ns.maximum() / 1000,
          cycle_callback_ns.Quantile(90, 100) / 1000,
          cycle_callback_ns.maximum() / 1000,
          cycle_release_ns.Quantile(90, 100) / 1000,
          cycle_release_ns.maximum() / 1000,
          hard_ready_miss_after_wake_reason[
              size_t(PresenterWakeReason::kGuardStart)],
          hard_ready_miss_after_wake_reason[
              size_t(PresenterWakeReason::kPlannedDispatch)],
          hard_ready_miss_after_wake_reason[
              size_t(PresenterWakeReason::kGpuPoll)],
          hard_ready_miss_after_wake_reason[
              size_t(PresenterWakeReason::kSemanticDeadline)],
          hard_ready_miss_after_wake_reason[
              size_t(PresenterWakeReason::kProducerEvent)]);
      XELOGE(
          "ZeroFGC0TerminalHardReadyAttribution miss none/capture_poll/"
          "generation_poll/post_poll/capture_submit/recovery/post/generation="
          "{}/{}/{}/{}/{}/{}/{}/{} crossed={}/{}/{}/{}/{}/{}/{}/{}",
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kNone)],
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kCapturePoll)],
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kGenerationPoll)],
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kPostPoll)],
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kCaptureSubmit)],
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kRecoveryPost)],
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kNormalPost)],
          hard_ready_miss_by_blocking_operation[
              size_t(BlockingOperation::kGeneration)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kNone)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kCapturePoll)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kGenerationPoll)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kPostPoll)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kCaptureSubmit)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kRecoveryPost)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kNormalPost)],
          hard_ready_miss_crossed_by_blocking_operation[
              size_t(BlockingOperation::kGeneration)]);
      XELOGE(
          "ZeroFGC0TerminalInvariants accepted_real_drop/apply_before_planned/"
          "multi_blocking_submit/multi_host_driver/first_pump_before_driver/"
          "future_cursor_by_real_defer="
          "{}/{}/{}/{}/{}/{}",
          accepted_real_drop_violation_total,
          apply_before_planned_violation_total,
          multi_blocking_submit_quantum_violation_total,
          multi_host_driver_op_quantum_violation_total,
          first_pump_before_driver_violation_total,
          future_cursor_advanced_by_real_defer_total);
      XELOGE(
          "ZeroFGC0TerminalChainWarmup attempts/submitted/completed/armed="
          "{}/{}/{}/{} samples={} residence_us p50/p90/p99/max="
          "{}/{}/{}/{} predicted_bypassed={} S applied/hard_drop={}/{} "
          "Real hard_miss={} source/residency_drop={}/{} capture_near_miss={} "
          "live/physical_capacity={}/{}",
          synthetic_chain_warmup_attempt_total,
          synthetic_chain_warmup_submitted_total,
          synthetic_chain_warmup_completed_total,
          SyntheticChainEstimatorArmed(), synthetic_chain_residence_ns.count(),
          synthetic_chain_residence_ns.Quantile(50, 100) / 1000,
          synthetic_chain_residence_ns.Quantile(90, 100) / 1000,
          synthetic_chain_residence_ns.Quantile(99, 100) / 1000,
          synthetic_chain_residence_ns.WindowMaximum() / 1000,
          synthetic_chain_predicted_bypassed_total,
          synthetic_chain_warmup_applied_total,
          synthetic_chain_warmup_hard_drop_total,
          synthetic_chain_warmup_real_hard_miss_total,
          synthetic_chain_warmup_source_protection_drop_total,
          synthetic_chain_warmup_residency_pressure_drop_total,
          SyntheticChainWarmupCaptureNearMisses(),
          CountLiveSyntheticChains(), kSyntheticPoolSize);
      XELOGE(
          "ZeroFGC0TerminalProduction jit_shadow=counterfactual "
          "would_defer S_gen/S_post(shadow)/R_post(real)="
          "{}/{}/{} eager_submit gen/post={}/{} S_final_ready_lead_us "
          "p50/p90/p99/max={}/{}/{}/{} S_presentation_hard_miss={} "
          "live/high={} / {}",
          synthetic_generation_would_jit_defer_total,
          synthetic_post_would_jit_defer_total,
          real_post_would_jit_defer_total,
          synthetic_generation_eager_submit_total,
          synthetic_post_eager_submit_total,
          synthetic_final_ready_lead_ns.Quantile(50, 100) / 1000,
          synthetic_final_ready_lead_ns.Quantile(90, 100) / 1000,
          synthetic_final_ready_lead_ns.Quantile(99, 100) / 1000,
          synthetic_final_ready_lead_ns.maximum() / 1000,
          hard_target_miss_synthetic_total,
          CountLiveSyntheticChains(), generation_in_flight_high_water);
      XELOGE(
          "ZeroFGC0TerminalActivePoll calls completion/generation/post="
          "{}/{}/{} "
          "host_us capture_completion p50/p90/p99/max={}/{}/{}/{} "
          "generation={}/{}/{}/{} "
          "post={}/{}/{}/{} allowed head/source={}/{} "
          "not_due generation/post={}/{} "
          "retry generation/post={}/{} observation_est_us generation/post="
          "{}/{} head_not_due={} bounded_retry={} "
          "retry_min_us={}",
          active_capture_poll_total, active_generation_poll_total,
          active_post_poll_total,
          active_capture_poll_host_ns.Quantile(50, 100) / 1000,
          active_capture_poll_host_ns.Quantile(90, 100) / 1000,
          active_capture_poll_host_ns.Quantile(99, 100) / 1000,
          active_capture_poll_host_ns.maximum() / 1000,
          active_generation_poll_host_ns.Quantile(50, 100) / 1000,
          active_generation_poll_host_ns.Quantile(90, 100) / 1000,
          active_generation_poll_host_ns.Quantile(99, 100) / 1000,
          active_generation_poll_host_ns.maximum() / 1000,
          active_post_poll_host_ns.Quantile(50, 100) / 1000,
          active_post_poll_host_ns.Quantile(90, 100) / 1000,
          active_post_poll_host_ns.Quantile(99, 100) / 1000,
          active_post_poll_host_ns.maximum() / 1000,
          poll_allowed_head_critical_total,
          poll_allowed_source_critical_total,
          generation_poll_not_due_total,
          post_poll_not_due_total, generation_poll_retry_scheduled_total,
          post_poll_retry_scheduled_total,
          GenerationObservationEstimateNs() / 1000,
          PostObservationEstimateNs() / 1000,
          head_observation_not_due_total,
          observation_bounded_retry_wake_total,
          kObservationDueRetryIntervalNs / 1000);
      XELOGE(
          "ZeroFGC0TerminalCaptureReadyProvenance issue_observed_us "
          "p50/p90/p99/max={}/{}/{}/{} issue_submit_us={}/{}/{}/{} "
          "submit_first_poll_us={}/{}/{}/{} submit_ready_poll_us="
          "{}/{}/{}/{}",
          issue_to_candidate_ready_ns.Quantile(50, 100) / 1000,
          issue_to_candidate_ready_ns.Quantile(90, 100) / 1000,
          issue_to_candidate_ready_ns.Quantile(99, 100) / 1000,
          issue_to_candidate_ready_ns.maximum() / 1000,
          capture_issue_to_transfer_submit_ns.Quantile(50, 100) / 1000,
          capture_issue_to_transfer_submit_ns.Quantile(90, 100) / 1000,
          capture_issue_to_transfer_submit_ns.Quantile(99, 100) / 1000,
          capture_issue_to_transfer_submit_ns.maximum() / 1000,
          capture_submit_to_first_poll_begin_ns.Quantile(50, 100) / 1000,
          capture_submit_to_first_poll_begin_ns.Quantile(90, 100) / 1000,
          capture_submit_to_first_poll_begin_ns.Quantile(99, 100) / 1000,
          capture_submit_to_first_poll_begin_ns.maximum() / 1000,
          capture_submit_to_ready_poll_begin_ns.Quantile(50, 100) / 1000,
          capture_submit_to_ready_poll_begin_ns.Quantile(90, 100) / 1000,
          capture_submit_to_ready_poll_begin_ns.Quantile(99, 100) / 1000,
          capture_submit_to_ready_poll_begin_ns.maximum() / 1000);
      XELOGE(
          "ZeroFGC0TerminalCaptureReadyObservation poll_host_us "
          "p50/p90/p99/max={}/{}/{}/{} submit_observed_us={}/{}/{}/{}",
          active_capture_poll_host_ns.Quantile(50, 100) / 1000,
          active_capture_poll_host_ns.Quantile(90, 100) / 1000,
          active_capture_poll_host_ns.Quantile(99, 100) / 1000,
          active_capture_poll_host_ns.maximum() / 1000,
          capture_transfer_residence_ns.Quantile(50, 100) / 1000,
          capture_transfer_residence_ns.Quantile(90, 100) / 1000,
          capture_transfer_residence_ns.Quantile(99, 100) / 1000,
          capture_transfer_residence_ns.maximum() / 1000);
      XELOGE(
          "ZeroFGC0TerminalCaptureLiveness free_min={} committed_wait="
          "{}/{} cpu_promotions={} source_timeline_cpu_poll={} "
          "direct_timeline_wait_submits={} proactive attempts/submits={}/{} "
          "emergency attempts/success={}/{} reserve_recycle_us "
          "p50/p90/p99/max={}/{}/{}/{}",
          capture_free_min.load(std::memory_order_relaxed),
          CountCapturesWaitingForResidency(),
          capture_committed_waiting_residency_high_water.load(
              std::memory_order_relaxed),
          committed_capture_cpu_promotion_total.load(
              std::memory_order_relaxed),
          source_timeline_cpu_poll_total.load(std::memory_order_relaxed),
          direct_capture_timeline_wait_submit_total.load(
              std::memory_order_relaxed),
          capture_proactive_turnover_attempt_total,
          capture_proactive_turnover_submit_total,
          capture_emergency_progress_attempt_total,
          capture_emergency_progress_success_total,
          capture_reserve_to_recycle_ns.Quantile(50, 100) / 1000,
          capture_reserve_to_recycle_ns.Quantile(90, 100) / 1000,
          capture_reserve_to_recycle_ns.Quantile(99, 100) / 1000,
          capture_reserve_to_recycle_ns.maximum() / 1000);
      XELOGE(
          "ZeroFGC0TerminalSyntheticAdmission residency surplus/eager/"
          "active_pressure_drop={}/{}/{} eager_generation={} "
          "recovery_generation attempt/submit/poll/drop/physical_release="
          "{}/{}/{}/{}/{}",
          synthetic_residency_surplus_admission_total,
          synthetic_residency_eager_admission_total,
          synthetic_residency_active_pressure_drop_total,
          synthetic_generation_eager_residency_total,
          residency_recovery_generation_attempt_total,
          residency_recovery_generation_submit_total,
          residency_recovery_generation_poll_total,
          residency_recovery_generation_drop_total,
          residency_recovery_generation_physical_release_total);
      XELOGE(
          "ZeroFGC0TerminalLiveness residency free/ingress/used/capacity/"
          "capture_near_miss/residency_exceeded/max={}/{}/{}/{}/{}/{}/{} "
          "unfunded R/S={}/{} starvation_mask=0x{:X} "
          "early_real attempt/success/blocked_funding/blocked_no_final/"
          "with_prior_debt={}/{}/{}/{}/{} S_funding_yield={} "
          "recovery_debt current/high/discharged/recycled={}/{}/{}/{}",
          pipeline_snapshot_candidate_free.load(std::memory_order_relaxed),
          pipeline_snapshot_candidate_ingress.load(
              std::memory_order_relaxed),
          pipeline_snapshot_candidate_residency.load(
              std::memory_order_relaxed),
          kRealResidencyCapacity,
          capture_reserve_near_miss_total.load(
              std::memory_order_relaxed),
          residency_capacity_exceeded_total.load(
              std::memory_order_relaxed),
          residency_capacity_excess_high_water.load(
              std::memory_order_relaxed),
          pipeline_snapshot_real_unfunded.load(std::memory_order_relaxed),
          pipeline_snapshot_synthetic_unfunded.load(
              std::memory_order_relaxed),
          pipeline_snapshot_resource_starvation_mask.load(
              std::memory_order_relaxed),
          early_real_transfer_attempt_total.load(std::memory_order_relaxed),
          early_real_transfer_success_total.load(std::memory_order_relaxed),
          early_real_transfer_blocked_funding_total.load(
              std::memory_order_relaxed),
          early_real_transfer_blocked_final_pool_total.load(
              std::memory_order_relaxed),
          residency_recovery_with_prior_debt_total.load(
              std::memory_order_relaxed),
          synthetic_real_funding_yield_total.load(
              std::memory_order_relaxed),
          CountResidencyRecoveryDebt(),
          residency_recovery_debt_high_water.load(std::memory_order_relaxed),
          residency_recovery_debt_discharged_total.load(
              std::memory_order_relaxed),
          residency_recovery_debt_recycled_before_apply_total.load(
              std::memory_order_relaxed));
      XELOGE(
          "ZeroFGC0TerminalResidencyRecovery generation_advance={} "
          "early_release actual/no_release/compound/blocked_ownership="
          "{}/{}/{}/{}",
          residency_recovery_generation_advance_total.load(
              std::memory_order_relaxed),
          early_real_transfer_actual_release_total.load(
              std::memory_order_relaxed),
          early_real_transfer_no_release_total.load(
              std::memory_order_relaxed),
          early_real_transfer_compound_pending_total.load(
              std::memory_order_relaxed),
          early_real_transfer_blocked_ownership_total.load(
              std::memory_order_relaxed));
      for (uint32_t i = 0; i < kPoolSize; ++i) {
        XELOGE(
            "ZeroFGC0TerminalCandidateSlot index={} source/ref/state="
            "{}/{}/{}",
            i,
            pipeline_snapshot_candidate_source_ids[i].load(
                std::memory_order_relaxed),
            pipeline_snapshot_candidate_owner_refs[i].load(
                std::memory_order_relaxed),
            pipeline_snapshot_candidate_states[i].load(
                std::memory_order_relaxed));
      }
    }
    std::shared_ptr<EventBridge> local_bridge = bridge;
    SignalPresenterWake(local_bridge, true);
  }

  bool IsReleaseFenceReady(int fence_fd,
                           bool* observed_signal_out = nullptr,
                           bool* observation_error_out = nullptr) {
    if (observed_signal_out) {
      *observed_signal_out = false;
    }
    if (observation_error_out) {
      *observation_error_out = false;
    }
    if (fence_fd < 0) {
      return true;
    }
    pollfd descriptor = {fence_fd, POLLIN, 0};
    const int result = poll(&descriptor, 1, 0);
    const bool observed_signal =
        result > 0 && (descriptor.revents & POLLIN) &&
        !(descriptor.revents & (POLLERR | POLLNVAL));
    if (observed_signal_out) {
      *observed_signal_out = observed_signal;
    }
    if (observation_error_out) {
      *observation_error_out =
          result < 0 || (result > 0 && !observed_signal);
    }
    // Preserve the existing lifecycle readiness path for callers that do not
    // request physical proof; B1 never scores from an error-only poll event.
    return result > 0;
  }

  // Wait-side falsifier classes, ordered so a Generation's class is the
  // maximum of its two inputs' classes.
  enum InputReadiness : size_t {
    kInputComplete = 0,
    kInputUnknown = 1,
    kInputPending = 2,
    kInputReadinessCount = 3,
  };

  // Whether a Residency input is still pending on the GPU when the presenter
  // builds the Generation request, read without waiting. This is before
  // recording and queue acquisition, so an input can still retire before
  // vkQueueSubmit. An observed-ready slot is complete, a transfer with an
  // observable sync_fd (C2) is polled, anything else is unknown. An fd in
  // error is left for PollCaptureTransfers to abandon.
  size_t ClassifyResidencyInput(uint32_t slot_index) {
    if (slot_index >= kPoolSize) {
      return kInputUnknown;
    }
    const Slot& slot = slots[slot_index];
    const SlotState state = slot.state.load(std::memory_order_acquire);
    if (state == SlotState::kReady) {
      return kInputComplete;
    }
    if (state != SlotState::kTransferSubmitted ||
        !slot.completion.sync_fd_observable()) {
      return kInputUnknown;
    }
    ++generation_input_poll_total;
    switch (slot.completion.PollSyncFd()) {
      case ZeroFGCompletionOwner::SyncFdPoll::kSignaled:
        return kInputComplete;
      case ZeroFGCompletionOwner::SyncFdPoll::kPending:
        return kInputPending;
      default:
        return kInputUnknown;
    }
  }

  void ReleaseCandidateSlot(uint32_t slot_index) {
    if (slot_index >= kPoolSize) {
      return;
    }
    Slot& slot = slots[slot_index];
    slot.source_id.store(0, std::memory_order_relaxed);
    slot.issue_time_ns.store(0, std::memory_order_relaxed);
    slot.policy_wait_ns.store(0, std::memory_order_relaxed);
    slot.publish_time_ns.store(0, std::memory_order_relaxed);
    slot.timeline_value.store(0, std::memory_order_relaxed);
    slot.ready_timeline = VK_NULL_HANDLE;
    slot.source_content_extent = {};
    slot.storage_content_extent = {};
    slot.display_aspect_ratio_x = 0;
    slot.display_aspect_ratio_y = 0;
    slot.is_8bpc = false;
    slot.ready_time_ns = 0;
    slot.transfer_signal_value = 0;
    slot.transfer_submit_time_ns = 0;
    slot.transfer_first_poll_begin_time_ns = 0;
    slot.observed_pending_time_ns = 0;
    slot.observed_ready_time_ns = 0;
    slot.apply_time_ns = 0;
    slot.owner_refs = 0;
    slot.runtime_admitted = false;
    // The sync_fd ends with this occupancy. A pending recreation is kept: a
    // sync semaphore whose export failed still holds its signal and is
    // replaced before the slot's next capture-transfer submit.
    slot.completion.RetireSyncFd();
    slot.state.store(SlotState::kFree, std::memory_order_release);
    ++candidate_release_total;
  }

  void RetainRealSlot(uint32_t slot_index) {
    if (slot_index < kPoolSize) {
      ++slots[slot_index].owner_refs;
    }
  }

  void ReleaseRealSlot(uint32_t slot_index) {
    if (slot_index >= kPoolSize) {
      return;
    }
    Slot& slot = slots[slot_index];
    if (!slot.owner_refs) {
      Terminalize(TerminalReason::kInvalidHandoff,
                  slot.source_id.load(std::memory_order_relaxed));
      return;
    }
    if (!--slot.owner_refs && slot.runtime_admitted &&
        slot.state.load(std::memory_order_acquire) == SlotState::kReady) {
      ReleaseCandidateSlot(slot_index);
    }
  }

  void ReleaseFinalOutputSlotNow(uint32_t slot_index) {
    if (slot_index >= kFinalOutputPoolSize) {
      return;
    }
    FinalOutputSlot& slot = final_outputs[slot_index];
    if (slot.apply_time_ns) {
      release_latency_ns.Add(PresenterMonotonicTimeNs() - slot.apply_time_ns);
    }
    const uint64_t source_id = slot.source_id;
    const CandidateKind kind = slot.candidate_kind;
    slot.source_id = 0;
    slot.sequence_id = 0;
    slot.pair_a_source_id = 0;
    slot.latency_epoch = 0;
    slot.candidate_kind = CandidateKind::kReal;
    slot.semantic_target_time_ns = 0;
    slot.target_time_ns = 0;
    slot.semantic_epoch_origin_ns = 0;
    slot.semantic_output_quantum_ns = 0;
    slot.semantic_tick_index = 0;
    slot.apply_time_ns = 0;
    slot.post_completion_semaphore = VK_NULL_HANDLE;
    slot.post_completion_value = 0;
    slot.semantic_forward_skipped = false;
    slot.operating_latency_ns = 0;
    slot.final_ready_time_ns = 0;
    slot.final_ready_soft_deadline_ns = 0;
    slot.dispatch_lead_ns = 0;
    slot.actual_apply_lead_ns = 0;
    slot.ordered_ready_wait_ns = 0;
    if (slot.residency_recovery_debt.exchange(false,
                                              std::memory_order_acq_rel)) {
      ++residency_recovery_debt_recycled_before_apply_total;
    }
    slot.free_time_ns = PresenterMonotonicTimeNs();
    slot.last_released_kind = kind;
    slot.last_release_fenced = false;
    slot.state.store(FinalOutputState::kFree, std::memory_order_release);
    if (kind == CandidateKind::kReal) {
      last_released_source_id.store(source_id, std::memory_order_release);
    }
    ++release_event_total;
  }

  void ReleaseFinalOutputSlot(uint32_t slot_index) {
    if (slot_index >= kFinalOutputPoolSize) {
      return;
    }
    for (PostJob& job : post_jobs) {
      if (!job.active || job.final_output_index != slot_index) {
        continue;
      }
      // The Surface transaction may be abandoned (late S / shutdown) while
      // the GPU still owns this physical buffer. Keep it non-free until the
      // duplicated sync_file proves Post completion.
      job.release_output_on_completion = true;
      if (final_outputs[slot_index].state.load(std::memory_order_acquire) ==
          FinalOutputState::kReady) {
        final_outputs[slot_index].state.store(
            FinalOutputState::kPostProcessing, std::memory_order_release);
      }
      ++post_output_release_deferred_total;
      return;
    }
    ReleaseFinalOutputSlotNow(slot_index);
  }

  uint32_t FindFreeResidencySlot() const {
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      if (slots[i].state.load(std::memory_order_acquire) ==
          SlotState::kFree) {
        return i;
      }
    }
    return UINT32_MAX;
  }

  void ReleaseCaptureSlot(uint32_t capture_index) {
    if (capture_index >= kCapturePoolSize) {
      return;
    }
    CaptureSlot& capture = capture_slots[capture_index];
    capture.source_id = 0;
    capture.reserve_time_ns = 0;
    capture.issue_time_ns = 0;
    capture.publish_time_ns = 0;
    capture.source_timeline = VK_NULL_HANDLE;
    capture.source_timeline_value = 0;
    capture.generation = 0;
    capture.source_content_extent = {};
    capture.storage_content_extent = {};
    capture.display_aspect_ratio_x = 0;
    capture.display_aspect_ratio_y = 0;
    capture.is_8bpc = false;
    capture.ingress_context_index = UINT32_MAX;
    capture.state.store(CaptureState::kFree, std::memory_order_release);
  }

  bool SubmitCaptureTransfer(uint32_t capture_index,
                             uint32_t residency_index) {
    // Teardown may have made device B physically idle while a queued capture
    // callback is still being drained. Reject before changing slot ownership,
    // so no new B submission can be accepted after the idle latch.
    if (vulkan_device->RejectZeroFGSubmitAfterTeardownIdle()) {
      return false;
    }
    CaptureSlot& capture = capture_slots[capture_index];
    Slot& residency = slots[residency_index];
    SlotState expected = SlotState::kFree;
    if (!residency.state.compare_exchange_strong(
            expected, SlotState::kTransferSubmitted,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
      return false;
    }
    const VulkanDevice::Functions& dfn = vulkan_device->functions();
    const uint32_t queue_family =
        vulkan_device->queue_family_graphics_compute();
    // V2-2: signal the Residency slot's own readiness timeline. Its previous
    // point belongs to the slot's previous capture transfer, which was
    // observed complete before the slot became free, so the signal's cleanup
    // never waits on another slot's pending transfer.
    const bool per_slot_readiness = residency.completion.owns_timeline();
    const VkSemaphore readiness_timeline =
        per_slot_readiness ? residency.completion.timeline()
                           : capture_transfer_timeline;
    const uint64_t signal_value =
        per_slot_readiness ? residency.completion.ClaimTimelineValue()
                           : next_capture_transfer_value++;
    if (residency.completion.occupied()) {
      ++capture_owner_stats.reuse_before_retire;
    }
    // The accepted ingress submission contains the immutable GuestOutput ->
    // Capture blit, final internal-layout barrier and timeline signal before
    // this dependency submission can be built. The
    // Capture and free Residency images are exchangeable device-local storage,
    // so no second full-frame copy is necessary. This empty GPU dependency
    // submit converts the Source signal into the presenter-owned readiness
    // timeline. On q0 it orders after the earlier Source submission; on q1 it
    // supplies the required cross-queue dependency without a CPU poll.
    // Wait on the timeline the ingress copy signaled (the shared one, or its
    // context's own with C3).
    const VkSemaphore source_timeline =
        capture.source_timeline != VK_NULL_HANDLE ? capture.source_timeline
                                                  : timeline;
    const uint64_t source_wait_value = capture.source_timeline_value;
    const VkPipelineStageFlags source_wait_stage =
        VK_PIPELINE_STAGE_TRANSFER_BIT;
    // A sync semaphore whose export failed on an earlier use still holds that
    // signal. The slot was free, so its previous GPU work has retired; replace
    // the semaphore before signaling it again.
    RecreateCaptureSyncSemaphoreIfNeeded(residency, residency_index);
    VkSemaphore signal_semaphores[2] = {readiness_timeline,
                                        residency.completion.sync_semaphore()};
    const bool signal_capture_sync_fd =
        residency.completion.can_signal_sync_fd();
    const uint32_t signal_semaphore_count = signal_capture_sync_fd ? 2 : 1;
    // VUID-VkSubmitInfo-pNext-03241: with a timeline among the signaled
    // semaphores, provide one value per signal semaphore. The value paired
    // with the binary sync_fd semaphore is ignored.
    const uint64_t signal_values[2] = {signal_value, 0};
    VkTimelineSemaphoreSubmitInfo timeline_submit = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline_submit.waitSemaphoreValueCount = 1;
    timeline_submit.pWaitSemaphoreValues = &source_wait_value;
    timeline_submit.signalSemaphoreValueCount = signal_semaphore_count;
    timeline_submit.pSignalSemaphoreValues = signal_values;
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline_submit;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &source_timeline;
    submit.pWaitDstStageMask = &source_wait_stage;
    submit.signalSemaphoreCount = signal_semaphore_count;
    submit.pSignalSemaphores = signal_semaphores;
    const uint64_t queue_request_ns = PresenterMonotonicTimeNs();
    uint64_t submit_begin_ns = 0;
    {
      const VulkanDevice::Queue::Acquisition queue =
          vulkan_device->AcquireQueue(
              queue_family, vulkan_device->queue_index_zerofg_presenter());
      submit_begin_ns = PresenterMonotonicTimeNs();
      capture_transfer_queue_wait_ns.Add(submit_begin_ns - queue_request_ns);
      const VkResult result =
          dfn.vkQueueSubmit(queue.queue(), 1, &submit, VK_NULL_HANDLE);
      const uint64_t submit_host_ns =
          PresenterMonotonicTimeNs() - submit_begin_ns;
      capture_transfer_submit_host_ns.Add(submit_host_ns);
      if (result != VK_SUCCESS) {
        residency.state.store(SlotState::kFree, std::memory_order_release);
        Terminalize(TerminalReason::kCaptureTransferFailure,
                    capture.source_id);
        return false;
      }
      capture_owner_stats.CountSubmit(per_slot_readiness, submit_host_ns);
    }
    residency.completion.MarkSubmitted(readiness_timeline);

    if (signal_capture_sync_fd) {
      const uint64_t export_begin_ns = PresenterMonotonicTimeNs();
      const bool exported = residency.completion.ExportSyncFd(vulkan_device);
      capture_sync_fd_export_host_ns.Add(PresenterMonotonicTimeNs() -
                                         export_begin_ns);
      ++capture_sync_fd_export_total;
      if (!exported) {
        ++capture_sync_fd_export_failure_total;
        ++capture_sync_fd_fallback_total;
      }
    } else {
      residency.completion.SetSyncFdFallback();
      ++capture_sync_fd_fallback_total;
    }

    // Exchange ownership of the physical images only after queue acceptance.
    // The image written by Source becomes the in-flight Residency image. The
    // previously-free Residency image becomes Capture storage and is reusable
    // immediately; no GPU operation references it.
    std::swap(capture.image, residency.image);
    std::swap(capture.memory, residency.memory);
    std::swap(capture.view, residency.view);
    std::swap(capture.ever_written, residency.ever_written);
    residency.source_id.store(capture.source_id, std::memory_order_relaxed);
    residency.handoff_ready =
        ingress_copy_contexts[capture.ingress_context_index].handoff_ready;
    residency.issue_time_ns.store(capture.issue_time_ns,
                                  std::memory_order_relaxed);
    // B1: born with the Real, beside its issue time. A Residency slot is
    // recycled, and a Real promoted without a sidecar hold never writes this
    // field, so without the reset it inherited the previous occupant's hold.
    residency.policy_wait_ns.store(0, std::memory_order_relaxed);
    residency.publish_time_ns.store(capture.publish_time_ns,
                                    std::memory_order_relaxed);
    residency.timeline_value.store(signal_value, std::memory_order_relaxed);
    residency.ready_timeline = readiness_timeline;
    residency.generation.store(capture.generation,
                               std::memory_order_relaxed);
    residency.source_content_extent = capture.source_content_extent;
    residency.storage_content_extent = capture.storage_content_extent;
    residency.display_aspect_ratio_x = capture.display_aspect_ratio_x;
    residency.display_aspect_ratio_y = capture.display_aspect_ratio_y;
    residency.is_8bpc = capture.is_8bpc;
    residency.ingress_context_index = capture.ingress_context_index;
    residency.transfer_signal_value = signal_value;
    if (capture.ingress_context_index < ingress_copy_contexts.size()) {
      IngressCopyContext& ingress =
          ingress_copy_contexts[capture.ingress_context_index];
      ingress.capture_transfer_retirement_value = signal_value;
      ingress.capture_transfer_retirement_timeline = readiness_timeline;
    }
    residency.transfer_submit_time_ns = PresenterMonotonicTimeNs();
    residency.transfer_first_poll_begin_time_ns = 0;
    residency.observed_pending_time_ns = 0;
    residency.observed_ready_time_ns = 0;
    residency.owner_refs = 0;
    residency.runtime_admitted = false;
    const uint64_t issue = capture.issue_time_ns ? capture.issue_time_ns
                                                : capture.publish_time_ns;
    if (issue && residency.transfer_submit_time_ns >= issue) {
      capture_issue_to_transfer_submit_ns.Add(
          residency.transfer_submit_time_ns - issue);
    }
    if (capture.reserve_time_ns &&
        residency.transfer_submit_time_ns >= capture.reserve_time_ns) {
      capture_reserve_to_recycle_ns.Add(residency.transfer_submit_time_ns -
                                        capture.reserve_time_ns);
    }
    ReleaseCaptureSlot(capture_index);
    ++capture_transfer_submitted_total;
    ++direct_capture_timeline_wait_submit_total;
    UpdateHighWater(CountUsedSlots());
    return true;
  }

  bool SubmitOneReadyCaptureTransfer() {
    uint32_t selected_capture = UINT32_MAX;
    uint64_t oldest_source = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < kCapturePoolSize; ++i) {
      const CaptureSlot& capture = capture_slots[i];
      if (capture.state.load(std::memory_order_acquire) !=
              CaptureState::kReadyWaitingResidency ||
          !capture.source_id) {
        continue;
      }
      if (capture.source_id < oldest_source) {
        selected_capture = i;
        oldest_source = capture.source_id;
      }
    }
    if (selected_capture == UINT32_MAX) {
      return false;
    }
    const uint32_t residency = FindFreeResidencySlot();
    if (residency == UINT32_MAX) {
      return false;
    }
    BeginBlockingOperation(BlockingOperation::kCaptureSubmit);
    const bool submitted = SubmitCaptureTransfer(selected_capture, residency);
    EndBlockingOperation();
    return submitted;
  }

  // Capture observation: a non-waiting poll of each in-flight capture
  // transfer's sync_fd, at most once per kCaptureObservationIntervalNs. It
  // records when the transfer was last seen pending and first seen complete,
  // and changes nothing else: readiness, reclamation and wakes still follow
  // PollCaptureTransfers.
  void ObserveCaptureTransfersForTelemetry() {
    const uint64_t now = PresenterMonotonicTimeNs();
    if (capture_observation_last_poll_ns &&
        now - capture_observation_last_poll_ns <
            kCaptureObservationIntervalNs) {
      return;
    }
    capture_observation_last_poll_ns = now;
    for (Slot& residency : slots) {
      if (residency.observed_ready_time_ns ||
          residency.state.load(std::memory_order_acquire) !=
              SlotState::kTransferSubmitted ||
          !residency.completion.sync_fd_observable()) {
        continue;
      }
      ++capture_observation_poll_total;
      switch (residency.completion.PollSyncFd()) {
        case ZeroFGCompletionOwner::SyncFdPoll::kSignaled:
          residency.observed_ready_time_ns = now;
          ++capture_observation_ready_total;
          break;
        case ZeroFGCompletionOwner::SyncFdPoll::kPending:
          residency.observed_pending_time_ns = now;
          break;
        default:
          // Left to PollCaptureTransfers, which abandons the fd.
          ++capture_observation_error_total;
          break;
      }
    }
  }

  DriverPollResult PollCaptureTransfers() {
    DriverPollResult poll;
    if (!CountCaptureTransfers()) {
      return poll;
    }
    poll.invoked = true;
    BeginHostDriverOperation(BlockingOperation::kCapturePoll);
    const uint64_t poll_begin_ns = PresenterMonotonicTimeNs();
    uint64_t completed = 0;
    // C2: poll the observable sync_fds first. An fd the kernel answers with
    // an error is abandoned, and its slot joins the counter query below.
    std::array<bool, kPoolSize> sync_fd_ready = {};
    {
      for (uint32_t i = 0; i < kPoolSize; ++i) {
        Slot& residency = slots[i];
        if (residency.state.load(std::memory_order_acquire) !=
                SlotState::kTransferSubmitted ||
            !residency.completion.sync_fd_observable()) {
          continue;
        }
        const uint64_t fd_poll_begin_ns = PresenterMonotonicTimeNs();
        const ZeroFGCompletionOwner::SyncFdPoll poll_state =
            residency.completion.PollSyncFd();
        capture_sync_fd_poll_host_ns.Add(PresenterMonotonicTimeNs() -
                                         fd_poll_begin_ns);
        ++capture_sync_fd_poll_total;
        if (poll_state == ZeroFGCompletionOwner::SyncFdPoll::kError) {
          ++capture_owner_stats.sync_fd_errors;
          ++capture_sync_fd_fallback_total;
          residency.completion.AbandonSyncFd();
          continue;
        }
        sync_fd_ready[i] =
            poll_state == ZeroFGCompletionOwner::SyncFdPoll::kSignaled;
        if (sync_fd_ready[i]) {
          ++capture_sync_fd_ready_total;
        } else {
          // Capture observation: a pending answer bounds completion from
          // below.
          residency.observed_pending_time_ns = fd_poll_begin_ns;
        }
      }
    }
    // Slots without an observable sync_fd, including one abandoned above,
    // take the exceptional counter query on the timeline their transfer
    // signaled: one query shared by the slots on the shared timeline (every
    // slot when sync_fd observation is off), one per slot on its own V2-2
    // timeline.
    bool need_timeline_query = false;
    std::array<bool, kPoolSize> owner_query_needed = {};
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      const Slot& residency = slots[i];
      if (vulkan_device->zerofg_teardown_idle()) {
        sync_fd_ready[i] = true;
        continue;
      }
      if (residency.state.load(std::memory_order_acquire) !=
              SlotState::kTransferSubmitted ||
          residency.completion.sync_fd_observable()) {
        continue;
      }
      // No pending driver query on B. Retain ownership, fail open, then
      // explicitly drain B in shutdown before recycling these resources.
      EndHostDriverOperation();
      Terminalize(TerminalReason::kTimelineQueryFailure,
                  residency.source_id.load());
      return poll;
      if (residency.ready_timeline == VK_NULL_HANDLE ||
          residency.ready_timeline == capture_transfer_timeline) {
        need_timeline_query = true;
      } else {
        owner_query_needed[i] = true;
      }
    }
    VkResult timeline_result = VK_SUCCESS;
    if (need_timeline_query) {
      ++capture_owner_stats.timeline_queries;
      timeline_result = vulkan_device->vkGetSemaphoreCounterValue()(
          vulkan_device->device(), capture_transfer_timeline, &completed);
    }
    std::array<bool, kPoolSize> owner_query_ready = {};
    for (uint32_t i = 0; i < kPoolSize && timeline_result == VK_SUCCESS;
         ++i) {
      if (!owner_query_needed[i]) {
        continue;
      }
      ++capture_owner_stats.timeline_queries;
      uint64_t slot_completed = 0;
      timeline_result = vulkan_device->vkGetSemaphoreCounterValue()(
          vulkan_device->device(), slots[i].ready_timeline, &slot_completed);
      owner_query_ready[i] =
          slot_completed >= slots[i].transfer_signal_value;
    }
    const uint64_t poll_end_ns = PresenterMonotonicTimeNs();
    active_capture_poll_host_ns.Add(poll_end_ns - poll_begin_ns);
    ++active_capture_poll_total;
    EndHostDriverOperation();
    if (timeline_result != VK_SUCCESS) {
      Terminalize(TerminalReason::kTimelineQueryFailure, 0);
      return poll;
    }
    for (Slot& residency : slots) {
      if (residency.state.load(std::memory_order_acquire) ==
              SlotState::kTransferSubmitted &&
          !residency.transfer_first_poll_begin_time_ns) {
        residency.transfer_first_poll_begin_time_ns = poll_begin_ns;
      }
    }
    completed_capture_transfer_value = completed;
    const uint64_t now = poll_end_ns;
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      Slot& residency = slots[i];
      if (residency.state.load(std::memory_order_acquire) !=
          SlotState::kTransferSubmitted) {
        continue;
      }
      const bool ready =
          vulkan_device->zerofg_teardown_idle() ? true
          : residency.completion.sync_fd_observable()
              ? sync_fd_ready[i]
          : owner_query_needed[i]
              ? owner_query_ready[i]
              : completed >= residency.transfer_signal_value;
      if (!ready) {
        continue;
      }
      residency.ready_time_ns = now;
      residency.completion.CloseSyncFd();
      residency.completion.MarkRetired();
      RecreateCaptureSyncSemaphoreIfNeeded(residency, i);
      RetireIngressContext(residency.ingress_context_index,
                           residency.transfer_signal_value);
      residency.ingress_context_index = UINT32_MAX;
      residency.transfer_signal_value = 0;
      residency.state.store(SlotState::kReady, std::memory_order_release);
      const bool release_after_observation =
          residency.runtime_admitted && residency.owner_refs == 0;
      const uint64_t issue =
          residency.issue_time_ns.load(std::memory_order_relaxed)
              ? residency.issue_time_ns.load(std::memory_order_relaxed)
              : residency.publish_time_ns.load(std::memory_order_relaxed);
      if (issue && now >= issue) {
        issue_to_candidate_ready_ns.Add(now - issue);
        readiness_wait_ns.Add(now - issue);
      }
      if (residency.transfer_submit_time_ns &&
          now >= residency.transfer_submit_time_ns) {
        if (residency.transfer_first_poll_begin_time_ns >=
            residency.transfer_submit_time_ns) {
          capture_submit_to_first_poll_begin_ns.Add(
              residency.transfer_first_poll_begin_time_ns -
              residency.transfer_submit_time_ns);
        }
        if (poll_begin_ns >= residency.transfer_submit_time_ns) {
          capture_submit_to_ready_poll_begin_ns.Add(
              poll_begin_ns - residency.transfer_submit_time_ns);
        }
        capture_transfer_residence_ns.Add(
            now - residency.transfer_submit_time_ns);
        // The first sight of completion: the telemetry poll, or this poll
        // when the telemetry poll had not seen it. Completion lies after
        // the last pending answer and no later than the first sight.
        const uint64_t first_ready_ns =
            residency.observed_ready_time_ns &&
                    residency.observed_ready_time_ns <= now
                ? residency.observed_ready_time_ns
                : now;
        if (first_ready_ns >= residency.transfer_submit_time_ns) {
          capture_first_seen_ready_ns.Add(
              first_ready_ns - residency.transfer_submit_time_ns);
        }
        capture_reclaim_after_first_seen_ns.Add(now - first_ready_ns);
        if (residency.observed_pending_time_ns >=
                residency.transfer_submit_time_ns &&
            residency.observed_pending_time_ns <= first_ready_ns) {
          capture_last_seen_pending_ns.Add(
              residency.observed_pending_time_ns -
              residency.transfer_submit_time_ns);
          capture_observation_uncertainty_ns.Add(
              first_ready_ns - residency.observed_pending_time_ns);
        }
      }
      residency.transfer_submit_time_ns = 0;
      residency.transfer_first_poll_begin_time_ns = 0;
      residency.observed_pending_time_ns = 0;
      residency.observed_ready_time_ns = 0;
      ++readiness_wait_total;
      ++capture_transfer_completed_total;
      UpdateHighWater(CountUsedSlots());
      if (release_after_observation) {
        ReleaseCandidateSlot(i);
      }
      poll.progress = true;
    }
    return poll;
  }

  void FailStructuralTimingEpoch(const char* reason, uint64_t source_id) {
    ++structural_timing_failure_total;
    last_pair_synthetic_applied = false;
    ResetPhaseDebt();
    // No synchronous flush or partial clock reset with live sovereign Reals.
    // The normal drop/drain paths retain all submitted GPU ownership.
    DropSyntheticFromOldPacingEpoch(SyntheticDropReason::kLifecycleEpoch);
    XELOGE("ZeroFGStructuralTimingEpoch reason={} source={} action=fail_open",
           reason, source_id);
    Terminalize(TerminalReason::kInvalidHandoff, source_id);
  }

  void DropSyntheticFromOldPacingEpoch(SyntheticDropReason reason) {
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& logical = logical_outputs[i];
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kSynthetic) {
        DropSyntheticLogical(i, reason);
      }
    }
  }

  bool SourcePeriodWindowClustered(const PresenterSampleWindow<32>& window,
                                   uint64_t tolerance_divisor) const {
    const uint64_t median = window.Quantile(50, 100);
    return median &&
           window.Quantile(90, 100) - window.Quantile(10, 100) <=
               median / tolerance_divisor;
  }

  bool SourceTransitionWindowClustered() const {
    const uint64_t median = source_transition_samples_ns.Quantile(50, 100);
    return median &&
           source_transition_samples_ns.Quantile(90, 100) -
                   source_transition_samples_ns.Quantile(10, 100) <=
               median / 10;
  }

  bool PhysicalRecoveryProbePendingOrActive() const {
    return source_rate_probe_recovery_pending ||
           (SourceRateProbeActive() &&
            source_rate_probe_reason ==
                SourceRateProbeReason::kPhysicalOperatingPointRecovery);
  }

  bool H1cHasPreConfirmationState() const {
    return transition_planned_space_state ==
           TransitionPlannedSpaceState::kFastSequence;
  }

  static const char* H1cResetReasonName(H1cResetReason reason) {
    switch (reason) {
      case H1cResetReason::kRawSlow:
        return "raw_slow";
      case H1cResetReason::kCleanNonFast:
        return "clean_non_fast";
      case H1cResetReason::kIssueGap:
        return "issue_sequence_gap";
      case H1cResetReason::kTimeout:
        return "fast_sequence_timeout";
      case H1cResetReason::kRecoveryProbe:
        return "physical_recovery_probe";
      case H1cResetReason::kPeriodTransition:
        return "period_transition";
      case H1cResetReason::kCount:
        break;
    }
    return "unknown";
  }

  void FinishH1cFastSequence(H1cResetReason reason,
                             const char* log_reason = nullptr) {
    if (!H1cHasPreConfirmationState()) {
      return;
    }
    if (reason == H1cResetReason::kTimeout) {
      ++h1c_timeout_total;
    } else {
      ++h1c_reset_total;
    }
    ++h1c_episode_end_reason_total[size_t(reason)];
    h1c_false_positive_pair_total = SaturatingAddNs(
        h1c_false_positive_pair_total, h1c_episode_real_only_pairs);
    transition_planned_space_state = TransitionPlannedSpaceState::kIdle;
    h1c_fast_sequence_start_ns = 0;
    h1c_fast_sequence_start_issue = 0;
    h1c_episode_real_only_pairs = 0;
    h1c_episode_quarantined_pairs = 0;
    h1c_episode_s_at_drain_start = 0;
  }

  uint64_t HighestLiveSyntheticSourceId() const {
    uint64_t watermark = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kSynthetic) {
        watermark = std::max(watermark, logical.candidate.source_id);
      }
    }
    return watermark;
  }

  void FinishH1cDrain(const char* reason, bool timeout = false) {
    if (transition_planned_space_state !=
        TransitionPlannedSpaceState::kPostConfirmDrain) {
      return;
    }
    if (timeout) {
      ++h1c_timeout_total;
    } else if (!std::strcmp(reason, "drained")) {
      ++h1c_drain_complete_total;
    } else {
      ++h1c_reset_total;
    }
    const uint64_t live_s = CountLogicalOutputs(CandidateKind::kSynthetic);
    const uint64_t retired = h1c_episode_s_at_drain_start > live_s
                                 ? h1c_episode_s_at_drain_start - live_s
                                 : 0;
    h1c_s_retired_during_drain_total = SaturatingAddNs(
        h1c_s_retired_during_drain_total, retired);
    transition_planned_space_state = TransitionPlannedSpaceState::kIdle;
    h1c_drain_start_ns = 0;
    h1c_drain_deadline_issue = 0;
    h1c_drain_synthetic_source_watermark = 0;
    h1c_episode_s_at_start = 0;
    h1c_episode_s_at_drain_start = 0;
    h1c_episode_frontier_at_drain_start = 0;
    h1c_episode_chains_at_drain_start = 0;
    h1c_episode_quarantined_pairs = 0;
  }

  void UpdateH1cEpisode(uint64_t now_ns, uint64_t issue_sequence) {
    if (PhysicalRecoveryProbePendingOrActive()) {
      if (H1cHasPreConfirmationState()) {
        ++h1c_recovery_probe_inhibit_total;
        FinishH1cFastSequence(H1cResetReason::kRecoveryProbe);
      } else if (transition_planned_space_state ==
                 TransitionPlannedSpaceState::kPostConfirmDrain) {
        ++h1c_recovery_probe_inhibit_total;
        FinishH1cDrain("physical_recovery_probe");
      }
      return;
    }
    if (H1cHasPreConfirmationState() &&
        ((h1c_fast_sequence_start_ns && now_ns > SaturatingAddNs(
                                               h1c_fast_sequence_start_ns,
                                               kTransitionPlannedSpaceTimeoutNs)) ||
         (issue_sequence >= h1c_fast_sequence_start_issue &&
          issue_sequence - h1c_fast_sequence_start_issue >
              kTransitionPlannedSpaceIssueBudget))) {
      FinishH1cFastSequence(H1cResetReason::kTimeout);
    } else if (transition_planned_space_state ==
               TransitionPlannedSpaceState::kPostConfirmDrain) {
      if (now_ns > SaturatingAddNs(
                       h1c_drain_start_ns,
                       kTransitionPlannedSpaceTimeoutNs) ||
          (h1c_drain_deadline_issue &&
           issue_sequence > h1c_drain_deadline_issue)) {
        FinishH1cDrain("drain_timeout", true);
      } else if (!CountLogicalOutputs(CandidateKind::kSynthetic) &&
                 !CountLiveSyntheticChains()) {
        FinishH1cDrain("drained");
      }
    }
  }

  bool BeginH1cFastSequence(uint64_t issue_sequence, uint64_t now_ns,
                            uint64_t arm_issue_sequence) {
    // While §7 owns the delivered P the normal estimator is frozen, so a
    // faster Source has no confirmation lag for H1c to bridge: the §7
    // recovery probe measures it. H1c withholding S there would also
    // manufacture the slack that requests that probe.
    if (!SplitAlwaysSArmed() || PhysicalRecoveryProbePendingOrActive() ||
        physical_op_confirmed ||
        transition_planned_space_state != TransitionPlannedSpaceState::kIdle) {
      return false;
    }
    transition_planned_space_state =
        TransitionPlannedSpaceState::kFastSequence;
    h1c_fast_sequence_start_ns = now_ns;
    h1c_fast_sequence_start_issue = issue_sequence;
    h1c_episode_real_only_pairs = 0;
    h1c_episode_quarantined_pairs = 0;
    h1c_episode_s_at_start =
        CountLogicalOutputs(CandidateKind::kSynthetic);
    h1c_episode_s_at_drain_start = h1c_episode_s_at_start;
    h1c_episode_frontier_at_start = CountSyntheticProductionFrontier();
    h1c_episode_chains_at_start = CountLiveSyntheticChains();
    ++h1c_fast_sequence_start_total;
    return true;
  }

  // An evidence sequence that ends without ever arming H1c. A single
  // clean-fast sample is the jitter/catch-up case gate B exists to absorb.
  void NoteH1cUnarmedSequenceEnd(H1cResetReason reason,
                                 uint64_t issue_sequence) {
    if (!h1c_fast_evidence_valid || h1c_fast_evidence_attempted ||
        h1c_fast_evidence_inhibited || h1c_fast_evidence_clean_count != 1) {
      return;
    }
    ++h1c_single_sample_sequence_total;
    ++h1c_single_sample_end_reason_total[size_t(reason)];
    if (h1c_evidence_log_total < 16) {
      ++h1c_evidence_log_total;
    }
  }

  void ResetH1cFastSequenceEvidence(H1cResetReason reason,
                                   uint64_t issue_sequence) {
    if (H1cHasPreConfirmationState()) {
      FinishH1cFastSequence(reason);
    } else {
      NoteH1cUnarmedSequenceEnd(reason, issue_sequence);
    }
    h1c_fast_evidence_valid = false;
    h1c_fast_evidence_attempted = false;
    h1c_fast_evidence_inhibited = false;
    h1c_fast_evidence_clean_count = 0;
    h1c_fast_evidence_start_ns = 0;
    h1c_fast_evidence_start_issue = 0;
    h1c_fast_evidence_last_issue = issue_sequence;
  }

  // Gate A. The upper median of the short raw window against the delivered P
  // (source_period_ns, not learned P). The window mean is telemetry only: a
  // slow median with a mean at or below P is the BP/catch-up alternation
  // signature the next runtime must be able to see.
  void EvaluateH1cShortRawDirection() {
    h1c_short_raw_ready =
        h1c_short_raw_interval_ns.count() >= kH1cShortRawWindow;
    if (!h1c_short_raw_ready || !source_period_ns) {
      h1c_short_raw_median_ns = 0;
      h1c_short_raw_mean_ns = 0;
      h1c_short_raw_slow = false;
      return;
    }
    h1c_short_raw_median_ns = h1c_short_raw_interval_ns.Quantile(50, 100);
    h1c_short_raw_mean_ns = h1c_short_raw_interval_ns.Mean();
    h1c_short_raw_slow =
        SaturatingMultiplyNs(h1c_short_raw_median_ns, 100) >
        SaturatingMultiplyNs(source_period_ns, kH1cRawSlowPercent);
  }

  void ObserveH1cFastSequenceEvidence(uint64_t issue_sequence,
                                     uint64_t interval_ns,
                                     uint64_t bp_wait_ns,
                                     bool censored_interval) {
    const bool issue_contiguous =
        !h1c_fast_evidence_last_issue ||
        (issue_sequence > h1c_fast_evidence_last_issue &&
         issue_sequence - h1c_fast_evidence_last_issue == 1);
    // Gate A input: every consumed raw interval, clean, BP and post-BP. Lost
    // intervals are not fabricated, so a sequence gap restarts the window.
    if (!issue_contiguous) {
      h1c_short_raw_interval_ns = {};
    }
    h1c_short_raw_interval_ns.Add(interval_ns);
    if (!source_period_ns) {
      h1c_fast_evidence_last_issue = issue_sequence;
      return;
    }
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    UpdateH1cEpisode(now_ns, issue_sequence);
    if (h1c_fast_evidence_valid && !issue_contiguous) {
      ResetH1cFastSequenceEvidence(H1cResetReason::kIssueGap,
                                   issue_sequence);
    }
    h1c_fast_evidence_last_issue = issue_sequence;
    const bool clean_fast =
        !censored_interval && !bp_wait_ns &&
        interval_ns >= kSourcePeriodMinimumPlausibleNs &&
        interval_ns <= kSourcePeriodMaximumPlausibleNs &&
        interval_ns < source_period_ns - source_period_ns / 5;

    // Gate A is judged on every interval, censored ones included, before the
    // censorship return below: a sequence must not survive a slow plant by
    // hiding behind BP/post-BP intervals. It ends only pre-confirmation
    // evidence; a confirmed epoch's post-confirm drain is not touched.
    EvaluateH1cShortRawDirection();
    if (h1c_short_raw_slow) {
      if (h1c_fast_evidence_valid || H1cHasPreConfirmationState()) {
        ++h1c_raw_slow_reset_total;
        if (h1c_short_raw_mean_ns <= source_period_ns) {
          ++h1c_raw_slow_bimodal_suspect_total;
        }
        ResetH1cFastSequenceEvidence(H1cResetReason::kRawSlow,
                                     issue_sequence);
      } else if (clean_fast) {
        ++h1c_raw_slow_blocked_sample_total;
      }
      return;
    }

    if (PhysicalRecoveryProbePendingOrActive()) {
      if (!censored_interval && !bp_wait_ns && clean_fast &&
          !h1c_fast_evidence_valid) {
        h1c_fast_evidence_valid = true;
        h1c_fast_evidence_start_ns = now_ns;
        h1c_fast_evidence_start_issue = issue_sequence;
        h1c_fast_evidence_attempted = true;
        h1c_fast_evidence_clean_count = 1;
      } else if (!censored_interval && !bp_wait_ns && !clean_fast &&
                 h1c_fast_evidence_valid) {
        ResetH1cFastSequenceEvidence(H1cResetReason::kCleanNonFast,
                                     issue_sequence);
      }
      if (h1c_fast_evidence_valid) {
        h1c_fast_evidence_inhibited = true;
        if (H1cHasPreConfirmationState()) {
          ++h1c_recovery_probe_inhibit_total;
          FinishH1cFastSequence(H1cResetReason::kRecoveryProbe);
        } else if (transition_planned_space_state ==
                   TransitionPlannedSpaceState::kPostConfirmDrain) {
          ++h1c_recovery_probe_inhibit_total;
          FinishH1cDrain("physical_recovery_probe");
        }
      }
      return;
    }
    if (censored_interval || bp_wait_ns) {
      // Censored intervals neither count as fast evidence nor end it.
      return;
    }
    if (!clean_fast) {
      if (h1c_fast_evidence_valid) {
        ResetH1cFastSequenceEvidence(H1cResetReason::kCleanNonFast,
                                     issue_sequence);
      }
      return;
    }
    if (!h1c_short_raw_ready) {
      // Gate A cannot judge this sample's direction yet: not qualified.
      return;
    }
    if (!h1c_fast_evidence_valid) {
      h1c_fast_evidence_valid = true;
      h1c_fast_evidence_attempted = false;
      h1c_fast_evidence_inhibited = false;
      h1c_fast_evidence_clean_count = 1;
      h1c_fast_evidence_start_ns = now_ns;
      h1c_fast_evidence_start_issue = issue_sequence;
      ++h1c_evidence_sequence_total;
    } else if (h1c_fast_evidence_clean_count <
               std::numeric_limits<uint32_t>::max()) {
      ++h1c_fast_evidence_clean_count;
      if (h1c_fast_evidence_clean_count == kH1cArmCleanFastSamples) {
        ++h1c_second_sample_total;
      }
    }
    // Gate B: the first TransitionRealOnly needs a second qualified sample of
    // this same sequence; each one was judged by gate A as not slow.
    if (h1c_fast_evidence_clean_count < kH1cArmCleanFastSamples ||
        h1c_fast_evidence_attempted || h1c_fast_evidence_inhibited ||
        transition_planned_space_state != TransitionPlannedSpaceState::kIdle) {
      return;
    }
    if (physical_op_confirmed) {
      ++h1c_physical_op_block_total;
      return;
    }
    if (BeginH1cFastSequence(h1c_fast_evidence_start_issue,
                             h1c_fast_evidence_start_ns, issue_sequence)) {
      h1c_fast_evidence_attempted = true;
    }
  }

  void OnH1cPeriodTransition(uint64_t previous_period_ns,
                             uint64_t new_period_ns,
                             uint64_t confirmation_issue_sequence) {
    NoteH1cUnarmedSequenceEnd(H1cResetReason::kPeriodTransition,
                              confirmation_issue_sequence);
    h1c_fast_evidence_valid = false;
    h1c_fast_evidence_attempted = false;
    h1c_fast_evidence_inhibited = false;
    h1c_fast_evidence_clean_count = 0;
    h1c_fast_evidence_start_ns = 0;
    h1c_fast_evidence_start_issue = 0;
    h1c_fast_evidence_last_issue = confirmation_issue_sequence;
    if (!H1cHasPreConfirmationState()) {
      return;
    }
    if (new_period_ns >= previous_period_ns) {
      FinishH1cFastSequence(H1cResetReason::kPeriodTransition,
                            "non_faster_epoch");
      return;
    }
    transition_planned_space_state =
        TransitionPlannedSpaceState::kPostConfirmDrain;
    h1c_confirm_issue = confirmation_issue_sequence;
    h1c_drain_start_ns = PresenterMonotonicTimeNs();
    h1c_drain_deadline_issue = SaturatingAddNs(
        confirmation_issue_sequence, kTransitionPlannedSpaceIssueBudget);
    h1c_drain_synthetic_source_watermark = HighestLiveSyntheticSourceId();
    h1c_episode_s_at_drain_start =
        CountLogicalOutputs(CandidateKind::kSynthetic);
    h1c_episode_frontier_at_drain_start = CountSyntheticProductionFrontier();
    h1c_episode_chains_at_drain_start = CountLiveSyntheticChains();
    UpdateH1cEpisode(h1c_drain_start_ns, confirmation_issue_sequence);
  }

  void OnD1FastSequenceReset(const char* reason, uint64_t issue_sequence) {
    // D1's one-shot trace and H1c's reusable fast-sequence authority have
    // independent lifetimes. H1c is reset by its own interval observer.
    (void)reason;
    (void)issue_sequence;
  }

  bool H1cMayCreateTransitionRealOnlyPair(uint64_t now_ns,
                                           uint64_t issue_sequence) {
    UpdateH1cEpisode(now_ns, issue_sequence);
    // Gates A and B stay explicit here: no new TransitionRealOnly while the
    // raw direction is slow, before a second qualified sample, or while §7
    // owns the delivered P.
    return H1cHasPreConfirmationState() &&
           !PhysicalRecoveryProbePendingOrActive() && !physical_op_confirmed &&
           !h1c_short_raw_slow && h1c_fast_evidence_valid &&
           !h1c_fast_evidence_inhibited &&
           h1c_fast_evidence_clean_count >= kH1cArmCleanFastSamples;
  }

  void RememberPlannedSpaceFastEvidence(uint64_t issue_sequence,
                                        uint64_t interval_ns,
                                        uint64_t bp_wait_ns,
                                        bool censored_interval) {
    PlannedSpaceTrace& trace = planned_space_trace;
    if (trace.started || !source_period_ns) {
      return;
    }

    const uint64_t observed_ns = PresenterMonotonicTimeNs();
    if (trace.fast_evidence_valid) {
      const bool sequence_contiguous =
          issue_sequence > trace.fast_evidence_last_issue_sequence &&
          issue_sequence - trace.fast_evidence_last_issue_sequence == 1;
      if (sequence_contiguous) {
        // Each raw period interval is an elapsed Source-issue interval. A
        // censored interval may contribute to elapsed-time reconstruction,
        // but never to the positive fast-evidence count below.
        trace.fast_evidence_delay_ns =
            SaturatingAddNs(trace.fast_evidence_delay_ns, interval_ns);
        ++trace.fast_evidence_delay_issues;
      } else {
        trace.fast_evidence_delay_exact = false;
        if (issue_sequence > trace.fast_evidence_last_issue_sequence) {
          trace.fast_evidence_missing_issues +=
              issue_sequence - trace.fast_evidence_last_issue_sequence - 1;
        }
        // A gap/overrun prevents proving that the old run is still current.
        // Start a new sequence only if this very interval is clean and fast.
        if (trace.fast_sequence_event_log_total < 16) {
          ++trace.fast_sequence_event_log_total;
        }
        trace.fast_evidence_valid = false;
        OnD1FastSequenceReset("issue_sequence_gap", issue_sequence);
        trace.fast_evidence_delay_exact = true;
        trace.fast_evidence_delay_ns = 0;
        trace.fast_evidence_delay_issues = 0;
        trace.fast_evidence_missing_issues = 0;
        trace.fast_evidence_censored_issues = 0;
        trace.fast_evidence_clean_count = 0;
      }
      trace.fast_evidence_last_issue_sequence = issue_sequence;
    }

    const bool clean_fast =
        !censored_interval && !bp_wait_ns &&
        interval_ns >= kSourcePeriodMinimumPlausibleNs &&
        interval_ns <= kSourcePeriodMaximumPlausibleNs &&
        interval_ns < source_period_ns - source_period_ns / 5;
    if (censored_interval || bp_wait_ns) {
      if (trace.fast_evidence_valid) {
        ++trace.fast_evidence_censored_issues;
      }
      // In particular, the first apparently clean interval after BP is not
      // allowed to start or extend the positive fast sequence.
      return;
    }
    if (!clean_fast) {
      if (trace.fast_evidence_valid) {
        trace.fast_reset_observed_ns = observed_ns;
        trace.fast_reset_issue_sequence = issue_sequence;
        trace.fast_reset_interval_ns = interval_ns;
        if (trace.fast_sequence_event_log_total < 16) {
          ++trace.fast_sequence_event_log_total;
        }
        trace.fast_evidence_valid = false;
        OnD1FastSequenceReset("clean_non_fast", issue_sequence);
        trace.fast_evidence_delay_exact = true;
        trace.fast_evidence_delay_ns = 0;
        trace.fast_evidence_delay_issues = 0;
        trace.fast_evidence_missing_issues = 0;
        trace.fast_evidence_censored_issues = 0;
        trace.fast_evidence_clean_count = 0;
      }
      return;
    }

    if (!trace.fast_evidence_valid) {
      trace.fast_evidence_valid = true;
      trace.fast_evidence_delay_exact = true;
      trace.fast_evidence_time_ns = observed_ns;
      trace.fast_evidence_issue_sequence = issue_sequence;
      trace.fast_evidence_last_issue_sequence = issue_sequence;
      trace.fast_evidence_interval_ns = interval_ns;
      trace.fast_evidence_bp_wait_ns = bp_wait_ns;
      trace.fast_evidence_old_period_ns = source_period_ns;
      trace.fast_evidence_delay_ns = 0;
      trace.fast_evidence_delay_issues = 0;
      trace.fast_evidence_missing_issues = 0;
      trace.fast_evidence_censored_issues = 0;
      trace.fast_evidence_clean_count = 1;
      trace.fast_evidence_frontier = CountSyntheticProductionFrontier();
      trace.last_frontier = uint32_t(trace.fast_evidence_frontier);
      if (trace.last_frontier <= kLogicalSyntheticCapacity) {
        const uint32_t level = trace.last_frontier;
        trace.frontier_levels_seen |= uint32_t(1) << level;
        trace.frontier_first_time_ns[level] = observed_ns;
        trace.frontier_first_issue_sequence[level] = issue_sequence;
      }
      if (trace.fast_sequence_event_log_total < 16) {
        ++trace.fast_sequence_event_log_total;
      }
    } else {
      ++trace.fast_evidence_clean_count;
    }
  }

  void BeginPlannedSpaceTrace(uint64_t previous_period_ns,
                              uint64_t new_period_ns,
                              uint64_t confirmation_issue_sequence) {
    PlannedSpaceTrace& trace = planned_space_trace;
    if (trace.started || !previous_period_ns ||
        !new_period_ns || new_period_ns >= previous_period_ns) {
      return;
    }
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    // The rate probe can confirm before ConsumeSourceIssuePeriods advances its
    // cursor; the normal estimator confirms afterward. Both paths carry the
    // exact issue being processed, so never infer it from that cursor here.
    const uint64_t current_issue_sequence = confirmation_issue_sequence;
    const bool fast_sequence_current =
        trace.fast_evidence_valid &&
        confirmation_issue_sequence >=
                                         trace.fast_evidence_issue_sequence &&
        confirmation_issue_sequence ==
            trace.fast_evidence_last_issue_sequence;
    const bool fast_sequence_in_window =
        fast_sequence_current && now_ns >= trace.fast_evidence_time_ns &&
        now_ns - trace.fast_evidence_time_ns <=
            kPlannedSpaceTraceWindowNs &&
        current_issue_sequence >= trace.fast_evidence_issue_sequence &&
        current_issue_sequence - trace.fast_evidence_issue_sequence <=
            kPlannedSpaceTraceIssueWindow;
    const uint64_t trace_start_time_ns =
        fast_sequence_in_window ? trace.fast_evidence_time_ns : now_ns;
    const uint64_t trace_start_issue_sequence =
        fast_sequence_in_window ? trace.fast_evidence_issue_sequence
                                : current_issue_sequence;
    const uint32_t frontier_at_confirmation =
        CountSyntheticProductionFrontier();
    if (!fast_sequence_in_window) {
      // Do not invent a source interval at the confirmation point. The event
      // trace may still start here, but its fast-sequence provenance is absent.
      trace.fast_evidence_valid = false;
      trace.last_frontier = frontier_at_confirmation;
      trace.frontier_levels_seen = 0;
      trace.frontier_first_time_ns = {};
      trace.frontier_first_issue_sequence = {};
      trace.frontier_first_source_id = {};
      trace.frontier_first_logical_sequence = {};
      if (frontier_at_confirmation <= kLogicalSyntheticCapacity) {
        trace.frontier_levels_seen |=
            uint32_t(1) << frontier_at_confirmation;
        trace.frontier_first_time_ns[frontier_at_confirmation] = now_ns;
        trace.frontier_first_issue_sequence[frontier_at_confirmation] =
            current_issue_sequence;
      }
    }
    ObservePlannedSpaceFrontier(CountSyntheticProductionFrontier(),
                                current_issue_sequence, 0,
                                "period_confirmation", current_issue_sequence);
    trace.started = true;
    trace.active = true;
    trace.start_time_ns = trace_start_time_ns;
    trace.start_issue_sequence = trace_start_issue_sequence;
    trace.confirmation_time_ns = now_ns;
    trace.confirmation_issue_sequence = confirmation_issue_sequence;
    trace.confirmation_source_id = confirmation_issue_sequence;
    trace.confirmation_period_epoch = period_epoch;
    trace.frontier_at_confirmation = frontier_at_confirmation;
    trace.live_chains_at_confirmation = CountLiveSyntheticChains();
    trace.actual_compacted_at_confirmation =
        latency_depth_decay_actual_compacted_total;
    trace.old_period_ns = previous_period_ns;
    trace.new_period_ns = new_period_ns;
    trace.old_quantum_ns = std::max<uint64_t>(previous_period_ns / 2, 1);
    trace.new_quantum_ns = std::max<uint64_t>(new_period_ns / 2, 1);
    ++planned_space_trace.event_count;
    for (uint32_t level = 0; level <= kLogicalSyntheticCapacity; ++level) {
      if (!(trace.frontier_levels_seen & (uint32_t(1) << level)) ||
          trace.event_count >= kPlannedSpaceTraceMaxEvents) {
        continue;
      }
      ++trace.event_count;
    }
    if (trace.event_count < kPlannedSpaceTraceMaxEvents) {
      ++trace.event_count;
    }
  }

  void FinishPlannedSpaceTrace(const char* reason) {
    PlannedSpaceTrace& trace = planned_space_trace;
    if (!trace.started || trace.summary_logged) {
      return;
    }
    trace.active = false;
    trace.summary_logged = true;
    trace.frontier_at_finish = CountSyntheticProductionFrontier();
    trace.live_chains_at_finish = CountLiveSyntheticChains();
  }

  bool PlannedSpaceTraceWithinWindow(uint64_t now_ns,
                                     uint64_t source_issue_sequence) {
    PlannedSpaceTrace& trace = planned_space_trace;
    if (!trace.active) {
      return false;
    }
    if (now_ns < trace.start_time_ns ||
        now_ns - trace.start_time_ns > kPlannedSpaceTraceWindowNs ||
        source_issue_sequence < trace.start_issue_sequence ||
        source_issue_sequence - trace.start_issue_sequence >
            kPlannedSpaceTraceIssueWindow) {
      FinishPlannedSpaceTrace(now_ns >= trace.start_time_ns &&
                                      now_ns - trace.start_time_ns >
                                          kPlannedSpaceTraceWindowNs
                                  ? "time_window_expired"
                                  : "issue_window_expired");
      return false;
    }
    return true;
  }

  bool PlannedSpaceTraceCanRecord(uint64_t now_ns,
                                 uint64_t source_issue_sequence) {
    if (!PlannedSpaceTraceWithinWindow(now_ns, source_issue_sequence)) {
      return false;
    }
    if (planned_space_trace.event_count >= kPlannedSpaceTraceMaxEvents) {
      FinishPlannedSpaceTrace("event_limit");
      return false;
    }
    ++planned_space_trace.event_count;
    return true;
  }

  void ObservePlannedSpaceFrontier(uint32_t frontier, uint64_t source_id,
                                  uint64_t logical_sequence,
                                  const char* cause,
                                  uint64_t observed_issue_sequence = 0) {
    PlannedSpaceTrace& trace = planned_space_trace;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    const uint64_t issue_sequence =
        observed_issue_sequence ? observed_issue_sequence
                                : last_processed_source_issue_period_sequence;
    if (!trace.active && !trace.started && trace.fast_evidence_valid) {
      if (now_ns < trace.fast_evidence_time_ns ||
          now_ns - trace.fast_evidence_time_ns > kPlannedSpaceTraceWindowNs ||
          issue_sequence < trace.fast_evidence_issue_sequence ||
          issue_sequence - trace.fast_evidence_issue_sequence >
              kPlannedSpaceTraceIssueWindow) {
        return;
      }
      const uint32_t candidate_frontier =
          std::min(frontier, kLogicalSyntheticCapacity + 1);
      // Only the sampled level was observed at this event. Marking every
      // crossed integer on a jump would fabricate first-reached timestamps.
      if (candidate_frontier <= kLogicalSyntheticCapacity) {
        const uint32_t bit = uint32_t(1) << candidate_frontier;
        if (!(trace.frontier_levels_seen & bit)) {
          trace.frontier_levels_seen |= bit;
          trace.frontier_first_time_ns[candidate_frontier] = now_ns;
          trace.frontier_first_issue_sequence[candidate_frontier] =
              issue_sequence;
          trace.frontier_first_source_id[candidate_frontier] = source_id;
          trace.frontier_first_logical_sequence[candidate_frontier] =
              logical_sequence;
        }
      }
      trace.last_frontier = candidate_frontier;
      return;
    }
    if (!PlannedSpaceTraceWithinWindow(now_ns, issue_sequence)) {
      return;
    }
    const uint32_t bounded_frontier =
        std::min(frontier, kLogicalSyntheticCapacity + 1);
    if (trace.last_frontier == UINT32_MAX) {
      trace.last_frontier = bounded_frontier;
      return;
    }
    if (bounded_frontier <= kLogicalSyntheticCapacity) {
      const uint32_t bit = uint32_t(1) << bounded_frontier;
      if (!(trace.frontier_levels_seen & bit)) {
        if (trace.event_count >= kPlannedSpaceTraceMaxEvents) {
          FinishPlannedSpaceTrace("event_limit");
          return;
        }
        trace.frontier_levels_seen |= bit;
        ++trace.event_count;
      }
    }
    trace.last_frontier = bounded_frontier;
  }

  void ObservePlannedSpaceNewQuantumIssue(uint64_t source_id,
                                          uint64_t issue_ns) {
    PlannedSpaceTrace& trace = planned_space_trace;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (!trace.active || trace.new_quantum_issue_seen ||
        period_epoch != trace.confirmation_period_epoch ||
        stable_output_quantum_ns != trace.new_quantum_ns ||
        !PlannedSpaceTraceCanRecord(
            now_ns, last_processed_source_issue_period_sequence)) {
      return;
    }
    trace.new_quantum_issue_seen = true;
    trace.new_quantum_issue_source_id = source_id;
    trace.new_quantum_issue_ns = issue_ns;
  }

  void ObservePlannedSpaceSourceInterval(uint64_t issue_sequence,
                                         uint64_t interval_ns,
                                         bool censored_interval) {
    PlannedSpaceTrace& trace = planned_space_trace;
    if (!trace.active ||
        !PlannedSpaceTraceWithinWindow(
            PresenterMonotonicTimeNs(), issue_sequence)) {
      return;
    }
    if (censored_interval) {
      ++trace.episode_censored_source_interval_total;
      return;
    }
    ++trace.episode_clean_source_interval_total;
    if (interval_ns > trace.natural_source_gap_max_ns) {
      trace.natural_source_gap_max_ns = interval_ns;
      trace.natural_source_gap_max_issue_sequence = issue_sequence;
    }
  }

  void ObservePlannedSpaceOrderedRefusal(
      uint64_t source_id, uint64_t issue_ns, uint64_t previous_target_ns,
      uint64_t desired_future_anchor_ns, uint64_t issue_anchor_ns,
      uint64_t live_floor_ns, uint64_t ordered_floor_ns,
      uint64_t base_safe_floor_ns, uint64_t executive_tail_ns,
      uint64_t accounted_ns, uint64_t applied_ns) {
    PlannedSpaceTrace& trace = planned_space_trace;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (trace.ordered_refusal_seen ||
        !PlannedSpaceTraceCanRecord(
            now_ns, last_processed_source_issue_period_sequence)) {
      return;
    }
    trace.ordered_refusal_seen = true;
    trace.first_ordered_refusal_time_ns = now_ns;
    trace.first_ordered_refusal_source_id = source_id;
    trace.first_ordered_refusal_target_ns = previous_target_ns;
    trace.live_chains_at_first_ordered_refusal = CountLiveSyntheticChains();

    const uint32_t frontier = CountSyntheticProductionFrontier();
    trace.frontier_at_first_ordered_refusal = frontier;
    ObservePlannedSpaceFrontier(frontier, source_id, 0, "ordered_refusal");
  }

  void ObservePlannedSpaceSyntheticOpportunity(const LogicalOutput& synthetic,
                                               uint64_t b_floor_ns,
                                               uint32_t frontier_before) {
    PlannedSpaceTrace& trace = planned_space_trace;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    const uint64_t source_id = synthetic.candidate.source_id;
    const uint32_t frontier_after = CountSyntheticProductionFrontier();
    ObservePlannedSpaceFrontier(frontier_after, source_id,
                                synthetic.sequence_id, "s_obligation_created");
    if (!trace.ordered_refusal_seen ||
        source_id < trace.first_ordered_refusal_source_id ||
        frontier_before >= kLogicalSyntheticCapacity ||
        !PlannedSpaceTraceWithinWindow(
            now_ns, last_processed_source_issue_period_sequence)) {
      return;
    }
    ++trace.prospective_s_opportunities;
    if (frontier_before == 3 && frontier_after >= 4) {
      ++trace.later_s_refill_3_to_4_total;
    }
    if (trace.prospective_s_logged < trace.prospective_s_source_ids.size() &&
        PlannedSpaceTraceCanRecord(
            now_ns, last_processed_source_issue_period_sequence)) {
      trace.prospective_s_source_ids[trace.prospective_s_logged++] = source_id;
    }
  }

  bool IsPlannedSpaceLoggedProspectiveSource(uint64_t source_id) const {
    for (uint64_t logged_source_id :
         planned_space_trace.prospective_s_source_ids) {
      if (logged_source_id == source_id) {
        return true;
      }
    }
    return false;
  }

  void ObservePlannedSpaceRealCommit(const LogicalOutput& real) {
    PlannedSpaceTrace& trace = planned_space_trace;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (!trace.ordered_refusal_seen ||
        real.candidate.source_id < trace.first_ordered_refusal_source_id ||
        !IsPlannedSpaceLoggedProspectiveSource(real.candidate.source_id) ||
        !PlannedSpaceTraceCanRecord(
            now_ns, last_processed_source_issue_period_sequence)) {
      return;
    }
  }

  void ObservePlannedSpaceRealEnqueue(const LogicalOutput& real) {
    PlannedSpaceTrace& trace = planned_space_trace;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (!trace.ordered_refusal_seen ||
        real.candidate.source_id < trace.first_ordered_refusal_source_id ||
        !IsPlannedSpaceLoggedProspectiveSource(real.candidate.source_id) ||
        !PlannedSpaceTraceCanRecord(
            now_ns, last_processed_source_issue_period_sequence)) {
      return;
    }
  }

  void ResetRealOnlyC2Window(uint64_t epoch) {
    source_real_only_c2_rate_timestamps_ns = {};
    source_real_only_c2_window_bp_wait_ns = 0;
    source_real_only_c2_window_epoch = epoch;
    source_real_only_c2_window_remaining = kRealOnlyValidationSamples;
    // C2 owns completion of the bootstrap while enabled. Any discarded or
    // incomplete C2 window starts a fresh qualification window; it cannot let
    // the independent legacy countdown release Synthetic.
    if (source_bootstrap_real_only_remaining.load(
            std::memory_order_acquire)) {
      SetSourceBootstrapRealOnlyRemaining(kRealOnlyValidationSamples);
    }
  }

  void SetSourceBootstrapRealOnlyRemaining(
      uint32_t remaining, bool c2_comparison_approved = false) {
    const uint32_t previous = source_bootstrap_real_only_remaining.load(
        std::memory_order_relaxed);
    if (previous && !remaining && !c2_comparison_approved) {
      // Fail closed if another path ever tries to complete a C2-owned cycle.
      // Keep one unit armed so the invariant violation cannot release S.
      ++c2_bootstrap_completed_without_comparison_total;
      remaining = 1;
    }
    source_bootstrap_real_only_remaining.store(remaining,
                                               std::memory_order_release);
  }

  void CompleteSourceBootstrapFromC2Comparison() {
    if (!source_bootstrap_real_only_remaining.load(
            std::memory_order_acquire)) {
      return;
    }
    ++c2_bootstrap_completed_by_comparison_total;
    SetSourceBootstrapRealOnlyRemaining(0, true);
  }

  void AcquireRealOnlyC2Baseline(uint64_t period_ns, uint64_t epoch) {
    source_real_only_c2_baseline_valid = period_ns != 0;
    source_real_only_c2_baseline_period_ns =
        source_real_only_c2_baseline_valid ? period_ns : 0;
    source_real_only_c2_baseline_epoch = epoch;
    source_real_only_baseline_period_ns =
        source_real_only_c2_baseline_period_ns;
    if (source_real_only_c2_baseline_valid) {
      ++source_real_only_c2_baseline_acquired_total;
    }
    source_real_only_ratio_per_mille =
        source_real_only_c2_baseline_valid ? 1000 : 0;
  }

  void InvalidateRealOnlyC2BaselineForEpoch(uint64_t epoch) {
    if (source_real_only_c2_baseline_valid) {
      ++source_real_only_c2_baseline_invalidated_total;
    }
    source_real_only_c2_baseline_valid = false;
    source_real_only_c2_baseline_period_ns = 0;
    source_real_only_c2_baseline_epoch = epoch;
    source_real_only_c2_native_reference_valid = false;
    source_real_only_baseline_period_ns = 0;
    ResetRealOnlyC2Window(epoch);
  }

  bool ObserveRealOnlyC2Interval(uint64_t sequence, uint64_t raw_start_ns,
                                 uint64_t interval_ns, uint64_t bp_wait_ns,
                                 bool bootstrap_pending_for_issue) {
    // ObserveSourcePeriod and the independent free-run probe run first for
    // this issue. An epoch number alone is not regime evidence while either
    // transition authority still has an unsettled candidate.
    const bool regime_unsettled =
        source_transition_samples_ns.count() != 0 ||
        SourceRateProbeActive() ||
        source_real_only_c2_window_epoch != period_epoch;
    if (regime_unsettled) {
      if (source_real_only_c2_rate_timestamps_ns.count() > 1) {
        ++source_real_only_c2_comparison_skipped_regime_mismatch_total;
      }
      ResetRealOnlyC2Window(period_epoch);
      return false;
    }
    if (!source_real_only_c2_rate_timestamps_ns.count()) {
      source_real_only_c2_window_epoch = period_epoch;
      source_real_only_c2_rate_timestamps_ns.Add(raw_start_ns);
    }
    source_real_only_c2_rate_timestamps_ns.Add(
        SaturatingAddNs(raw_start_ns, interval_ns));
    source_real_only_c2_window_bp_wait_ns = SaturatingAddNs(
        source_real_only_c2_window_bp_wait_ns, bp_wait_ns);
    const size_t count =
        source_real_only_c2_rate_timestamps_ns.count() - 1;
    source_real_only_c2_window_remaining =
        count < kRealOnlyValidationSamples
            ? uint32_t(kRealOnlyValidationSamples - count)
            : 0;
    if (count < kRealOnlyValidationSamples) {
      // Mirror C2's actual qualified-window progress into the gate consumed
      // by SourceProtectionAllowsSynthetic/SplitAlwaysSArmed. Keep the last
      // unit armed until the completed window has been classified.
      if (source_bootstrap_real_only_remaining.load(
              std::memory_order_acquire)) {
        SetSourceBootstrapRealOnlyRemaining(
            source_real_only_c2_window_remaining);
      }
      return false;
    }

    const uint64_t measured =
        source_real_only_c2_rate_timestamps_ns.AveragePeriodNs();
    const bool stable = measured &&
        source_real_only_c2_rate_timestamps_ns.EffectiveRateStable(
            kSourceEffectiveRateToleranceDivisor);
    source_real_only_c2_last_bp_explained_ns = 0;
    bool qualified_same_regime_comparison = false;
    if (!stable) {
      ++source_real_only_c2_unstable_window_total;
    } else if (regime_unsettled) {
      ++source_real_only_c2_comparison_skipped_regime_mismatch_total;
    } else if (!source_real_only_c2_baseline_valid ||
               source_real_only_c2_baseline_epoch != period_epoch) {
      if (source_real_only_c2_baseline_valid) {
        ++source_real_only_c2_baseline_invalidated_total;
      }
      AcquireRealOnlyC2Baseline(measured, period_epoch);
    } else {
      const uint64_t baseline = source_real_only_c2_baseline_period_ns;
      const bool slower =
          SaturatingMultiplyNs(measured, 100) >
          SaturatingMultiplyNs(baseline, 115);
      const bool faster =
          SaturatingMultiplyNs(measured, 100) <
          SaturatingMultiplyNs(baseline, 85);
      source_real_only_ratio_per_mille =
          baseline ? SaturatingMultiplyNs(measured, 1000) / baseline : 0;
      if (faster) {
        // A faster window is possible transition evidence, not authority to
        // replace the frozen reference inside the current confirmed epoch.
        ++source_real_only_c2_comparison_skipped_regime_mismatch_total;
        ++source_real_only_c2_faster_regime_evidence_total;
      } else if (slower) {
        qualified_same_regime_comparison = true;
        const uint64_t expected_ns = SaturatingMultiplyNs(
            baseline, kRealOnlyValidationSamples);
        const uint64_t measured_ns = SaturatingMultiplyNs(
            measured, kRealOnlyValidationSamples);
        const uint64_t slowdown_excess_ns =
            measured_ns > expected_ns ? measured_ns - expected_ns : 0;
        source_real_only_c2_last_bp_explained_ns =
            std::min(source_real_only_c2_window_bp_wait_ns,
                     slowdown_excess_ns);
        ++source_real_only_c2_comparison_same_regime_total;
        ++source_real_only_c2_slower_regime_evidence_total;
        const bool backpressure_explains_full_excess =
            slowdown_excess_ns &&
            source_real_only_c2_window_bp_wait_ns >= slowdown_excess_ns;
        if (backpressure_explains_full_excess) {
          ++source_real_only_c2_bp_attributed_slowdown_total;
        }
        // Preserve the original sovereignty falsifier only inside the
        // existing bootstrap/re-arm cycle, against the still-valid native
        // reference from the same confirmed epoch. The existing failure
        // criterion needs both a qualified >15% slowdown and BP that explains
        // the whole measured excess; BP by itself is never a failure.
        if (bootstrap_pending_for_issue &&
            source_real_only_c2_native_reference_valid &&
            source_real_only_c2_baseline_epoch == period_epoch &&
            backpressure_explains_full_excess &&
            SaturatingMultiplyNs(measured, 100) >
                SaturatingMultiplyNs(baseline, 115)) {
          ++source_real_only_validation_fail_total;
          XELOGE(
              "ZeroFGC2RealOnly bootstrap_fail sequence={} epoch={} "
              "baseline_us={} measured_us={} ratio_per_mille={} "
              "bp_us={} native_reference=true",
              sequence, period_epoch, baseline / 1000, measured / 1000,
              source_real_only_ratio_per_mille,
              source_real_only_c2_window_bp_wait_ns / 1000);
          Terminalize(TerminalReason::kSourceSovereignty, sequence);
          ResetRealOnlyC2Window(period_epoch);
          return true;
        }
        // Outside bootstrap/re-arm this remains diagnostic. In particular,
        // no same-epoch window replaces or weakens the frozen reference.
      } else {
        qualified_same_regime_comparison = true;
        ++source_real_only_c2_comparison_same_regime_total;
        ++source_real_only_validation_pass_total;
      }
    }
    // A completed, stable comparison against a same-epoch C2 reference is
    // the sole authority that may release this bootstrap cycle. A newly
    // acquired diagnostic baseline, transition evidence, instability, or an
    // unsettled regime is not a comparison and leaves the cycle armed.
    if (qualified_same_regime_comparison && bootstrap_pending_for_issue) {
      CompleteSourceBootstrapFromC2Comparison();
    }
    ResetRealOnlyC2Window(period_epoch);
    return false;
  }

  // Adopts a new Source operating period. The caller owns the evidence: a
  // clustered transition window of effective rates, or a rate probe's free-run
  // window, or a §7 physical path. Applied only when the candidate really is
  // a different regime for the authority that commands it.
  bool ConfirmSourcePeriodTransition(
      uint64_t candidate, uint64_t confirmation_issue_sequence,
      SourcePeriodAuthority authority = SourcePeriodAuthority::kNormal) {
    if (!candidate || !source_period_ns ||
        !SourcePeriodTransitionAuthorized(candidate, authority)) {
      return false;
    }
    const uint64_t previous_period = source_period_ns;
    source_period_ns = candidate;
    // Learned P is the clean-only regime. It follows the normal authority and
    // a full §7 recovery (a free-run measurement back inside it); a §7
    // physical P, initial, refined or partially recovered, never enters it.
    if (authority == SourcePeriodAuthority::kNormal ||
        authority == SourcePeriodAuthority::kPhysicalRecoveryFull) {
      source_period_learned_ns = candidate;
      source_period_learned_samples_ns = {};
      for (uint32_t i = 0; i < 8; ++i) {
        source_period_learned_samples_ns.Add(candidate);
      }
    }
    source_period_samples_ns = {};
    // Blocker #3. The refill has to satisfy the gate the reanchor must pass.
    // SourcePhaseWindowStable() requires kSourcePhaseStableSamples clustered
    // samples; this refilled 8 against a gate of 16, so a confirmed
    // transition could never satisfy it on its own and had to wait for 8
    // further Source intervals to accumulate. Under our own backpressure
    // those intervals are censored, which is how a confirmed 60 Hz Source sat
    // behind a 60 Hz output lattice for about 16 seconds while it wanted 120.
    //
    // A slower transition already bypassed the gate explicitly. The faster
    // direction is the urgent one - every frame is wrong while it waits - and
    // had no bypass at all. Matching the refill to the gate removes the
    // asymmetry without adding a second special case.
    for (uint32_t i = 0; i < kSourcePhaseStableSamples; ++i) {
      source_period_samples_ns.Add(candidate);
    }
    if (candidate < previous_period) {
      ++source_transition_faster_refilled_total;
    }
    source_transition_samples_ns = {};
    source_transition_sample_total = 0;
    if (planned_space_trace.started) {
      // A bounded H1b episode belongs to one confirmed regime. Close it
      // before the next epoch and allow a separate transition-local trace;
      // the current confirmation may have no preceding fast-sequence record.
      FinishPlannedSpaceTrace("epoch_change");
      planned_space_trace = {};
    }
    ++period_epoch;
    InvalidateRealOnlyC2BaselineForEpoch(period_epoch);
    ++source_transition_confirmed_total;
    if (candidate > previous_period) {
      ++source_transition_slower_confirmed_total;
    }
    source_protection_baseline_ns = candidate;
    source_protection_treatment_ns = {};
    source_protection_strikes = 0;
    ResetPhaseDebt();
    // Reconcile frequency without unarming live Real commitments.
    source_rate_transition_reanchor_pending = stable_latency_metronome_armed;
    source_rate_transition_slower_pending =
        stable_latency_metronome_armed && candidate > previous_period;
    BeginPlannedSpaceTrace(previous_period, candidate,
                           confirmation_issue_sequence);
    OnH1cPeriodTransition(previous_period, candidate,
                          confirmation_issue_sequence);
    OnPhysicalOperatingPointPeriodTransition(authority, previous_period,
                                             confirmation_issue_sequence);
    // A normal rate transition does not invalidate an admitted S. Future
    // opportunities use the reconciled frequency after the ordered boundary.
    return true;
  }

  // The one decision of whether a candidate is a new operating period. The
  // normal authority keeps its global regime gate: within a fifth of P it is
  // the same regime. §7's own distinction rule is reachable only through a §7
  // authority, with that path's state prerequisites and its own separately
  // qualified physical evidence; it never relaxes the normal gate.
  bool SourcePeriodTransitionAuthorized(
      uint64_t candidate, SourcePeriodAuthority authority) const {
    const uint64_t candidate_delta = candidate > source_period_ns
                                         ? candidate - source_period_ns
                                         : source_period_ns - candidate;
    const bool physical_owner =
        split_physical_operating_point_enabled;
    switch (authority) {
      case SourcePeriodAuthority::kNormal:
        return candidate_delta > source_period_ns / 5;
      case SourcePeriodAuthority::kPhysicalInitial:
        return physical_owner && !physical_op_confirmed &&
               candidate > source_period_ns &&
               PhysicalOperatingPointPeriodDistinct(candidate);
      case SourcePeriodAuthority::kPhysicalRefinement:
        return physical_owner && physical_op_confirmed &&
               candidate > source_period_ns &&
               PhysicalOperatingPointPeriodDistinct(candidate);
      case SourcePeriodAuthority::kPhysicalRecoveryPartial:
        return physical_owner && physical_op_confirmed &&
               candidate < source_period_ns &&
               PhysicalOperatingPointPeriodDistinct(candidate) &&
               !PhysicalRecoveryReachesLearned(candidate);
      case SourcePeriodAuthority::kPhysicalRecoveryFull:
        // Existing policy: a measurement back in the learned regime ends the
        // physical operating point whatever its distance from the delivered
        // physical P.
        return physical_owner && physical_op_confirmed &&
               PhysicalRecoveryReachesLearned(candidate);
    }
    return false;
  }

  // §7's distinction rule: a factual P at least the distinction percent away
  // from the delivered P. It equals the maintenance floor, so a candidate
  // that stayed legitimately inside the maintenance band is distinct.
  bool PhysicalOperatingPointPeriodDistinct(uint64_t candidate_ns) const {
    if (!candidate_ns || !source_period_ns) {
      return false;
    }
    const uint64_t delta = candidate_ns > source_period_ns
                               ? candidate_ns - source_period_ns
                               : source_period_ns - candidate_ns;
    return SaturatingMultiplyNs(delta, 100) >=
           SaturatingMultiplyNs(source_period_ns,
                                kPhysicalOperatingPointDistinctPercent);
  }

  // A recovery measurement below the maintenance floor over learned P is no
  // longer physically distinct from the learned clean regime.
  bool PhysicalRecoveryReachesLearned(uint64_t candidate_ns) const {
    return source_period_learned_ns &&
           SaturatingMultiplyNs(candidate_ns, 100) <
               SaturatingMultiplyNs(source_period_learned_ns,
                                    kPhysicalOperatingPointMaintenancePercent);
  }

  // Which authority a §7 recovery-probe measurement goes to. Back in the
  // learned regime: full recovery. Faster than the delivered physical P but
  // still distinct from learned: partial recovery, §7 keeps ownership. Not
  // faster: only the normal global gate may judge it, as before.
  SourcePeriodAuthority PhysicalRecoveryProbeAuthority(
      uint64_t candidate_ns) const {
    if (PhysicalRecoveryReachesLearned(candidate_ns)) {
      return SourcePeriodAuthority::kPhysicalRecoveryFull;
    }
    if (candidate_ns < source_period_ns) {
      return SourcePeriodAuthority::kPhysicalRecoveryPartial;
    }
    return SourcePeriodAuthority::kNormal;
  }

  static const char* PhysicalRecoveryOutcomeName(
      SourcePeriodAuthority authority) {
    switch (authority) {
      case SourcePeriodAuthority::kPhysicalRecoveryFull:
        return "full_recovery";
      case SourcePeriodAuthority::kPhysicalRecoveryPartial:
        return "partial_recovery";
      case SourcePeriodAuthority::kNormal:
        return "normal_regime";
      case SourcePeriodAuthority::kPhysicalInitial:
      case SourcePeriodAuthority::kPhysicalRefinement:
        break;
    }
    return "none";
  }

  // Every period transition, whoever commands it. §7 evidence is relative to
  // the delivered P, so a new epoch restarts it, and a live candidate or
  // refinement not owned by this authority is aborted. Recovery outcomes are
  // applied here: partial keeps §7 confirmed at the new physical P; full (or
  // the normal authority taking P) ends it.
  void OnPhysicalOperatingPointPeriodTransition(
      SourcePeriodAuthority authority, uint64_t previous_period_ns,
      uint64_t issue_sequence) {
    if (!split_physical_operating_point_enabled) {
      return;
    }
    physical_op_observations = {};
    physical_op_observation_count = 0;
    physical_op_observation_next = 0;
    physical_op_slack_streak = 0;
    if (authority != SourcePeriodAuthority::kPhysicalInitial) {
      AbortPhysicalOperatingPointCandidate(
          PhysicalOperatingPointAbortReason::kReferenceChanged,
          issue_sequence);
    }
    if (authority != SourcePeriodAuthority::kPhysicalRefinement) {
      AbortPhysicalOperatingPointRefinement(
          PhysicalOperatingPointAbortReason::kReferenceChanged,
          issue_sequence);
    }
    if (!physical_op_confirmed) {
      return;
    }
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (authority == SourcePeriodAuthority::kPhysicalRecoveryPartial) {
      ++physical_op_recovery_partial_total;
      physical_op_probe_cooldown_until_ns =
          SaturatingAddNs(now_ns, kPhysicalOperatingPointCooldownNs);
      return;
    }
    if (authority != SourcePeriodAuthority::kPhysicalRecoveryFull &&
        authority != SourcePeriodAuthority::kNormal) {
      return;
    }
    if (authority == SourcePeriodAuthority::kPhysicalRecoveryFull) {
      ++physical_op_recovery_full_total;
    } else {
      ++physical_op_recovery_normal_exit_total;
    }
    physical_op_confirmed = false;
    physical_operating_point_state = PhysicalOperatingPointState::kIdle;
    physical_op_last_exit_ns = now_ns;
    physical_op_idle_since_ns = physical_op_last_exit_ns;
    ++physical_op_exit_total;
    physical_op_probe_cooldown_until_ns = SaturatingAddNs(
        physical_op_last_exit_ns, kPhysicalOperatingPointCooldownNs);
  }

  void ConsumeSourceIssuePeriods() {
    if (!accepting.load(std::memory_order_acquire)) {
      return;
    }
    ObserveResidualPhaseShadowEpoch(PresenterMonotonicTimeNs());
    if (!source_period_ns) {
      source_period_ns =
          qualified_native_source_period_ns.load(std::memory_order_acquire);
      if (source_period_ns) {
        source_period_learned_ns = source_period_ns;
        source_period_learned_samples_ns = {};
        for (uint32_t i = 0; i < 8; ++i) {
          source_period_samples_ns.Add(source_period_ns);
          source_period_learned_samples_ns.Add(source_period_ns);
        }
        source_protection_baseline_ns = source_period_ns;
      }
    }
    if (!source_issue_cursor_initialized) {
      last_processed_source_issue_period_sequence =
          source_validation_start_sequence.load(std::memory_order_acquire);
      source_issue_cursor_initialized = true;
    }
    const auto snapshot = xe::GetSourceBoundarySnapshot();
    const auto& periods = snapshot.issue_periods;
    if (!periods.consistent) {
      ++source_issue_snapshot_inconsistent_total;
      ResetPhaseDebt();
      PauseResidualPhaseShadow(PresenterMonotonicTimeNs());
      return;
    }
    if (!periods.count ||
        periods.ending_sequence <= last_processed_source_issue_period_sequence) {
      return;
    }
    const uint64_t first = periods.ending_sequence - periods.count + 1;
    if (last_processed_source_issue_period_sequence + 1 < first) {
      source_issue_period_overrun_total +=
          first - last_processed_source_issue_period_sequence - 1;
      OnD1FastSequenceReset("source_issue_overrun", first - 1);
      if (SourceRateProbeActive()) {
        ++source_rate_probe_episode_expired_total;
        if (source_rate_probe_reason ==
            SourceRateProbeReason::kPhysicalOperatingPointRecovery) {
          ++source_rate_probe_recovery_expired_total;
          ++source_rate_probe_recovery_abort_total;
        }
        FinishSourceRateProbe(PresenterMonotonicTimeNs(),
                              kSourceRateProbeCooldownAbortNs);
      }
      // Lost intervals are not fabricated. Rebuild evidence, retaining P and
      // the live lattice. REAL_ONLY must collect a new complete window.
      source_effective_rate_timestamps_ns = {};
      source_transition_samples_ns = {};
      source_transition_sample_total = 0;
      ++source_real_only_c2_window_overrun_total;
      ResetRealOnlyC2Window(period_epoch);
      ResetPhaseDebt();
      PauseResidualPhaseShadow(PresenterMonotonicTimeNs());
      last_processed_source_issue_period_sequence = first - 1;
    }
    for (uint64_t sequence = last_processed_source_issue_period_sequence + 1;
         sequence <= periods.ending_sequence; ++sequence) {
      const uint64_t interval = periods.intervals_ns[size_t(sequence - first)];
      const uint64_t raw_start_ns = source_issue_raw_accumulator_ns;
      const bool bootstrap_pending_for_issue =
          source_bootstrap_real_only_remaining.load(
              std::memory_order_acquire) != 0;
      // P learns only from clean Source intervals. An interval holding a
      // Source wait ZeroFG itself imposed (lossless handoff backpressure) is
      // endogenous, and the first clean one after it may be a catch-up: both
      // are censored and P keeps its last reliable value. REAL_ONLY
      // sovereignty, the phase paths and telemetry keep every raw interval.
      const uint64_t bp_wait = periods.bp_wait_ns[size_t(sequence - first)];
      const bool post_bp_clean = !bp_wait && source_issue_previous_bp;
      const bool censor_period = bp_wait || post_bp_clean;
      source_issue_previous_bp = bp_wait != 0;
      ObserveH1cFastSequenceEvidence(sequence, interval, bp_wait,
                                     censor_period);
      RememberPlannedSpaceFastEvidence(sequence, interval, bp_wait,
                                       censor_period);
      source_issue_raw_interval_ns.Add(interval);
      ObserveSourceIssueCleanRun(bp_wait);
      ObservePhysicalOperatingPointInterval(sequence, interval, bp_wait,
                                             post_bp_clean, !censor_period);
      ObserveSourceRateProbeInterval(sequence, interval, bp_wait);
      if (bp_wait) {
        ++source_issue_bp_interval_total;
        if (split_sidecar_hold_begin_ns) {
          // Source backpressure while the accepted sidecar Real was held:
          // pressure that rose upstream from the shallow executive.
          ++split_sidecar_bp_interval_total;
        }
        source_issue_bp_wait_ns.Add(bp_wait);
        source_issue_bp_wait_ns_total =
            SaturatingAddNs(source_issue_bp_wait_ns_total, bp_wait);
        if (bp_wait >= interval) {
          ++source_issue_bp_clamped_total;
        }
        ++source_issue_p_censored_bp_total;
      } else if (post_bp_clean) {
        ++source_issue_p_censored_post_bp_total;
      } else {
        source_issue_learned_interval_ns.Add(interval);
      }
      if (censor_period && source_transition_samples_ns.count()) {
        // Blocker #4. The samples already collected were measured on CLEAN
        // intervals; a censored one carries no information about them. The
        // old rule wiped them anyway, and this plant alternates about one
        // censored interval in two, so evidence almost never survived to
        // the four samples a confirmation needs (the old reset counter read
        // 31 in one run).
        //
        // Straddling two regimes is the real risk, and there is already an
        // authority for it: SourceTransitionWindowClustered() requires
        // p90-p10 within a tenth of the median, which a window spanning a
        // regime change cannot satisfy. Let the evidence stand and let that
        // test judge it.
        ++source_issue_p_censored_transition_kept_total;
      }
      last_processed_source_issue_period_sequence = sequence;
      ++source_issue_period_consumed_total;
      source_issue_raw_accumulator_ns =
          SaturatingAddNs(source_issue_raw_accumulator_ns, interval);
      if (!censor_period) {
        ObserveSourcePeriod(interval, sequence);
      }
      if (ObserveRealOnlyC2Interval(sequence, raw_start_ns, interval, bp_wait,
                                    bootstrap_pending_for_issue)) {
        return;
      }
      ObservePlannedSpaceSourceInterval(sequence, interval, censor_period);
      ObservePhaseDebt(interval);
      ObserveResidualPhaseShadow(interval, PresenterMonotonicTimeNs());
    }
    MaybeArmSourceRateProbe(PresenterMonotonicTimeNs());
  }

  // Consecutive Source intervals free of our own backpressure. P learns only
  // from a run: the first clean interval after backpressure is censored as a
  // possible catch-up, and a transition needs four clustered samples after
  // that, with no censored interval in between. A plant that alternates
  // backpressure and clean intervals therefore teaches P nothing at all, which
  // is what these runs measure.
  void ObserveSourceIssueCleanRun(uint64_t bp_wait) {
    source_issue_recent_bp_mask = (source_issue_recent_bp_mask << 1) |
                                  (bp_wait ? uint32_t(1) : uint32_t(0));
    if (source_issue_recent_bp_count < kSourceRateProbeBpWindow) {
      ++source_issue_recent_bp_count;
    }
    if (bp_wait) {
      if (source_issue_clean_run_length) {
        source_issue_clean_run_samples.Add(source_issue_clean_run_length);
        source_issue_clean_run_length = 0;
      }
      return;
    }
    ++source_issue_clean_run_length;
    if (source_issue_clean_run_length > source_issue_clean_run_max) {
      source_issue_clean_run_max = source_issue_clean_run_length;
    }
    if (source_issue_clean_run_length == kSourceTransitionCleanRunIntervals) {
      ++source_issue_clean_run_learnable_total;
    }
  }

  // The Source's own rate is unobservable when we pace it and no clean run in
  // the recent window is long enough to teach a transition. A rate-matched
  // handoff produces exactly that: the Source is released once per acceptance
  // and waits again, so backpressure and clean intervals alternate and every
  // clean one is a post-backpressure one. Requiring mostly-backpressure
  // windows would miss it (2026-09-19 Rayman runtime: 8 of 16).
  void SourceRecentIssueWindow(uint32_t& bp_intervals,
                               uint32_t& longest_clean_run) const {
    bp_intervals = 0;
    longest_clean_run = 0;
    uint32_t clean_run = 0;
    for (uint32_t i = 0; i < kSourceRateProbeBpWindow; ++i) {
      if (source_issue_recent_bp_mask & (uint32_t(1) << i)) {
        ++bp_intervals;
        clean_run = 0;
        continue;
      }
      ++clean_run;
      if (clean_run > longest_clean_run) {
        longest_clean_run = clean_run;
      }
    }
  }

  bool SourceRateUnobservable() const {
    if (source_issue_recent_bp_count < kSourceRateProbeBpWindow) {
      return false;
    }
    uint32_t bp_intervals = 0;
    uint32_t longest_clean_run = 0;
    SourceRecentIssueWindow(bp_intervals, longest_clean_run);
    return bp_intervals >= kSourceRateProbePacedIntervals &&
           longest_clean_run < kSourceTransitionCleanRunIntervals;
  }

  static bool PhysicalOperatingPointRawAtLeast(uint64_t raw_ns,
                                               uint64_t reference_ns,
                                               uint64_t percent) {
    return reference_ns && SaturatingMultiplyNs(raw_ns, 100) >=
                               SaturatingMultiplyNs(reference_ns, percent);
  }

  // The dwell is time-based. At a fast delivered P, 128 issues elapse before
  // 2.5 s, and with the lowered entry ratio a maintained candidate would
  // expire on issues before it could ever confirm. The budget therefore
  // covers the dwell's issue span at the delivered cadence plus the
  // confirmation minimum; for a delivered P of 31.25 ms or slower it is 128.
  static uint64_t PhysicalOperatingPointIssueBudget(uint64_t reference_ns) {
    if (!reference_ns) {
      return kPhysicalOperatingPointConfirmIssueBudget;
    }
    const uint64_t dwell_issues =
        (kPhysicalOperatingPointConfirmDwellNs + reference_ns - 1) /
        reference_ns;
    return std::max<uint64_t>(
        kPhysicalOperatingPointConfirmIssueBudget,
        SaturatingAddNs(dwell_issues,
                        kPhysicalOperatingPointConfirmMinimumIssues));
  }

  static const char* PhysicalOperatingPointAbortReasonName(
      PhysicalOperatingPointAbortReason reason) {
    switch (reason) {
      case PhysicalOperatingPointAbortReason::kEvidenceRecovered:
        return "evidence_recovered";
      case PhysicalOperatingPointAbortReason::kPressureReleased:
        return "pressure_released";
      case PhysicalOperatingPointAbortReason::kImplausible:
        return "implausible";
      case PhysicalOperatingPointAbortReason::kIssueBudget:
        return "issue_budget";
      case PhysicalOperatingPointAbortReason::kPeriodNotDistinct:
        return "period_not_distinct";
      case PhysicalOperatingPointAbortReason::kRateProbeActive:
        return "rate_probe_active";
      case PhysicalOperatingPointAbortReason::kRecoveryProbe:
        return "recovery_probe";
      case PhysicalOperatingPointAbortReason::kReferenceChanged:
        return "reference_changed";
      case PhysicalOperatingPointAbortReason::kCount:
        break;
    }
    return "unknown";
  }

  void AbortPhysicalOperatingPointCandidate(
      PhysicalOperatingPointAbortReason reason, uint64_t issue_sequence) {
    if (physical_operating_point_state !=
        PhysicalOperatingPointState::kCandidate) {
      return;
    }
    ++physical_op_candidate_abort_total;
    ++physical_op_candidate_abort_reason_total[size_t(reason)];
    physical_operating_point_state = PhysicalOperatingPointState::kIdle;
    physical_op_candidate_start_ns = 0;
    physical_op_candidate_start_issue = 0;
    physical_op_candidate_reference_ns = 0;
    physical_op_candidate_windows = 0;
    physical_op_candidate_below_entry_windows = 0;
    physical_op_candidate_below_entry_logged = false;
  }

  void ClearPhysicalOperatingPointRefinement() {
    physical_op_refinement_active = false;
    physical_op_refinement_start_ns = 0;
    physical_op_refinement_start_issue = 0;
    physical_op_refinement_candidate_period_ns = 0;
    physical_op_refinement_windows = 0;
    physical_op_refinement_below_entry_windows = 0;
    physical_op_refinement_below_entry_logged = false;
  }

  void AbortPhysicalOperatingPointRefinement(
      PhysicalOperatingPointAbortReason reason, uint64_t issue_sequence) {
    if (!physical_op_refinement_active) {
      return;
    }
    ++physical_op_refinement_abort_total;
    ++physical_op_refinement_abort_reason_total[size_t(reason)];
    ClearPhysicalOperatingPointRefinement();
  }

  void ObservePhysicalOperatingPointInterval(uint64_t issue_sequence,
                                              uint64_t interval_ns,
                                              uint64_t bp_wait_ns,
                                              bool post_bp,
                                              bool clean) {
    if (!split_physical_operating_point_enabled ||
        !source_period_learned_ns || !interval_ns) {
      return;
    }
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    const uint64_t real_finalready_total =
        physical_op_real_finalready_before_target_total.load(
            std::memory_order_acquire);
    const bool real_finalready_before_target =
        physical_op_real_finalready_snapshot_initialized &&
        real_finalready_total > physical_op_real_finalready_observed_total;
    physical_op_real_finalready_observed_total = real_finalready_total;
    physical_op_real_finalready_snapshot_initialized = true;
    physical_op_last_recovery_signal = real_finalready_before_target;
    if (source_rate_probe_recovery_pending && !SourceRateProbeActive() &&
        ((source_rate_probe_recovery_requested_ns &&
          now_ns > SaturatingAddNs(
                       source_rate_probe_recovery_requested_ns,
                       kPhysicalOperatingPointProbeArmTimeoutNs)) ||
         (issue_sequence >= source_rate_probe_recovery_requested_issue &&
          issue_sequence - source_rate_probe_recovery_requested_issue >
              kPhysicalOperatingPointProbeArmIssueBudget))) {
      source_rate_probe_recovery_pending = false;
      ++source_rate_probe_recovery_expired_total;
      physical_op_slack_streak = 0;
      physical_op_probe_cooldown_until_ns = SaturatingAddNs(
          now_ns, kSourceRateProbeCooldownAbortNs);
      source_rate_probe_recovery_requested_ns = 0;
      source_rate_probe_recovery_requested_issue = 0;
    }
    if (PhysicalRecoveryProbePendingOrActive()) {
      AbortPhysicalOperatingPointRefinement(
          PhysicalOperatingPointAbortReason::kRecoveryProbe, issue_sequence);
      return;
    }
    if (SourceRateProbeActive()) {
      AbortPhysicalOperatingPointRefinement(
          PhysicalOperatingPointAbortReason::kRateProbeActive, issue_sequence);
      AbortPhysicalOperatingPointCandidate(
          PhysicalOperatingPointAbortReason::kRateProbeActive, issue_sequence);
      return;
    }
    if (!physical_op_confirmed && !physical_op_idle_since_ns) {
      physical_op_idle_since_ns = now_ns;
    }
    PhysicalOperatingPointObservation observation;
    observation.raw_period_ns = interval_ns;
    observation.bp = bp_wait_ns != 0;
    observation.post_bp = post_bp;
    observation.clean = clean;
    observation.frontier = CountSyntheticProductionFrontier();
    observation.live_chains = CountLiveSyntheticChains();
    observation.free_final_outputs = CountFreeFinalOutputs();
    observation.real_final_ready_before_target =
        real_finalready_before_target;
    const uint64_t holds_now = synthetic_presentation_hold_total;
    observation.hold = physical_op_hold_snapshot_initialized &&
                       holds_now > physical_op_previous_hold_total;
    physical_op_previous_hold_total = holds_now;
    physical_op_hold_snapshot_initialized = true;
    observation.downstream_pressure =
        observation.bp || observation.post_bp || observation.hold ||
        observation.frontier >= kLogicalSyntheticCapacity ||
        observation.live_chains >= 2 || observation.free_final_outputs <= 1;
    physical_op_observations[physical_op_observation_next] = observation;
    physical_op_observation_next =
        (physical_op_observation_next + 1) % kPhysicalOperatingPointWindow;
    physical_op_observation_count = std::min(
        physical_op_observation_count + 1, kPhysicalOperatingPointWindow);

    if (physical_op_observation_count < kPhysicalOperatingPointWindow) {
      return;
    }
    uint32_t bp_count = 0;
    uint32_t post_bp_count = 0;
    uint32_t clean_count = 0;
    uint32_t censored_count = 0;
    uint32_t hold_count = 0;
    uint32_t pressure_count = 0;
    std::array<uint64_t, kPhysicalOperatingPointWindow> raw_periods = {};
    for (uint32_t i = 0; i < kPhysicalOperatingPointWindow; ++i) {
      const auto& sample = physical_op_observations[i];
      bp_count += sample.bp;
      post_bp_count += sample.post_bp;
      clean_count += sample.clean;
      censored_count += sample.bp || sample.post_bp;
      hold_count += sample.hold;
      pressure_count += sample.downstream_pressure;
      raw_periods[i] = sample.raw_period_ns;
    }
    std::sort(raw_periods.begin(), raw_periods.end());
    const uint64_t raw_median_ns =
        raw_periods[kPhysicalOperatingPointWindow / 2];
    const uint32_t window = kPhysicalOperatingPointWindow;
    const auto per_mille = [window](uint32_t value) {
      return uint64_t(value) * 1000 / window;
    };
    physical_op_last_raw_period_ns = raw_median_ns;
    physical_op_last_bp_fraction_per_mille = per_mille(bp_count);
    physical_op_last_post_bp_fraction_per_mille = per_mille(post_bp_count);
    physical_op_last_censored_fraction_per_mille =
        per_mille(censored_count);
    physical_op_last_clean_fraction_per_mille = per_mille(clean_count);
    physical_op_last_hold_fraction_per_mille = per_mille(hold_count);
    physical_op_last_pressure_fraction_per_mille = per_mille(pressure_count);

    // The raw direction is judged against the delivered P. Outside a
    // confirmed §7 it equals the learned clean P; a refinement compares with
    // the delivered physical P. Censorship and pressure floors are unchanged.
    const uint64_t delivered_ns = source_period_ns;
    physical_op_last_raw_ratio_per_mille =
        delivered_ns
            ? SaturatingMultiplyNs(raw_median_ns, 1000) / delivered_ns
            : 0;
    const bool raw_plausible =
        raw_median_ns >= kSourcePeriodMinimumPlausibleNs &&
        raw_median_ns <= kSourcePeriodMaximumPlausibleNs;
    const bool pressure_floors = censored_count * 100 >= window * 40 &&
                                 pressure_count * 100 >= window * 40;
    const bool sustained_physical_pressure = raw_plausible && pressure_floors;
    const bool raw_at_entry = PhysicalOperatingPointRawAtLeast(
        raw_median_ns, delivered_ns, kPhysicalOperatingPointEntryPercent);
    const bool raw_maintained = PhysicalOperatingPointRawAtLeast(
        raw_median_ns, delivered_ns,
        kPhysicalOperatingPointMaintenancePercent);
    const bool evidence_maintained =
        sustained_physical_pressure && raw_maintained;
    const PhysicalOperatingPointAbortReason evidence_loss =
        !raw_plausible
            ? PhysicalOperatingPointAbortReason::kImplausible
            : (!raw_maintained
                   ? PhysicalOperatingPointAbortReason::kEvidenceRecovered
                   : PhysicalOperatingPointAbortReason::kPressureReleased);
    const uint64_t issue_budget =
        PhysicalOperatingPointIssueBudget(delivered_ns);

    if (physical_operating_point_state ==
        PhysicalOperatingPointState::kConfirmed) {
      // Headroom H1c creates by withholding S is not physical slack.
      const bool downstream_slack =
          !observation.hold &&
          observation.real_final_ready_before_target &&
          transition_planned_space_state == TransitionPlannedSpaceState::kIdle;
      if (downstream_slack) {
        physical_op_slack_streak = std::min(
            physical_op_slack_streak + 1,
            kPhysicalOperatingPointSlackIntervals);
      } else {
        physical_op_slack_streak = 0;
      }
      if (physical_op_slack_streak >=
              kPhysicalOperatingPointSlackIntervals &&
          !source_rate_probe_recovery_pending && !SourceRateProbeActive() &&
          now_ns >= physical_op_probe_cooldown_until_ns) {
        source_rate_probe_recovery_pending = true;
        source_rate_probe_recovery_requested_ns = now_ns;
        source_rate_probe_recovery_requested_issue = issue_sequence;
        ++source_rate_probe_recovery_requested_total;
        UpdateH1cEpisode(now_ns, issue_sequence);
      }
      // Refinement: the same hysteresis and distinction rule as the initial
      // candidate, against the currently delivered physical P.
      if (physical_op_refinement_active &&
          PhysicalRecoveryProbePendingOrActive()) {
        AbortPhysicalOperatingPointRefinement(
            PhysicalOperatingPointAbortReason::kRecoveryProbe, issue_sequence);
      } else if (physical_op_refinement_active && !evidence_maintained) {
        AbortPhysicalOperatingPointRefinement(evidence_loss, issue_sequence);
      } else if (!physical_op_refinement_active &&
                 sustained_physical_pressure && raw_at_entry &&
                 !PhysicalRecoveryProbePendingOrActive()) {
        physical_op_refinement_active = true;
        physical_op_refinement_start_ns = now_ns;
        physical_op_refinement_start_issue = issue_sequence;
        physical_op_refinement_windows = 0;
        physical_op_refinement_below_entry_windows = 0;
        physical_op_refinement_below_entry_logged = false;
        ++physical_op_refinement_enter_total;
      }
      if (!physical_op_refinement_active) {
        return;
      }
      physical_op_refinement_candidate_period_ns = raw_median_ns;
      if (physical_op_refinement_windows <
          std::numeric_limits<uint32_t>::max()) {
        ++physical_op_refinement_windows;
      }
      if (!raw_at_entry) {
        ++physical_op_refinement_below_entry_window_total;
        if (physical_op_refinement_below_entry_windows <
            std::numeric_limits<uint32_t>::max()) {
          ++physical_op_refinement_below_entry_windows;
        }
        if (!physical_op_refinement_below_entry_logged) {
          physical_op_refinement_below_entry_logged = true;
        }
      }
      const uint64_t refinement_issues =
          issue_sequence >= physical_op_refinement_start_issue
              ? issue_sequence - physical_op_refinement_start_issue
              : 0;
      if (refinement_issues > issue_budget) {
        AbortPhysicalOperatingPointRefinement(
            PhysicalOperatingPointAbortReason::kIssueBudget, issue_sequence);
        return;
      }
      if (now_ns < SaturatingAddNs(physical_op_refinement_start_ns,
                                   kPhysicalOperatingPointConfirmDwellNs) ||
          refinement_issues < kPhysicalOperatingPointConfirmMinimumIssues) {
        return;
      }
      if (!ConfirmSourcePeriodTransition(
              raw_median_ns, issue_sequence,
              SourcePeriodAuthority::kPhysicalRefinement)) {
        AbortPhysicalOperatingPointRefinement(
            PhysicalOperatingPointAbortReason::kPeriodNotDistinct,
            issue_sequence);
        return;
      }
      ++physical_op_refinement_confirm_total;
      ClearPhysicalOperatingPointRefinement();
      return;
    }

    // Initial candidate: born at the entry ratio, kept alive down to the
    // maintenance floor without restarting its dwell.
    if (physical_operating_point_state == PhysicalOperatingPointState::kIdle &&
        sustained_physical_pressure && raw_at_entry) {
      physical_operating_point_state = PhysicalOperatingPointState::kCandidate;
      physical_op_candidate_start_ns = now_ns;
      physical_op_candidate_start_issue = issue_sequence;
      physical_op_candidate_reference_ns = delivered_ns;
      physical_op_candidate_windows = 0;
      physical_op_candidate_below_entry_windows = 0;
      physical_op_candidate_below_entry_logged = false;
      ++physical_op_candidate_enter_total;
    }
    if (physical_operating_point_state !=
        PhysicalOperatingPointState::kCandidate) {
      return;
    }
    if (!evidence_maintained) {
      AbortPhysicalOperatingPointCandidate(evidence_loss, issue_sequence);
      return;
    }
    physical_op_candidate_period_ns = raw_median_ns;
    if (physical_op_candidate_windows <
        std::numeric_limits<uint32_t>::max()) {
      ++physical_op_candidate_windows;
    }
    if (!raw_at_entry) {
      ++physical_op_candidate_below_entry_window_total;
      if (physical_op_candidate_below_entry_windows <
          std::numeric_limits<uint32_t>::max()) {
        ++physical_op_candidate_below_entry_windows;
      }
      if (!physical_op_candidate_below_entry_logged) {
        physical_op_candidate_below_entry_logged = true;
      }
    }
    const uint64_t candidate_issues =
        issue_sequence >= physical_op_candidate_start_issue
            ? issue_sequence - physical_op_candidate_start_issue
            : 0;
    if (candidate_issues > issue_budget) {
      AbortPhysicalOperatingPointCandidate(
          PhysicalOperatingPointAbortReason::kIssueBudget, issue_sequence);
      return;
    }
    if (now_ns < SaturatingAddNs(physical_op_candidate_start_ns,
                                 kPhysicalOperatingPointConfirmDwellNs) ||
        candidate_issues < kPhysicalOperatingPointConfirmMinimumIssues) {
      return;
    }
    if (!ConfirmSourcePeriodTransition(
            raw_median_ns, issue_sequence,
            SourcePeriodAuthority::kPhysicalInitial)) {
      AbortPhysicalOperatingPointCandidate(
          PhysicalOperatingPointAbortReason::kPeriodNotDistinct,
          issue_sequence);
      return;
    }
    physical_operating_point_state = PhysicalOperatingPointState::kConfirmed;
    physical_op_confirmed = true;
    physical_op_last_entry_ns = now_ns;
    physical_op_idle_since_ns = 0;
    physical_op_slack_streak = 0;
    ++physical_op_confirm_total;
    if (physical_op_exit_total) {
      ++physical_op_reentry_total;
    }
    physical_op_candidate_start_ns = 0;
    physical_op_candidate_start_issue = 0;
    physical_op_candidate_reference_ns = 0;
    physical_op_candidate_windows = 0;
    physical_op_candidate_below_entry_windows = 0;
    physical_op_candidate_below_entry_logged = false;
  }

  bool SourceRateProbeActive() const {
    return source_rate_probe_state != SourceRateProbeState::kIdle;
  }

  void MaybeArmSourceRateProbe(uint64_t now_ns) {
    const bool recovery_probe = source_rate_probe_recovery_pending &&
                                physical_op_confirmed;
    if (SourceRateProbeActive() || !SplitAlwaysSArmed() ||
        !stable_latency_metronome_armed || !source_period_ns ||
        (!recovery_probe && physical_op_confirmed) ||
        source_transition_samples_ns.count() ||
        source_bootstrap_real_only_remaining.load(std::memory_order_acquire) ||
        (!recovery_probe && !SourceRateUnobservable())) {
      return;
    }
    if (source_rate_probe_next_due_ns &&
        now_ns < source_rate_probe_next_due_ns) {
      return;
    }
    source_rate_probe_state = SourceRateProbeState::kDraining;
    source_rate_probe_reason = recovery_probe
                                   ? SourceRateProbeReason::kPhysicalOperatingPointRecovery
                                   : SourceRateProbeReason::kNormal;
    source_rate_probe_recovery_pending = false;
    source_rate_probe_begin_ns = now_ns;
    source_rate_probe_interval_budget = kSourceRateProbeIntervalBudget;
    source_rate_probe_samples_ns = {};
    source_rate_probe_reference_period_ns = source_period_ns;
    source_rate_probe_min_candidate_ns = 0;
    source_rate_probe_max_candidate_ns = 0;
    source_rate_probe_start_issue_sequence =
        last_processed_source_issue_period_sequence;
    source_rate_probe_deadline_issue_sequence = SaturatingAddNs(
        source_rate_probe_start_issue_sequence,
        kSourceRateProbeIntervalBudget);
    source_rate_probe_min_candidate_ns = kSourcePeriodMinimumPlausibleNs;
    source_rate_probe_max_candidate_ns = kSourcePeriodMaximumPlausibleNs;
    ++source_rate_probe_attempt_total;
    ++source_rate_probe_episode_started_total;
    if (recovery_probe) {
      ++source_rate_probe_recovery_attempt_total;
      UpdateH1cEpisode(now_ns, source_rate_probe_start_issue_sequence);
    }
  }

  void FinishSourceRateProbe(uint64_t now_ns, uint64_t cooldown_ns) {
    source_rate_probe_state = SourceRateProbeState::kIdle;
    source_rate_probe_reason = SourceRateProbeReason::kNormal;
    source_rate_probe_samples_ns = {};
    source_rate_probe_begin_ns = 0;
    source_rate_probe_interval_budget = 0;
    source_rate_probe_next_due_ns = SaturatingAddNs(now_ns, cooldown_ns);
  }

  // One interval seen while a probe is open. The drain keeps the Source free,
  // so a clean interval here is the Source's own cadence; the first one after
  // the release is still a catch-up and is never sampled.
  void ObserveSourceRateProbeInterval(uint64_t issue_sequence,
                                     uint64_t interval, uint64_t bp_wait) {
    if (!SourceRateProbeActive()) {
      return;
    }
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (source_rate_probe_interval_budget) {
      --source_rate_probe_interval_budget;
    }
    const bool recovery_probe =
        source_rate_probe_reason ==
        SourceRateProbeReason::kPhysicalOperatingPointRecovery;
    if (issue_sequence > source_rate_probe_deadline_issue_sequence) {
      ++source_rate_probe_episode_expired_total;
      if (source_rate_probe_reason ==
          SourceRateProbeReason::kPhysicalOperatingPointRecovery) {
        ++source_rate_probe_recovery_expired_total;
      }
      FinishSourceRateProbe(now_ns, kSourceRateProbeCooldownAbortNs);
      return;
    }
    if (bp_wait > kSourceRateProbeCleanToleranceNs) {
      // Still paced by us: the drain has not caught up with the Source yet.
      source_rate_probe_state = SourceRateProbeState::kDraining;
      source_rate_probe_samples_ns = {};
    } else if (source_rate_probe_state == SourceRateProbeState::kDraining) {
      source_rate_probe_state = SourceRateProbeState::kMeasuring;
    } else if (interval > bp_wait) {
      source_rate_probe_samples_ns.Add(interval - bp_wait);
    }
    if (source_rate_probe_samples_ns.count() >= kSourceRateProbeSamples) {
      const uint64_t median = source_rate_probe_samples_ns.Quantile(50, 100);
      const bool clustered =
          median && source_rate_probe_samples_ns.Quantile(90, 100) -
                            source_rate_probe_samples_ns.Quantile(10, 100) <=
                        median / 10;
      source_rate_probe_last_candidate_ns = median;
      if (!clustered) {
        ++source_rate_probe_abort_total;
        if (source_rate_probe_reason ==
            SourceRateProbeReason::kPhysicalOperatingPointRecovery) {
          ++source_rate_probe_recovery_abort_total;
        }
        FinishSourceRateProbe(now_ns, kSourceRateProbeCooldownAbortNs);
        return;
      }
      if (median < source_rate_probe_min_candidate_ns ||
          median > source_rate_probe_max_candidate_ns) {
        ++source_rate_probe_candidate_out_of_range_total;
        ++source_rate_probe_no_change_total;
        if (source_rate_probe_reason ==
            SourceRateProbeReason::kPhysicalOperatingPointRecovery) {
          ++source_rate_probe_recovery_no_change_total;
        }
        FinishSourceRateProbe(now_ns, kSourceRateProbeCooldownNoChangeNs);
        return;
      }
      ++source_rate_probe_candidate_in_range_total;
      // A §7 recovery measurement is classified before it is judged: back in
      // the learned regime is a full recovery, faster but still physically
      // distinct is a partial one that keeps §7, anything else only the
      // normal global gate may accept.
      const SourcePeriodAuthority authority =
          recovery_probe ? PhysicalRecoveryProbeAuthority(median)
                         : SourcePeriodAuthority::kNormal;
      if (ConfirmSourcePeriodTransition(median, issue_sequence, authority)) {
        // Measured free of our pacing: the effective-rate evidence of the old
        // regime describes a cadence we imposed and is stale by construction.
        source_effective_rate_timestamps_ns = {};
        ++source_rate_probe_confirm_total;
        ++source_rate_probe_confirmed_transition_total;
        if (recovery_probe) {
          ++source_rate_probe_recovery_confirm_total;
          physical_op_recovery_last_candidate_ns = median;
          physical_op_recovery_last_outcome =
              PhysicalRecoveryOutcomeName(authority);
        }
        FinishSourceRateProbe(now_ns, kSourceRateProbeCooldownConfirmNs);
      } else {
        ++source_rate_probe_no_change_total;
        if (recovery_probe) {
          ++source_rate_probe_recovery_no_change_total;
          physical_op_recovery_last_candidate_ns = median;
          physical_op_recovery_last_outcome =
              authority == SourcePeriodAuthority::kPhysicalRecoveryPartial
                  ? "no_change_not_distinct"
                  : "no_change_not_faster";
        }
        FinishSourceRateProbe(now_ns, kSourceRateProbeCooldownNoChangeNs);
      }
      return;
    }
    if (!source_rate_probe_interval_budget ||
        now_ns > SaturatingAddNs(source_rate_probe_begin_ns,
                                 kSourceRateProbeWallBudgetNs)) {
      ++source_rate_probe_episode_expired_total;
      ++source_rate_probe_abort_total;
      if (source_rate_probe_reason ==
          SourceRateProbeReason::kPhysicalOperatingPointRecovery) {
        ++source_rate_probe_recovery_expired_total;
        ++source_rate_probe_recovery_abort_total;
      }
      FinishSourceRateProbe(now_ns, kSourceRateProbeCooldownAbortNs);
    }
  }

  bool ObserveSourcePeriod(uint64_t interval, uint64_t issue_sequence) {
    // Only new upstream intervals enter the existing span/count estimator.
    const uint64_t previous_issue = source_issue_interval_accumulator_ns;
    source_issue_interval_accumulator_ns =
        SaturatingAddNs(previous_issue, interval);
    const uint64_t current_issue = source_issue_interval_accumulator_ns;
    if (interval < kSourcePeriodMinimumPlausibleNs ||
        interval > kSourcePeriodMaximumPlausibleNs) {
      ++source_rejected_sample_total;
      source_effective_rate_timestamps_ns = {};
      source_effective_rate_timestamps_ns.Add(current_issue);
      if (last_processed_source_issue_period_sequence >
          source_protection_after_issue_sequence) {
        last_pair_synthetic_applied = false;
      }
      return false;
    }
    ++period_sample_total;
    source_pair_interval_ns.Add(interval);
    if (!source_effective_rate_timestamps_ns.count()) {
      source_effective_rate_timestamps_ns.Add(previous_issue);
    }
    source_effective_rate_timestamps_ns.Add(current_issue);
    const bool effective_rate_stable =
        source_effective_rate_timestamps_ns.EffectiveRateStable(
            kSourceEffectiveRateToleranceDivisor);
    const uint64_t effective_period =
        effective_rate_stable
            ? source_effective_rate_timestamps_ns.AveragePeriodNs()
            : 0;
    if (effective_period) {
      source_effective_rate_last_ns = effective_period;
      ++source_effective_rate_observation_total;
    } else if (source_effective_rate_timestamps_ns.count() >=
               kSourceEffectiveRateTimestampSamples) {
      ++source_effective_rate_unstable_total;
    }

    const uint64_t qualified_native_period =
        qualified_native_source_period_ns.load(std::memory_order_acquire);
    if (!source_period_ns) {
      if (qualified_native_period) {
        // The native normal-presenter window is the uncontaminated bootstrap
        // authority. Once ZeroFG is active, a robust effective cadence may
        // still become the new operating point through the transition/phase
        // machinery below.
        source_period_ns = qualified_native_period;
        source_period_learned_ns = qualified_native_period;
        source_period_samples_ns = {};
        source_period_learned_samples_ns = {};
        for (uint32_t i = 0; i < 8; ++i) {
          source_period_samples_ns.Add(qualified_native_period);
          source_period_learned_samples_ns.Add(qualified_native_period);
        }
        source_protection_baseline_ns = qualified_native_period;
      }
    }
    if (!source_period_ns) {
      if (!effective_period) {
        last_pair_synthetic_applied = false;
        return false;
      }
      source_period_samples_ns.Add(effective_period);
      if (source_period_samples_ns.count() < 8 ||
          !SourcePeriodWindowClustered(source_period_samples_ns, 5)) {
        last_pair_synthetic_applied = false;
        return false;
      }
      source_period_ns = source_period_samples_ns.Quantile(50, 100);
      source_period_learned_ns = source_period_ns;
      source_period_learned_samples_ns = {};
      for (uint32_t i = 0; i < 8; ++i) {
        source_period_learned_samples_ns.Add(source_period_ns);
      }
      source_protection_baseline_ns = source_period_ns;
      last_pair_synthetic_applied = false;
      return true;
    }

    // The normal estimator continues to accept only its existing clean
    // observations while §7 owns the delivered operating point. Keep its P
    // memory separate; do not let a clean-only sample silently replace the
    // confirmed physical authority before a recovery probe.
    if (physical_op_confirmed) {
      if (effective_period) {
        source_period_learned_samples_ns.Add(effective_period);
        if (source_period_learned_samples_ns.count() >= 8 &&
            SourcePeriodWindowClustered(source_period_learned_samples_ns, 20)) {
          source_period_learned_ns =
              source_period_learned_samples_ns.Quantile(50, 100);
        }
      }
      return source_period_ns != 0;
    }

    const uint32_t real_only_remaining =
        source_bootstrap_real_only_remaining.load(std::memory_order_acquire);

    if (source_period_ns) {
      if (interval * 100 < source_period_ns * 80) {
        ++source_pair_shorter_than_rate_total;
      } else if (interval * 100 > source_period_ns * 120) {
        ++source_pair_longer_than_rate_total;
      }
    }

    if (effective_period) {
      if (!real_only_remaining) {
        // Competition is measured against the sustained produced cadence, not
        // against an individual frame-skipping interval.
        UpdateSourceProtection(effective_period);
      }
      const uint64_t difference =
          effective_period > source_period_ns
              ? effective_period - source_period_ns
              : source_period_ns - effective_period;
      if (difference > source_period_ns / 5) {
        // A large effective-rate transition is confirmed from multiple stable
        // span/count estimates. Bimodal pair intervals never enter this path
        // individually.
        const uint64_t candidate_median =
            source_transition_samples_ns.Quantile(50, 100);
        if (candidate_median) {
          const uint64_t candidate_difference =
              effective_period > candidate_median
                  ? effective_period - candidate_median
                  : candidate_median - effective_period;
          if (candidate_difference > candidate_median / 8) {
            source_transition_samples_ns = {};
          }
        }
        source_transition_samples_ns.Add(effective_period);
        ++source_transition_sample_total;
        if (source_transition_samples_ns.count() >= 4 &&
            SourceTransitionWindowClustered()) {
          ConfirmSourcePeriodTransition(
              source_transition_samples_ns.Quantile(50, 100),
              issue_sequence);
        }
      } else {
        source_transition_samples_ns = {};
        source_transition_sample_total = 0;
        source_period_samples_ns.Add(effective_period);
        if (SourcePeriodWindowClustered(source_period_samples_ns, 20)) {
          source_period_ns = source_period_samples_ns.Quantile(50, 100);
          source_period_learned_ns = source_period_ns;
          source_period_learned_samples_ns.Add(effective_period);
        }
      }
    }

    // A valid A->B interval remains causal pair evidence even while the
    // robust Source estimator is confirming a new operating point. The
    // current P/O epoch remains authoritative until its future-only reanchor
    // is applied; transition confirmation is not an S admission veto.
    if (last_processed_source_issue_period_sequence >
        source_protection_after_issue_sequence) {
      last_pair_synthetic_applied = false;
    }
    return source_period_ns != 0 && !source_transition_samples_ns.count();
  }

  void UpdateSourceProtection(uint64_t effective_period_ns) {
    if (!source_protection_baseline_ns) {
      source_protection_baseline_ns = source_period_ns;
    }
    if (!last_pair_synthetic_applied ||
        last_processed_source_issue_period_sequence <=
            source_protection_after_issue_sequence) {
      return;
    }
    source_protection_treatment_ns.Add(effective_period_ns);
    if (source_protection_treatment_ns.count() < 16) {
      return;
    }
    const uint64_t treatment_period =
        source_protection_treatment_ns.Quantile(50, 100);
    if (source_protection_baseline_ns &&
        treatment_period * 100 > source_protection_baseline_ns * 110) {
      ++source_protection_strikes;
      ++source_protection_would_strike_total;
    } else {
      source_protection_strikes = 0;
    }
    source_protection_treatment_ns = {};
    if (source_protection_strikes >= 2) {
      // Competition evidence is observation-only. Under the current policy it
      // must not create an S-off probe or cooldown merely because disabling FG
      // could restore native Source cadence.
      source_protection_strikes = 0;
      ++source_protection_would_probe_total;
    }
  }

  bool SourceProtectionAllowsSynthetic() const {
    // Initial Real-only validation remains a real qualification boundary.
    // Dynamic CPU/GPU/queue competition is telemetry, not an S admission veto.
    return !source_bootstrap_real_only_remaining.load(
        std::memory_order_acquire);
  }

  bool IsStablePacingState() const {
    return source_period_ns && !source_transition_samples_ns.count() &&
           accepting.load(std::memory_order_acquire) &&
           !detach_requested.load(std::memory_order_acquire);
  }

  bool SplitAlwaysSArmed() const {
    // Once latency/metronome qualification has completed, a brief source
    // transition must not re-enable shared-device drops for an admitted split
    // obligation. Normal timing eligibility still gates new S creation.
    return stable_latency_metronome_armed &&
           accepting.load(std::memory_order_acquire) && !detach_requested.load(
               std::memory_order_acquire) &&
           !source_bootstrap_real_only_remaining.load(
               std::memory_order_acquire);
  }

  uint32_t LogicalSyntheticCapacity() const {
    return kLogicalSyntheticCapacity;
  }

  bool IsValidSyntheticExecutionState() const {
    return source_period_ns && accepting.load(std::memory_order_acquire) &&
           !detach_requested.load(std::memory_order_acquire);
  }

  uint64_t GenerationCostEstimateNs() const {
    if (generation_service_gpu_ns.count() < kGenerationBootstrapSamples) {
      return synthetic_cost_seed_ns;
    }
    return std::max(synthetic_cost_seed_ns,
                    generation_service_gpu_ns.Quantile(90, 100));
  }

  uint64_t PostCostEstimateNs() const {
    return std::max<uint64_t>(1000000ull,
                              post_service_gpu_ns.Quantile(90, 100));
  }

  uint64_t StageObservationBootstrapNs() const {
    if (stable_output_quantum_ns) {
      return stable_output_quantum_ns;
    }
    if (source_period_ns) {
      return std::max<uint64_t>(source_period_ns / 2, 4000000ull);
    }
    return 8000000ull;
  }

  uint64_t GenerationObservationEstimateNs() const {
    // H13: generation_submit_to_ready_ns is submit-to-CPU-observation wall
    // clock, so it contains this scheduler's own observation delay. Using its
    // p90 to schedule the next observation was self-referential by
    // construction: a late poll inflated the sample, which delayed the next
    // poll, which inflated it again. Rayman showed the result — service p50
    // ~6.85 ms against residence p50 ~30.5 ms / p90 ~37.7 ms, 68 actual polls
    // against 7536 not_due — GPU work finished long before the CPU looked,
    // so the pool appeared occupied after the useful work had ended. That
    // hides real demand from the plant exactly like a production veto does.
    //
    // The cadence now comes from the causal service estimate with the existing
    // bounded bootstrap floor. GenerationCostEstimateNs() is used rather than
    // generation_service_gpu_ns directly because it already encapsulates the
    // seed fallback for the known chained Gen->Post timestamp defect, which
    // can leave the raw window with zero samples. That defect is deliberately
    // not repaired here.
    //
    // generation_submit_to_ready_ns remains excellent residence telemetry; it
    // simply no longer controls when the next observation happens.
    return std::max(GenerationCostEstimateNs(), StageObservationBootstrapNs());
  }

  uint64_t PostObservationEstimateNs() const {
    if (post_submit_to_ready_ns.count() >= kGenerationBootstrapSamples) {
      return std::max<uint64_t>(
          1000000ull, post_submit_to_ready_ns.Quantile(90, 100));
    }
    return std::max(PostCostEstimateNs(), StageObservationBootstrapNs());
  }

  uint64_t StageObservationTimeNs(uint64_t base_ns, uint64_t interval_ns,
                                  uint64_t soft_deadline_ns,
                                  uint64_t hard_deadline_ns) const {
    const uint64_t minimum_interval_ns = 1000000ull;
    uint64_t observation_ns = SaturatingAddNs(
        base_ns, std::max(interval_ns, minimum_interval_ns));
    const auto clamp_to_live_deadline =
        [base_ns, &observation_ns](uint64_t deadline_ns) {
          if (deadline_ns > base_ns && deadline_ns < observation_ns) {
            observation_ns = deadline_ns;
          }
        };
    clamp_to_live_deadline(soft_deadline_ns);
    clamp_to_live_deadline(hard_deadline_ns);
    return observation_ns;
  }

  uint64_t StageObservationRetryIntervalNs() const {
    if (stable_output_quantum_ns) {
      return stable_output_quantum_ns;
    }
    if (source_period_ns) {
      return std::max<uint64_t>(source_period_ns / 2, 1);
    }
    return 8000000ull;
  }

  uint64_t GenerationResidenceEstimateNs() const {
    return generation_submit_to_ready_ns.count()
               ? generation_submit_to_ready_ns.Quantile(90, 100)
               : synthetic_cost_seed_ns;
  }

  uint64_t PostResidenceEstimateNs() const {
    return post_submit_to_ready_ns.count()
               ? post_submit_to_ready_ns.Quantile(90, 100)
               : PostCostEstimateNs();
  }

  bool SyntheticChainEstimatorArmed() const {
    return synthetic_chain_estimator_armed;
  }

  uint64_t SyntheticChainResidenceEstimateNs() const {
    return SyntheticChainEstimatorArmed()
               ? synthetic_chain_residence_ns.Quantile(90, 100)
               : 0;
  }

  bool SyntheticChainWarmupActive() const {
    return synthetic_chain_warmup_started &&
           !SyntheticChainEstimatorArmed();
  }

  void BeginSyntheticChainWarmup() {
    if (synthetic_chain_warmup_started) {
      return;
    }
    synthetic_chain_warmup_started = true;
    synthetic_chain_warmup_capture_near_miss_baseline =
        capture_reserve_near_miss_total.load(std::memory_order_relaxed);
  }

  uint64_t SyntheticChainWarmupCaptureNearMisses() const {
    if (!synthetic_chain_warmup_started) {
      return 0;
    }
    if (SyntheticChainEstimatorArmed()) {
      return synthetic_chain_warmup_capture_near_miss_final;
    }
    const uint64_t current =
        capture_reserve_near_miss_total.load(std::memory_order_relaxed);
    return current >= synthetic_chain_warmup_capture_near_miss_baseline
               ? current - synthetic_chain_warmup_capture_near_miss_baseline
               : 0;
  }

  void ObserveSyntheticChainResidence(uint64_t generation_submit_time_ns,
                                      uint64_t completion_observed_time_ns,
                                      bool warmup_submission) {
    if (!generation_submit_time_ns ||
        completion_observed_time_ns < generation_submit_time_ns) {
      return;
    }
    synthetic_chain_residence_ns.Add(completion_observed_time_ns -
                                     generation_submit_time_ns);
    if (warmup_submission) {
      ++synthetic_chain_warmup_completed_total;
    }
    if (!SyntheticChainEstimatorArmed() &&
        synthetic_chain_residence_ns.count() >=
            kSyntheticChainEstimatorArmSamples) {
      synthetic_chain_estimator_armed = true;
      const uint64_t current =
          capture_reserve_near_miss_total.load(std::memory_order_relaxed);
      synthetic_chain_warmup_capture_near_miss_final =
          current >= synthetic_chain_warmup_capture_near_miss_baseline
              ? current - synthetic_chain_warmup_capture_near_miss_baseline
              : 0;
    }
  }

  uint64_t IssueToReadyEstimateNs() const {
    return issue_to_candidate_ready_ns.Quantile(90, 100);
  }

  uint64_t IssueToCaptureDependencyEstimateNs() const {
    // IssueSwap to capture reservation only. The later handoff into residency
    // waits for a Real to release its slot, which takes longer the deeper D
    // is; counting that wait here let D sustain its own requirement.
    // Known ZeroFG-imposed waits are excluded (see the sample site).
    return capture_issue_to_reserve_dependency_ns.Quantile(90, 100);
  }

  static uint64_t SaturatingAddNs(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs
               ? std::numeric_limits<uint64_t>::max()
               : lhs + rhs;
  }

  static uint64_t SaturatingMultiplyNs(uint64_t value,
                                       uint64_t multiplier) {
    return multiplier &&
                   value > std::numeric_limits<uint64_t>::max() / multiplier
               ? std::numeric_limits<uint64_t>::max()
               : value * multiplier;
  }

  static int64_t SaturatingAddSignedNs(int64_t lhs, int64_t rhs) {
    if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) {
      return std::numeric_limits<int64_t>::max();
    }
    if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs) {
      return std::numeric_limits<int64_t>::min();
    }
    return lhs + rhs;
  }

  static int64_t SaturatingSubtractSignedNs(int64_t lhs, int64_t rhs) {
    if (rhs == std::numeric_limits<int64_t>::min()) {
      return lhs >= 0 ? std::numeric_limits<int64_t>::max()
                      : lhs - rhs;
    }
    return SaturatingAddSignedNs(lhs, -rhs);
  }

  static uint64_t SignedMagnitudeNs(int64_t value) {
    if (value >= 0) {
      return uint64_t(value);
    }
    return uint64_t(-(value + 1)) + 1;
  }

  uint64_t LatencyDepthCushionNs() const {
    const uint64_t output_quantum_ns =
        stable_output_quantum_ns
            ? stable_output_quantum_ns
            : source_period_ns ? std::max<uint64_t>(source_period_ns / 2, 1)
                               : 8000000ull;
    // Deliberate slack, not a symbolic one-millisecond margin. Half an output
    // quantum absorbs ordinary host/service variance while the structural
    // bound prevents this cushion from becoming semantic queue depth.
    return std::clamp<uint64_t>(output_quantum_ns / 2, 4000000ull,
                                12000000ull);
  }

  uint64_t GenerationResidenceOverServiceEstimateNs() const {
    const uint64_t residence_ns = GenerationResidenceEstimateNs();
    const uint64_t service_ns = GenerationCostEstimateNs();
    return residence_ns > service_ns ? residence_ns - service_ns : 0;
  }

  uint64_t PostResidenceOverServiceEstimateNs() const {
    const uint64_t residence_ns = PostResidenceEstimateNs();
    const uint64_t service_ns = PostCostEstimateNs();
    return residence_ns > service_ns ? residence_ns - service_ns : 0;
  }

  uint64_t ServicePathLatencyDepthNs() const {
    // D is the causal/service runway needed to construct a 2x pair, not the
    // time work happened to reside behind other work. Preserve one Source
    // period as the causal floor and add each measured service component once.
    uint64_t service_ns = IssueToCaptureDependencyEstimateNs();
    service_ns = SaturatingAddNs(service_ns, GenerationCostEstimateNs());
    service_ns = SaturatingAddNs(service_ns, PostCostEstimateNs());
    service_ns = SaturatingAddNs(service_ns, dispatch_lead_ns);
    service_ns = SaturatingAddNs(service_ns, LatencyDepthCushionNs());
    return std::max(service_ns, source_period_ns);
  }

  uint64_t OperatingLatencyExcessNs() const {
    const uint64_t required_ns = BoundedRequiredLatencyDepthNs();
    return operating_latency_ns > required_ns
               ? operating_latency_ns - required_ns
               : 0;
  }

  uint64_t RequiredLatencyDepthNs() const {
    // Queue/scheduler residence remains valuable scheduling/congestion/service
    // evidence, but it may not inflate structural semantic depth.
    // Display refresh and latch residence likewise never enter D. Once the
    // chain estimator is armed, a future pair also needs causal runway for its
    // Synthetic chain after B is captured; that runway is the chain service
    // cost (Generation + Post). The measured chain residence (submit to
    // observed completion) also holds scheduling, queue and observation time,
    // so it stays telemetry: counting it let one scheduler spike raise D by
    // estimate alone, with no Synthetic ever short of runway.
    //
    // Phase 5 commit C (#16, and the cushion half of #17). Before the chain
    // estimator arms, the service path is the only bootstrap there is, P floor
    // and cushion included. Once it arms, causal_required_ns - already AB + S
    // in code, O + dep + Gen + Post + lead - stands ALONE. The old
    // max(service, causal) kept a rival estimator alive that carried a static
    // cushion and a floor on source_period_ns, so D inherited P's error.
    // Runtime-inert on every plant measured (service won 0 of 222, the cushion
    // clamped 0 of 222); removed because it is wrong, not because it cost.
    if (!SyntheticChainEstimatorArmed()) {
      return ServicePathLatencyDepthNs();
    }
    uint64_t causal_required_ns = stable_output_quantum_ns;
    causal_required_ns = SaturatingAddNs(
        causal_required_ns, IssueToCaptureDependencyEstimateNs());
    causal_required_ns = SaturatingAddNs(
        causal_required_ns,
        SaturatingAddNs(GenerationCostEstimateNs(), PostCostEstimateNs()));
    causal_required_ns =
        SaturatingAddNs(causal_required_ns, dispatch_lead_ns);
    return causal_required_ns;
  }

  uint64_t LatencyDepthUpperBoundNs() const {
    const uint64_t baseline_upper_ns =
        source_period_ns
            ? SaturatingMultiplyNs(source_period_ns, 2)
            : 64000000ull;
    uint64_t upper_ns =
        std::max<uint64_t>(LatencyDepthCushionNs(), baseline_upper_ns);
    const uint64_t required_ns = RequiredLatencyDepthNs();
    const uint64_t output_quantum_ns =
        stable_output_quantum_ns
            ? stable_output_quantum_ns
            : source_period_ns ? std::max<uint64_t>(source_period_ns / 2, 1)
                               : 0;
    if (required_ns && output_quantum_ns) {
      const uint64_t rounded_ns =
          SaturatingAddNs(required_ns, output_quantum_ns - 1);
      const uint64_t quanta = rounded_ns / output_quantum_ns;
      upper_ns = std::max(upper_ns,
                          SaturatingMultiplyNs(quanta, output_quantum_ns));
    }
    return upper_ns;
  }

  uint64_t LatencyDepthBootstrapFloorNs() const {
    const uint64_t output_quantum_ns =
        source_period_ns ? std::max<uint64_t>(source_period_ns / 2, 1)
                         : LatencyDepthCushionNs();
    return std::min<uint64_t>(
        SaturatingMultiplyNs(output_quantum_ns,
                             kLatencyDepthBootstrapQuanta),
        LatencyDepthUpperBoundNs());
  }

  uint64_t BoundedRequiredLatencyDepthNs() const {
    // Commit C. The cushion is no longer the lower clamp: it is a pure
    // function of O, measured against nothing, and at 120 Hz it alone was
    // half a tick of fat before any learning. The requirement already
    // contains O once armed, so it cannot fall to zero; only the upper rail
    // remains.
    return std::min(RequiredLatencyDepthNs(), LatencyDepthUpperBoundNs());
  }

  uint64_t PhaseReserveTargetNs(uint64_t quantum_ns) const {
    // Better D owns the measured margin in F. A second fixed reserve would
    // add unmeasured depth to every future projection. The issue+D, live and
    // ordered safety floors remain authoritative.
    (void)quantum_ns;
    return 0;
  }

  uint64_t PhaseReserveMinimumNs(uint64_t target_ns) const {
    return target_ns ? std::max<uint64_t>(1000000ull, target_ns / 2) : 0;
  }

  void RefreshPhaseReserveTelemetry(uint64_t now_ns) {
    phase_reserve_target_ns = PhaseReserveTargetNs(stable_output_quantum_ns);
    phase_reserve_minimum_ns =
        PhaseReserveMinimumNs(phase_reserve_target_ns);
    const uint64_t floor_ns =
        SaturatingAddNs(now_ns, dispatch_lead_ns);
    phase_reserve_actual_ns =
        next_semantic_target_ns > floor_ns ? next_semantic_target_ns - floor_ns
                                           : 0;
    phase_reserve_deficit_ns =
        phase_reserve_target_ns > phase_reserve_actual_ns
            ? phase_reserve_target_ns - phase_reserve_actual_ns
            : 0;
  }

  void ResetPhaseDebt(bool count_reset = true) {
    if (phase_debt_block_intervals && count_reset) {
      ++phase_debt_partial_discard_total;
    }
    if (count_reset &&
        (phase_debt_ns || phase_debt_block_intervals ||
         phase_debt_block_error_ns)) {
      ++phase_debt_reset_total;
    }
    phase_debt_ns = 0;
    phase_debt_block_error_ns = 0;
    phase_debt_block_span_ns = 0;
    phase_debt_block_period_ns = 0;
    phase_debt_period_epoch = 0;
    phase_debt_block_intervals = 0;
  }

  void ResetSourcePhaseTelemetry() {
    source_issue_to_publish_ns = {};
    source_phase_epoch_pair_period_last_ns = 0;
    source_phase_current_source_period_last_ns = 0;
    source_phase_period_error_last_ns = 0;
    source_phase_source_anchor_last_ns = 0;
    source_phase_next_semantic_target_last_ns = 0;
    source_phase_error_last_ns = 0;
    source_phase_lead_high_ns = 0;
    source_phase_lag_high_ns = 0;
    source_phase_observation_total = 0;
    source_phase_stable_source_sample_total = 0;
    source_phase_previous_valid = false;
    source_phase_previous_error_ns = 0;
    source_phase_confirmation_start_issue_ns = 0;
    source_phase_confirmation_period_error_ns = 0;
    source_phase_confirmation_frequency_accumulated_ns = 0;
    source_phase_confirmation_samples = 0;
    source_phase_confirmation_consistent_samples = 0;
    source_phase_confirmation_forward_skip_current = 0;
    source_phase_confirmation_forward_skip_total = 0;
    source_phase_confirmation_direction_restart_total = 0;
    source_phase_reanchor_pending = false;
    pending_source_reanchor_period_ns = 0;
    source_reanchor_provenance = {};
    source_phase_reanchor_request_total = 0;
    source_phase_reanchor_faster_source_request_total = 0;
    source_phase_reanchor_slower_source_request_total = 0;
    source_phase_reanchor_total = 0;
    source_phase_reanchor_faster_source_total = 0;
    source_phase_reanchor_slower_source_total = 0;
    source_phase_reanchor_old_output_quantum_ns = 0;
    source_phase_reanchor_new_output_quantum_ns = 0;
    source_phase_reanchor_phase_last_ns = 0;
    source_phase_reanchor_frequency_evidence_last_ns = 0;
    source_phase_reanchor_forward_skip_confirmation_last = 0;
    source_phase_reanchor_live_real_invalidation_total = 0;
    source_phase_frequency_drift_total_ns = 0;
    source_phase_reanchor_impulse_total_ns = 0;
    source_phase_residual_total_ns = 0;
    phase_reserve_target_ns = 0;
    phase_reserve_actual_ns = 0;
    phase_reserve_minimum_ns = 0;
    phase_reserve_deficit_ns = 0;
    ResetPhaseDebt(false);
    phase_debt_blocks_total = 0;
    phase_debt_fire_total = 0;
    phase_debt_reset_total = 0;
    phase_debt_partial_discard_total = 0;
  }

  uint64_t SemanticTargetForTick(uint64_t tick_index) const {
    return SaturatingAddNs(
        semantic_epoch_origin_ns,
        SaturatingMultiplyNs(stable_output_quantum_ns, tick_index));
  }

  static bool TickForSemanticTarget(uint64_t target_ns, uint64_t origin_ns,
                                    uint64_t quantum_ns,
                                    uint64_t& tick_out) {
    if (!target_ns || !origin_ns || !quantum_ns || target_ns < origin_ns) {
      return false;
    }
    const uint64_t distance_ns = target_ns - origin_ns;
    if (distance_ns % quantum_ns) {
      return false;
    }
    tick_out = distance_ns / quantum_ns;
    return true;
  }

  static uint64_t CandidateExecutiveDebtTicks(
      const OutputCandidate& candidate) {
    uint64_t nominal_tick = 0;
    uint64_t assigned_tick = 0;
    if (!TickForSemanticTarget(candidate.semantic_target_time_ns,
                               candidate.semantic_epoch_origin_ns,
                               candidate.semantic_output_quantum_ns,
                               nominal_tick) ||
        !TickForSemanticTarget(candidate.target_time_ns,
                               candidate.semantic_epoch_origin_ns,
                               candidate.semantic_output_quantum_ns,
                               assigned_tick) ||
        assigned_tick < nominal_tick) {
      return 0;
    }
    return assigned_tick - nominal_tick;
  }

  const OutputCandidate* FindLiveExecutiveTailCandidate() const {
    const OutputCandidate* tail = nullptr;
    uint64_t tail_sequence = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped ||
          !logical.candidate.target_time_ns ||
          logical.sequence_id <= tail_sequence) {
        continue;
      }
      tail = &logical.candidate;
      tail_sequence = logical.sequence_id;
    }
    return tail;
  }

  uint64_t ExecutiveTailTargetNs() const {
    uint64_t tail_ns = std::max(last_applied_semantic_target_ns,
                                last_accounted_semantic_target_ns);
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped) {
        continue;
      }
      tail_ns = std::max(tail_ns, logical.candidate.target_time_ns);
    }
    return tail_ns;
  }

  uint64_t CurrentRealExecutiveDebtTicks() const {
    uint64_t debt_ticks = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped ||
          logical.candidate.kind != CandidateKind::kReal ||
          !logical.candidate.target_time_ns) {
        continue;
      }
      debt_ticks =
          std::max(debt_ticks, CandidateExecutiveDebtTicks(logical.candidate));
    }
    return debt_ticks;
  }

  void ObserveNewRealExecutiveDebt(uint64_t previous_debt_ticks,
                                   uint64_t new_debt_ticks) {
    real_exec_debt_high_water =
        std::max(real_exec_debt_high_water, new_debt_ticks);
    if (new_debt_ticks < previous_debt_ticks) {
      real_exec_debt_discharge_tick_total = SaturatingAddNs(
          real_exec_debt_discharge_tick_total,
          previous_debt_ticks - new_debt_ticks);
      if (!new_debt_ticks) {
        ++real_exec_debt_zero_transition_total;
      }
    }
  }

  // Which term produced base_safe_floor_ns. std::max cannot report a tie, so
  // this checks in a fixed order and the first match wins; that only skews the
  // histogram between terms that were equal anyway.
  enum class ReanchorFloorTerm : uint8_t {
    kIssueAnchor,
    kLive,
    kOrdered,
    kCount,
  };

  static const char* ReanchorFloorTermName(ReanchorFloorTerm term) {
    switch (term) {
      case ReanchorFloorTerm::kIssueAnchor:
        return "issue_anchor";
      case ReanchorFloorTerm::kLive:
        return "live";
      case ReanchorFloorTerm::kOrdered:
        return "ordered";
      default:
        return "none";
    }
  }

  static ReanchorFloorTerm ReanchorFloorWinner(
      uint64_t base_safe_floor_ns, uint64_t live_floor_ns,
      uint64_t ordered_floor_ns, uint64_t desired_future_anchor_ns) {
    ReanchorFloorTerm winner = ReanchorFloorTerm::kIssueAnchor;
    if (base_safe_floor_ns == live_floor_ns) {
      winner = ReanchorFloorTerm::kLive;
    }
    if (base_safe_floor_ns == ordered_floor_ns) {
      winner = ReanchorFloorTerm::kOrdered;
    }
    if (base_safe_floor_ns == desired_future_anchor_ns) {
      winner = ReanchorFloorTerm::kIssueAnchor;
    }
    return winner;
  }

  // Telemetry only: decomposes the floor the reanchor just composed. Read-only
  // on values the caller already computed, so it cannot move a target.
  void ObserveReanchorFloorTerms(uint64_t previous_target_ns,
                                 uint64_t desired_future_anchor_ns,
                                 uint64_t live_floor_ns,
                                 uint64_t ordered_floor_ns,
                                 uint64_t base_safe_floor_ns,
                                 uint64_t phase_reserve_ns,
                                 uint64_t executive_tail_ns,
                                 uint64_t last_accounted_ns,
                                 uint64_t last_applied_ns) {
    if (!previous_target_ns) {
      return;
    }
    const ReanchorFloorTerm winner = ReanchorFloorWinner(
        base_safe_floor_ns, live_floor_ns, ordered_floor_ns,
        desired_future_anchor_ns);
    ++reanchor_floor_term_observations_total;
    ++reanchor_floor_winner_total[size_t(winner)];
    // The decisive split. If the composed floor is already behind the cursor,
    // the only thing preventing a compaction is the phase reserve added on top
    // of it, and that is a parameter rather than a causal constraint. If it is
    // not, the winning term is a real constraint and no amount of reserve
    // tuning will help.
    const bool reserve_only_block = base_safe_floor_ns < previous_target_ns;
    if (reserve_only_block) {
      ++reanchor_floor_reserve_only_block_total;
    }
  }

  bool ApplyPendingLatencyDepthReanchor(uint64_t source_id,
                                        uint64_t issue_ns,
                                        uint64_t desired_future_anchor_ns) {
    if (!latency_depth_reanchor_pending || !stable_latency_metronome_armed ||
        !stable_output_quantum_ns || !issue_ns ||
        !desired_future_anchor_ns) {
      return false;
    }

    const bool armed_by_decay = latency_depth_reanchor_armed_by_decay;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    const uint64_t live_floor_ns =
        SaturatingAddNs(now_ns, dispatch_lead_ns);
    const uint64_t executive_tail_ns = ExecutiveTailTargetNs();
    const uint64_t ordered_boundary_ns =
        std::max({executive_tail_ns, last_accounted_semantic_target_ns,
                  last_applied_semantic_target_ns});
    const uint64_t ordered_floor_ns =
        ordered_boundary_ns
            ? SaturatingAddNs(ordered_boundary_ns, stable_output_quantum_ns)
            : 0;
    const uint64_t previous_target_ns = next_semantic_target_ns;
    const uint64_t base_safe_floor_ns =
        std::max({desired_future_anchor_ns, live_floor_ns, ordered_floor_ns});
    const uint64_t reserve_floor_ns = SaturatingAddNs(
        base_safe_floor_ns, PhaseReserveTargetNs(stable_output_quantum_ns));
    // Blocker #2, second half. next_semantic_target_ns is where the cursor
    // currently is, not a constraint on where it may go. Clamping to it made
    // every reanchor forward-only, so a reduced depth could never compact
    // anything. The genuine floors are all in base_safe_floor_ns already:
    // live (not in the past), ordered (not before the committed boundary) and
    // the issue anchor. Committed targets are untouched either way - this
    // moves the future cursor, not existing commitments.
    const uint64_t first_target_ns =
        reserve_floor_ns;
    if (!first_target_ns ||
        first_target_ns == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    // Blocker #2, observational only, and deliberately ABOVE the decay gate
    // below: a refused decay is the most interesting event there is, and
    // decomposing only the ones that got through left the 21 no-ops of a
    // Rayman run invisible while the summary showed 5 terms.
    ObserveReanchorFloorTerms(previous_target_ns, desired_future_anchor_ns,
                              live_floor_ns, ordered_floor_ns,
                              base_safe_floor_ns,
                              PhaseReserveTargetNs(stable_output_quantum_ns),
                              executive_tail_ns,
                              last_accounted_semantic_target_ns,
                              last_applied_semantic_target_ns);
    // Blocker #2. A DECAY MAY NEVER INCREASE THE INSTALLED FUTURE FRONTIER.
    // The 40 reanchor-term observations proved the composed floor already
    // lands ahead of the cursor - issue + D measured +7 to +39 ms later than
    // it - so arming a reanchor from a decay pushed the lattice OUT by about
    // 20.7 ms every time it fired. The contraction mechanism was causing
    // dilation. No floor is at fault: ordered and issue are causal
    // constraints, and the phase reserve was exonerated at 0 of 40.
    //
    // F_old and F_new measure what the depth change allows, but they cannot
    // authorise a write on their own: with cursor < F_new < F_old the
    // projection has compacted while installing F_new would still move the
    // real lattice forward. So the write is gated on the frontier actually
    // installed, which is previous_target_ns. An attack is untouched - it may
    // legitimately deepen D and carry the future out with it.
    if (armed_by_decay) {
      const uint64_t f_new_ns = desired_future_anchor_ns;
      const uint64_t f_old_ns = SaturatingAddNs(
          issue_ns, latency_depth_reanchor_previous_operating_ns);
      latency_depth_decay_projection_compactable_ns = SaturatingAddNs(
          latency_depth_decay_projection_compactable_ns,
          f_old_ns > f_new_ns ? f_old_ns - f_new_ns : 0);
      if (first_target_ns >= previous_target_ns) {
        // Which reason this is decides the next patch, so they are counted
        // apart rather than as one refusal. Under compaction first_target is
        // reserve_floor, so reaching here already means reserve_floor is at or
        // past the installed frontier; these three cases are exhaustive.
        if (f_new_ns >= previous_target_ns) {
          // The lattice was already tighter than the reduced depth demands.
          // There was nothing to compact and the old code expanded it anyway.
          ++latency_depth_decay_already_tighter_total;
        } else if (base_safe_floor_ns >= previous_target_ns) {
          // The reduction had room and a causal floor consumed it. This is
          // the only refusal that argues about ordered or live.
          ++latency_depth_decay_blocked_immutable_total;
          if (ReanchorFloorWinner(base_safe_floor_ns, live_floor_ns,
                                  ordered_floor_ns,
                                  desired_future_anchor_ns) ==
              ReanchorFloorTerm::kOrdered) {
            ObservePlannedSpaceOrderedRefusal(
                source_id, issue_ns, previous_target_ns,
                desired_future_anchor_ns, desired_future_anchor_ns,
                live_floor_ns,
                ordered_floor_ns, base_safe_floor_ns, executive_tail_ns,
                last_accounted_semantic_target_ns,
                last_applied_semantic_target_ns);
          }
        } else {
          // Everything causal was already behind the frontier and only the
          // phase reserve added on top put it past. A parameter, not a
          // constraint - and not attributable before this split existed.
          ++latency_depth_decay_blocked_reserve_only_total;
        }
        latency_depth_reanchor_pending = false;
        latency_depth_reanchor_armed_by_decay = false;
        return false;
      }
      ++latency_depth_decay_actual_compacted_total;
    } else if (desired_future_anchor_ns <= previous_target_ns) {
      // Better D, the symmetric invariant: A RISE THE INSTALLED FRONTIER
      // ALREADY COVERS MOVES NOTHING. If issue + D_new is not past the cursor,
      // the next Real already gets at least D_new, and re-laying the lattice
      // would only add the phase reserve on top of the ordered floor - a
      // few-ms push for a sub-ms deficit, which is blocker #1's ceil rebuilt
      // through the actuator.
      latency_depth_reanchor_pending = false;
      ++better_d_rise_already_deep_total;
      return false;
    }

    std::array<uint64_t, kLogicalOutputCapacity> live_real_targets = {};
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (logical.candidate.kind == CandidateKind::kReal &&
          logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped) {
        live_real_targets[i] = logical.candidate.target_time_ns;
      }
    }

    semantic_epoch_origin_ns = first_target_ns;
    next_semantic_tick_index = 0;
    next_semantic_target_ns = first_target_ns;
    latency_depth_reanchor_pending = false;
    latency_depth_reanchor_armed_by_decay = false;
    ++latency_depth_reanchor_total;
    latency_depth_reanchor_future_anchor_ns = first_target_ns;
    latency_depth_reanchor_impulse_ns =
        PresenterSignedDeltaNs(first_target_ns, previous_target_ns);
    if (armed_by_decay && latency_depth_reanchor_impulse_ns > 0) {
      // Hard gate, structurally unreachable: the decay gate above returns
      // before the write. Any count here is the invariant broken.
      ++latency_depth_decay_forward_dilation_total;
    }
    if (latency_depth_reanchor_impulse_ns < 0) {
      // The gate: the future cursor actually moved back.
      ++latency_depth_lattice_compacted_total;
      latency_depth_lattice_compacted_ns = SaturatingAddNs(
          latency_depth_lattice_compacted_ns,
          uint64_t(-latency_depth_reanchor_impulse_ns));
    }

    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      if (live_real_targets[i] &&
          logical_outputs[i].candidate.target_time_ns !=
              live_real_targets[i]) {
        ++latency_depth_reanchor_live_real_invalidation_total;
      }
    }
    return true;
  }

  // E1F-SHADOW: telemetry-only mirror of the future-only D reanchor safe-floor
  // composition in ApplyPendingLatencyDepthReanchor(). It reuses the same
  // helpers and saturating arithmetic and returns what that reanchor's
  // first_target_ns would be right now, but it never requires or consumes
  // latency_depth_reanchor_pending and, being const, cannot write any pacing
  // state. shadow_now_ns is a neutral monotonic anchor: this runs inside
  // PumpLogicalOrder() with no Source Issue event, so there is no issue_ns and
  // the live path's issue-derived desired_future_anchor term is replaced by
  // shadow_now_ns. Lateness magnitude is intentionally not added; physical
  // time is already represented by shadow_now_ns and shadow_live_floor_ns.
  uint64_t ComputeE1FShadowFirstTargetNs(uint64_t shadow_now_ns) const {
    const uint64_t shadow_live_floor_ns =
        SaturatingAddNs(shadow_now_ns, dispatch_lead_ns);
    const uint64_t shadow_ordered_boundary_ns =
        std::max({ExecutiveTailTargetNs(), last_accounted_semantic_target_ns,
                  last_applied_semantic_target_ns});
    const uint64_t shadow_ordered_floor_ns =
        shadow_ordered_boundary_ns
            ? SaturatingAddNs(shadow_ordered_boundary_ns,
                              stable_output_quantum_ns)
            : 0;
    const uint64_t shadow_base_safe_floor_ns = std::max(
        {shadow_now_ns, shadow_live_floor_ns, shadow_ordered_floor_ns});
    const uint64_t shadow_reserve_floor_ns = SaturatingAddNs(
        shadow_base_safe_floor_ns,
        PhaseReserveTargetNs(stable_output_quantum_ns));
    return std::max(next_semantic_target_ns, shadow_reserve_floor_ns);
  }

  RunwayShadowFuture FindUnifiedRunwayShadowFuture() const {
    const LogicalOutput* first = nullptr;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped ||
          logical.candidate.kind != CandidateKind::kReal ||
          logical.candidate.target_time_ns) {
        continue;
      }
      if (!first || logical.sequence_id < first->sequence_id) {
        first = &logical;
      }
    }
    RunwayShadowFuture future;
    if (first) {
      future.source_id = first->candidate.source_id;
      future.sequence_id = first->sequence_id;
      for (const LogicalOutput& logical : logical_outputs) {
        // H12: "unanchored" is the absence of a target, not a production
        // state. Keying this on the retired kWaitingPairAnchor state would
        // silently never match now that admission enters kWaitingGeneration.
        if (logical.state != LogicalOutputState::kFree &&
            logical.state != LogicalOutputState::kDropped &&
            logical.candidate.kind == CandidateKind::kSynthetic &&
            !logical.candidate.target_time_ns &&
            logical.candidate.pair_a_source_id == future.source_id) {
          future.pair_a = logical.candidate.pair_a_source_id;
          future.pair_b = logical.candidate.source_id;
          break;
        }
      }
    } else if (incoming_real.occupied()) {
      future.source_id = incoming_real.source_id;
    }
    // Never invent an unaccepted Source identity or a pair across mailbox gaps.
    return future;
  }

  RunwayShadowProposal ComputeUnifiedRunwayShadowProposal(
      uint64_t now_ns, uint64_t proposed_period_ns, bool qualified) const {
    RunwayShadowProposal proposal;
    proposal.time_ns = now_ns;
    proposal.period_epoch = period_epoch;
    proposal.epoch_origin_ns = semantic_epoch_origin_ns;
    proposal.quantum_ns = stable_output_quantum_ns;
    proposal.proposed_period_ns = proposed_period_ns;
    proposal.proposed_quantum_ns = proposed_period_ns / 2;
    proposal.future = FindUnifiedRunwayShadowFuture();
    if (qualified && proposal.proposed_quantum_ns) {
      const int64_t period_delta_ns = PresenterSignedDeltaNs(
          proposed_period_ns, SaturatingMultiplyNs(stable_output_quantum_ns, 2));
      // Deadband labels only: retain raw P/O in the counterfactual geometry.
      proposal.kind = SignedMagnitudeNs(period_delta_ns) <=
                              kSourcePhaseMinimumFrequencyErrorNs
                          ? RunwayShadowKind::kPhaseOnly
                      : period_delta_ns > 0 ? RunwayShadowKind::kSlower
                                            : RunwayShadowKind::kFaster;
      const uint64_t boundary_ns =
          std::max({ExecutiveTailTargetNs(), last_accounted_semantic_target_ns,
                    last_applied_semantic_target_ns});
      const uint64_t ordered_floor_ns =
          boundary_ns ? SaturatingAddNs(boundary_ns,
                                       proposal.proposed_quantum_ns)
                      : 0;
      const uint64_t safe_floor_ns = std::max(
          SaturatingAddNs(now_ns, dispatch_lead_ns), ordered_floor_ns);
      proposal.first_target_ns = std::max(
          next_semantic_target_ns,
          SaturatingAddNs(safe_floor_ns,
                         PhaseReserveTargetNs(proposal.proposed_quantum_ns)));
      proposal.impulse_ns = proposal.first_target_ns - next_semantic_target_ns;
    }

    // Read-only counterparts of the existing pipeline starvation classes.
    // Active GPU is context, not proof of pressure; no class vetoes a proposal.
    proposal.free_final = CountFreeFinalOutputs();
    proposal.free_residency = CountFreeCandidateSlots();
    proposal.capture_used = CountSourceIngressSlots();
    proposal.generation_jobs = CountGenerationJobs();
    proposal.post_jobs = CountPostJobs();
    proposal.candidate_pressure = CandidateReserveWarning();
    if (ResidencyBackpressureRequiresProgress()) {
      proposal.pressure_mask |= kResourceStarvationResidency;
    }
    if (!proposal.free_final) {
      proposal.pressure_mask |= kResourceStarvationFinalOutput;
    }
    const uint32_t waiting_post = FindOldestWaitingPostRaw();
    if (waiting_post != UINT32_MAX && proposal.free_final &&
        proposal.free_final <=
            RequiredFinalOutputSurplusFor(logical_outputs[waiting_post])) {
      proposal.pressure_mask |= kResourceStarvationFunding;
    }
    const LogicalOutput* head = FindLogicalBySequence(next_apply_sequence);
    if (head && head->state != LogicalOutputState::kFinalReady &&
        head->state != LogicalOutputState::kDropped) {
      for (const LogicalOutput& logical : logical_outputs) {
        if (logical.state == LogicalOutputState::kFinalReady &&
            logical.sequence_id > head->sequence_id) {
          proposal.pressure_mask |= kResourceStarvationOrdering;
          break;
        }
      }
      if (head->state == LogicalOutputState::kWaitingGeneration ||
          head->state == LogicalOutputState::kWaitingPost) {
        proposal.scheduler_present =
            ClassifyUnsubmittedLatencyMiss(*head) == LatencyMissCause::kScheduler;
      }
    }
    if (proposal.generation_jobs || proposal.post_jobs) {
      proposal.pressure_mask |= kResourceStarvationActiveGpu;
    }
    proposal.timing_only_candidate =
        qualified && !proposal.candidate_pressure &&
        !(proposal.pressure_mask & ~uint32_t(kResourceStarvationActiveGpu));
    return proposal;
  }

  void ObserveUnifiedRunwayShadowEpoch(uint64_t now_ns) {
    auto& shadow = unified_runway_shadow;
    if (shadow.epoch_observed &&
        (shadow.observed_period_epoch != period_epoch ||
         shadow.observed_quantum_ns != stable_output_quantum_ns ||
         shadow.observed_epoch_origin_ns != semantic_epoch_origin_ns)) {
      ++shadow.epoch_reset_total;
      if (shadow.episode_active) {
        shadow.episode_duration_ns = now_ns - shadow.episode_start_ns;
        shadow.episode_generation_late_delta =
            synthetic_drop_total[size_t(SyntheticDropReason::kGenerationLate)] -
            shadow.episode_generation_late_start;
      }
      // Invalidate episode evidence and any pending future witness. Cumulative
      // counters/H distributions remain session diagnostics, never evidence
      // for qualifying another episode. The independent upstream Source window
      // and its freshness cursor are untouched.
      shadow.episode_active = false;
      shadow.episode_fired = false;
      shadow.qualification_pending = false;
      shadow.deficit_streak = 0;
      shadow.awaiting_commit_boundary = false;
      shadow.episode_start_ns = 0;
      shadow.episode_duration_ns = 0;
      shadow.episode_h_min_ns = 0;
      shadow.episode_deficit_max_ns = 0;
      shadow.episode_start_period_ns = 0;
      shadow.episode_current_period_ns = 0;
      shadow.episode_period_min_ns = 0;
      shadow.episode_period_max_ns = 0;
      shadow.episode_generation_late_start = 0;
      shadow.episode_generation_late_delta = 0;
      shadow.proposal = {};
      shadow.last_sample_ns = 0;
    }
    shadow.epoch_observed = true;
    shadow.observed_period_epoch = period_epoch;
    shadow.observed_quantum_ns = stable_output_quantum_ns;
    shadow.observed_epoch_origin_ns = semantic_epoch_origin_ns;
    shadow.epoch_period_ns = SaturatingMultiplyNs(stable_output_quantum_ns, 2);
  }

  void ObserveUnifiedRunwayShadow(uint64_t now_ns) {
    auto& shadow = unified_runway_shadow;
    if (!accepting.load(std::memory_order_acquire) ||
        terminal_reason.load(std::memory_order_acquire) !=
            TerminalReason::kNone) {
      return;
    }
    // Check even between samples, before a pre-commit witness can be recorded.
    ObserveUnifiedRunwayShadowEpoch(now_ns);
    if (!stable_latency_metronome_armed || !stable_output_quantum_ns ||
        !next_semantic_target_ns ||
        now_ns < SaturatingAddNs(shadow.last_sample_ns,
                                 kUnifiedRunwayShadowSampleNs)) {
      return;
    }
    shadow.last_sample_ns = now_ns;
    const uint64_t ordered_boundary_ns =
        std::max({ExecutiveTailTargetNs(), last_accounted_semantic_target_ns,
                  last_applied_semantic_target_ns});
    const uint64_t ordered_floor_ns =
        ordered_boundary_ns
            ? SaturatingAddNs(ordered_boundary_ns, stable_output_quantum_ns)
            : 0;
    const uint64_t live_floor_ns = SaturatingAddNs(now_ns, dispatch_lead_ns);
    // Signed extension of canonical phase_reserve_actual_ns. Pair-local
    // commitments ahead of the cursor constrain reform, not live runway.
    shadow.h_live_ns = PresenterSignedDeltaNs(next_semantic_target_ns, live_floor_ns);
    shadow.ordered_gap_ns =
        PresenterSignedDeltaNs(next_semantic_target_ns, ordered_floor_ns);
    shadow.reform_gap_ns = PresenterSignedDeltaNs(
        next_semantic_target_ns, std::max(live_floor_ns, ordered_floor_ns));
    shadow.h_live_min_ns = shadow.sample_total
                              ? std::min(shadow.h_live_min_ns, shadow.h_live_ns)
                              : shadow.h_live_ns;
    ++shadow.sample_total;
    shadow.AddHSample(shadow.h_live_ns);
    shadow.h_target_ns = PhaseReserveTargetNs(stable_output_quantum_ns);
    shadow.h_deficit_ns = uint64_t(std::max<int64_t>(
        0, SaturatingSubtractSignedNs(int64_t(shadow.h_target_ns),
                                      shadow.h_live_ns)));

    // This raw 31-timestamp window is built from sequence-deduplicated Source
    // Issue intervals, before robust clustering, and resets on invalid input
    // or ring overrun. Its span/count remains observable during a ramp. Do not
    // use source_period_samples_ns: those samples are already stability-gated.
    shadow.observed_period_ns =
        source_effective_rate_timestamps_ns.AveragePeriodNs();
    shadow.source_clustered =
        source_effective_rate_timestamps_ns.EffectiveRateStable(
            kSourceEffectiveRateToleranceDivisor);
    shadow.source_fresh = last_processed_source_issue_period_sequence >
                          shadow.last_issue_sequence;
    shadow.last_issue_sequence = last_processed_source_issue_period_sequence;
    const bool deficit =
        shadow.h_live_ns < int64_t(PhaseReserveMinimumNs(shadow.h_target_ns));
    if (!shadow.episode_active && deficit) {
      shadow.episode_active = true;
      ++shadow.episode_total;
      shadow.episode_start_ns = now_ns;
      shadow.episode_h_min_ns = shadow.h_live_ns;
      shadow.episode_deficit_max_ns = 0;
      shadow.episode_start_period_ns =
          shadow.source_fresh ? shadow.observed_period_ns : 0;
      shadow.episode_current_period_ns = shadow.episode_start_period_ns;
      shadow.episode_period_min_ns = shadow.episode_start_period_ns;
      shadow.episode_period_max_ns = shadow.episode_start_period_ns;
      shadow.episode_generation_late_start =
          synthetic_drop_total[size_t(SyntheticDropReason::kGenerationLate)];
      shadow.deficit_streak = 0;
    }
    if (!shadow.episode_active) {
      return;
    }
    shadow.episode_duration_ns = now_ns - shadow.episode_start_ns;
    shadow.episode_h_min_ns = std::min(shadow.episode_h_min_ns, shadow.h_live_ns);
    shadow.episode_deficit_max_ns =
        std::max(shadow.episode_deficit_max_ns, shadow.h_deficit_ns);
    if (shadow.source_fresh && shadow.observed_period_ns) {
      shadow.episode_current_period_ns = shadow.observed_period_ns;
      shadow.episode_period_min_ns =
          shadow.episode_period_min_ns
              ? std::min(shadow.episode_period_min_ns, shadow.observed_period_ns)
              : shadow.observed_period_ns;
      shadow.episode_period_max_ns =
          std::max(shadow.episode_period_max_ns, shadow.observed_period_ns);
    }
    shadow.episode_generation_late_delta =
        synthetic_drop_total[size_t(SyntheticDropReason::kGenerationLate)] -
        shadow.episode_generation_late_start;
    // Observation hysteresis: touching H_min is not recovery. End only at a
    // sampled H >= the full current H_target; no reserve is replenished here.
    if (shadow.h_live_ns >= int64_t(shadow.h_target_ns)) {
      shadow.episode_active = false;
      shadow.episode_fired = false;
      shadow.qualification_pending = false;
      shadow.deficit_streak = 0;
      shadow.awaiting_commit_boundary = false;
      return;
    }
    if (shadow.episode_fired) {
      return;
    }
    if (!shadow.qualification_pending) {
      shadow.deficit_streak = deficit ? shadow.deficit_streak + 1 : 0;
      if (shadow.deficit_streak < kUnifiedRunwayShadowFireSamples) {
        return;
      }
    }
    if (!shadow.source_fresh || !shadow.source_clustered ||
        !shadow.observed_period_ns) {
      if (!shadow.qualification_pending) {
        shadow.qualification_pending = true;
        ++shadow.kind_total[size_t(RunwayShadowKind::kUnqualified)];
      }
      return;
    }
    shadow.deficit_streak = 0;
    shadow.qualification_pending = false;
    shadow.episode_fired = true;
    ++shadow.qualified_proposal_total;
    shadow.proposal = ComputeUnifiedRunwayShadowProposal(
        now_ns, shadow.observed_period_ns,
        shadow.source_fresh && shadow.source_clustered);
    ++shadow.kind_total[size_t(shadow.proposal.kind)];
    if (shadow.proposal.kind != RunwayShadowKind::kUnqualified) {
      shadow.impulse_ns.Add(shadow.proposal.impulse_ns);
    }
    if (shadow.proposal.candidate_pressure ||
        (shadow.proposal.pressure_mask &
         (kResourceStarvationResidency | kResourceStarvationFinalOutput |
          kResourceStarvationFunding))) {
      ++shadow.physical_pressure_total;
    }
    if (!shadow.proposal.future.source_id) {
      ++shadow.no_affectable_future_total;
    }
    shadow.awaiting_commit_boundary =
        shadow.proposal.kind != RunwayShadowKind::kUnqualified;
    // E2A implementation/telemetry remain historical; E2B is the sole canary.
    RequestE2BClockCanary();
  }

  void ObserveUnifiedRunwayShadowCommitBoundary(const LogicalOutput& logical,
                                                uint64_t now_ns) {
    auto& shadow = unified_runway_shadow;
    if (!shadow.awaiting_commit_boundary || logical.candidate.target_time_ns ||
        logical.candidate.kind != CandidateKind::kReal) {
      return;
    }
    shadow.awaiting_commit_boundary = false;
    ++shadow.commit_boundary_observed_total;
  }

  bool E2BResidualEvidenceValid() const {
    const auto& residual = residual_phase_shadow;
    // Reuse the shadow's material episode, never turn its balance into a
    // correction amount. Qualification flicker still belongs to the shadow.
    return residual.epoch_observed && !residual.paused && residual.fired &&
           residual.blocks_total && residual.period_epoch == period_epoch &&
           residual.quantum_ns == stable_output_quantum_ns &&
           residual.origin_ns == semantic_epoch_origin_ns &&
           SignedMagnitudeNs(residual.residual_ns) >=
               std::max<uint64_t>(stable_output_quantum_ns / 4, 1);
  }

  void RequestE2BClockCanary() {
    const auto& shadow = unified_runway_shadow;
    if (e2b.pending || shadow.proposal.kind != RunwayShadowKind::kPhaseOnly ||
        !shadow.source_fresh || !shadow.source_clustered ||
        !shadow.proposal.timing_only_candidate ||
        !shadow.episode_generation_late_delta) {
      return;
    }
    if (!E2BResidualEvidenceValid()) {
      ++e2b.blocked_residual_invalid;
      return;
    }
    if (!SourcePhaseWindowStable()) {
      ++e2b.blocked_source_unqualified;
      return;
    }
    const uint64_t old_period_ns =
        SaturatingMultiplyNs(shadow.proposal.quantum_ns, 2);
    const uint64_t robust_period_ns = source_period_ns;
    const int64_t robust_delta_ns =
        PresenterSignedDeltaNs(robust_period_ns, old_period_ns);
    const int32_t robust_sign = (robust_delta_ns > 0) - (robust_delta_ns < 0);
    const int32_t residual_sign =
        (residual_phase_shadow.residual_ns > 0) -
        (residual_phase_shadow.residual_ns < 0);
    if (!robust_period_ns ||
        SignedMagnitudeNs(robust_delta_ns) >
            kSourcePhaseMinimumFrequencyErrorNs ||
        robust_sign != residual_sign) {
      ++e2b.blocked_residual_period_incoherent;
      return;
    }
    e2b.witness = shadow.proposal;
    e2b.episode = shadow.episode_total;
    e2b.issue_sequence = last_processed_source_issue_period_sequence;
    e2b.source_window_count = source_effective_rate_timestamps_ns.count();
    e2b.robust_sample_count = source_period_samples_ns.count();
    e2b.source_qualified = shadow.source_clustered && shadow.source_fresh;
    e2b.raw_period_witness_ns =
        source_effective_rate_timestamps_ns.AveragePeriodNs();
    e2b.robust_period_witness_ns = robust_period_ns;
    e2b.robust_delta_witness_ns = robust_delta_ns;
    const auto& residual = residual_phase_shadow;
    e2b.residual_ns = residual.residual_ns;
    e2b.residual_sign = (residual.residual_ns > 0) - (residual.residual_ns < 0);
    e2b.residual_blocks = residual.blocks_total;
    e2b.residual_period_epoch = residual.period_epoch;
    e2b.residual_quantum_ns = residual.quantum_ns;
    e2b.residual_origin_ns = residual.origin_ns;
    e2b.residual_resets = residual.structural_reset_total;
    e2b.source_reanchors = source_phase_reanchor_total;
    e2b.latency_reanchors = latency_depth_reanchor_total;
    e2b.generation_late_start = shadow.episode_generation_late_start;
    e2b.generation_late_snapshot =
        synthetic_drop_total[size_t(SyntheticDropReason::kGenerationLate)];
    e2b.pending = true;
    ++e2b.request_total;
  }

  void RecordLatencyFeedbackExclusion(LatencyMissCause cause) {
    ++latency_feedback_excluded_total;
    switch (cause) {
      case LatencyMissCause::kResidencyPressure:
        ++latency_feedback_excluded_residency_pressure_total;
        break;
      case LatencyMissCause::kFinalOutputPressure:
        ++latency_feedback_excluded_final_output_pressure_total;
        break;
      case LatencyMissCause::kFunding:
        ++latency_feedback_excluded_funding_total;
        break;
      case LatencyMissCause::kOrdering:
        ++latency_feedback_excluded_ordering_total;
        break;
      case LatencyMissCause::kScheduler:
        ++latency_feedback_excluded_scheduler_total;
        break;
      case LatencyMissCause::kTransition:
        ++latency_feedback_excluded_transition_total;
        break;
      case LatencyMissCause::kTrueFeasibility:
        break;
    }
  }

  void ObserveLatencyFeasibilityMiss(
      const LogicalOutput& logical,
      LatencyMissCause cause = LatencyMissCause::kTrueFeasibility,
      LatencyMissProvenance provenance =
          LatencyMissProvenance::kSemanticHardLate) {
    ++latency_miss_provenance_total[size_t(provenance)];
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (logical.candidate.target_time_ns && stable_output_quantum_ns &&
        now_ns < SaturatingAddNs(logical.candidate.target_time_ns,
                                 stable_output_quantum_ns)) {
      ++latency_miss_provenance_one_quantum_help_total[size_t(provenance)];
    }
    if (cause != LatencyMissCause::kTrueFeasibility) {
      RecordLatencyFeedbackExclusion(cause);
      return;
    }
    if (!stable_latency_metronome_armed ||
        !logical.stable_pair ||
        logical.latency_epoch != latency_epoch ||
        logical.semantic_forward_skipped) {
      RecordLatencyFeedbackExclusion(LatencyMissCause::kTransition);
      return;
    }
    ++latency_feedback_true_feasibility_total;
    ++latency_feasibility_miss_total;
    // Counted, and no more. Under Better D a real deficit arrives as a
    // qualified C_pair sample and raises F by exactly its size; a miss
    // streak attacking by a whole O is the quantized rise §1.6 forbids.
  }

  // One latency judgment per S: its physical FinalReady time against the soft
  // deadline derived from its anchor. At FinalReady for an S already anchored;
  // when the anchor arrives for one that was FinalReady first (deferred). The
  // anchor time itself teaches nothing.
  void ScoreSyntheticFinalReadyLatency(LogicalOutput& logical,
                                       uint64_t ready_judgement_ns,
                                       bool deferred_anchor) {
    if (!ready_judgement_ns || logical.latency_feedback_scored) {
      return;
    }
    logical.latency_feedback_scored = true;
    if (!deferred_anchor &&
        logical.candidate.target_time_ns > ready_judgement_ns) {
      synthetic_final_ready_lead_ns.Add(logical.candidate.target_time_ns -
                                        ready_judgement_ns);
    }
    if (logical.final_ready_soft_deadline_ns &&
        ready_judgement_ns >= logical.final_ready_soft_deadline_ns) {
      ++final_ready_miss_total;
      if (deferred_anchor) {
        ++synthetic_deferred_feedback_miss_total;
      }
      ObserveLatencyFeasibilityMiss(
          logical, LatencyMissCause::kScheduler,
          LatencyMissProvenance::kFinalReadySoftLate);
    } else {
      ObserveLatencySuccess(logical);
      ++final_ready_success_total;
      if (deferred_anchor) {
        ++synthetic_deferred_feedback_success_total;
      }
    }
  }

  void MaybeScoreSyntheticPostCompletionLatency(LogicalOutput& logical,
                                                bool deferred_anchor) {
    if (logical.candidate.kind != CandidateKind::kSynthetic ||
        logical.synthetic_presentation_retired ||
        !logical.candidate.target_time_ns ||
        !logical.final_ready_soft_deadline_ns ||
        !logical.post_completion_evidence_observed) {
      return;
    }
    if (!logical.post_completion_physical_proven ||
        !logical.post_completion_proven_time_ns) {
      if (!logical.latency_judgement_skipped_no_proof_recorded) {
        logical.latency_judgement_skipped_no_proof_recorded = true;
        ++final_ready_judgement_skipped_no_proof_total;
      }
      return;
    }
    if (logical.latency_feedback_scored) {
      ++final_ready_judgement_duplicate_prevented_total;
      return;
    }
    ScoreSyntheticFinalReadyLatency(
        logical, logical.post_completion_proven_time_ns, deferred_anchor);
  }

  void RecordPostCompletionReady(LogicalOutput& logical,
                                 uint64_t completion_ns,
                                 PostCompletionEvidence evidence) {
    logical.post_completion_evidence_observed = true;
    const bool physically_proven =
        evidence == PostCompletionEvidence::kSyncFile ||
        evidence == PostCompletionEvidence::kTimeline ||
        evidence == PostCompletionEvidence::kTeardownIdle;
    if (physically_proven) {
      if (!logical.post_completion_physical_proven) {
        logical.post_completion_physical_proven = true;
        ++post_completion_proven_total;
      }
      // The first actual proof is the B1 observation time. An earlier
      // lifecycle-only fallback/error timestamp must never become the score.
      if (!logical.post_completion_proven_time_ns && completion_ns) {
        logical.post_completion_proven_time_ns = completion_ns;
      }
    } else if (evidence == PostCompletionEvidence::kUnprovenFallback) {
      if (!logical.post_completion_fallback_observed) {
        logical.post_completion_fallback_observed = true;
        ++completion_legacy_unproven_total;
      }
    } else if (!logical.post_completion_observation_error) {
      logical.post_completion_observation_error = true;
      ++post_completion_observation_error_total;
    }

    if (completion_ns) {
      // Preserve the existing lifecycle timestamp and its downstream
      // PhysicalPresentNotBeforeNs effect regardless of B1.
      logical.post_completion_ready_time_ns = std::max(
          logical.post_completion_ready_time_ns, completion_ns);
    }
    if (logical.post_completion_physical_proven &&
        logical.post_completion_proven_time_ns &&
        !logical.post_completion_sensor_recorded) {
      ObservePostCompletionEvidence(
          logical, logical.post_completion_proven_time_ns);
      logical.post_completion_sensor_recorded = true;
    }
    MaybeScoreSyntheticPostCompletionLatency(logical, false);
  }

  void ObserveLatencySuccess(const LogicalOutput& logical) {
    if (!stable_latency_metronome_armed || !logical.stable_pair ||
        logical.latency_epoch != latency_epoch ||
        logical.semantic_forward_skipped) {
      ++latency_feedback_excluded_total;
      return;
    }
    // Counted only. No streak exists to qualify a decay, so none can die on
    // a reset: F decays from the sliding tail instead.
    ++latency_success_total;
  }

  // ---------------------------------------------------------------------
  // Phase 5 Better D.  D_operating = D_causal + F.
  //
  //   D_causal  robust location (median) of qualified D_need samples of the
  //             current Source period epoch; until there are enough, the
  //             component estimate, causal_required alone since commit C.
  //             That prior is built from service estimates that exclude
  //             queue residence by design, so it cannot learn backlog.
  //   F         the only term that breathes. Continuous in ns - the lattice
  //             quantizes once, in the actuator. It rises only on a
  //             measured p95 target, only with a full, fresh ring; decay
  //             approaches that target, never by consecutive successes.
  //
  // What this replaces, and why #10 dies with it: the legacy decay waited
  // for 60 consecutive successes and then asked whether an aged Real's
  // issue plus the shallower depth still lay in the future. The Real aged
  // during its own qualification, the answer was no, and the streak reset -
  // every request refused while the plant was deep (§5.14). There is no
  // streak here and no aged Real: a setpoint change is only ever a setpoint
  // change, and whether it may move the lattice is the Phase 4 actuator's
  // question, already guarded by the live floor and the no-dilation gate.
  // ---------------------------------------------------------------------

  bool BetterDWindowReady() const {
    return better_d_window_epoch == period_epoch &&
           better_d_need_ns.count() >= kBetterDMinSamples;
  }

  bool BetterDWindowFull() const {
    return better_d_window_epoch == period_epoch &&
           better_d_need_ns.count() == kBetterDWindowCapacity;
  }

  uint64_t BetterDCausalNs() const {
    return BetterDWindowReady() ? better_d_need_ns.Quantile(50, 100)
                                : RequiredLatencyDepthNs();
  }

  // With no qualified tail yet, the location is all that is known, so the
  // decay may only take F down to the margin above it.
  // B4: a tail that is no longer being re-measured stops justifying F. In
  // the deep state every sample is censored, the window freezes holding the
  // tail that leaked in on the way down, and F sat at its ceiling for the
  // whole deep epoch in both titles (Rayman f4004-f5054, Arkham f1697-end) -
  // so the next reanchor would lay the lattice at that depth. The
  // architecture's "F has no sacred history" made concrete: once no sample
  // has qualified for kBetterDStaleIssues Source issues, the location is all
  // that is known and F decays toward the margin above it. Re-freshing the
  // age alone does not restore p95 authority: the frozen ring must be fully
  // renewed after each stale episode.
  bool BetterDTailFresh() const {
    return BetterDWindowReady() &&
           better_d_issues_since_qualified < kBetterDStaleIssues;
  }

  bool BetterDTailStale() const {
    return BetterDWindowReady() &&
           better_d_issues_since_qualified >= kBetterDStaleIssues;
  }

  // A fresh last sample does not make a mostly old ring fresh again. Start a
  // new renewal episode only on the transition into stale, then require one
  // full ring of qualified samples before restoring distributive authority.
  void UpdateBetterDTailStaleState() {
    const bool stale = BetterDTailStale();
    if (stale == better_d_tail_stale_active) {
      return;
    }
    better_d_tail_stale_active = stale;
    if (stale) {
      better_d_stale_renewal_pending = true;
      better_d_qualified_since_stale = 0;
      ++better_d_stale_episode_total;
    }
  }

  void RecordBetterDStaleRenewalSample() {
    if (!better_d_stale_renewal_pending) {
      return;
    }
    if (better_d_qualified_since_stale < kBetterDWindowCapacity) {
      ++better_d_qualified_since_stale;
    }
    if (better_d_qualified_since_stale == kBetterDWindowCapacity) {
      better_d_stale_renewal_pending = false;
      ++better_d_stale_renewal_recovered_total;
    }
  }

  uint64_t BetterDFTargetNs() const {
    // A partial p95 is too close to the largest individual samples. Neither
    // attack nor distributive decay may use it. Bootstrap/stale decay only
    // knows the margin; it never raises F to that margin by itself.
    if (!BetterDWindowFull() || !BetterDTailFresh() ||
        better_d_stale_renewal_pending) {
      return kBetterDMarginNs;
    }
    const uint64_t tail_ns = better_d_need_ns.Quantile(95, 100);
    const uint64_t causal_ns = BetterDCausalNs();
    return std::max(kBetterDMarginNs,
                    tail_ns > causal_ns ? tail_ns - causal_ns : 0);
  }

  // B3c. The architecture's D_causal <= D_operating <= D_max_contract, made
  // explicit. The value is still the implicit one - D_causal + F_max -
  // because the first run that could have chosen it was contaminated; this
  // names the contract so the next clean run can set it deliberately.
  uint64_t BetterDMaxContractNs() const {
    return SaturatingAddNs(BetterDCausalNs(), kBetterDFMaxNs);
  }

  uint64_t BetterDStepNs() const {
    return std::max<uint64_t>(stable_output_quantum_ns / kBetterDStepDivisor,
                              1);
  }

  // A Source period transition kills the distribution: samples of another P
  // are not evidence about this one, and F was insurance for the tail of a
  // regime that no longer exists. Resetting F here is also what stops D
  // being carried across 30 -> 60 at the old depth - the entry into the deep
  // state in both 2026-09-22 Rayman runs.
  void SyncBetterDEpoch() {
    if (better_d_window_epoch == period_epoch) {
      return;
    }
    better_d_window_epoch = period_epoch;
    better_d_need_ns = {};
    better_d_deficit_ns = {};
    better_d_f_ns = 0;
    // Seed arming from the new epoch's setpoint in ApplyBetterDSetpoint,
    // without comparing it against depth armed for the previous regime.
    better_d_armed_depth_ns = 0;
    better_d_issues_since_qualified = 0;
    better_d_tail_stale_active = false;
    better_d_stale_renewal_pending = false;
    better_d_qualified_since_stale = 0;
    ++better_d_epoch_reset_total;
  }

  // B3a. The sample is taken at Post completion, but a pair is placed only
  // when A receives its semantic commitment, so an S that finishes first has
  // pair_lattice_distance 0 - unplaced, not a gap. The first build censored
  // all of those as gaps: the fast samples went, the slow ones taught D, and
  // a deeper D delays A's commitment, so the bias fed itself. It is held on
  // the logical instead and scored where the anchor resolves it, the same
  // way H12 defers the latency feedback of an S that was FinalReady first.
  void ObserveBetterDSample(LogicalOutput& logical, uint64_t clean_ns,
                            uint64_t completion_ns) {
    if (logical.better_d_sample_scored || logical.better_d_sample_pending) {
      ++better_d_double_score_total;
      return;
    }
    if (!logical.candidate.pair_lattice_distance) {
      logical.better_d_sample_pending = true;
      logical.better_d_pending_clean_ns = clean_ns;
      logical.better_d_pending_completion_ns = completion_ns;
      ++better_d_deferred_total;
      return;
    }
    ScoreBetterDSample(logical, clean_ns, completion_ns, 0, false);
  }

  void ResolveDeferredBetterDSample(LogicalOutput& logical) {
    if (!logical.better_d_sample_pending) {
      return;
    }
    logical.better_d_sample_pending = false;
    ++better_d_deferred_resolved_total;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    const uint64_t anchor_lag_ns =
        now_ns > logical.better_d_pending_completion_ns
            ? now_ns - logical.better_d_pending_completion_ns
            : 0;
    better_d_anchor_lag_ns.Add(anchor_lag_ns);
    ScoreBetterDSample(logical, logical.better_d_pending_clean_ns,
                       logical.better_d_pending_completion_ns, anchor_lag_ns,
                       true);
  }

  // anchor_lag_ns and deferred are telemetry since B4: the anchor wait is
  // recorded and printed with every rise, and censors nothing.
  void ScoreBetterDSample(LogicalOutput& logical, uint64_t clean_ns,
                          uint64_t completion_ns, uint64_t anchor_lag_ns,
                          bool deferred) {
    if (logical.better_d_sample_scored) {
      ++better_d_double_score_total;
      return;
    }
    logical.better_d_sample_scored = true;
    // Count B7 incidence independently of reason precedence: kUnstable,
    // kGap, kSourceLate or kBacklog may censor the same marked S first.
    if (logical.funding_backlog_marked) {
      ++funding_backlog_censored_total;
    }
    // A -> B as the Source delivered it. On an S, candidate.issue_time_ns is
    // B's issue and pair_a_issue_time_ns is A's.
    const uint64_t a_issue_ns = logical.candidate.pair_a_issue_time_ns;
    const uint64_t b_issue_ns = logical.candidate.issue_time_ns;
    const uint64_t ab_ns =
        a_issue_ns && b_issue_ns > a_issue_ns ? b_issue_ns - a_issue_ns : 0;
    BetterDCensor reason = BetterDCensor::kCount;
    if (!stable_latency_metronome_armed || !stable_output_quantum_ns) {
      reason = BetterDCensor::kUnarmed;
    } else if (!logical.stable_pair) {
      reason = BetterDCensor::kUnstable;
    } else if (logical.source_period_epoch != period_epoch) {
      reason = BetterDCensor::kEpoch;
    } else if (logical.candidate.pair_lattice_distance != 1) {
      reason = BetterDCensor::kGap;
    } else if (logical.semantic_forward_skipped ||
               logical.synthetic_presentation_retired) {
      reason = BetterDCensor::kSkippedOrRetired;
    } else if (source_period_ns &&
               ab_ns > SaturatingAddNs(source_period_ns,
                                       source_period_ns / 2)) {
      reason = BetterDCensor::kSourceLate;
      better_d_source_late_ab_ns.Add(ab_ns);
    } else if (logical.in_production_during_saturation) {
      reason = BetterDCensor::kBacklog;
    } else if (logical.funding_backlog_marked) {
      reason = BetterDCensor::kFundingBacklog;
    }
    if (reason != BetterDCensor::kCount) {
      ++better_d_censored_total[size_t(reason)];
      return;
    }
    SyncBetterDEpoch();
    // D_need_i = C_pair_clean_i - S_offset + dispatch_lead. A consecutive
    // pair at 2x places S one output quantum after A.
    const uint64_t s_offset_ns = stable_output_quantum_ns;
    const uint64_t lead_ns =
        logical.dispatch_lead_ns ? logical.dispatch_lead_ns : dispatch_lead_ns;
    const uint64_t need_ns = SaturatingAddNs(
        clean_ns > s_offset_ns ? clean_ns - s_offset_ns : 0, lead_ns);
    better_d_need_ns.Add(need_ns);
    ++better_d_qualified_total;
    better_d_issues_since_qualified = 0;
    UpdateBetterDTailStaleState();
    RecordBetterDStaleRenewalSample();
    const uint64_t f_before_ns = better_d_f_ns;
    const uint64_t d_ns = SaturatingAddNs(BetterDCausalNs(), f_before_ns);
    const uint64_t f_target_ns = BetterDFTargetNs();
    const bool target_authoritative =
        BetterDWindowFull() && BetterDTailFresh() &&
        !better_d_stale_renewal_pending;
    const uint64_t f_after_ns = std::min(f_target_ns, kBetterDFMaxNs);
    // D_causal still qualifies at 32 samples, but F requires all 128. A
    // single qualified deficit is diagnostic, never an additive F request.
    if (need_ns > d_ns && !target_authoritative) {
      ++better_d_rise_unready_total;
    }
    if (target_authoritative && f_after_ns > f_before_ns) {
      const uint64_t deficit_ns = f_after_ns - f_before_ns;
      better_d_deficit_ns.Add(deficit_ns);
      ++better_d_rise_total;
      better_d_rise_ns_total = SaturatingAddNs(better_d_rise_ns_total,
                                               deficit_ns);
    }
    if (target_authoritative && f_target_ns > f_before_ns) {
      // Preserve the unclamped request for the existing ceiling-state and
      // excess telemetry. Apply clamps it; repeated ceiling requests are not
      // counted as new rises when F is already at the cap.
      better_d_f_ns = f_target_ns;
    }
    ApplyBetterDSetpoint();
  }

  // Once per Source issue, so the rate in ms/s is the same at every P.
  // Approach the p95 target without crossing it. Before full qualification
  // or when stale, retain margin-only decay; no partial quantile authority.
  void DecayBetterD() {
    if (better_d_window_epoch != period_epoch) {
      // ApplyBetterDSetpoint owns the epoch reset immediately after this.
      return;
    }
    const uint64_t target_ns = BetterDFTargetNs();
    if (better_d_f_ns > target_ns) {
      better_d_f_ns -= std::min(BetterDStepNs(), better_d_f_ns - target_ns);
      ++better_d_decay_step_total;
    }
  }

  void ApplyBetterDSetpoint() {
    if (!stable_latency_metronome_armed ||
        !stable_output_quantum_ns) {
      return;
    }
    SyncBetterDEpoch();
    // B3c. The ceiling is a controller STATE, entered and left, not a clamp
    // counted per call - the first build's F_ceiling=50 meant 50 calls, not 50
    // episodes. While in it the problem is no longer solvable by runway;
    // escalation (operating point, capacity, fail-open) is deliberately not
    // wired yet, because the first run's ceiling was reached on samples that
    // should never have qualified.
    if (better_d_f_ns > kBetterDFMaxNs) {
      better_d_f_ceiling_excess_max_ns =
          std::max(better_d_f_ceiling_excess_max_ns,
                   better_d_f_ns - kBetterDFMaxNs);
      better_d_f_ns = kBetterDFMaxNs;
      if (!better_d_f_at_ceiling) {
        better_d_f_at_ceiling = true;
        ++better_d_f_ceiling_enter_total;
      }
    } else if (better_d_f_at_ceiling && better_d_f_ns < kBetterDFMaxNs) {
      better_d_f_at_ceiling = false;
      ++better_d_f_ceiling_exit_total;
    }
    const uint64_t causal_ns = BetterDCausalNs();
    const uint64_t d_ns = SaturatingAddNs(causal_ns, better_d_f_ns);
    better_d_causal_last_ns = causal_ns;
    latency_required_last_ns = causal_ns;
    operating_latency_ns = d_ns;
    // Hand the actuator a change only once it can matter. The Phase 4
    // actuator does the rounding, refuses any decay that would dilate the
    // installed frontier, and classifies why; a rise that the installed
    // frontier already covers is refused there too.
    const uint64_t arm_ns =
        std::max<uint64_t>(stable_output_quantum_ns / kBetterDArmDivisor, 1);
    if (!better_d_armed_depth_ns) {
      better_d_armed_depth_ns = d_ns;
    } else if (SaturatingAddNs(d_ns, arm_ns) <= better_d_armed_depth_ns) {
      latency_depth_reanchor_pending = true;
      latency_depth_reanchor_armed_by_decay = true;
      latency_depth_reanchor_previous_operating_ns = better_d_armed_depth_ns;
      ++latency_depth_decay_armed_reanchor_total;
      ++better_d_arm_decay_total;
      better_d_armed_depth_ns = d_ns;
    } else if (d_ns >= SaturatingAddNs(better_d_armed_depth_ns, arm_ns)) {
      latency_depth_reanchor_pending = true;
      latency_depth_reanchor_armed_by_decay = false;
      ++better_d_arm_rise_total;
      better_d_armed_depth_ns = d_ns;
    }
  }

  // Censoring signal for Better D. Cheap: runs only when capacity is full.
  //
  // B4: every live S still in production, not only the ones waiting for
  // Generation. All four entries into the deep state in the 20260922-173054
  // run (Rayman f1604 and f4004, Arkham f647 and f1697) let through S whose
  // need p90 read 86-126 ms while the frontier sat at 4: they were already
  // past Generation when the pool filled, and finished inside the backlog.
  // An S already physically complete is left alone - its C_pair closed
  // before the saturation and saw none of it.
  void MarkSyntheticsWaitingForProductionCapacity() {
    for (LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped ||
          logical.candidate.kind != CandidateKind::kSynthetic ||
          logical.post_completion_ready_time_ns ||
          logical.better_d_sample_scored ||
          logical.better_d_sample_pending ||
          logical.in_production_during_saturation) {
        continue;
      }
      logical.in_production_during_saturation = true;
      ++better_d_backlog_marked_total;
    }
  }

  bool SourcePhaseWindowStable() const {
    return source_period_ns &&
           source_period_samples_ns.count() >= kSourcePhaseStableSamples &&
           SourcePeriodWindowClustered(source_period_samples_ns, 20) &&
           !source_transition_samples_ns.count() && IsStablePacingState();
  }

  static const char* SourceReanchorReasonName(SourceReanchorRequestReason reason) {
    switch (reason) {
      case SourceReanchorRequestReason::kPhaseDebt:
        return "phase_debt";
      case SourceReanchorRequestReason::kFrequencyConfirmation:
        return "frequency_confirmation";
      case SourceReanchorRequestReason::kConfirmedTransition:
        return "confirmed_transition";
      case SourceReanchorRequestReason::kOther:
        return "other";
      default:
        return "none";
    }
  }

  void RequestSourceReanchor(uint64_t source_period,
                             SourceReanchorRequestReason reason) {
    if (!source_period) {
      return;
    }
    auto& provenance = source_reanchor_provenance;
    if (source_phase_reanchor_pending) {
      ++provenance.overwritten_total;
      provenance.last_overwritten = provenance.pending;
    }
    provenance.pending = reason;
    ++provenance.requests[size_t(reason)];
    pending_source_reanchor_period_ns = source_period;
    source_phase_reanchor_pending = true;
    ++source_phase_reanchor_request_total;
  }

  void BeginSourcePhaseConfirmation(uint64_t issue_ns,
                                    int64_t period_error_ns,
                                    bool forward_skipped) {
    source_phase_confirmation_start_issue_ns = issue_ns;
    source_phase_confirmation_period_error_ns = period_error_ns;
    source_phase_confirmation_frequency_accumulated_ns =
        SignedMagnitudeNs(period_error_ns);
    source_phase_confirmation_samples = 1;
    source_phase_confirmation_consistent_samples = 1;
    source_phase_confirmation_forward_skip_current = forward_skipped ? 1 : 0;
  }

  void ClearSourcePhaseConfirmation() {
    source_phase_confirmation_start_issue_ns = 0;
    source_phase_confirmation_period_error_ns = 0;
    source_phase_confirmation_frequency_accumulated_ns = 0;
    source_phase_confirmation_samples = 0;
    source_phase_confirmation_consistent_samples = 0;
    source_phase_confirmation_forward_skip_current = 0;
  }

  // Shadow identity is independent of live Phase Debt resets/confirmation.
  // Return true when an interval straddling an observed epoch change must not
  // be attributed to the new reference. Totals survive; epoch evidence does not.
  bool ObserveResidualPhaseShadowEpoch(uint64_t now_ns) {
    auto& shadow = residual_phase_shadow;
    const bool changed =
        shadow.epoch_observed &&
        (shadow.period_epoch != period_epoch ||
         shadow.quantum_ns != stable_output_quantum_ns ||
         shadow.origin_ns != semantic_epoch_origin_ns);
    if (changed) {
      ++shadow.structural_reset_total;
      if (shadow.paused) {
        shadow.pause_duration_ns = SaturatingAddNs(
            shadow.pause_duration_ns, now_ns - shadow.pause_begin_ns);
      }
      shadow.residual_ns = 0;
      shadow.minimum_ns = 0;
      shadow.maximum_ns = 0;
      shadow.block_period_ns = 0;
      shadow.block_span_ns = 0;
      shadow.block_intervals = 0;
      shadow.fired = false;
      shadow.paused = false;
      shadow.pause_begin_ns = 0;
      shadow.previous_h_valid = false;
      shadow.previous_h_ns = 0;
    }
    if (shadow.epoch_observed ||
        (stable_latency_metronome_armed && stable_output_quantum_ns &&
         semantic_epoch_origin_ns)) {
      shadow.epoch_observed = true;
      shadow.period_epoch = period_epoch;
      shadow.quantum_ns = stable_output_quantum_ns;
      shadow.origin_ns = semantic_epoch_origin_ns;
    }
    return changed;
  }

  void PauseResidualPhaseShadow(uint64_t now_ns) {
    auto& shadow = residual_phase_shadow;
    if (!shadow.epoch_observed || shadow.paused) {
      return;
    }
    shadow.paused = true;
    shadow.pause_begin_ns = now_ns;
    ++shadow.pause_total;
    // No partial-span/count, residual or fire-latch mutation on qualification
    // flicker (including an inconsistent Issue snapshot or ring overrun).
  }

  void ObserveResidualPhaseShadow(uint64_t interval_ns, uint64_t now_ns) {
    auto& shadow = residual_phase_shadow;
    if (ObserveResidualPhaseShadowEpoch(now_ns) || !shadow.epoch_observed) {
      return;
    }
    // Same qualification as live Debt; invalid Issue intervals are excluded
    // just as in ObserveSourcePeriod. Unlike live Debt, PAUSE retains evidence.
    if (!stable_latency_metronome_armed || !stable_output_quantum_ns ||
        !source_period_ns || interval_ns < 5000000ull ||
        interval_ns > 100000000ull ||
        !source_effective_rate_timestamps_ns.EffectiveRateStable(
            kSourceEffectiveRateToleranceDivisor) ||
        source_transition_samples_ns.count()) {
      PauseResidualPhaseShadow(now_ns);
      ++shadow.paused_interval_total;
      return;
    }
    if (shadow.paused) {
      shadow.pause_duration_ns = SaturatingAddNs(
          shadow.pause_duration_ns, now_ns - shadow.pause_begin_ns);
      shadow.paused = false;
      shadow.pause_begin_ns = 0;
      ++shadow.resume_total;
    }
    if (!shadow.block_intervals) {
      shadow.block_period_ns =
          SaturatingMultiplyNs(stable_output_quantum_ns, 2);
    }
    // Sum only delivered, qualified Issue intervals: never bridge pause time
    // or fabricate missing intervals. A resumed partial block keeps its P_epoch.
    shadow.block_span_ns = SaturatingAddNs(shadow.block_span_ns, interval_ns);
    ++shadow.block_intervals;
    if (shadow.block_intervals < kPhaseDebtBlockIntervals) {
      return;
    }
    const uint64_t expected_span_ns =
        SaturatingMultiplyNs(shadow.block_period_ns, shadow.block_intervals);
    const int64_t block_error_ns =
        PresenterSignedDeltaNs(shadow.block_span_ns, expected_span_ns);
    shadow.residual_ns =
        SaturatingAddSignedNs(shadow.residual_ns, block_error_ns);
    shadow.minimum_ns = std::min(shadow.minimum_ns, shadow.residual_ns);
    shadow.maximum_ns = std::max(shadow.maximum_ns, shadow.residual_ns);
    ++shadow.blocks_total;
    shadow.positive_block_total += block_error_ns > 0;
    shadow.negative_block_total += block_error_ns < 0;

    const uint64_t threshold_ns =
        std::max<uint64_t>(stable_output_quantum_ns / 4, 1);
    const uint64_t magnitude_ns = SignedMagnitudeNs(shadow.residual_ns);
    // Observational hysteresis only: one fire until magnitude falls below
    // half the live Debt threshold (or the structural epoch changes).
    if (magnitude_ns < std::max<uint64_t>(threshold_ns / 2, 1)) {
      shadow.fired = false;
    }
    const bool would_fire = !shadow.fired && magnitude_ns >= threshold_ns;
    if (would_fire) {
      shadow.fired = true;
      if (shadow.residual_ns > 0) {
        ++shadow.would_fire_positive_total;
      } else {
        ++shadow.would_fire_negative_total;
      }
    }

    const uint64_t live_floor_ns = SaturatingAddNs(now_ns, dispatch_lead_ns);
    const int64_t h_live_ns =
        PresenterSignedDeltaNs(next_semantic_target_ns, live_floor_ns);
    shadow.previous_h_valid = true;
    shadow.previous_h_ns = h_live_ns;
    shadow.block_intervals = 0;
    shadow.block_span_ns = 0;
    shadow.block_period_ns = 0;
  }

  void ObservePhaseDebt(uint64_t interval_ns) {
    if (!stable_latency_metronome_armed || !stable_output_quantum_ns ||
        !source_period_ns ||
        !source_effective_rate_timestamps_ns.EffectiveRateStable(
            kSourceEffectiveRateToleranceDivisor) ||
        source_transition_samples_ns.count()) {
      if (phase_debt_block_intervals) {
        ResetPhaseDebt();
      }
      return;
    }

    if (phase_debt_block_intervals &&
        phase_debt_period_epoch != period_epoch) {
      ResetPhaseDebt();
    }
    if (!phase_debt_block_intervals) {
      phase_debt_block_period_ns =
          SaturatingMultiplyNs(stable_output_quantum_ns, 2);
      phase_debt_period_epoch = period_epoch;
    }
    phase_debt_block_span_ns = SaturatingAddNs(
        phase_debt_block_span_ns, interval_ns);
    ++phase_debt_block_intervals;
    if (phase_debt_block_intervals < kPhaseDebtBlockIntervals) {
      return;
    }

    const uint64_t expected_span_ns = SaturatingMultiplyNs(
        phase_debt_block_period_ns, kPhaseDebtBlockIntervals);
    const int64_t block_error_ns =
        PresenterSignedDeltaNs(phase_debt_block_span_ns, expected_span_ns);
    phase_debt_block_error_ns = block_error_ns;
    ++phase_debt_blocks_total;
    const int64_t debt_sum = SaturatingAddSignedNs(
        int64_t(std::min<uint64_t>(phase_debt_ns,
                                   uint64_t(std::numeric_limits<int64_t>::max()))),
        block_error_ns);
    phase_debt_ns = debt_sum > 0 ? uint64_t(debt_sum) : 0;
    phase_debt_block_span_ns = 0;
    phase_debt_block_period_ns = 0;
    phase_debt_period_epoch = 0;
    phase_debt_block_intervals = 0;

    const uint64_t threshold_ns =
        std::max<uint64_t>(stable_output_quantum_ns / 4, 1);
    if (phase_debt_ns >= threshold_ns && !source_phase_reanchor_pending) {
      ++phase_debt_fire_total;
      RequestSourceReanchor(source_period_ns,
                            SourceReanchorRequestReason::kPhaseDebt);
      ResetPhaseDebt();
    }
  }

  void ObserveSourcePhase(uint64_t source_id, uint64_t issue_ns,
                          uint64_t source_anchor_ns,
                          bool forward_skipped) {
    if (!issue_ns || !source_anchor_ns || !stable_output_quantum_ns ||
        !next_semantic_target_ns) {
      ClearSourcePhaseConfirmation();
      source_phase_previous_valid = false;
      return;
    }

    const uint64_t epoch_pair_period_ns =
        SaturatingMultiplyNs(stable_output_quantum_ns, 2);
    const int64_t period_error_ns =
        PresenterSignedDeltaNs(epoch_pair_period_ns, source_period_ns);
    const int64_t phase_error_ns =
        PresenterSignedDeltaNs(next_semantic_target_ns, source_anchor_ns);
    const int64_t previous_phase_error_ns = source_phase_previous_error_ns;
    const bool had_previous_phase = source_phase_previous_valid;
    const uint64_t elapsed_periods =
        had_previous_phase && source_id > source_phase_previous_source_id
            ? source_id - source_phase_previous_source_id : 1;
    source_phase_previous_source_id = source_id;
    const uint64_t drift_magnitude = std::min<uint64_t>(
        SaturatingMultiplyNs(SignedMagnitudeNs(period_error_ns),
                             elapsed_periods),
        uint64_t(std::numeric_limits<int64_t>::max()));
    const int64_t interval_drift_ns =
        period_error_ns < 0 ? -int64_t(drift_magnitude)
                            : int64_t(drift_magnitude);

    source_phase_epoch_pair_period_last_ns = epoch_pair_period_ns;
    source_phase_current_source_period_last_ns = source_period_ns;
    source_phase_period_error_last_ns = period_error_ns;
    source_phase_source_anchor_last_ns = source_anchor_ns;
    source_phase_next_semantic_target_last_ns = next_semantic_target_ns;
    source_phase_error_last_ns = phase_error_ns;
    ++source_phase_observation_total;
    if (phase_error_ns >= 0) {
      source_phase_lead_high_ns =
          std::max(source_phase_lead_high_ns, uint64_t(phase_error_ns));
    } else {
      source_phase_lag_high_ns =
          std::max(source_phase_lag_high_ns,
                   SignedMagnitudeNs(phase_error_ns));
    }

    if (had_previous_phase) {
      const int64_t observed_delta_ns = SaturatingSubtractSignedNs(
          phase_error_ns, previous_phase_error_ns);
      const int64_t explained_delta_ns = interval_drift_ns;
      source_phase_frequency_drift_total_ns = SaturatingAddSignedNs(
          source_phase_frequency_drift_total_ns, interval_drift_ns);
      source_phase_residual_total_ns = SaturatingAddSignedNs(
          source_phase_residual_total_ns,
          SaturatingSubtractSignedNs(observed_delta_ns,
                                     explained_delta_ns));
    }
    source_phase_previous_error_ns = phase_error_ns;
    source_phase_previous_valid = true;

    if (!SourcePhaseWindowStable()) {
      ClearSourcePhaseConfirmation();
      return;
    }
    ++source_phase_stable_source_sample_total;

    const uint64_t minimum_frequency_error_ns = std::max<uint64_t>(
        kSourcePhaseMinimumFrequencyErrorNs, source_period_ns / 1000);
    const uint64_t period_error_magnitude_ns =
        SignedMagnitudeNs(period_error_ns);
    if (!period_error_ns ||
        period_error_magnitude_ns < minimum_frequency_error_ns) {
      ClearSourcePhaseConfirmation();
      return;
    }

    // The already-qualified effective Source rate is the sole frequency
    // authority. ForwardSkip is an independent, legitimate phase-position
    // impulse: it must remain visible in decomposition telemetry, but must not
    // erase evidence that the epoch pair period and P_rate have different
    // frequencies.
    if (forward_skipped) {
      ++source_phase_confirmation_forward_skip_total;
    }
    const bool confirmation_direction_matches =
        source_phase_confirmation_period_error_ns &&
        ((source_phase_confirmation_period_error_ns > 0) ==
         (period_error_ns > 0));
    if (!source_phase_confirmation_start_issue_ns ||
        !confirmation_direction_matches) {
      if (source_phase_confirmation_start_issue_ns &&
          !confirmation_direction_matches) {
        ++source_phase_confirmation_direction_restart_total;
      }
      BeginSourcePhaseConfirmation(issue_ns, period_error_ns, forward_skipped);
      return;
    }

    source_phase_confirmation_samples = uint32_t(std::min<uint64_t>(
        uint64_t(source_phase_confirmation_samples) + elapsed_periods,
        UINT32_MAX));
    source_phase_confirmation_consistent_samples = uint32_t(
        std::min<uint64_t>(
            uint64_t(source_phase_confirmation_consistent_samples) +
                elapsed_periods, UINT32_MAX));
    source_phase_confirmation_period_error_ns = period_error_ns;
    source_phase_confirmation_frequency_accumulated_ns = SaturatingAddNs(
        source_phase_confirmation_frequency_accumulated_ns,
        SaturatingMultiplyNs(period_error_magnitude_ns, elapsed_periods));
    if (forward_skipped) {
      ++source_phase_confirmation_forward_skip_current;
    }

    const uint64_t elapsed_ns =
        issue_ns > source_phase_confirmation_start_issue_ns
            ? issue_ns - source_phase_confirmation_start_issue_ns
            : 0;
    const uint64_t half_quantum_ns =
        std::max<uint64_t>(stable_output_quantum_ns / 2, 1);
    const uint64_t periods_to_half_quantum =
        half_quantum_ns / period_error_magnitude_ns +
        (half_quantum_ns % period_error_magnitude_ns ? 1 : 0);
    const uint64_t confirmation_periods = std::clamp<uint64_t>(
        periods_to_half_quantum,
        kSourcePhaseMinimumConfirmPeriods,
        kSourcePhaseMaximumConfirmPeriods);
    const uint64_t confirmation_time_ns =
        SaturatingMultiplyNs(source_period_ns, confirmation_periods);
    const uint64_t accumulated_frequency_error_ns =
        source_phase_confirmation_frequency_accumulated_ns;
    const bool normal_confirmation =
        accumulated_frequency_error_ns >= half_quantum_ns &&
        elapsed_ns >= confirmation_time_ns;
    const bool urgent_confirmation =
        accumulated_frequency_error_ns >= stable_output_quantum_ns &&
        source_phase_confirmation_samples >=
            kSourcePhaseUrgentConfirmSamples;
    if (normal_confirmation || urgent_confirmation) {
      RequestSourceReanchor(source_period_ns,
                            SourceReanchorRequestReason::kFrequencyConfirmation);
      if (period_error_ns > 0) {
        ++source_phase_reanchor_faster_source_request_total;
      } else {
        ++source_phase_reanchor_slower_source_request_total;
      }
    }
  }

  bool ApplyPendingSourcePhaseReanchor(uint64_t issue_ns,
                                       uint64_t source_anchor_ns,
                                       uint64_t safe_future_anchor_ns,
                                       bool confirmed_slower_transition =
                                           false) {
    const uint64_t requested_source_period_ns =
        pending_source_reanchor_period_ns ? pending_source_reanchor_period_ns
                                          : source_period_ns;
    if (!source_phase_reanchor_pending ||
        (!confirmed_slower_transition && !SourcePhaseWindowStable()) ||
        !issue_ns || !source_anchor_ns || !safe_future_anchor_ns ||
        !stable_output_quantum_ns || !requested_source_period_ns) {
      return false;
    }

    const uint64_t old_output_quantum_ns = stable_output_quantum_ns;
    const uint64_t new_output_quantum_ns =
        std::max<uint64_t>(requested_source_period_ns / 2, 1);
    if (new_output_quantum_ns == old_output_quantum_ns) {
      ClearSourcePhaseConfirmation();
      source_phase_reanchor_pending = false;
      pending_source_reanchor_period_ns = 0;
      ++source_reanchor_provenance.equal_quantum_noop_total;
      source_reanchor_provenance.pending = SourceReanchorRequestReason::kNone;
      ResetPhaseDebt();
      return false;
    }

    struct LiveRealTimingSnapshot {
      bool live = false;
      uint64_t semantic_epoch_origin_ns = 0;
      uint64_t semantic_output_quantum_ns = 0;
      uint64_t semantic_tick_index = 0;
      uint64_t semantic_target_time_ns = 0;
      uint64_t target_time_ns = 0;
    };
    std::array<LiveRealTimingSnapshot, kLogicalOutputCapacity>
        live_real_timing = {};
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (logical.candidate.kind == CandidateKind::kReal &&
          logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped) {
        live_real_timing[i] = {
            true,
            logical.candidate.semantic_epoch_origin_ns,
            logical.candidate.semantic_output_quantum_ns,
            logical.candidate.semantic_tick_index,
            logical.candidate.semantic_target_time_ns,
            logical.candidate.target_time_ns,
        };
      }
    }

    const uint64_t ordered_boundary_ns =
        std::max({ExecutiveTailTargetNs(),
                  last_accounted_semantic_target_ns,
                  last_applied_semantic_target_ns});
    uint64_t first_target_ns = safe_future_anchor_ns;
    if (ordered_boundary_ns) {
      first_target_ns = std::max(
          first_target_ns,
          SaturatingAddNs(ordered_boundary_ns, new_output_quantum_ns));
    }
    first_target_ns = SaturatingAddNs(
        first_target_ns, PhaseReserveTargetNs(new_output_quantum_ns));
    if (!first_target_ns ||
        first_target_ns == std::numeric_limits<uint64_t>::max()) {
      return false;
    }

    const int64_t phase_before_ns = PresenterSignedDeltaNs(
        next_semantic_target_ns, source_anchor_ns);
    stable_output_quantum_ns = new_output_quantum_ns;
    semantic_epoch_origin_ns = first_target_ns;
    next_semantic_tick_index = 0;
    next_semantic_target_ns = first_target_ns;
    ++source_phase_reanchor_total;
    if (new_output_quantum_ns < old_output_quantum_ns) {
      ++source_phase_reanchor_faster_source_total;
    } else {
      ++source_phase_reanchor_slower_source_total;
    }
    source_phase_reanchor_old_output_quantum_ns = old_output_quantum_ns;
    source_phase_reanchor_new_output_quantum_ns = new_output_quantum_ns;
    source_phase_reanchor_phase_last_ns = phase_before_ns;

    const int64_t phase_after_ns = PresenterSignedDeltaNs(
        next_semantic_target_ns, source_anchor_ns);
    const int64_t reanchor_impulse_ns = SaturatingSubtractSignedNs(
        phase_after_ns, phase_before_ns);
    source_phase_reanchor_impulse_total_ns = SaturatingAddSignedNs(
        source_phase_reanchor_impulse_total_ns, reanchor_impulse_ns);
    source_phase_epoch_pair_period_last_ns =
        SaturatingMultiplyNs(new_output_quantum_ns, 2);
    source_phase_period_error_last_ns = PresenterSignedDeltaNs(
        source_phase_epoch_pair_period_last_ns, source_period_ns);
    source_phase_next_semantic_target_last_ns = next_semantic_target_ns;
    source_phase_error_last_ns = phase_after_ns;
    if (phase_after_ns >= 0) {
      source_phase_lead_high_ns =
          std::max(source_phase_lead_high_ns, uint64_t(phase_after_ns));
    } else {
      source_phase_lag_high_ns =
          std::max(source_phase_lag_high_ns,
                   SignedMagnitudeNs(phase_after_ns));
    }
    source_phase_previous_error_ns = phase_after_ns;
    source_phase_previous_valid = true;
    source_phase_reanchor_frequency_evidence_last_ns =
        source_phase_confirmation_frequency_accumulated_ns;
    source_phase_reanchor_forward_skip_confirmation_last =
        source_phase_confirmation_forward_skip_current;
    auto& provenance = source_reanchor_provenance;
    provenance.last_applied = provenance.pending;
    ++provenance.applied[size_t(provenance.pending)];
    provenance.pending = SourceReanchorRequestReason::kNone;
    source_phase_reanchor_pending = false;
    pending_source_reanchor_period_ns = 0;
    ResetPhaseDebt();
    ClearSourcePhaseConfirmation();

    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LiveRealTimingSnapshot& before = live_real_timing[i];
      const OutputCandidate& after = logical_outputs[i].candidate;
      if (before.live &&
          (after.semantic_epoch_origin_ns != before.semantic_epoch_origin_ns ||
           after.semantic_output_quantum_ns !=
               before.semantic_output_quantum_ns ||
           after.semantic_tick_index != before.semantic_tick_index ||
           after.semantic_target_time_ns != before.semantic_target_time_ns ||
           after.target_time_ns != before.target_time_ns)) {
        ++source_phase_reanchor_live_real_invalidation_total;
      }
    }
    return true;
  }

  bool EnsureStableLatencyMetronome(uint64_t source_id, uint64_t issue_ns,
                                    bool& forward_skipped_out) {
    forward_skipped_out = false;
    if (!source_period_ns || !issue_ns) {
      return false;
    }
    if (!stable_latency_metronome_armed) {
      stable_output_quantum_ns =
          std::max<uint64_t>(source_period_ns / 2, 1);
      latency_required_last_ns = BoundedRequiredLatencyDepthNs();
      operating_latency_ns =
          std::max(latency_required_last_ns, LatencyDepthBootstrapFloorNs());
      const uint64_t anchor_ns =
          SaturatingAddNs(issue_ns, operating_latency_ns);
      const uint64_t previous_semantic_boundary_ns =
          std::max({last_semantic_target_ns,
                    last_accounted_semantic_target_ns,
                    ExecutiveTailTargetNs()});
      semantic_epoch_origin_ns =
          previous_semantic_boundary_ns
              ? std::max(anchor_ns,
                         SaturatingAddNs(previous_semantic_boundary_ns,
                                         stable_output_quantum_ns))
              : anchor_ns;
      semantic_epoch_origin_ns = SaturatingAddNs(
          semantic_epoch_origin_ns,
          PhaseReserveTargetNs(stable_output_quantum_ns));
      next_semantic_tick_index = 0;
      next_semantic_target_ns = SemanticTargetForTick(0);
      stable_latency_metronome_armed = true;
      ++latency_metronome_arm_total;
      RefreshPhaseReserveTelemetry(PresenterMonotonicTimeNs());
      ObserveSourcePhase(source_id, issue_ns, anchor_ns, false);
      return true;
    }
    // Before the anchor is composed, so this issue already uses the new D.
    if (better_d_issues_since_qualified < kBetterDStaleIssues) {
      ++better_d_issues_since_qualified;
    }
    UpdateBetterDTailStaleState();
    DecayBetterD();
    ApplyBetterDSetpoint();
    const uint64_t issue_anchor_ns =
        SaturatingAddNs(issue_ns, operating_latency_ns);
    const uint64_t current_time_ns = PresenterMonotonicTimeNs();
    const uint64_t live_anchor_ns = SaturatingAddNs(
        current_time_ns, dispatch_lead_ns);
    // A pending Source-phase reanchor anchors at max(issue, live): when the
    // live anchor wins, promotion time (e.g. sidecar residence) entered it.
    const bool live_anchor_wins = live_anchor_ns > issue_anchor_ns;
    RefreshPhaseReserveTelemetry(current_time_ns);
    // Source lateness and D attacks are observed without moving the absolute
    // semantic cursor. No policy path may silently burn a whole S/R pair.
    forward_skipped_out = false;
    bool source_reanchor_applied = false;
    const uint64_t pre_reanchor_semantic_target_ns = next_semantic_target_ns;
    if (source_rate_transition_reanchor_pending) {
      RequestSourceReanchor(source_period_ns,
                            SourceReanchorRequestReason::kConfirmedTransition);
      if (stable_output_quantum_ns ==
          std::max<uint64_t>(source_period_ns / 2, 1)) {
        source_rate_transition_reanchor_pending = false;
        source_rate_transition_slower_pending = false;
      } else if (ApplyPendingSourcePhaseReanchor(
                     issue_ns, issue_anchor_ns,
                     std::max(issue_anchor_ns, live_anchor_ns),
                     source_rate_transition_slower_pending)) {
        source_rate_transition_reanchor_pending = false;
        source_rate_transition_slower_pending = false;
        source_reanchor_applied = true;
        ObserveSourceReanchorAnchor(live_anchor_wins);
      }
    }
    if (source_reanchor_applied) {
      // The source-rate reanchor used this cycle's current operating D and
      // already reforms the future epoch once for both changes.
      if (latency_depth_reanchor_pending) {
        ++latency_depth_reanchor_total;
        latency_depth_reanchor_future_anchor_ns = next_semantic_target_ns;
        latency_depth_reanchor_impulse_ns = PresenterSignedDeltaNs(
            next_semantic_target_ns, pre_reanchor_semantic_target_ns);
      }
      latency_depth_reanchor_pending = false;
      latency_depth_reanchor_armed_by_decay = false;
    } else {
      ApplyPendingLatencyDepthReanchor(source_id, issue_ns, issue_anchor_ns);
    }
    ObserveSourcePhase(source_id, issue_ns, issue_anchor_ns, forward_skipped_out);
    if (ApplyPendingSourcePhaseReanchor(
            issue_ns, issue_anchor_ns,
            std::max(issue_anchor_ns, live_anchor_ns))) {
      ObserveSourceReanchorAnchor(live_anchor_wins);
    }
    return true;
  }

  uint64_t DispatchLeadUpperBoundNs() const {
    // The dispatch lead covers only work under ZeroFG's control before the
    // transaction has
    // been handed to SurfaceFlinger. Keep it independent of R, VRR and latch
    // residence. At very high Source rates, never let it consume a full
    // semantic quantum.
    const uint64_t semantic_bound =
        stable_output_quantum_ns
            ? std::max<uint64_t>(kDispatchLeadFloorNs,
                                 stable_output_quantum_ns - 1)
            : kDispatchLeadCeilingNs;
    return std::clamp<uint64_t>(semantic_bound, kDispatchLeadFloorNs,
                                kDispatchLeadCeilingNs);
  }

  bool DispatchLeadEstimatorArmed() const {
    return dispatch_lead_controller_armed;
  }

  const char* DispatchLeadEstimatorSource() const {
    return DispatchLeadEstimatorArmed() ? "host_prepare_apply_p90"
                                        : "bounded_host_bootstrap";
  }

  uint64_t DispatchLeadEstimateNs() const {
    return std::clamp<uint64_t>(dispatch_lead_ns, kDispatchLeadFloorNs,
                                DispatchLeadUpperBoundNs());
  }

  void RefreshDispatchLeadEstimate() {
    if (transaction_prepare_ns.count() < kDispatchLeadArmSamples ||
        surface_apply_host_ns.count() < kDispatchLeadArmSamples) {
      dispatch_lead_ns = DispatchLeadEstimateNs();
      return;
    }
    const uint64_t evidence_ns = SaturatingAddNs(
        SaturatingAddNs(transaction_prepare_ns.Quantile(90, 100),
                        surface_apply_host_ns.Quantile(90, 100)),
        kDispatchLeadMarginNs);
    dispatch_lead_ns = std::clamp<uint64_t>(
        evidence_ns, kDispatchLeadFloorNs, DispatchLeadUpperBoundNs());
    dispatch_lead_controller_armed = true;
    // The rolling p90 windows are the stability mechanism. The causal host
    // estimate may rise or fall as old samples leave them. Dispatch lateness
    // remains telemetry and cannot feed this estimate back into itself.
  }

  bool FindRealSemanticCommitment(uint64_t source_id,
                                  OutputCandidate& commitment_out) const {
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kReal &&
          logical.candidate.source_id == source_id &&
          logical.candidate.target_time_ns &&
          logical.candidate.semantic_epoch_origin_ns &&
          logical.candidate.semantic_output_quantum_ns) {
        commitment_out = logical.candidate;
        return true;
      }
    }
    if (last_applied_real_commitment.source_id == source_id &&
        last_applied_real_commitment.target_time_ns &&
        last_applied_real_commitment.semantic_epoch_origin_ns &&
        last_applied_real_commitment.semantic_output_quantum_ns) {
      commitment_out = last_applied_real_commitment;
      return true;
    }
    return false;
  }

  // Accepted adjacency chooses the pair. d = B.source_id - A.source_id is the
  // number of guest IssueSwaps the pair spans: its identity and the most
  // lattice space it may own. It is a count, not elapsed Source time; how far
  // the pair actually reaches is bounded by Source phase in
  // PhaseBoundedPairDistance().
  static bool AcceptedPairSourceDistance(uint64_t a_source_id,
                                         uint64_t b_source_id,
                                         uint64_t& distance_out) {
    distance_out = 0;
    if (!a_source_id || b_source_id <= a_source_id) {
      return false;
    }
    distance_out = b_source_id - a_source_id;
    return true;
  }

  static uint64_t SyntheticPairSourceDistance(const OutputCandidate& candidate) {
    uint64_t distance = 0;
    return candidate.kind == CandidateKind::kSynthetic &&
                   AcceptedPairSourceDistance(candidate.pair_a_source_id,
                                              candidate.source_id, distance)
               ? distance
               : 0;
  }

  // The IssueSwap distance d is not elapsed Source time. When IssueSwaps were
  // bunched (a Source hitch followed by a burst whose intermediate
  // publications the latest-wins mailbox never delivered), d*P overstates the
  // Source time that passed between A and B. T_A already embeds A's latency,
  // so B's pair floor T_A + 2*d*O equals issue_B + latency exactly when d
  // periods really elapsed; any surplus would become B's latency and, through
  // T_B, the latency of every later pair, since the rolling chain does not
  // drain it. Source issue timing is the phase authority, so it bounds the
  // pair: B's boundary may not land more than one pair step beyond B's own
  // Source phase target (issue_B + D + phase reserve). The result lies in
  // [1, d]: d=1 is never changed, evenly spaced gaps keep d, and a burst only
  // shrinks toward the consecutive spacing. Issue timing never proposes a
  // spacing; it only refuses spacing that did not elapse.
  uint64_t PhaseBoundedPairDistance(const OutputCandidate& committed_a,
                                    uint64_t source_distance,
                                    uint64_t b_issue_ns) const {
    const uint64_t quantum_ns = committed_a.semantic_output_quantum_ns;
    if (source_distance <= 1 || !b_issue_ns || !quantum_ns ||
        !committed_a.target_time_ns || !operating_latency_ns) {
      return source_distance;
    }
    const uint64_t pair_step_ns = SaturatingMultiplyNs(quantum_ns, 2);
    const uint64_t phase_bound_ns = SaturatingAddNs(
        SaturatingAddNs(b_issue_ns, operating_latency_ns),
        SaturatingAddNs(PhaseReserveTargetNs(quantum_ns), pair_step_ns));
    if (phase_bound_ns <= committed_a.target_time_ns) {
      return 1;
    }
    return std::clamp<uint64_t>(
        (phase_bound_ns - committed_a.target_time_ns) / pair_step_ns, 1,
        source_distance);
  }

  // Decides a pair's lattice distance the first time it is placed and records
  // how much IssueSwap distance Source phase refused.
  uint64_t DecidePairLatticeDistance(const OutputCandidate& committed_a,
                                     uint64_t source_distance,
                                     uint64_t b_issue_ns) {
    const uint64_t lattice_distance =
        PhaseBoundedPairDistance(committed_a, source_distance, b_issue_ns);
    if (lattice_distance < source_distance) {
      ++accepted_gap_pair_phase_capped_total;
      accepted_gap_pair_phase_capped_tick_total = SaturatingAddNs(
          accepted_gap_pair_phase_capped_tick_total,
          SaturatingMultiplyNs(source_distance - lattice_distance, 2));
    }
    return lattice_distance;
  }

  // Witness: how far a pair-local B commitment sits beyond B's Source phase
  // target (issue_B + D + phase reserve). Consecutive pairs expose latency the
  // rolling chain already carries; a gap pair stays within one pair step
  // unless its d_lat was clamped to 1 behind an A that already carried it.
  void ObservePairBoundaryPhaseExcess(uint64_t source_distance,
                                      uint64_t boundary_ns, uint64_t b_issue_ns,
                                      uint64_t quantum_ns) {
    if (!source_distance || !boundary_ns || !b_issue_ns || !quantum_ns) {
      return;
    }
    const uint64_t phase_target_ns = SaturatingAddNs(
        SaturatingAddNs(b_issue_ns, operating_latency_ns),
        PhaseReserveTargetNs(quantum_ns));
    if (boundary_ns <= phase_target_ns) {
      return;
    }
    uint64_t& excess_max_ns = source_distance > 1
                                  ? accepted_gap_b_phase_excess_max_ns
                                  : consecutive_pair_b_phase_excess_max_ns;
    excess_max_ns = std::max(excess_max_ns, boundary_ns - phase_target_ns);
  }

  // For a pair at lattice distance d_lat (PhaseBoundedPairDistance): S at
  // A.tick + d_lat (T_A + d_lat*O) and B's factual floor at A.tick + 2*d_lat
  // (T_A + 2*d_lat*O). The d_lat-1 ticks on either side of S are semantic HOLD
  // positions accounted by AccountSemanticHoldsBefore(); they are never
  // replayed, reconstructed or compressed. d_lat=1 is the nominal consecutive
  // pair and produces exactly the historical values. d_lat is decided once, as
  // the pair's initial contract, and never extends, rescues or retargets S.
  bool ComputePairTimingFromCommittedReal(
      const OutputCandidate& committed_a, uint64_t lattice_distance,
      uint64_t& synthetic_semantic_out, uint64_t& real_earliest_eligible_out,
      uint64_t& synthetic_target_out, uint64_t& synthetic_tick_out,
      bool transition_real_only = false) const {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const uint64_t quantum_ns = committed_a.semantic_output_quantum_ns;
    if (!lattice_distance || lattice_distance > kMax / 2 ||
        !committed_a.target_time_ns || !committed_a.semantic_epoch_origin_ns ||
        !quantum_ns) {
      return false;
    }
    const uint64_t pair_ticks =
        lattice_distance * (transition_real_only ? 1 : 2);
    if (quantum_ns > kMax / pair_ticks ||
        committed_a.target_time_ns > kMax - quantum_ns * pair_ticks ||
        committed_a.semantic_tick_index > kMax - pair_ticks) {
      return false;
    }
    const uint64_t synthetic_offset_ns = quantum_ns * lattice_distance;
    if (transition_real_only) {
      // An explicit alternate pair contract, not a compressed A->S->B pair:
      // there is no S position, and B's factual floor is A + ld*O.
      synthetic_semantic_out = 0;
      synthetic_target_out = 0;
      synthetic_tick_out = 0;
      real_earliest_eligible_out =
          committed_a.target_time_ns + synthetic_offset_ns;
      return real_earliest_eligible_out > committed_a.target_time_ns &&
             real_earliest_eligible_out - committed_a.target_time_ns ==
                 synthetic_offset_ns;
    }
    synthetic_tick_out = committed_a.semantic_tick_index + lattice_distance;
    synthetic_target_out = committed_a.target_time_ns + synthetic_offset_ns;
    synthetic_semantic_out = synthetic_target_out;
    real_earliest_eligible_out = synthetic_target_out + synthetic_offset_ns;
    return synthetic_target_out > committed_a.target_time_ns &&
           synthetic_target_out - committed_a.target_time_ns ==
               synthetic_offset_ns &&
           real_earliest_eligible_out - synthetic_target_out ==
               synthetic_offset_ns;
  }

  bool ComputeAcceptedRealTiming(const Slot& current,
                         uint64_t pair_a_source_id,
                         uint64_t& synthetic_semantic_out,
                         uint64_t& real_earliest_eligible_out,
                         uint64_t& synthetic_target_out,
                         uint64_t& synthetic_tick_out,
                         uint64_t& synthetic_epoch_origin_out,
                         uint64_t& synthetic_quantum_out,
                         uint64_t& pair_lattice_distance_out,
                         bool& forward_skipped_out,
                         bool transition_real_only = false) {
    synthetic_semantic_out = 0;
    real_earliest_eligible_out = 0;
    synthetic_target_out = 0;
    synthetic_tick_out = 0;
    synthetic_epoch_origin_out = 0;
    synthetic_quantum_out = 0;
    pair_lattice_distance_out = 0;
    forward_skipped_out = false;
    const uint64_t issue = current.issue_time_ns.load(std::memory_order_relaxed);
    const uint64_t source_id =
        current.source_id.load(std::memory_order_relaxed);
    const bool metronome_ready =
        EnsureStableLatencyMetronome(source_id, issue, forward_skipped_out);
    ObservePlannedSpaceNewQuantumIssue(source_id, issue);
    if (!metronome_ready ||
        !stable_output_quantum_ns ||
        stable_output_quantum_ns >
            std::numeric_limits<uint64_t>::max() / 2 ||
        next_semantic_target_ns >
            std::numeric_limits<uint64_t>::max() -
                stable_output_quantum_ns * 2) {
      return false;
    }
    // Accepted Real timing remains available even when its preceding Real has
    // not yet received a shallow commitment. Synthetic timing, however, is
    // pair-local: only A's concrete commitment may assign S_AB and B's floor.
    real_earliest_eligible_out =
        SemanticTargetForTick(SaturatingAddNs(next_semantic_tick_index, 1));
    OutputCandidate committed_a;
    if (pair_a_source_id &&
        FindRealSemanticCommitment(pair_a_source_id, committed_a)) {
      // The pair is placed from A's commitment at its phase-bounded lattice
      // distance.
      uint64_t source_distance = 0;
      if (!AcceptedPairSourceDistance(
              pair_a_source_id,
              current.source_id.load(std::memory_order_relaxed),
              source_distance)) {
        return false;
      }
      const uint64_t lattice_distance =
          DecidePairLatticeDistance(committed_a, source_distance, issue);
      if (!ComputePairTimingFromCommittedReal(
              committed_a, lattice_distance, synthetic_semantic_out,
              real_earliest_eligible_out, synthetic_target_out,
              synthetic_tick_out, transition_real_only)) {
        return false;
      }
      pair_lattice_distance_out = lattice_distance;
      synthetic_epoch_origin_out = committed_a.semantic_epoch_origin_ns;
      synthetic_quantum_out = committed_a.semantic_output_quantum_ns;
    }
    return true;
  }

  void RecordSyntheticDrop(SyntheticDropReason reason) {
    ++synthetic_drop_total[size_t(reason)];
    if (!SyntheticChainWarmupActive()) {
      return;
    }
    if (reason == SyntheticDropReason::kSourceProtection) {
      ++synthetic_chain_warmup_source_protection_drop_total;
    } else if (reason == SyntheticDropReason::kResidencyPressure) {
      ++synthetic_chain_warmup_residency_pressure_drop_total;
    }
  }

  void RecordStableSyntheticDrop(LogicalOutput& logical) {
    if (logical.stable_pair && !logical.stable_drop_recorded) {
      logical.stable_drop_recorded = true;
      ++stable_synthetic_dropped_total;
    }
  }

  OutputCandidate MakeRealCandidate(
      uint32_t slot_index, uint64_t earliest_eligible_time_ns = 0) const {
    const Slot& slot = slots[slot_index];
    OutputCandidate candidate;
    candidate.kind = CandidateKind::kReal;
    candidate.image = slot.image;
    candidate.view = slot.view;
    candidate.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    candidate.storage_extent = slot.storage_content_extent;
    candidate.content_extent = slot.source_content_extent;
    candidate.display_aspect_ratio_x = slot.display_aspect_ratio_x;
    candidate.display_aspect_ratio_y = slot.display_aspect_ratio_y;
    candidate.is_8bpc = slot.is_8bpc;
    candidate.real_slot = slot_index;
    candidate.source_id = slot.source_id.load(std::memory_order_relaxed);
    candidate.issue_time_ns =
        slot.issue_time_ns.load(std::memory_order_relaxed);
    candidate.publish_time_ns =
        slot.publish_time_ns.load(std::memory_order_relaxed);
    candidate.ready_time_ns = slot.ready_time_ns;
    candidate.earliest_eligible_time_ns = earliest_eligible_time_ns;
    return candidate;
  }

  uint32_t CountFreeFinalOutputs() const {
    return kFinalOutputPoolSize - CountUsedFinalOutputs();
  }

  uint32_t AllocateLogicalOutput(const OutputCandidate& candidate,
                                 LogicalOutputState state,
                                 uint64_t post_submit_soft_deadline_ns = 0,
                                 uint64_t final_ready_soft_deadline_ns = 0) {
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& output = logical_outputs[i];
      if (output.state != LogicalOutputState::kFree) {
        continue;
      }
      output = {};
      output.state = state;
      output.token = next_logical_token++;
      output.sequence_id = next_logical_sequence++;
      output.post_submit_soft_deadline_ns = post_submit_soft_deadline_ns;
      output.final_ready_soft_deadline_ns = final_ready_soft_deadline_ns;
      output.dispatch_lead_ns = dispatch_lead_ns;
      output.source_period_epoch = period_epoch;
      output.latency_epoch = latency_epoch;
      output.operating_latency_ns = operating_latency_ns;
      output.candidate = candidate;
      logical_output_high_water =
          std::max(logical_output_high_water, CountLogicalOutputs());
      if (candidate.kind == CandidateKind::kSynthetic) {
        logical_synthetic_high_water = std::max(
            logical_synthetic_high_water,
            CountLogicalOutputs(CandidateKind::kSynthetic));
      }
      return i;
    }
    return UINT32_MAX;
  }

  void ConfigureLogicalStaging(LogicalOutput& logical,
                               bool includes_generation) {
    // Pending Real uses its absolute eligibility floor only to prepare
    // production. Presentation authority remains absent until epoch, nominal
    // target and target_time_ns are assigned at the ordered head.
    uint64_t target_ns = logical.candidate.target_time_ns;
    if (!target_ns) {
      target_ns = logical.candidate.semantic_target_time_ns;
    }
    if (!target_ns) {
      target_ns = logical.candidate.earliest_eligible_time_ns;
    }
    if (!target_ns) {
      return;
    }
    if (!logical.staging_cushion_ns) {
      logical.staging_cushion_ns = LatencyDepthCushionNs();
    }
    logical.final_ready_soft_deadline_ns =
        target_ns > logical.dispatch_lead_ns
            ? target_ns - logical.dispatch_lead_ns
            : target_ns;
    logical.post_submit_soft_deadline_ns =
        logical.final_ready_soft_deadline_ns >
                logical.post_residence_budget_ns
            ? logical.final_ready_soft_deadline_ns -
                  logical.post_residence_budget_ns
            : logical.final_ready_soft_deadline_ns;
    logical.post_submit_not_before_ns =
        logical.post_submit_soft_deadline_ns > logical.staging_cushion_ns
            ? logical.post_submit_soft_deadline_ns -
                  logical.staging_cushion_ns
            : 0;
    if (includes_generation) {
      logical.generation_submit_soft_deadline_ns =
          logical.post_submit_soft_deadline_ns >
                  logical.generation_residence_budget_ns
              ? logical.post_submit_soft_deadline_ns -
                    logical.generation_residence_budget_ns
              : logical.post_submit_soft_deadline_ns;
      logical.generation_submit_not_before_ns =
          logical.generation_submit_soft_deadline_ns >
                  logical.staging_cushion_ns
              ? logical.generation_submit_soft_deadline_ns -
                    logical.staging_cushion_ns
              : 0;
    }
  }

  uint32_t CountLogicalOutputs() const {
    uint32_t count = 0;
    for (const LogicalOutput& output : logical_outputs) {
      count += output.state != LogicalOutputState::kFree;
    }
    return count;
  }

  // H12: fully produced Synthetics parked at the presentation boundary because
  // their pair-local anchor has not arrived yet. A persistently nonzero value
  // means anchors are late, not that production is failing.
  uint32_t CountUnanchoredFinalReadySynthetics() const {
    uint32_t count = 0;
    for (const LogicalOutput& output : logical_outputs) {
      count += output.state == LogicalOutputState::kFinalReady &&
               output.candidate.kind == CandidateKind::kSynthetic &&
               !output.candidate.target_time_ns;
    }
    return count;
  }

  uint32_t CountLogicalOutputs(CandidateKind kind) const {
    uint32_t count = 0;
    for (const LogicalOutput& output : logical_outputs) {
      count += output.state != LogicalOutputState::kFree &&
               output.candidate.kind == kind;
    }
    return count;
  }

  uint32_t CountEarlierUnresolvedLogicalOutputs(uint64_t sequence_id) const {
    uint32_t count = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      count += logical.state != LogicalOutputState::kFree &&
               logical.state != LogicalOutputState::kDropped &&
               logical.sequence_id < sequence_id;
    }
    return count;
  }

  static bool LogicalNeedsFutureFinalOutput(const LogicalOutput& logical) {
    // H13: a presentation-retired ticket may still need a physical FinalOutput
    // to finish Post, but it holds no presentation obligation any more. It must
    // not reserve a slot ahead of B through RequiredFinalOutputSurplusFor()
    // merely because its old sequence is lower. It still acquires a slot
    // normally when one is actually free.
    if (logical.synthetic_presentation_retired) {
      return false;
    }
    // H12: kWaitingPairAnchor is unreachable; retained only for
    // telemetry/state-index stability. Not production authority.
    return logical.state == LogicalOutputState::kWaitingPairAnchor ||
           logical.state == LogicalOutputState::kWaitingGeneration ||
           logical.state == LogicalOutputState::kWaitingPost;
  }

  uint32_t CountEarlierUnfundedFinalOutputs(
      uint64_t sequence_id,
      std::optional<CandidateKind> kind = std::nullopt) const {
    uint32_t count = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      count += logical.sequence_id < sequence_id &&
               LogicalNeedsFutureFinalOutput(logical) &&
               (!kind || logical.candidate.kind == *kind);
    }
    return count;
  }

  uint32_t CountUnfundedFinalOutputs(
      std::optional<CandidateKind> kind = std::nullopt) const {
    uint32_t count = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      count += LogicalNeedsFutureFinalOutput(logical) &&
               (!kind || logical.candidate.kind == *kind);
    }
    return count;
  }

  uint32_t RequiredFinalOutputSurplusFor(
      const LogicalOutput& logical) const {
    // FinalOutput funding follows temporal order, not candidate kind. A later
    // accepted Real is a sovereign delivery obligation, but it is not an
    // obligation to occupy a physical FinalOutput before an earlier S can be
    // posted/applied. Reserve only for earlier unfunded logical outputs. Real
    // Residency liveness is handled explicitly by the bounded physical
    // recovery path.
    return CountEarlierUnfundedFinalOutputs(logical.sequence_id);
  }

  uint32_t CountFreeCandidateSlots() const {
    return kPoolSize - CountUsedSlots();
  }

  bool CandidateReserveWarning() const {
    return CountFreeCandidateSlots() < kCandidateReserveTarget;
  }

  bool FinalOutputHasLiveLogicalOwner(uint32_t final_output_index) const {
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFinalReady &&
          logical.final_output_index == final_output_index) {
        return true;
      }
    }
    return false;
  }

  uint32_t CountOrphanReadyFinalOutputs() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < kFinalOutputPoolSize; ++i) {
      count += final_outputs[i].state.load(std::memory_order_acquire) ==
                   FinalOutputState::kReady &&
               !FinalOutputHasLiveLogicalOwner(i);
    }
    return count;
  }

  bool RecoverOrphanReadyFinalOutputs() {
    bool progress = false;
    for (uint32_t i = 0; i < kFinalOutputPoolSize; ++i) {
      if (final_outputs[i].state.load(std::memory_order_acquire) !=
              FinalOutputState::kReady ||
          FinalOutputHasLiveLogicalOwner(i)) {
        continue;
      }
      // kReady proves post completion and no Surface transaction has consumed
      // the buffer. With no live logical owner, immediate recycle is safe and
      // prevents a diagnostic ownership defect from becoming pool deadlock.
      ReleaseFinalOutputSlot(i);
      orphan_ready_final_output_total.fetch_add(1,
                                                std::memory_order_relaxed);
      progress = true;
    }
    return progress;
  }

  uint32_t FindOldestWaitingPostRaw(
      std::optional<CandidateKind> kind = std::nullopt) const {
    uint32_t logical_index = UINT32_MAX;
    uint64_t oldest_sequence = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (logical.state == LogicalOutputState::kWaitingPost &&
          (!kind || logical.candidate.kind == *kind) &&
          logical.sequence_id < oldest_sequence) {
        logical_index = i;
        oldest_sequence = logical.sequence_id;
      }
    }
    return logical_index;
  }

  bool RealCandidatePostCanReleaseSlot(const OutputCandidate& candidate) const {
    if (candidate.kind != CandidateKind::kReal ||
        candidate.real_slot >= kPoolSize) {
      return false;
    }
    const Slot& slot = slots[candidate.real_slot];
    return slot.runtime_admitted && slot.owner_refs == 1 &&
           slot.state.load(std::memory_order_acquire) != SlotState::kFree;
  }

  bool RealLogicalPostCanReleaseSlot(const LogicalOutput& logical) const {
    return logical.state == LogicalOutputState::kWaitingPost &&
           RealCandidatePostCanReleaseSlot(logical.candidate);
  }

  bool CandidateSlotHasActiveGeneration(uint32_t slot_index) const {
    if (slot_index >= kPoolSize) {
      return false;
    }
    for (const GenerationJob& job : generation_jobs) {
      if (job.active && (job.previous_real_slot == slot_index ||
                         job.current_real_slot == slot_index)) {
        return true;
      }
    }
    return false;
  }

  bool RealLogicalPostCanAdvanceResidencyRecovery(
      const LogicalOutput& logical) const {
    return logical.state == LogicalOutputState::kWaitingPost &&
           logical.candidate.kind == CandidateKind::kReal &&
           (RealCandidatePostCanReleaseSlot(logical.candidate) ||
            CandidateSlotHasActiveGeneration(
                logical.candidate.real_slot));
  }

  bool CandidateSlotHasOutstandingReal(uint32_t slot_index) const {
    if (slot_index >= kPoolSize) {
      return false;
    }
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.candidate.kind == CandidateKind::kReal &&
          logical.candidate.real_slot == slot_index &&
          (logical.state == LogicalOutputState::kWaitingPost ||
           logical.state == LogicalOutputState::kPostSubmitted)) {
        return true;
      }
    }
    return false;
  }

  bool CandidateSlotProgressedByOneGeneration(uint32_t slot_index) const {
    if (slot_index >= kPoolSize) {
      return false;
    }
    const Slot& slot = slots[slot_index];
    if (!slot.runtime_admitted) {
      return false;
    }
    // One Synthetic reference may be the final physical owner, or it may be
    // the only owner preventing the sovereign Real post from becoming the
    // final release. In both cases Generation completion (or a pre-submit S
    // burn) makes concrete Residency progress without projected credit.
    return slot.owner_refs == 1 ||
           (slot.owner_refs == 2 &&
            CandidateSlotHasOutstandingReal(slot_index));
  }

  bool SyntheticGenerationEnablesCandidateRelease(
      const LogicalOutput& logical) const {
    return logical.state == LogicalOutputState::kWaitingGeneration &&
           logical.candidate.kind == CandidateKind::kSynthetic &&
           (CandidateSlotProgressedByOneGeneration(
                logical.previous_real_slot) ||
            CandidateSlotProgressedByOneGeneration(
                logical.current_real_slot));
  }

  bool ResidencyRecoveryGenerationCanAdvance(
      const LogicalOutput& logical) const {
    // Once S owns A/B, a diagnostic throughput classification must never keep
    // those Reals pinned during actual Source backpressure. The job either
    // submits and releases them on completion or is burned by its existing
    // feasibility/resource checks.
    return SyntheticGenerationEnablesCandidateRelease(logical);
  }

  bool ResidencyBackpressureRequiresProgress() const {
    // A transfer already in flight is progress, not congestion. Override JIT
    // only after PublicationCommitted made an immutable Capture eligible and
    // it is physically stranded because Residency has no free destination.
    return !CountFreeCandidateSlots() &&
           CountCapturesWaitingForResidency() != 0;
  }

  bool SyntheticRetentionHasPhysicalRunway() const {
    // Retaining A/B does not allocate another Residency slot. Refuse optional
    // ownership only after a committed Capture is physically stranded with no
    // destination. Merely observing a full Residency store while Capture
    // turnover is already in flight is not Source backpressure.
    return !ResidencyBackpressureRequiresProgress();
  }

  uint32_t CountUnrepresentedRealObligations() const {
    return incoming_real.occupied() ? 1u : 0u;
  }

  bool HasLogicalCapacityForSynthetic() const {
    const uint32_t logical_total = CountLogicalOutputs();
    const uint32_t logical_synthetic =
        CountLogicalOutputs(CandidateKind::kSynthetic);
    const uint32_t free = kLogicalOutputCapacity - logical_total;
    // Preserve capacity both for every accepted/reserved Real currently
    // lacking a logical entry and for the maximum bounded Real ownership
    // topology. A Synthetic can never consume a Real-reserved position.
    return logical_synthetic < LogicalSyntheticCapacity() &&
           free > CountUnrepresentedRealObligations();
  }

  LogicalOutput* FindLogicalBySequence(uint64_t sequence_id) {
    for (LogicalOutput& output : logical_outputs) {
      if (output.state != LogicalOutputState::kFree &&
          output.sequence_id == sequence_id) {
        return &output;
      }
    }
    return nullptr;
  }

  const LogicalOutput* FindLogicalBySequence(uint64_t sequence_id) const {
    for (const LogicalOutput& output : logical_outputs) {
      if (output.state != LogicalOutputState::kFree &&
          output.sequence_id == sequence_id) {
        return &output;
      }
    }
    return nullptr;
  }

  const LogicalOutput* FindRollingPairBoundaryReal(
      const LogicalOutput& synthetic) const {
    // B is identified by the pair identity the Synthetic already carries:
    // S.source_id is B's accepted source_id, strictly after pair_a. Numeric
    // adjacency is not required; the Source distance only places the pair.
    uint64_t source_distance = 0;
    if (synthetic.candidate.kind != CandidateKind::kSynthetic ||
        !AcceptedPairSourceDistance(synthetic.candidate.pair_a_source_id,
                                    synthetic.candidate.source_id,
                                    source_distance)) {
      return nullptr;
    }
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kReal &&
          logical.candidate.source_id == synthetic.candidate.source_id) {
        return &logical;
      }
    }
    return nullptr;
  }

  bool LogicalHasJob(uint32_t logical_index, uint64_t token) const {
    for (const GenerationJob& job : generation_jobs) {
      if (job.active && job.logical_index == logical_index &&
          job.logical_token == token) {
        return true;
      }
    }
    for (const PostJob& job : post_jobs) {
      if (job.active && job.logical_index == logical_index &&
          job.logical_token == token) {
        return true;
      }
    }
    return false;
  }

  void DropSyntheticLogical(uint32_t logical_index,
                            SyntheticDropReason reason) {
    if (logical_index >= kLogicalOutputCapacity) {
      return;
    }
    LogicalOutput& logical = logical_outputs[logical_index];
    if (logical.state != LogicalOutputState::kFree &&
        logical.state != LogicalOutputState::kDropped &&
        logical.candidate.kind == CandidateKind::kReal) {
      ++accepted_real_drop_violation_total;
      return;
    }
    if (logical.state == LogicalOutputState::kFree ||
        logical.state == LogicalOutputState::kDropped ||
        logical.candidate.kind != CandidateKind::kSynthetic) {
      return;
    }
    if (SplitAlwaysSArmed() &&
        reason != SyntheticDropReason::kLifecycleEpoch &&
        reason != SyntheticDropReason::kDiscontinuity &&
        reason != SyntheticDropReason::kStructuralOrphan) {
      ++split_forbidden_s_drop_total;
      XELOGE("ZeroFGSplitForbiddenSyntheticDrop source={} reason={} "
             "action=fail_open",
             logical.candidate.source_id, static_cast<int>(reason));
      Terminalize(TerminalReason::kOrderingFailure,
                  logical.candidate.source_id);
    }
    const LogicalOutputState state = logical.state;
    if (state == LogicalOutputState::kWaitingPost &&
        logical.candidate.synthetic_index < kSyntheticPoolSize &&
        !LogicalHasJob(logical_index, logical.token)) {
      ReleaseSyntheticBackingOwnership(logical.candidate.synthetic_index);
      synthetic_release_callback(logical.candidate.synthetic_index);
      synthetic_pool_occupancy = CountGenerationJobs();
    }
    // H12: kWaitingPairAnchor is unreachable; retained only for
    // telemetry/state-index stability. Not production authority.
    if ((state == LogicalOutputState::kWaitingPairAnchor ||
         state == LogicalOutputState::kWaitingGeneration) &&
        !LogicalHasJob(logical_index, logical.token)) {
      ReleaseRealSlot(logical.previous_real_slot);
      ReleaseRealSlot(logical.current_real_slot);
    }
    if (state == LogicalOutputState::kFinalReady &&
        logical.final_output_index < kFinalOutputPoolSize) {
      FinalOutputSlot& output = final_outputs[logical.final_output_index];
      if (output.state.load(std::memory_order_acquire) ==
          FinalOutputState::kReady) {
        ReleaseFinalOutputSlot(logical.final_output_index);
        logical.final_output_index = UINT32_MAX;
      } else {
        final_output_owner_mismatch_total.fetch_add(
            1, std::memory_order_relaxed);
      }
    }
    // PostSubmitted retains its FinalOutput until PollPostJobs proves GPU
    // completion. That path observes kDropped and performs the safe release.
    RecordStableSyntheticDrop(logical);
    logical.state = LogicalOutputState::kDropped;
    ObservePlannedSpaceFrontier(CountSyntheticProductionFrontier(),
                                logical.candidate.source_id,
                                logical.sequence_id, "s_dropped");
    RecordSyntheticDrop(reason);
  }

  // H13/H14 — PRESENTATION HOLD MUST NOT CANCEL ADMITTED SYNTHESIS.
  //
  // H14.1 permits this normal HOLD only when the sovereign B's pair-local
  // eligibility floor is factually due and B is FinalReady while S is
  // unfinished. B is still targetless here. This commits the semantic HOLD
  // exactly once, releases the ordered position so B may receive its shallow
  // head commitment, and leaves the production ticket fully alive: no state
  // change, no token invalidation, no job cancellation and no resource release
  // while Generation/Post still own them.
  //
  // The ordered position MUST be resolved here rather than by breaking out of
  // the pump: a still-producing S that stayed the logical head would prevent B
  // from ever becoming head and deadlock the executive.
  //
  // H13: head-only authority. The caller must have resolved this logical as
  // FindLogicalBySequence(next_apply_sequence) before retiring its
  // presentation position, because this advances next_apply_sequence. Both
  // current call sites are inside that pump loop.
  //
  // Historical presentation telemetry (hard_target_miss_synthetic,
  // safe_window_expired, nominal_late) is deliberately still recorded for
  // continuity, but has no authority over this decision. What is deliberately
  // NOT recorded is
  // ObserveLatencyFeasibilityMiss(): a HOLD is the intended outcome here, so
  // it must never teach D/latency or be read as a Production failure.
  void RetireSyntheticPresentation(uint32_t logical_index, uint64_t now_ns) {
    if (logical_index >= kLogicalOutputCapacity) {
      return;
    }
    LogicalOutput& logical = logical_outputs[logical_index];
    if (logical.candidate.kind != CandidateKind::kSynthetic ||
        logical.synthetic_presentation_retired ||
        logical.state == LogicalOutputState::kFree ||
        logical.state == LogicalOutputState::kDropped) {
      return;
    }
    const bool has_job = LogicalHasJob(logical_index, logical.token);
    RecordSyntheticSafeWindowExpired(logical, now_ns);
    RecordHardTargetMiss(logical, now_ns);
    switch (logical.state) {
      case LogicalOutputState::kWaitingGeneration:
        if (has_job) {
          ++presentation_expired_generation_inflight_total;
        } else {
          ++presentation_expired_before_generation_submit_total;
          // Historical H12 name for the same event. After H13 it no longer
          // means "Production failed": Generation is still submitted after
          // this point, which generation_submitted_after_presentation_hold
          // proves.
          ++anchored_s_reached_b_safe_without_generation_job_total;
        }
        break;
      case LogicalOutputState::kWaitingPost:
        ++presentation_expired_waiting_post_total;
        ++synthetic_post_submit_late_total;
        break;
      case LogicalOutputState::kPostSubmitted:
        ++presentation_expired_post_submitted_total;
        ++synthetic_post_ready_late_total;
        break;
      case LogicalOutputState::kFinalReady:
        ++presentation_expired_final_ready_total;
        break;
      default:
        break;
    }

    // Commit the semantic HOLD exactly once, identically to the historical
    // dropped-S path, then release the ordered presentation position.
    if (logical.candidate.target_time_ns &&
        logical.candidate.target_time_ns > last_accounted_semantic_target_ns) {
      AccountSemanticHoldsBefore(logical.candidate,
                                 logical.candidate.semantic_tick_index);
      ++semantic_hold_total;
      last_accounted_semantic_target_ns = logical.candidate.target_time_ns;
      RecordOrderedBoundaryProvenance(
          OrderedBoundaryKind::kExplicitSHold, logical.candidate.source_id,
          logical.sequence_id, logical.candidate.semantic_tick_index,
          logical.candidate.target_time_ns);
      AdvanceSemanticCursorToTick(
          logical.candidate,
          SaturatingAddNs(logical.candidate.semantic_tick_index, 1));
    }
    logical.synthetic_presentation_retired = true;
    ++synthetic_presentation_hold_total;
    if (SyntheticPairSourceDistance(logical.candidate) > 1) {
      ++accepted_gap_synthetic_hold_total;
    }
    ++next_apply_sequence;

    if (logical.state == LogicalOutputState::kFinalReady) {
      // Production already finished, so there is nothing left to produce.
      FinalizeRetiredSyntheticProduction(logical_index);
    }
  }

  // H13: production of a presentation-retired S has completed. It must never be
  // applied. Retire it through the existing bounded ownership path;
  // CleanupDroppedLogicalOutputs() frees the slot once no job/GPU owner
  // remains, and PollPostJobs()/ReleaseFinalOutputSlot() keep honouring
  // fence/timeline ownership. No synchronous wait is introduced.
  void FinalizeRetiredSyntheticProduction(uint32_t logical_index) {
    if (logical_index >= kLogicalOutputCapacity) {
      return;
    }
    LogicalOutput& logical = logical_outputs[logical_index];
    if (!logical.synthetic_presentation_retired ||
        logical.state != LogicalOutputState::kFinalReady) {
      return;
    }
    ++production_retired_after_presentation_hold_total;
    if (!logical.synthetic_generation_attempted) {
      ++admitted_s_retired_without_generation_attempt_total;
    }
    DropSyntheticLogical(logical_index,
                         SyntheticDropReason::kPresentationRetired);
  }

  uint64_t SyntheticPairSafeDeadlineFromReal(uint64_t real_target_ns,
                                             uint64_t dispatch_lead_ns) const {
    if (!real_target_ns) {
      return 0;
    }
    return real_target_ns > dispatch_lead_ns
               ? real_target_ns - dispatch_lead_ns
               : real_target_ns;
  }

  void UpdateSyntheticPairSafeDeadlineForReal(uint64_t real_source_id,
                                              uint64_t real_target_ns,
                                              uint64_t dispatch_lead_ns) {
    const uint64_t safe_deadline_ns =
        SyntheticPairSafeDeadlineFromReal(real_target_ns, dispatch_lead_ns);
    if (!safe_deadline_ns) {
      return;
    }
    for (LogicalOutput& logical : logical_outputs) {
      // H14: retain the historical B-safe observation only for an S that is
      // already anchored. It is not presentation authority and can never
      // cause HOLD/drop. AnchorSyntheticPairsToCommittedReal() remains the
      // sole authority that establishes S target/tick/epoch; no target means
      // there is no meaningful historical window to observe.
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kSynthetic &&
          logical.candidate.target_time_ns &&
          logical.candidate.source_id == real_source_id) {
        logical.synthetic_pair_safe_deadline_ns = safe_deadline_ns;
      }
    }
  }

  uint32_t CountLiveLateSyntheticSalvage() const {
    uint32_t count = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kSynthetic &&
          logical.synthetic_late_salvage_started) {
        ++count;
      }
    }
    return count;
  }

  void ObserveSyntheticNominalLateness(LogicalOutput& logical,
                                       uint64_t now_ns) {
    if (logical.candidate.kind != CandidateKind::kSynthetic ||
        !logical.candidate.target_time_ns ||
        now_ns < logical.candidate.target_time_ns) {
      return;
    }
    if (!logical.synthetic_nominal_late_recorded) {
      logical.synthetic_nominal_late_recorded = true;
      ++synthetic_nominal_late_total;
      synthetic_nominal_lateness_ns.Add(
          now_ns - logical.candidate.target_time_ns);
    }
    if (!logical.synthetic_late_salvage_started &&
        logical.synthetic_pair_safe_deadline_ns &&
        now_ns < logical.synthetic_pair_safe_deadline_ns) {
      logical.synthetic_late_salvage_started = true;
      ++late_salvage_started_total;
    }
  }

  void RecordSyntheticSafeWindowExpired(LogicalOutput& logical,
                                        uint64_t now_ns) {
    if (logical.candidate.kind == CandidateKind::kSynthetic &&
        logical.synthetic_pair_safe_deadline_ns &&
        now_ns >= logical.synthetic_pair_safe_deadline_ns &&
        !logical.synthetic_safe_window_expired_recorded) {
      logical.synthetic_safe_window_expired_recorded = true;
      ++late_salvage_safe_window_expired_total;
    }
  }

  void RecordHardTargetMiss(const LogicalOutput& logical, uint64_t now_ns) {
    if (logical.candidate.kind == CandidateKind::kSynthetic) {
      ++hard_target_miss_synthetic_total;
      if (!logical.candidate.target_time_ns) {
        // H12 falsifier: an unanchored S has no semantic contract and
        // therefore no semantic hard cutoff. This must remain zero; a nonzero
        // value proves a provisional deadline was reintroduced somewhere.
        ++synthetic_unanchored_semantic_hard_drop_total;
      }
      if (SyntheticChainWarmupActive() ||
          logical.synthetic_chain_warmup_submission) {
        ++synthetic_chain_warmup_hard_drop_total;
      }
      return;
    }

    ++hard_target_miss_real_total;
    if (SyntheticChainWarmupActive()) {
      ++synthetic_chain_warmup_real_hard_miss_total;
    }
    hard_real_miss_by_state[size_t(logical.state)].fetch_add(
        1, std::memory_order_relaxed);
    if (logical.candidate.target_time_ns &&
        now_ns >= logical.candidate.target_time_ns) {
      const uint64_t target_lateness_ns =
          now_ns - logical.candidate.target_time_ns;
      hard_real_target_lateness_ns.Add(target_lateness_ns);
      hard_real_target_lateness_last_ns.store(target_lateness_ns,
                                              std::memory_order_relaxed);
      uint64_t previous_max_ns =
          hard_real_target_lateness_max_ns.load(std::memory_order_relaxed);
      while (previous_max_ns < target_lateness_ns &&
             !hard_real_target_lateness_max_ns.compare_exchange_weak(
                 previous_max_ns, target_lateness_ns,
                 std::memory_order_relaxed, std::memory_order_relaxed)) {
      }
    }

    if (logical.state == LogicalOutputState::kWaitingPost) {
      if (logical.post_submit_not_before_ns &&
          now_ns >= logical.post_submit_not_before_ns) {
        const uint64_t lateness_ns =
            now_ns - logical.post_submit_not_before_ns;
        hard_real_waiting_post_not_before_lateness_ns.Add(lateness_ns);
        hard_real_waiting_post_not_before_lateness_last_ns.store(
            lateness_ns, std::memory_order_relaxed);
      }
      if (logical.post_submit_soft_deadline_ns &&
          now_ns >= logical.post_submit_soft_deadline_ns) {
        const uint64_t lateness_ns =
            now_ns - logical.post_submit_soft_deadline_ns;
        hard_real_waiting_post_soft_deadline_lateness_ns.Add(lateness_ns);
        hard_real_waiting_post_soft_deadline_lateness_last_ns.store(
            lateness_ns, std::memory_order_relaxed);
      }
      const uint32_t free_final_outputs = CountFreeFinalOutputs();
      const uint32_t required_surplus =
          RequiredFinalOutputSurplusFor(logical);
      const uint32_t earlier_unfunded =
          CountEarlierUnfundedFinalOutputs(logical.sequence_id);
      hard_real_waiting_post_free_final_outputs.Add(free_final_outputs);
      hard_real_waiting_post_required_surplus.Add(required_surplus);
      hard_real_waiting_post_earlier_unfunded.Add(earlier_unfunded);
      hard_real_waiting_post_free_final_outputs_last.store(
          free_final_outputs, std::memory_order_relaxed);
      hard_real_waiting_post_required_surplus_last.store(
          required_surplus, std::memory_order_relaxed);
      hard_real_waiting_post_earlier_unfunded_last.store(
          earlier_unfunded, std::memory_order_relaxed);
    } else if (logical.state == LogicalOutputState::kFinalReady) {
      if (logical.final_ready_time_ns && now_ns >= logical.final_ready_time_ns) {
        const uint64_t age_ns = now_ns - logical.final_ready_time_ns;
        hard_real_final_ready_age_ns.Add(age_ns);
        hard_real_final_ready_age_last_ns.store(age_ns,
                                               std::memory_order_relaxed);
      }
      const int64_t target_slack_ns =
          logical.final_ready_time_ns && logical.candidate.target_time_ns
              ? PresenterSignedDeltaNs(logical.candidate.target_time_ns,
                                logical.final_ready_time_ns)
              : 0;
      hard_real_final_ready_target_slack_last_ns.store(
          target_slack_ns, std::memory_order_relaxed);
      if (target_slack_ns >= 0) {
        hard_real_final_ready_runway_ns.Add(uint64_t(target_slack_ns));
      } else {
        hard_real_final_ready_late_at_ready_ns.Add(
            uint64_t(-target_slack_ns));
      }
    }
  }

  uint64_t CountLateReadyRealsBehindHead(uint64_t now_ns) const {
    uint64_t count = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped ||
          logical.candidate.kind != CandidateKind::kReal ||
          logical.state != LogicalOutputState::kFinalReady ||
          !logical.candidate.target_time_ns ||
          now_ns < logical.candidate.target_time_ns ||
          logical.sequence_id <= next_apply_sequence) {
        continue;
      }
      ++count;
    }
    return count;
  }

  uint32_t CountGenerationJobs() const {
    uint32_t count = 0;
    for (const GenerationJob& job : generation_jobs) {
      count += job.active;
    }
    return count;
  }

  uint32_t CountLiveSyntheticChains() const {
    // Blocker #22. The comment below held only for the chained path. When a
    // Generation completes on a poll before it is chained -- which
    // ScheduleResidencyRecoveryForSourceIngress() forces -- the completion
    // clears job.active and moves the logical to kWaitingPost WITHOUT calling
    // synthetic_release_callback, because the Post still needs the image. The
    // adapter's output_busy is cleared only by that callback, so the job count
    // can fall while physical occupancy stays at 3 of 3.
    //
    // Real output ownership is the honest authority: it is the mirror of the
    // release callback, which is exactly what the adapter keys on.
    return CountSyntheticBackingOwned();
  }

  // Physical Synthetic-chain capacity: the pool. A chain stays live from its
  // Generation submit through its Post's retirement.
  uint32_t SyntheticChainCapacity() const { return kSyntheticPoolSize; }

  bool SyntheticChainSubmissionCapacityAvailable() const {
    // Estimator state is observation-only. Physical Synthetic ownership is
    // bounded by the actual pool from the first qualified opportunity.
    const uint32_t capacity = SyntheticChainCapacity();
    const bool available = CountLiveSyntheticChains() < capacity;
    if (!available && CountGenerationJobs() < capacity) {
      // Generation ownership alone would have admitted this submit, the adapter
      // would have refused it with kSyntheticPool, and the host-driver
      // quantum would have been spent on an attempt that could not succeed
      // while the actionable Post waited. Gate B1's own counter.
      ++synthetic_backing_refusal_avoided_total;
    }
    return available;
  }

  bool GenerationNeedsCpuObservation(const GenerationJob& job) const {
    if (!job.active || job.chained_to_post) {
      return false;
    }
    if (job.logical_index >= kLogicalOutputCapacity) {
      return true;
    }
    const LogicalOutput& logical = logical_outputs[job.logical_index];
    // A live WaitingPost Synthetic is intentionally waiting for GPU-side
    // chaining, not for CPU readiness. Only orphaned or dropped Generation
    // needs the legacy timeline observation for safe retirement.
    return logical.token != job.logical_token ||
           logical.state == LogicalOutputState::kDropped;
  }

  uint32_t CountPostJobs() const {
    uint32_t count = 0;
    for (const PostJob& job : post_jobs) {
      count += job.active;
    }
    return count;
  }

  LatencyMissCause ClassifyUnsubmittedLatencyMiss(
      const LogicalOutput& logical) const {
    if (CandidateReserveWarning()) {
      return LatencyMissCause::kResidencyPressure;
    }
    if (logical.state == LogicalOutputState::kWaitingPost) {
      const uint32_t free_final_outputs = CountFreeFinalOutputs();
      if (!free_final_outputs) {
        return LatencyMissCause::kFinalOutputPressure;
      }
      if (free_final_outputs <= RequiredFinalOutputSurplusFor(logical)) {
        return LatencyMissCause::kFunding;
      }
    }
    if (CountEarlierUnresolvedLogicalOutputs(logical.sequence_id)) {
      return LatencyMissCause::kOrdering;
    }
    return LatencyMissCause::kScheduler;
  }

  void RefreshPipelineSnapshot() {
    const uint32_t live_late_salvage = CountLiveLateSyntheticSalvage();
    late_salvage_high_water =
        std::max(late_salvage_high_water, live_late_salvage);
    pipeline_snapshot_logical_total.store(CountLogicalOutputs(),
                                          std::memory_order_relaxed);
    pipeline_snapshot_logical_real.store(
        CountLogicalOutputs(CandidateKind::kReal),
        std::memory_order_relaxed);
    pipeline_snapshot_logical_synthetic.store(
        CountLogicalOutputs(CandidateKind::kSynthetic),
        std::memory_order_relaxed);
    pipeline_snapshot_logical_high_water.store(logical_output_high_water,
                                               std::memory_order_relaxed);
    pipeline_snapshot_generation_jobs.store(CountGenerationJobs(),
                                            std::memory_order_relaxed);
    pipeline_snapshot_post_jobs.store(CountPostJobs(),
                                      std::memory_order_relaxed);
    // Phase 0 sensors 2 and 9. Read-only observation on the existing refresh.
    ObserveSyntheticBackingAuthority();
    ObservePoolOccupancy();
    const uint64_t snapshot_now_ns = PresenterMonotonicTimeNs();
    std::array<std::array<uint32_t, size_t(LogicalOutputState::kCount)>, 2>
        logical_states = {};
    uint32_t max_earlier_unfunded = 0;
    uint32_t live_future_real_commitments = 0;
    uint32_t pending_uncommitted_real = 0;
    uint32_t pending_real_with_nominal_commitment = 0;
    uint32_t committed_real = 0;
    const LogicalOutput* first_live_real = nullptr;
    const LogicalOutput* last_live_real = nullptr;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree) {
        continue;
      }
      ++logical_states[size_t(logical.candidate.kind)]
                      [size_t(logical.state)];
      if (logical.state == LogicalOutputState::kWaitingPost) {
        max_earlier_unfunded = std::max(
            max_earlier_unfunded,
            CountEarlierUnfundedFinalOutputs(logical.sequence_id));
      }
      if (logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kReal) {
        if (!logical.candidate.target_time_ns) {
          ++pending_uncommitted_real;
          if (logical.candidate.semantic_target_time_ns ||
              logical.candidate.semantic_epoch_origin_ns ||
              logical.candidate.semantic_output_quantum_ns ||
              logical.candidate.semantic_tick_index) {
            ++pending_real_with_nominal_commitment;
          }
        } else {
          ++committed_real;
          if (logical.candidate.target_time_ns > snapshot_now_ns) {
            ++live_future_real_commitments;
          }
          if (!first_live_real ||
              logical.sequence_id < first_live_real->sequence_id) {
            first_live_real = &logical;
          }
          if (!last_live_real ||
              logical.sequence_id > last_live_real->sequence_id) {
            last_live_real = &logical;
          }
        }
      }
    }
    future_real_commitment_high_water =
        std::max(future_real_commitment_high_water,
                 live_future_real_commitments);
    uint64_t tail_real_target_minus_head_quanta = 0;
    if (first_live_real && last_live_real && stable_output_quantum_ns &&
        last_live_real->candidate.target_time_ns >=
            first_live_real->candidate.target_time_ns) {
      tail_real_target_minus_head_quanta =
          (last_live_real->candidate.target_time_ns -
           first_live_real->candidate.target_time_ns) /
          stable_output_quantum_ns;
    }
    pipeline_snapshot_live_future_real_commitments.store(
        live_future_real_commitments, std::memory_order_relaxed);
    pipeline_snapshot_future_real_commitment_high_water.store(
        future_real_commitment_high_water, std::memory_order_relaxed);
    pipeline_snapshot_tail_real_target_minus_head_quanta.store(
        tail_real_target_minus_head_quanta, std::memory_order_relaxed);
    committed_real_high_water =
        std::max(committed_real_high_water, committed_real);
    pipeline_snapshot_pending_uncommitted_real.store(
        pending_uncommitted_real, std::memory_order_relaxed);
    pipeline_snapshot_committed_real.store(committed_real,
                                           std::memory_order_relaxed);
    pipeline_snapshot_committed_real_high_water.store(
        committed_real_high_water, std::memory_order_relaxed);
    if (committed_real > 1) {
      if (!shallow_real_commitment_violation_latched) {
        ++shallow_real_commitment_violation_total;
        shallow_real_commitment_violation_latched = true;
      }
    } else {
      shallow_real_commitment_violation_latched = false;
    }
    if (pending_real_with_nominal_commitment) {
      if (!pending_real_nominal_violation_latched) {
        ++pending_real_nominal_violation_total;
        pending_real_nominal_violation_latched = true;
      }
    } else {
      pending_real_nominal_violation_latched = false;
    }
    for (size_t kind = 0; kind < logical_states.size(); ++kind) {
      for (size_t state = 0; state < logical_states[kind].size(); ++state) {
        pipeline_snapshot_logical_states[kind][state].store(
            logical_states[kind][state], std::memory_order_relaxed);
      }
    }
    const LogicalOutput* head = FindLogicalBySequence(next_apply_sequence);
    pipeline_snapshot_head_valid.store(head != nullptr,
                                       std::memory_order_relaxed);
    pipeline_snapshot_head_sequence.store(head ? head->sequence_id : 0,
                                          std::memory_order_relaxed);
    pipeline_snapshot_head_kind.store(
        head ? uint32_t(head->candidate.kind) : 0,
        std::memory_order_relaxed);
    pipeline_snapshot_head_state.store(
        head ? uint32_t(head->state) : uint32_t(LogicalOutputState::kFree),
        std::memory_order_relaxed);
    pipeline_snapshot_head_nominal_target_ns.store(
        head ? head->candidate.semantic_target_time_ns : 0,
        std::memory_order_relaxed);
    pipeline_snapshot_head_assigned_target_ns.store(
        head ? head->candidate.target_time_ns : 0,
        std::memory_order_relaxed);
    uint64_t head_nominal_tick = 0;
    uint64_t head_assigned_tick = 0;
    if (head) {
      TickForSemanticTarget(head->candidate.semantic_target_time_ns,
                            head->candidate.semantic_epoch_origin_ns,
                            head->candidate.semantic_output_quantum_ns,
                            head_nominal_tick);
      TickForSemanticTarget(head->candidate.target_time_ns,
                            head->candidate.semantic_epoch_origin_ns,
                            head->candidate.semantic_output_quantum_ns,
                            head_assigned_tick);
    }
    pipeline_snapshot_head_nominal_tick.store(head_nominal_tick,
                                              std::memory_order_relaxed);
    pipeline_snapshot_head_assigned_tick.store(head_assigned_tick,
                                               std::memory_order_relaxed);
    pipeline_snapshot_head_exec_debt.store(
        head ? CandidateExecutiveDebtTicks(head->candidate) : 0,
        std::memory_order_relaxed);
    const OutputCandidate* executive_tail =
        FindLiveExecutiveTailCandidate();
    uint64_t tail_nominal_tick = 0;
    uint64_t tail_assigned_tick = 0;
    if (executive_tail) {
      TickForSemanticTarget(executive_tail->semantic_target_time_ns,
                            executive_tail->semantic_epoch_origin_ns,
                            executive_tail->semantic_output_quantum_ns,
                            tail_nominal_tick);
      TickForSemanticTarget(executive_tail->target_time_ns,
                            executive_tail->semantic_epoch_origin_ns,
                            executive_tail->semantic_output_quantum_ns,
                            tail_assigned_tick);
    }
    pipeline_snapshot_tail_valid.store(executive_tail != nullptr,
                                       std::memory_order_relaxed);
    pipeline_snapshot_tail_kind.store(
        executive_tail ? uint32_t(executive_tail->kind) : 0,
        std::memory_order_relaxed);
    pipeline_snapshot_tail_source_id.store(
        executive_tail ? executive_tail->source_id : 0,
        std::memory_order_relaxed);
    pipeline_snapshot_tail_nominal_tick.store(tail_nominal_tick,
                                              std::memory_order_relaxed);
    pipeline_snapshot_tail_assigned_tick.store(tail_assigned_tick,
                                               std::memory_order_relaxed);
    pipeline_snapshot_tail_exec_debt.store(
        executive_tail ? CandidateExecutiveDebtTicks(*executive_tail) : 0,
        std::memory_order_relaxed);
    pipeline_snapshot_head_target_delta_ns.store(
        head && head->candidate.target_time_ns
            ? PresenterSignedDeltaNs(head->candidate.target_time_ns, snapshot_now_ns)
            : 0,
        std::memory_order_relaxed);
    pipeline_snapshot_last_applied_semantic_target_ns.store(
        last_applied_semantic_target_ns, std::memory_order_relaxed);
    pipeline_snapshot_last_accounted_semantic_target_ns.store(
        last_accounted_semantic_target_ns, std::memory_order_relaxed);
    pipeline_snapshot_next_semantic_target_delta_ns.store(
        next_semantic_target_ns
            ? PresenterSignedDeltaNs(next_semantic_target_ns, snapshot_now_ns)
            : 0,
        std::memory_order_relaxed);

    uint32_t real_final_ready_behind_head = 0;
    std::array<uint64_t, 3> next_ready_sequences = {
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max()};
    std::array<int64_t, 3> next_ready_target_deltas_ns = {};
    if (head) {
      for (const LogicalOutput& logical : logical_outputs) {
        if (logical.candidate.kind != CandidateKind::kReal ||
            logical.state != LogicalOutputState::kFinalReady ||
            logical.sequence_id <= head->sequence_id) {
          continue;
        }
        ++real_final_ready_behind_head;
        const int64_t target_delta_ns =
            logical.candidate.target_time_ns
                ? PresenterSignedDeltaNs(logical.candidate.target_time_ns,
                                  snapshot_now_ns)
                : 0;
        for (size_t position = 0; position < next_ready_sequences.size();
             ++position) {
          if (logical.sequence_id >= next_ready_sequences[position]) {
            continue;
          }
          for (size_t move = next_ready_sequences.size() - 1;
               move > position; --move) {
            next_ready_sequences[move] = next_ready_sequences[move - 1];
            next_ready_target_deltas_ns[move] =
                next_ready_target_deltas_ns[move - 1];
          }
          next_ready_sequences[position] = logical.sequence_id;
          next_ready_target_deltas_ns[position] = target_delta_ns;
          break;
        }
      }
    }
    pipeline_snapshot_real_final_ready_behind_head.store(
        real_final_ready_behind_head, std::memory_order_relaxed);
    for (size_t i = 0; i < next_ready_target_deltas_ns.size(); ++i) {
      pipeline_snapshot_real_final_ready_target_deltas_ns[i].store(
          next_ready_target_deltas_ns[i], std::memory_order_relaxed);
    }
    pipeline_snapshot_unfunded_total.store(CountUnfundedFinalOutputs(),
                                           std::memory_order_relaxed);
    pipeline_snapshot_real_unfunded.store(
        CountUnfundedFinalOutputs(CandidateKind::kReal),
        std::memory_order_relaxed);
    pipeline_snapshot_synthetic_unfunded.store(
        CountUnfundedFinalOutputs(CandidateKind::kSynthetic),
        std::memory_order_relaxed);
    pipeline_snapshot_max_earlier_unfunded.store(
        max_earlier_unfunded, std::memory_order_relaxed);
    const uint32_t candidate_free = CountFreeCandidateSlots();
    pipeline_snapshot_candidate_free.store(candidate_free,
                                            std::memory_order_relaxed);
    pipeline_snapshot_candidate_ingress.store(
        CountSourceIngressSlots(), std::memory_order_relaxed);
    pipeline_snapshot_candidate_residency.store(
        CountRealResidencySlots(), std::memory_order_relaxed);
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      pipeline_snapshot_candidate_owner_refs[i].store(
          slots[i].owner_refs, std::memory_order_relaxed);
      pipeline_snapshot_candidate_source_ids[i].store(
          slots[i].source_id.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      pipeline_snapshot_candidate_states[i].store(
          uint32_t(slots[i].state.load(std::memory_order_acquire)),
          std::memory_order_relaxed);
    }
    pipeline_snapshot_residency_pressure_drops.store(
        synthetic_drop_total[size_t(SyntheticDropReason::kResidencyPressure)],
        std::memory_order_relaxed);
    pipeline_snapshot_orphan_ready_final_outputs.store(
        CountOrphanReadyFinalOutputs(), std::memory_order_relaxed);
    std::array<uint32_t, size_t(FinalOutputState::kCount)> final_counts = {};
    for (const FinalOutputSlot& output : final_outputs) {
      ++final_counts[size_t(output.state.load(std::memory_order_acquire))];
    }
    for (size_t i = 0; i < final_counts.size(); ++i) {
      pipeline_snapshot_final_states[i].store(final_counts[i],
                                              std::memory_order_relaxed);
    }
    uint32_t starvation_mask = kResourceStarvationNone;
    if (ResidencyBackpressureRequiresProgress()) {
      starvation_mask |= kResourceStarvationResidency;
    }
    const uint32_t free_final_outputs =
        final_counts[size_t(FinalOutputState::kFree)];
    if (!free_final_outputs) {
      starvation_mask |= kResourceStarvationFinalOutput;
    }
    bool funding_starved = false;
    const uint32_t oldest_waiting_post = FindOldestWaitingPostRaw();
    if (oldest_waiting_post != UINT32_MAX && free_final_outputs) {
      const LogicalOutput& logical = logical_outputs[oldest_waiting_post];
      funding_starved =
          free_final_outputs <= RequiredFinalOutputSurplusFor(logical);
    }
    if (funding_starved) {
      starvation_mask |= kResourceStarvationFunding;
    }
    if (head && head->state != LogicalOutputState::kFinalReady &&
        head->state != LogicalOutputState::kDropped) {
      for (const LogicalOutput& logical : logical_outputs) {
        if (logical.state == LogicalOutputState::kFinalReady &&
            logical.sequence_id > head->sequence_id) {
          starvation_mask |= kResourceStarvationOrdering;
          break;
        }
      }
    }
    if (CountGenerationJobs() || CountPostJobs()) {
      starvation_mask |= kResourceStarvationActiveGpu;
    }
    pipeline_snapshot_resource_starvation_mask.store(
        starvation_mask, std::memory_order_relaxed);
  }

  bool SubmitGenerationJob(uint32_t logical_index,
                           uint32_t previous_slot_index,
                           uint32_t current_slot_index) {
    LogicalOutput& logical = logical_outputs[logical_index];
    const uint64_t now = PresenterMonotonicTimeNs();
    ObserveSyntheticNominalLateness(logical, now);
    // H13: the hard presentation target used to cancel this Generation before
    // the attempt was even counted. That was presentation authority leaking
    // into Production and is the primary round-9 defect: an S admitted with an
    // already-expired B-safe burned without a single Generation attempt. Only
    // concrete bounded physical capacity may refuse the attempt now.
    // #22: this read the job count directly, bypassing the backing authority
    // that SyntheticChainSubmissionCapacityAvailable() now uses. With the
    // switch on both must ask the same question.
    if (CountLiveSyntheticChains() >= SyntheticChainCapacity()) {
      ++split_s_waiting_for_physical_chain_total;
      return false;
      DropSyntheticLogical(logical_index,
                           SyntheticDropReason::kNoSyntheticSlot);
      return true;
    }
    const bool chain_warmup_attempt = !SyntheticChainEstimatorArmed();
    if (chain_warmup_attempt) {
      BeginSyntheticChainWarmup();
      ++synthetic_chain_warmup_attempt_total;
    }
    const uint64_t generation_residence =
        GenerationResidenceEstimateNs();
    const uint64_t post_residence = PostResidenceEstimateNs();
    const uint64_t remaining_to_hard_target =
        logical.candidate.target_time_ns > now
            ? logical.candidate.target_time_ns - now
            : 0;
    // H12: an unanchored S has no hard target, so remaining_to_hard_target is
    // 0 by absence, not by lateness. Scoring it would report a fictional
    // predicted miss on every pre-anchor submit.
    const bool legacy_prediction_would_drop =
        logical.candidate.target_time_ns &&
        SaturatingAddNs(generation_residence, post_residence) >
            remaining_to_hard_target;
    if (chain_warmup_attempt && legacy_prediction_would_drop) {
      // H12: the full-chain estimator is observational/counterfactual
      // evidence only and never becomes Synthetic admission authority, no
      // matter how many completed samples it accumulates. This counter records
      // what the retired predictor would have refused.
      ++synthetic_chain_predicted_bypassed_total;
    }

    const Slot& previous = slots[previous_slot_index];
    const Slot& current = slots[current_slot_index];
    GenerationRequest request;
    request.previous_image = previous.image;
    request.previous_view = previous.view;
    request.previous_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    request.current_image = current.image;
    request.current_view = current.view;
    request.current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    request.source_physical_extent = extent;
    request.extent = current.storage_content_extent;
    request.previous_source_id =
        previous.source_id.load(std::memory_order_relaxed);
    request.current_source_id =
        current.source_id.load(std::memory_order_relaxed);
    const uint64_t previous_ready_value =
        previous.timeline_value.load(std::memory_order_relaxed);
    const uint64_t current_ready_value =
        current.timeline_value.load(std::memory_order_relaxed);
    if (!previous_ready_value || !current_ready_value ||
        previous.ready_timeline == VK_NULL_HANDLE ||
        current.ready_timeline == VK_NULL_HANDLE) {
      Terminalize(TerminalReason::kInvalidHandoff,
                  request.current_source_id);
      return false;
    }
    // One wait per readiness timeline. A and B on the same timeline (the
    // shared one) fold into its higher value, exactly as before V2-2; on
    // their own V2-2 timelines each is waited on its own.
    if (previous.ready_timeline == current.ready_timeline) {
      request.input_wait_count = 1;
      request.input_wait_semaphores[0] = current.ready_timeline;
      request.input_wait_values[0] =
          std::max(previous_ready_value, current_ready_value);
    } else {
      request.input_wait_count = 2;
      request.input_wait_semaphores = {previous.ready_timeline,
                                       current.ready_timeline};
      request.input_wait_values = {previous_ready_value, current_ready_value};
    }
    const uint64_t logical_token = logical.token;
    request.queue_commit_allowed =
        [this, logical_index, logical_token]() {
          if (shutdown_requested.load(std::memory_order_acquire) ||
              detach_requested.load(std::memory_order_acquire) ||
              terminal_reason.load(std::memory_order_acquire) !=
                  TerminalReason::kNone ||
              !accepting.load(std::memory_order_acquire)) {
            return false;
          }
          const LogicalOutput& pending = logical_outputs[logical_index];
          return pending.token == logical_token &&
                 pending.state == LogicalOutputState::kWaitingGeneration &&
                 pending.candidate.kind == CandidateKind::kSynthetic;
        };
    ++generation_attempt_total;
    // H13: the admitted S reached a real Generation attempt. Recorded even
    // when its presentation window already closed, because that is exactly the
    // demand the plant must see.
    logical.synthetic_generation_attempted = true;
    if (logical.synthetic_presentation_retired) {
      ++generation_submitted_after_presentation_hold_total;
    }
    if (!generation_bootstrap_armed) {
      ++generation_bootstrap_attempt_total;
    }
    // v2 falsifier (wait side, H3): whether this Generation's A/B inputs were
    // still pending on the GPU when the request was built, read without
    // waiting. The cost is logged too: these polls also run with the v2
    // switches off whenever C2 exports capture sync_fds.
    const uint64_t input_class_begin_ns = PresenterMonotonicTimeNs();
    const size_t generation_input_class =
        std::max(ClassifyResidencyInput(previous_slot_index),
                 ClassifyResidencyInput(current_slot_index));
    generation_input_class_host_ns_total +=
        PresenterMonotonicTimeNs() - input_class_begin_ns;
    GenerationResult result;
    BeginBlockingOperation(BlockingOperation::kGeneration);
    const uint64_t generation_host_begin_ns = PresenterMonotonicTimeNs();
    const bool generation_submitted = generation_callback(request, result);
    generation_host_block_ns.Add(PresenterMonotonicTimeNs() -
                                 generation_host_begin_ns);
    EndBlockingOperation();
    profile_readback_attempt_total += result.profile_readback_attempts;
    profile_readback_success_total += result.profile_readback_successes;
    profile_readback_not_ready_total += result.profile_readback_not_ready;
    profile_readback_error_total += result.profile_readback_errors;
    if (result.drained_profile_sample_valid &&
        result.drained_profile_sample.timestamp_valid) {
      generation_service_gpu_ns.Add(result.drained_profile_sample.service_gpu_ns);
    }
    if (!generation_submitted) {
      last_generation_failure_stage = result.failure_stage;
      last_generation_status = result.zerofg_status;
      last_generation_vk_result = result.vk_result;
      last_generation_vk_result_valid = result.vk_result_valid;
      last_generation_submission_accepted = result.submission_accepted;
      if (!result.submission_accepted &&
          (result.failure_stage ==
               GenerationFailureStage::kPreparationYield ||
           result.failure_stage == GenerationFailureStage::kQueueBusy)) {
        // First-use setup is cooperatively split from submission, and q0 is
        // never waited for by the presenter. Keep the same logical Synthetic
        // pending and return authority to Source-critical Capture turnover.
        if (result.queue_wait_ns) {
          generation_queue_wait_ns.Add(result.queue_wait_ns);
        }
        return false;
      }
      if (result.failure_stage == GenerationFailureStage::kSyntheticPool) {
        // Physical chain capacity is transient pressure. Keep this exact S
        // obligation in WaitingGeneration and retry after an older chain
        // retires; never turn it into an A->B asymmetry.
        ++split_s_waiting_for_physical_chain_total;
        return false;
      }
      const bool fatal_initialization_failure =
          result.failure_stage == GenerationFailureStage::kAdapter ||
          result.failure_stage == GenerationFailureStage::kRecord ||
          (result.failure_stage == GenerationFailureStage::kResize &&
           result.zerofg_status != 0);
      if (result.submission_accepted || fatal_initialization_failure) {
        // The callback may have handed work to Vulkan before reporting the
        // failure, or the selected algorithm/backend may be unusable for this
        // session. Fail open rather than retrying initialization on every S.
        // Keeping A/B owned through teardown is also required if ownership may
        // have reached the GPU.
        Terminalize(TerminalReason::kGenerationFailure,
                    request.current_source_id);
        return false;
      }
      // Any other non-accepted failure after the split obligation exists is
      // an implementation/driver failure. Fail open visibly rather than
      // deleting S and continuing with B.
      ++split_forbidden_s_drop_total;
      Terminalize(TerminalReason::kGenerationFailure,
                  request.current_source_id);
      return false;
      // No GPU work owns A/B. DropSyntheticLogical is the single owner-release
      // authority for a pre-submit WaitingGeneration, avoiding a double
      // ReleaseRealSlot on this failure path.
      DropSyntheticLogical(
          logical_index,
          result.failure_stage == GenerationFailureStage::kSyntheticPool
              ? SyntheticDropReason::kNoSyntheticSlot
              : SyntheticDropReason::kGenerationFailure);
      return true;
    }
    if (result.submit_time_ns) {
      if (logical.generation_submit_soft_deadline_ns) {
        if (result.submit_time_ns <=
            logical.generation_submit_soft_deadline_ns) {
          generation_submit_ahead_of_latest_ns.Add(
              logical.generation_submit_soft_deadline_ns -
              result.submit_time_ns);
        } else {
          generation_submit_late_ns.Add(
              result.submit_time_ns -
              logical.generation_submit_soft_deadline_ns);
        }
      }
      if (logical.generation_submit_not_before_ns &&
          result.submit_time_ns >=
              logical.generation_submit_not_before_ns) {
        generation_submit_after_not_before_ns.Add(
            result.submit_time_ns -
            logical.generation_submit_not_before_ns);
      }
    }
    if (result.synthetic_index >= kSyntheticPoolSize ||
        generation_jobs[result.synthetic_index].active) {
      Terminalize(TerminalReason::kGenerationFailure,
                  request.current_source_id);
      return false;
    }
    // H12 evidence: production no longer waits for the semantic contract, so
    // record whether this Generation entered service before or after its
    // pair-local anchor arrived.
    if (logical.candidate.target_time_ns) {
      ++synthetic_generation_submitted_after_anchor_total;
    } else {
      ++synthetic_generation_submitted_before_anchor_total;
    }
    GenerationJob& job = generation_jobs[result.synthetic_index];
    job.active = true;
    // Phase 0 sensor 2: the adapter marked this output busy here, and only
    // synthetic_release_callback will clear it.
    AcquireSyntheticBackingOwnership(result.synthetic_index);
    job.synthetic_index = result.synthetic_index;
    job.logical_index = logical_index;
    job.logical_token = logical.token;
    job.previous_real_slot = previous_slot_index;
    job.current_real_slot = current_slot_index;
    job.chained_to_post = false;
    // The drained sample belongs to the previous occupant of this slot, not
    // to the logical Synthetic now being submitted. Keep it out of the job's
    // submission snapshot so delayed completion cannot misattribute it.
    result.drained_profile_sample_valid = false;
    result.drained_profile_sample = {};
    job.submit = result;
    const uint64_t generation_submit_ns =
        result.submit_time_ns ? result.submit_time_ns : PresenterMonotonicTimeNs();
    if (logical.generation_submit_not_before_ns &&
        generation_submit_ns < logical.generation_submit_not_before_ns) {
      ++synthetic_generation_eager_submit_total;
    }
    job.chain_submit_time_ns = generation_submit_ns;
    logical.generation_submit_time_ns = generation_submit_ns;
    job.synthetic_chain_warmup_submission = chain_warmup_attempt;
    logical.synthetic_chain_warmup_submission = chain_warmup_attempt;
    if (chain_warmup_attempt) {
      ++synthetic_chain_warmup_submitted_total;
    }
    // H13: a presentation-retired ticket holds no live presentation deadline,
    // so it contributes none to the observation schedule. Its cadence is
    // purely service/bootstrap. For a still-presentable S the deadlines may
    // only pull an observation earlier; they are never a permission gate.
    job.next_observation_time_ns = StageObservationTimeNs(
        generation_submit_ns, GenerationObservationEstimateNs(),
        logical.synthetic_presentation_retired
            ? 0
            : logical.post_submit_soft_deadline_ns,
        logical.synthetic_presentation_retired
            ? 0
            : logical.candidate.target_time_ns);
    generation_queue_wait_ns.Add(result.queue_wait_ns);
    generation_submit_host_ns.Add(result.submit_host_ns);
    last_generation_owner_stats = result.completion_owner_stats;
    ++generation_input_class_total[generation_input_class];
    generation_input_class_over_1ms[generation_input_class] +=
        result.submit_host_ns >= 1000000ull;
    if (result.completion_fd_export_attempted) {
      ++generation_sync_fd_export_total;
      generation_sync_fd_export_host_ns.Add(
          result.completion_fd_export_host_ns);
      generation_sync_fd_export_failure_total +=
          !result.completion_fd_exported;
    }
    generation_sync_fd_fallback_total += result.completion_fd_fallback;
    generation_setup_ns.Add(result.setup_ns);
    synthetic_pool_occupancy = CountGenerationJobs();
    synthetic_pool_high_water =
        std::max(synthetic_pool_high_water, synthetic_pool_occupancy);
    generation_in_flight_high_water =
        std::max(generation_in_flight_high_water, synthetic_pool_occupancy);
    // Source Protection tracks shared resource treatment even if this
    // optional output is later burned by its presentation cutoff.
    last_pair_synthetic_applied =
        logical.candidate.source_id >= source_protection_pair_floor;
    source_protection_after_issue_sequence =
        xe::internal::source_issue_period_ring().sequence.load(
            std::memory_order_acquire);
    // Generation has already published immutable image identity and a GPU
    // timeline dependency. Make the candidate Post-actionable immediately;
    // Post will wait on the dependency in Vulkan rather than requiring a CPU
    // completion observation between the two stages.
    logical.candidate.image = result.image;
    logical.candidate.view = result.view;
    logical.candidate.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    logical.candidate.storage_extent = result.extent;
    logical.candidate.synthetic_index = result.synthetic_index;
    logical.state = LogicalOutputState::kWaitingPost;
    logical.waiting_post_enter_time_ns = PresenterMonotonicTimeNs();
    MarkInheritedFundingBacklogOnPostEntry(logical);
    ++generation_to_post_chain_candidate_total;
    return true;
  }

  DriverPollResult PollGenerationJobs(
      uint32_t requested_logical_index = UINT32_MAX,
      bool only_if_observation_due = false) {
    DriverPollResult poll;
    GenerationJob* selected_job = nullptr;
    const uint64_t selection_now_ns = PresenterMonotonicTimeNs();
    const bool targeted = requested_logical_index < kLogicalOutputCapacity;
    if (targeted) {
      for (GenerationJob& job : generation_jobs) {
        if (!job.active || job.chained_to_post ||
            job.logical_index != requested_logical_index) {
          continue;
        }
        if (only_if_observation_due &&
            job.next_observation_time_ns > selection_now_ns) {
          ++generation_poll_not_due_total;
          poll.deferred_not_due = true;
          return poll;
        }
        {
          selected_job = &job;
          break;
        }
      }
      // A targeted liveness or head observation must never fall through and
      // query unrelated work merely because the requested job is absent.
      if (!selected_job) {
        return poll;
      }
    }
    if (!targeted) {
      uint64_t oldest_submit_ns = std::numeric_limits<uint64_t>::max();
      for (GenerationJob& job : generation_jobs) {
        if (!GenerationNeedsCpuObservation(job) ||
            (only_if_observation_due &&
             job.next_observation_time_ns > selection_now_ns)) {
          continue;
        }
        const uint64_t submit_ns =
            job.submit.submit_time_ns ? job.submit.submit_time_ns : 0;
        if (!selected_job || submit_ns < oldest_submit_ns) {
          selected_job = &job;
          oldest_submit_ns = submit_ns;
        }
      }
    }
    if (!selected_job) {
      if (only_if_observation_due && CountGenerationJobs()) {
        ++generation_poll_not_due_total;
      }
      return poll;
    }
    GenerationJob& job = *selected_job;
    poll.invoked = true;
    BeginHostDriverOperation(BlockingOperation::kGenerationPoll);
    const uint64_t poll_begin_ns = PresenterMonotonicTimeNs();
    GenerationResult completion;
    bool ready = false;
    const bool poll_succeeded =
        generation_poll_callback(job.synthetic_index, completion, ready);
    active_generation_poll_host_ns.Add(PresenterMonotonicTimeNs() - poll_begin_ns);
    ++active_generation_poll_total;
    if (poll_succeeded) {
      last_generation_owner_stats = completion.completion_owner_stats;
    }
    if (completion.completion_fd_polled) {
      ++generation_sync_fd_poll_total;
      generation_sync_fd_poll_host_ns.Add(
          completion.completion_fd_poll_host_ns);
      generation_sync_fd_ready_total += ready;
    }
    EndHostDriverOperation();
    if (!poll_succeeded) {
      Terminalize(TerminalReason::kGenerationFailure, 0);
      return poll;
    }
    if (!ready) {
      uint64_t soft_deadline_ns = 0;
      uint64_t hard_deadline_ns = 0;
      if (job.logical_index < kLogicalOutputCapacity) {
        const LogicalOutput& logical = logical_outputs[job.logical_index];
        // H13: no live presentation deadline once presentation is retired.
        if (logical.token == job.logical_token &&
            !logical.synthetic_presentation_retired) {
          soft_deadline_ns = logical.post_submit_soft_deadline_ns;
          hard_deadline_ns = logical.candidate.target_time_ns;
        }
      }
      job.next_observation_time_ns = StageObservationTimeNs(
          PresenterMonotonicTimeNs(), StageObservationRetryIntervalNs(),
          soft_deadline_ns, hard_deadline_ns);
      ++generation_poll_retry_scheduled_total;
      return poll;
    }
    poll.progress = true;
    generation_submit_to_ready_ns.Add(completion.submit_to_ready_ns);
    generation_gpu_completion_ns.Add(completion.submit_to_ready_ns);
    generation_total_ns.Add(completion.submit_to_ready_ns +
                            job.submit.setup_ns);
    if (!generation_bootstrap_armed) {
      generation_bootstrap_total_ns.Add(completion.submit_to_ready_ns);
      ++generation_bootstrap_success_total;
      if (generation_bootstrap_success_total >=
          kGenerationBootstrapSamples) {
        generation_bootstrap_estimate_ns =
            generation_bootstrap_total_ns.Quantile(50, 100);
        generation_bootstrap_armed = true;
      }
    } else {
      generation_steady_total_ns.Add(completion.submit_to_ready_ns);
    }
    ++synthetic_generated_total;
    const uint32_t logical_index = job.logical_index;
    const uint64_t logical_token = job.logical_token;
    ReleaseRealSlot(job.previous_real_slot);
    ReleaseRealSlot(job.current_real_slot);
    job.active = false;
    job.chained_to_post = false;
    synthetic_pool_occupancy = CountGenerationJobs();
    if (logical_index >= kLogicalOutputCapacity ||
        logical_outputs[logical_index].token != logical_token ||
        logical_outputs[logical_index].state == LogicalOutputState::kDropped) {
      ReleaseSyntheticBackingOwnership(completion.synthetic_index);
      synthetic_release_callback(completion.synthetic_index);
      return poll;
    }
    LogicalOutput& logical = logical_outputs[logical_index];
    const uint64_t generation_ready_time_ns = PresenterMonotonicTimeNs();
    ObserveSyntheticNominalLateness(logical, generation_ready_time_ns);
    // H13: a completed Generation used to be thrown away here when the hard
    // presentation target had passed. The pixels already exist; discarding
    // them is presentation authority cancelling finished Production. The
    // chain now always continues to Post, and the presentation decision is
    // made solely by the ordered executive.
    if (logical.synthetic_presentation_retired) {
      ++generation_completed_after_presentation_hold_total;
    }
    if (logical.post_submit_soft_deadline_ns &&
        generation_ready_time_ns >= logical.post_submit_soft_deadline_ns) {
      // Missing the planned post-entry point teaches internal latency, but
      // the admitted S remains salvageable until its hard semantic target.
      // T-A and the post-entry target are planning boundaries. Missing one is
      // useful scheduler telemetry, but an admitted S still has a live hard T
      // and this alone is not structural-D evidence.
      ObserveLatencyFeasibilityMiss(
          logical, LatencyMissCause::kScheduler,
          LatencyMissProvenance::kGenerationSoftLate);
    }
    logical.candidate.image = completion.image;
    logical.candidate.view = completion.view;
    logical.candidate.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    logical.candidate.storage_extent = completion.extent;
    logical.candidate.synthetic_index = completion.synthetic_index;
    logical.candidate.ready_time_ns = generation_ready_time_ns;
    logical.state = LogicalOutputState::kWaitingPost;
    if (!logical.waiting_post_enter_time_ns) {
      logical.waiting_post_enter_time_ns = generation_ready_time_ns;
    }
    MarkInheritedFundingBacklogOnPostEntry(logical);
    return poll;
  }

  bool GenerationJitActionable(LogicalOutput& logical, uint64_t now_ns) {
    if (logical.candidate.kind == CandidateKind::kSynthetic) {
      if (logical.generation_submit_not_before_ns &&
          now_ns < logical.generation_submit_not_before_ns &&
          !logical.generation_jit_deferred_recorded) {
        logical.generation_jit_deferred_recorded = true;
        ++synthetic_generation_would_jit_defer_total;
      }
      // H12: production is work-conserving. generation_submit_not_before_ns
      // remains counterfactual planning telemetry; it is not permission to
      // execute. Real JIT behavior below is unchanged. Actual bounded
      // ownership (pool, capture/residency, funding) stays authoritative and
      // is enforced by the submit path, not here.
      return true;
    }
    if (!logical.generation_submit_not_before_ns ||
        now_ns >= logical.generation_submit_not_before_ns) {
      return true;
    }
    if (ResidencyBackpressureRequiresProgress() &&
        ResidencyRecoveryGenerationCanAdvance(logical)) {
      if (!logical.generation_residency_recovery_recorded) {
        logical.generation_residency_recovery_recorded = true;
        ++residency_recovery_generation_advance_total;
      }
      return true;
    }
    if (!logical.generation_jit_deferred_recorded) {
      logical.generation_jit_deferred_recorded = true;
      ++generation_jit_deferral_total;
    }
    return false;
  }

  bool PostJitActionable(LogicalOutput& logical, uint64_t now_ns) {
    if (logical.candidate.kind == CandidateKind::kSynthetic) {
      if (logical.post_submit_not_before_ns &&
          now_ns < logical.post_submit_not_before_ns &&
          !logical.post_jit_deferred_recorded) {
        logical.post_jit_deferred_recorded = true;
        ++synthetic_post_would_jit_defer_total;
      }
      // H12: Synthetic Post is work-conserving for the same reason Generation
      // is. No safety reason survives static ownership analysis: SubmitPostJobs
      // still enforces free FinalOutput slots and the order-aware
      // RequiredFinalOutputSurplusFor() reservation for earlier unfunded
      // outputs, so an early Synthetic Post can never defund an earlier Real.
      // post_submit_not_before_ns stays counterfactual planning telemetry.
      return true;
    }
    // Split Production is eager/work-conserving. The not-before value is
    // retained for telemetry only; final presentation remains ordered and
    // metronome-controlled below.
    return true;
    if (!logical.post_submit_not_before_ns ||
        now_ns >= logical.post_submit_not_before_ns) {
      return true;
    }
    const bool recovery_required = ResidencyBackpressureRequiresProgress();
    const bool residency_recovery_post =
        logical.candidate.kind == CandidateKind::kReal &&
        RealLogicalPostCanAdvanceResidencyRecovery(logical) &&
        recovery_required;
    if (residency_recovery_post) {
      if (!logical.real_early_transfer_recorded) {
        logical.real_early_transfer_recorded = true;
        ++real_early_transfer_total;
        ++early_real_transfer_attempt_total;
      }
      return true;
    }
    if (logical.candidate.kind == CandidateKind::kReal &&
        recovery_required) {
      if (!logical.real_early_transfer_block_recorded) {
        logical.real_early_transfer_block_recorded = true;
        ++early_real_transfer_blocked_ownership_total;
      }
      return false;
    }
    if (!logical.post_jit_deferred_recorded) {
      logical.post_jit_deferred_recorded = true;
      ++post_jit_deferral_total;
      ++real_post_would_jit_defer_total;
    }
    return false;
  }

  uint32_t FindOldestWaitingPost(std::optional<CandidateKind> kind =
                                     std::nullopt) {
    uint32_t logical_index = UINT32_MAX;
    uint64_t oldest_sequence = std::numeric_limits<uint64_t>::max();
    bool selected_retired = true;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& candidate = logical_outputs[i];
      if (candidate.state != LogicalOutputState::kWaitingPost ||
          (kind && candidate.candidate.kind != *kind) ||
          !PostJitActionable(candidate, now_ns)) {
        continue;
      }
      // H13: a presentation-retired ticket has no presentation obligation, so
      // it must never outrank a still-presentable output — in particular a
      // sovereign Real — merely because its old semantic sequence is lower. It
      // is served only when nothing presentable is waiting, which keeps its
      // honest plant load without letting it reserve capacity ahead of B.
      const bool retired = candidate.synthetic_presentation_retired;
      if (retired != selected_retired) {
        if (retired) {
          continue;
        }
      } else if (candidate.sequence_id >= oldest_sequence) {
        continue;
      }
      logical_index = i;
      oldest_sequence = candidate.sequence_id;
      selected_retired = retired;
    }
    return logical_index;
  }

  uint32_t FindOldestCompoundResidencyRecoveryRealPost() const {
    uint32_t logical_index = UINT32_MAX;
    uint64_t oldest_sequence = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (RealLogicalPostCanAdvanceResidencyRecovery(logical) &&
          !RealLogicalPostCanReleaseSlot(logical) &&
          logical.sequence_id < oldest_sequence) {
        logical_index = i;
        oldest_sequence = logical.sequence_id;
      }
    }
    return logical_index;
  }

  uint32_t FindOldestDirectResidencyRecoveryRealPost() const {
    uint32_t logical_index = UINT32_MAX;
    uint64_t oldest_sequence = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (logical.candidate.kind == CandidateKind::kReal &&
          RealLogicalPostCanReleaseSlot(logical) &&
          logical.sequence_id < oldest_sequence) {
        logical_index = i;
        oldest_sequence = logical.sequence_id;
      }
    }
    return logical_index;
  }

  uint32_t FindOldestWaitingResidencyRecoveryGeneration() const {
    uint32_t logical_index = UINT32_MAX;
    uint64_t oldest_sequence = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (ResidencyRecoveryGenerationCanAdvance(logical) &&
          !LogicalHasJob(i, logical.token) &&
          logical.sequence_id < oldest_sequence) {
        logical_index = i;
        oldest_sequence = logical.sequence_id;
      }
    }
    return logical_index;
  }

  uint32_t FindOldestActiveResidencyRecoveryGeneration() const {
    uint32_t logical_index = UINT32_MAX;
    uint64_t oldest_sequence = std::numeric_limits<uint64_t>::max();
    for (const GenerationJob& job : generation_jobs) {
      if (!job.active || job.chained_to_post ||
          job.logical_index >= kLogicalOutputCapacity) {
        continue;
      }
      const LogicalOutput& logical = logical_outputs[job.logical_index];
      if (logical.token != job.logical_token ||
          logical.state != LogicalOutputState::kWaitingPost ||
          logical.candidate.kind != CandidateKind::kSynthetic ||
          (!CandidateSlotProgressedByOneGeneration(job.previous_real_slot) &&
           !CandidateSlotProgressedByOneGeneration(job.current_real_slot)) ||
          logical.sequence_id >= oldest_sequence) {
        continue;
      }
      logical_index = job.logical_index;
      oldest_sequence = logical.sequence_id;
    }
    return logical_index;
  }

  uint32_t FindOldestWaitingGeneration() {
    uint32_t logical_index = UINT32_MAX;
    uint64_t oldest_sequence = std::numeric_limits<uint64_t>::max();
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& logical = logical_outputs[i];
      if (logical.state == LogicalOutputState::kWaitingGeneration &&
          !LogicalHasJob(i, logical.token) &&
          GenerationJitActionable(logical, now_ns) &&
          logical.sequence_id < oldest_sequence) {
        logical_index = i;
        oldest_sequence = logical.sequence_id;
      }
    }
    return logical_index;
  }

  // Phase 5 B2, sensor only. funding_block was a COUNT - 41420 of them in a
  // Rayman run - and a count cannot say whether it explains the ~90 ms of
  // pre-submit residence that device_b_post_after_n_ready sees as an envelope
  // and no component sensor covers. The driver queue waits are 0 and GPU
  // service is a few ms, so the residence is before submit; this measures how
  // much of that wait is FinalOutput starvation, in time.
  //
  // SCOPE, deliberate: one timer for the whole plant, not one per logical. A
  // run of refusals followed by a submit records a single interval - how long
  // the plant was continuously starved of FinalOutput - which is exactly the
  // exploratory question. It is NOT sample-aligned, so it may not be
  // subtracted from C_pair_i the way B1 subtracts the sidecar hold. If the
  // number turns out to be large, the per-logical plumbing gets built then;
  // building it before knowing the magnitude would be plumbing on a guess.
  void OpenFundingBlock() {
    if (!funding_block_begin_ns) {
      funding_block_begin_ns = PresenterMonotonicTimeNs();
    }
  }

  void CloseFundingBlock() {
    if (!funding_block_begin_ns) {
      return;
    }
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (now_ns > funding_block_begin_ns) {
      const uint64_t held_ns = now_ns - funding_block_begin_ns;
      funding_block_ns.Add(held_ns);
      funding_block_total_ns = SaturatingAddNs(funding_block_total_ns, held_ns);
    }
    funding_block_begin_ns = 0;
  }

  const LogicalOutput* FindLogicalHoldingFinalOutput(
      uint32_t output_index) const {
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (logical.state == LogicalOutputState::kFinalReady &&
          logical.final_output_index == output_index) {
        return &logical;
      }
    }
    return nullptr;
  }

  FundingCensus CensusFinalOutputs(uint64_t now_ns) const {
    FundingCensus census;
    for (uint32_t i = 0; i < kFinalOutputPoolSize; ++i) {
      switch (final_outputs[i].state.load(std::memory_order_acquire)) {
        case FinalOutputState::kFree:
          ++census.free;
          break;
        case FinalOutputState::kPostProcessing:
          ++census.postprocessing;
          break;
        case FinalOutputState::kTransactionApplied:
          ++census.applied;
          break;
        case FinalOutputState::kReady: {
          // kReady does NOT mean GPU-complete: the split marks a Post
          // logically FinalReady before its GPU work necessarily finished.
          // The one thing that IS provable here is the pacing hold, because
          // the pump tests the clock before it tests completion.
          const LogicalOutput* holder = FindLogicalHoldingFinalOutput(i);
          if (holder && holder->candidate.target_time_ns &&
              now_ns < PlannedDispatchTimeNs(*holder)) {
            ++census.ready_future_target;
          } else {
            ++census.ready_time_passed;
          }
          break;
        }
        default:
          break;
      }
    }
    return census;
  }

  FundingWaitClass ClassifyFundingWaitButFor(
      const FundingCensus& census, const LogicalOutput& logical) const {
    // Counterfactual on the gate this logical actually failed. Every other
    // occupant - ready_time_passed, Applied, CompositorOwned, ReleasePending,
    // PostProcessing - stays exactly where it is. We are not pretending they
    // are ours. We ask only whether returning the slots unequivocally pinned
    // by our own dispatch clock would have let THIS logical through.
    return census.free + census.ready_future_target >
                   RequiredFinalOutputSurplusFor(logical)
               ? FundingWaitClass::kPolicyButFor
               : FundingWaitClass::kPhysicalResidual;
  }

  // Match the Post selector's presentable-before-retired, then sequence
  // ordering without invoking PostJitActionable (which records telemetry).
  // In the strict split every WaitingPost logical is Post-actionable.
  bool IsWaitingPostHead(const LogicalOutput& head) const {
    if (head.state != LogicalOutputState::kWaitingPost) {
      return false;
    }
    for (const LogicalOutput& other : logical_outputs) {
      if (&other == &head || other.state != LogicalOutputState::kWaitingPost) {
        continue;
      }
      if ((!other.synthetic_presentation_retired &&
           head.synthetic_presentation_retired) ||
          (other.synthetic_presentation_retired ==
               head.synthetic_presentation_retired &&
           other.sequence_id < head.sequence_id)) {
        return false;
      }
    }
    return true;
  }

  void MarkLaterSyntheticsBehindFundingHead(const LogicalOutput& head,
                                            FundingWaitClass wait_class) {
    if (!IsWaitingPostHead(head)) {
      return;
    }
    for (LogicalOutput& later : logical_outputs) {
      if (later.state != LogicalOutputState::kWaitingPost ||
          later.candidate.kind != CandidateKind::kSynthetic ||
          later.sequence_id <= head.sequence_id ||
          later.better_d_sample_scored) {
        continue;
      }
      if (wait_class == FundingWaitClass::kPolicyButFor &&
          !later.funding_backlog_marked) {
        later.funding_backlog_marked = true;
        ++funding_backlog_marked_total;
      } else if (wait_class == FundingWaitClass::kPhysicalResidual &&
                 !later.behind_physical) {
        later.behind_physical = true;
        ++behind_physical_total;
      }
    }
  }

  // A newly submitted S can enter WaitingPost between Post attempts. An open
  // per-logical funding wait is the bounded head-block identity; verify the
  // gate and but-for class afresh so a stale, already-drained block cannot
  // mark the newcomer. Logical reuse clears the wait and both provenance bits.
  void MarkInheritedFundingBacklogOnPostEntry(LogicalOutput& entering) {
    if (entering.candidate.kind != CandidateKind::kSynthetic ||
        entering.better_d_sample_scored) {
      return;
    }
    for (const LogicalOutput& head : logical_outputs) {
      if (head.state != LogicalOutputState::kWaitingPost ||
          !head.funding_wait_begin_ns ||
          head.sequence_id >= entering.sequence_id ||
          !IsWaitingPostHead(head)) {
        continue;
      }
      const FundingCensus census =
          CensusFinalOutputs(PresenterMonotonicTimeNs());
      if (census.free <= RequiredFinalOutputSurplusFor(head)) {
        MarkLaterSyntheticsBehindFundingHead(
            head, ClassifyFundingWaitButFor(census, head));
      }
      break;
    }
  }

  // Closes the open segment at now and restarts it there. A wait that begins
  // policy-but-for and becomes physical is not retroactively relabelled: each
  // part is charged to what was true while it elapsed, the way the sidecar
  // hold is measured rather than guessed.
  void AccumulateFundingSegment(LogicalOutput& logical, uint64_t now_ns) {
    if (!logical.funding_wait_begin_ns) {
      return;
    }
    const uint64_t held_ns = now_ns > logical.funding_wait_begin_ns
                                 ? now_ns - logical.funding_wait_begin_ns
                                 : 0;
    logical.funding_wait_begin_ns = now_ns;
    const FundingWaitClass segment =
        FundingWaitClass(logical.funding_wait_segment_class);
    if (!held_ns || segment == FundingWaitClass::kNone) {
      return;
    }
    logical.funding_wait_total_ns =
        SaturatingAddNs(logical.funding_wait_total_ns, held_ns);
    funding_class_total_ns[size_t(segment)] =
        SaturatingAddNs(funding_class_total_ns[size_t(segment)], held_ns);
    switch (segment) {
      case FundingWaitClass::kPolicyButFor:
        logical.funding_policy_wait_ns =
            SaturatingAddNs(logical.funding_policy_wait_ns, held_ns);
        funding_policy_wait_ns.Add(held_ns);
        break;
      case FundingWaitClass::kPhysicalResidual:
        logical.funding_physical_wait_ns =
            SaturatingAddNs(logical.funding_physical_wait_ns, held_ns);
        funding_physical_wait_ns.Add(held_ns);
        break;
      default:
        break;
    }
  }

  FundingWaitClass OpenLogicalFundingWait(LogicalOutput& logical,
                                         uint64_t now_ns) {
    last_funding_census = CensusFinalOutputs(now_ns);
    const FundingWaitClass observed =
        ClassifyFundingWaitButFor(last_funding_census, logical);
    if (!logical.funding_wait_begin_ns) {
      logical.funding_wait_begin_ns = now_ns;
      logical.funding_wait_segment_class = uint8_t(observed);
      ++funding_episode_open_total[size_t(observed)];
      return observed;
    }
    if (logical.funding_wait_segment_class != uint8_t(observed)) {
      AccumulateFundingSegment(logical, now_ns);
      logical.funding_wait_segment_class = uint8_t(observed);
      ++funding_segment_switch_total;
    }
    return observed;
  }

  void CloseLogicalFundingWait(LogicalOutput& logical, uint64_t now_ns) {
    if (!logical.funding_wait_begin_ns) {
      return;
    }
    AccumulateFundingSegment(logical, now_ns);
    logical.funding_wait_begin_ns = 0;
    logical.funding_wait_segment_class = uint8_t(FundingWaitClass::kNone);
  }

  // The funding wait of a logical ends when the physical funding condition
  // stops holding for IT, not when the arbiter eventually gets round to
  // selecting it. Waiting for selection is ordering or scheduler, and folding
  // that into funding would launder exactly what B2b was built to keep apart.
  // This also closes an abandoned wait, which is what leaked in B2b.
  void ObserveFundingWaitTransitions(uint64_t now_ns) {
    bool any_open = false;
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      if (logical_outputs[i].funding_wait_begin_ns) {
        any_open = true;
        break;
      }
    }
    if (!any_open) {
      return;
    }
    const FundingCensus census = CensusFinalOutputs(now_ns);
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& logical = logical_outputs[i];
      if (!logical.funding_wait_begin_ns) {
        continue;
      }
      if (logical.state != LogicalOutputState::kWaitingPost) {
        CloseLogicalFundingWait(logical, now_ns);
        ++funding_closed_by_state_total;
        continue;
      }
      if (census.free > RequiredFinalOutputSurplusFor(logical)) {
        CloseLogicalFundingWait(logical, now_ns);
        ++funding_closed_by_condition_total;
        continue;
      }
      const FundingWaitClass observed =
          ClassifyFundingWaitButFor(census, logical);
      if (logical.funding_wait_segment_class != uint8_t(observed)) {
        AccumulateFundingSegment(logical, now_ns);
        logical.funding_wait_segment_class = uint8_t(observed);
        ++funding_segment_switch_total;
      }
    }
  }

  bool SubmitPostJobs(uint32_t requested_logical_index = UINT32_MAX,
                      bool submit_one = false,
                      bool residency_recovery_post = false) {
    bool progress = false;
    // B2c: segment and retire open funding waits before this pass can change
    // the pool underneath them.
    ObserveFundingWaitTransitions(PresenterMonotonicTimeNs());
    for (;;) {
      uint32_t logical_index = requested_logical_index;
      if (logical_index >= kLogicalOutputCapacity ||
          logical_outputs[logical_index].state !=
              LogicalOutputState::kWaitingPost) {
        logical_index = FindOldestWaitingPost();
      }
      if (logical_index == UINT32_MAX) {
        break;
      }
      LogicalOutput& logical = logical_outputs[logical_index];
      const uint64_t now_ns = PresenterMonotonicTimeNs();
      if (!PostJitActionable(logical, now_ns)) {
        break;
      }
      const bool residency_recovery_early_transfer =
          logical.candidate.kind == CandidateKind::kReal &&
          logical.post_submit_not_before_ns &&
          now_ns < logical.post_submit_not_before_ns &&
          RealLogicalPostCanAdvanceResidencyRecovery(logical) &&
          ResidencyBackpressureRequiresProgress();
      const uint32_t recovery_debt_before_submit =
          residency_recovery_early_transfer ? CountResidencyRecoveryDebt()
                                            : 0;
      const bool residency_recovery_compound_transfer =
          residency_recovery_early_transfer &&
          !RealLogicalPostCanReleaseSlot(logical);
      // H13: Post used to be cancelled here when the hard presentation target
      // had passed. Generation had already paid its cost, so this destroyed
      // finished work for a presentation reason. Post now always proceeds;
      // physical FinalOutput funding and ordering below remain authoritative.
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        ObserveSyntheticNominalLateness(logical, PresenterMonotonicTimeNs());
      }
      const uint32_t required_surplus =
          RequiredFinalOutputSurplusFor(logical);
      uint32_t old_earlier_unfunded_high =
          final_output_earlier_unfunded_high_water.load(
              std::memory_order_relaxed);
      while (old_earlier_unfunded_high < required_surplus &&
             !final_output_earlier_unfunded_high_water.compare_exchange_weak(
                 old_earlier_unfunded_high, required_surplus,
                 std::memory_order_relaxed, std::memory_order_relaxed)) {
      }
      const uint32_t free_final_outputs = CountFreeFinalOutputs();
      if (!free_final_outputs) {
        final_output_funding_block_total.fetch_add(1,
                                                   std::memory_order_relaxed);
        OpenFundingBlock();
        MarkLaterSyntheticsBehindFundingHead(
            logical, OpenLogicalFundingWait(logical, now_ns));
        if (residency_recovery_early_transfer) {
          ++early_real_transfer_blocked_final_pool_total;
        }
        break;
      }
      if (free_final_outputs <= required_surplus) {
        // A transient lack of order-aware funding is a wait. S is dropped only
        // by its normal deadline or by explicit Residency-recovery liveness,
        // never merely because future accepted Reals exist.
        final_output_funding_block_total.fetch_add(1,
                                                   std::memory_order_relaxed);
        OpenFundingBlock();
        MarkLaterSyntheticsBehindFundingHead(
            logical, OpenLogicalFundingWait(logical, now_ns));
        if (residency_recovery_early_transfer) {
          ++early_real_transfer_blocked_funding_total;
        }
        break;
      }
      uint32_t output_index = UINT32_MAX;
      for (uint32_t i = 0; i < kFinalOutputPoolSize; ++i) {
        FinalOutputState expected = FinalOutputState::kFree;
        if (final_outputs[i].state.compare_exchange_strong(
                expected, FinalOutputState::kPostProcessing,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
          output_index = i;
          break;
        }
      }
      if (output_index == UINT32_MAX) {
        if (residency_recovery_early_transfer) {
          ++early_real_transfer_blocked_final_pool_total;
        }
        break;
      }
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        logical.final_output_acquire_time_ns = PresenterMonotonicTimeNs();
      }
      // STOP, backstop. ObserveFundingWaitTransitions normally closes the wait
      // earlier, the instant the condition passed. This catches the case where
      // the condition passed and was consumed inside the same pass.
      if (logical.funding_wait_begin_ns) {
        CloseLogicalFundingWait(logical, PresenterMonotonicTimeNs());
        ++funding_closed_by_acquire_total;
      }
      FinalOutputSlot& output = final_outputs[output_index];
      const Presenter::GuestOutputPaintConfig paint_config =
          paint_config_provider ? paint_config_provider()
                                : Presenter::GuestOutputPaintConfig();
      PostProcessRequest request;
      request.candidate_is_synthetic =
          logical.candidate.kind == CandidateKind::kSynthetic;
      request.source_id = logical.candidate.source_id;
      request.pair_a_source_id = logical.candidate.pair_a_source_id;
      request.logical_sequence = logical.sequence_id;
      request.candidate_image = logical.candidate.image;
      request.candidate_layout = logical.candidate.layout;
      request.candidate_storage_extent = logical.candidate.storage_extent;
      request.frontbuffer_width = logical.candidate.content_extent.width;
      request.frontbuffer_height = logical.candidate.content_extent.height;
      request.display_aspect_ratio_x =
          logical.candidate.display_aspect_ratio_x;
      request.display_aspect_ratio_y =
          logical.candidate.display_aspect_ratio_y;
      request.is_8bpc = logical.candidate.is_8bpc;
      request.config = paint_config;
      request.final_output_image = output.image;
      request.final_output_view = output.view;
      request.final_output_extent = extent;
      request.final_output_format = final_output_format;
      request.final_output_index = output_index;
      GenerationJob* chained_generation = nullptr;
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        if (logical.candidate.synthetic_index >= kSyntheticPoolSize) {
          ReleaseFinalOutputSlot(output_index);
          Terminalize(TerminalReason::kInvalidHandoff,
                      logical.candidate.source_id);
          return false;
        }
        GenerationJob& generation_job =
            generation_jobs[logical.candidate.synthetic_index];
        if (generation_job.active &&
            (generation_job.chained_to_post ||
             generation_job.logical_index != logical_index ||
             generation_job.logical_token != logical.token)) {
          ReleaseFinalOutputSlot(output_index);
          Terminalize(TerminalReason::kInvalidHandoff,
                      logical.candidate.source_id);
          return false;
        }
        if (generation_job.active) {
          request.candidate_wait_semaphore =
              generation_job.submit.completion_semaphore;
          request.candidate_wait_value = generation_job.submit.signal_value;
          chained_generation = &generation_job;
        }
      } else {
        if (logical.candidate.real_slot >= kPoolSize) {
          ReleaseFinalOutputSlot(output_index);
          Terminalize(TerminalReason::kInvalidHandoff,
                      logical.candidate.source_id);
          return false;
        }
        const Slot& real = slots[logical.candidate.real_slot];
        const uint64_t ready_value =
            real.timeline_value.load(std::memory_order_relaxed);
        if (!ready_value || real.ready_timeline == VK_NULL_HANDLE) {
          ReleaseFinalOutputSlot(output_index);
          Terminalize(TerminalReason::kInvalidHandoff,
                      logical.candidate.source_id);
          return false;
        }
        // The timeline this Real's capture transfer signaled (the shared one,
        // or its Residency slot's own with V2-2).
        request.candidate_wait_semaphore = real.ready_timeline;
        request.candidate_wait_value = ready_value;
      }
      output.source_id = logical.candidate.source_id;
      output.pair_a_source_id = logical.candidate.pair_a_source_id;
      output.latency_epoch = logical.latency_epoch;
      output.operating_latency_ns = logical.operating_latency_ns;
      output.candidate_kind = logical.candidate.kind;
      output.semantic_target_time_ns =
          logical.candidate.semantic_target_time_ns;
      output.target_time_ns = logical.candidate.target_time_ns;
      output.semantic_epoch_origin_ns =
          logical.candidate.semantic_epoch_origin_ns;
      output.semantic_output_quantum_ns =
          logical.candidate.semantic_output_quantum_ns;
      output.semantic_tick_index = logical.candidate.semantic_tick_index;
      output.sequence_id = logical.sequence_id;
      output.semantic_forward_skipped = logical.semantic_forward_skipped;
      output.final_ready_soft_deadline_ns =
          logical.final_ready_soft_deadline_ns;
      PostProcessResult submit;
      BeginBlockingOperation(residency_recovery_early_transfer ||
                                     residency_recovery_post
                                 ? BlockingOperation::kRecoveryPost
                                 : BlockingOperation::kNormalPost);
      const uint64_t post_host_begin_ns = PresenterMonotonicTimeNs();
      const bool post_submitted = post_process_callback(request, submit);
      post_host_block_ns.Add(PresenterMonotonicTimeNs() - post_host_begin_ns);
      EndBlockingOperation();
      const bool accepted_timeline_fallback =
          submit.submission_accepted && !submit.acquire_fence_exported &&
          submit.completion_semaphore != VK_NULL_HANDLE &&
          submit.signal_value != 0;
      if (!post_submitted && !accepted_timeline_fallback) {
        if (!submit.submission_accepted) {
          ReleaseFinalOutputSlot(output_index);
          if (logical.candidate.kind == CandidateKind::kSynthetic) {
            // Once an S obligation exists, a non-retryable pre-submit Post
            // failure must fail open. Dropping here would silently turn the
            // ordered A->S->B contract into A->B.
            ++split_forbidden_s_drop_total;
            Terminalize(TerminalReason::kPostProcessFailure,
                        logical.candidate.source_id, submit.failure_stage,
                        submit.vk_result_valid, submit.vk_result, false);
            return false;
            DropSyntheticLogical(logical_index,
                                 SyntheticDropReason::kGenerationFailure);
            progress = true;
            continue;
          }
        }
        Terminalize(TerminalReason::kPostProcessFailure,
                    logical.candidate.source_id, submit.failure_stage,
                    submit.vk_result_valid, submit.vk_result,
                    submit.submission_accepted);
        return false;
      }
      if (!submit.acquire_fence_exported && !accepted_timeline_fallback) {
        Terminalize(TerminalReason::kPostProcessFailure,
                    logical.candidate.source_id,
                    PostFailureStage::kAcquireFenceExport, false, VK_SUCCESS,
                    true);
        return false;
      }
      // The Post's single sync_file is its completion authority (a
      // non-waiting poll); the egress copy waits on the Post timeline on the
      // GPU. Nothing else consumes it.
      int completion_fence_fd = -1;
      if (submit.acquire_fence_exported && submit.acquire_fence_fd >= 0) {
        completion_fence_fd = submit.acquire_fence_fd;
      }
      submit.acquire_fence_fd = -1;
      if (submit.submit_time_ns) {
        if (logical.post_submit_soft_deadline_ns) {
          if (submit.submit_time_ns <=
              logical.post_submit_soft_deadline_ns) {
            post_submit_ahead_of_latest_ns.Add(
                logical.post_submit_soft_deadline_ns -
                submit.submit_time_ns);
          } else {
            post_submit_late_ns.Add(
                submit.submit_time_ns -
                logical.post_submit_soft_deadline_ns);
            if (logical.candidate.kind == CandidateKind::kSynthetic) {
              ObserveLatencyFeasibilityMiss(
                  logical, LatencyMissCause::kScheduler,
                  LatencyMissProvenance::kPostSoftLate);
            }
          }
        }
        if (logical.post_submit_not_before_ns &&
            submit.submit_time_ns >= logical.post_submit_not_before_ns) {
          post_submit_after_not_before_ns.Add(
              submit.submit_time_ns -
              logical.post_submit_not_before_ns);
        }
      }
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        if (logical.candidate.target_time_ns) {
          ++synthetic_post_submitted_after_anchor_total;
        } else {
          ++synthetic_post_submitted_before_anchor_total;
        }
        if (logical.synthetic_presentation_retired) {
          ++post_submitted_after_presentation_hold_total;
        }
      }
      PostJob& job = post_jobs[output_index];
      job.active = true;
      job.final_output_index = output_index;
      job.logical_index = logical_index;
      job.logical_token = logical.token;
      const uint64_t post_submit_ns =
          submit.submit_time_ns ? submit.submit_time_ns : PresenterMonotonicTimeNs();
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        logical.post_submit_time_ns = post_submit_ns;
      }
      if (logical.candidate.kind == CandidateKind::kSynthetic &&
          logical.post_submit_not_before_ns &&
          post_submit_ns < logical.post_submit_not_before_ns) {
        ++synthetic_post_eager_submit_total;
      }
      // H13: no live presentation deadline once presentation is retired.
      job.next_observation_time_ns = StageObservationTimeNs(
          post_submit_ns, PostObservationEstimateNs(),
          logical.synthetic_presentation_retired
              ? 0
              : logical.final_ready_soft_deadline_ns,
          logical.synthetic_presentation_retired
              ? 0
              : logical.candidate.target_time_ns);
      job.candidate = logical.candidate;
      job.submit = submit;
      job.completion_fence_fd = completion_fence_fd;
      job.timeline_fallback = accepted_timeline_fallback;
      job.release_output_on_completion = false;
      job.residency_recovery_early_transfer =
          residency_recovery_early_transfer;
      job.residency_recovery_compound_transfer =
          residency_recovery_compound_transfer;
      job.residency_recovery_post =
          residency_recovery_post;
      output.residency_recovery_debt.store(
          residency_recovery_early_transfer, std::memory_order_release);
      if (residency_recovery_early_transfer) {
        UpdateResidencyRecoveryDebtHighWater();
      }
      logical.final_output_index = output_index;
      output.post_completion_semaphore = submit.completion_semaphore;
      output.post_completion_value = submit.signal_value;
      const uint64_t final_ready_time_ns = PresenterMonotonicTimeNs();
      output.final_ready_time_ns = final_ready_time_ns;
      logical.final_ready_time_ns = final_ready_time_ns;
      if (split_physical_operating_point_enabled && logical.candidate.kind == CandidateKind::kReal &&
          logical.candidate.target_time_ns &&
          final_ready_time_ns < logical.candidate.target_time_ns) {
        physical_op_real_finalready_before_target_total.fetch_add(
            1, std::memory_order_release);
      }
      output.state.store(FinalOutputState::kReady,
                         std::memory_order_release);
      logical.state = LogicalOutputState::kFinalReady;
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        ObservePlannedSpaceFrontier(CountSyntheticProductionFrontier(),
                                    logical.candidate.source_id,
                                    logical.sequence_id, "s_final_ready");
      }
      // Phase 5 B2. A Post got out: whatever funding wait preceded it is over,
      // and its duration is the part of pre-submit residence that starvation
      // of the FinalOutput pool explains. This is the ONLY place a Post
      // reaches FinalReady, so it is the only close the normal topology ever
      // takes; the acquire-fence fail-open below overwrites this state but
      // runs after it, so the interval is already closed there.
      CloseFundingBlock();
      // H13: a presentation-retired ticket that reaches FinalReady is retired
      // by RetireCompletedPresentationRetiredSynthetics() on the next cycle
      // pass, never here — this submit path still owes the chained-generation,
      // fence and sample-window bookkeeping below, and must not be short-cut.
      if (logical.candidate.kind == CandidateKind::kSynthetic &&
          !logical.synthetic_presentation_retired) {
        if (!logical.candidate.target_time_ns) {
          // H12 MANDATORY: this S reached FinalReady before its pair-local
          // anchor existed, so it had no target, no soft deadline and no
          // planned dispatch. final_ready_soft_deadline_ns == 0 previously
          // fell into the else branch and was scored as a latency SUCCESS,
          // teaching D/decay from a contract that never existed. No temporal
          // contract yet means no temporal learning now: the recorded
          // FinalReady time is judged when the anchor arrives
          // (AnchorSyntheticPairsToCommittedReal).
          ++synthetic_final_ready_before_anchor_total;
        }
      }
      if (chained_generation) {
        chained_generation->chained_to_post = true;
        ++generation_to_post_chain_submit_total;
      }
      if (submit.acquire_fence_exported) {
        ++post_acquire_fence_ready_total;
      }
      post_queue_wait_ns.Add(submit.queue_wait_ns);
      post_submit_host_ns.Add(submit.submit_host_ns);
      post_export_host_ns.Add(submit.export_host_ns);
      last_post_owner_stats = submit.completion_owner_stats;
      if (submit.input_readiness_known) {
        if (submit.input_complete_at_submit) {
          post_submit_host_input_complete_ns.Add(submit.submit_host_ns);
        } else {
          post_submit_host_input_pending_ns.Add(submit.submit_host_ns);
        }
      }
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        post_submit_host_synthetic_ns.Add(submit.submit_host_ns);
      } else {
        post_submit_host_real_ns.Add(submit.submit_host_ns);
      }
      post_effect_count = submit.effect_count;
      ++post_submit_total;
      if (residency_recovery_early_transfer) {
        ++early_real_transfer_success_total;
        if (recovery_debt_before_submit) {
          ++residency_recovery_with_prior_debt_total;
        }
      }
      UpdateFinalOutputHighWater();
      post_in_flight_high_water =
          std::max(post_in_flight_high_water, CountPostJobs());
      progress = true;
      if (accepted_timeline_fallback) {
        // The Post's completion cannot be observed without waiting. Keep the
        // accepted GPU work represented as PostSubmitted so a same-cycle Pump
        // cannot apply it unproven; teardown retires the job first.
        job.release_output_on_completion = true;
        output.state.store(FinalOutputState::kPostProcessing,
                           std::memory_order_release);
        logical.state = LogicalOutputState::kPostSubmitted;
        Terminalize(TerminalReason::kPostProcessFailure,
                    logical.candidate.source_id,
                    PostFailureStage::kAcquireFenceExport, false, VK_SUCCESS,
                    true);
        return false;
      }
      requested_logical_index = UINT32_MAX;
      if (submit_one) {
        break;
      }
    }
    return progress;
  }

  bool SubmitResidencyRecoveryRealPost(uint32_t logical_index) {
    if (logical_index >= kLogicalOutputCapacity) {
      return false;
    }
    const LogicalOutput& recovery = logical_outputs[logical_index];
    const uint32_t free_final_outputs = CountFreeFinalOutputs();
    if (!free_final_outputs ||
        free_final_outputs <= RequiredFinalOutputSurplusFor(recovery)) {
      return false;
    }

    const uint32_t post_jobs_before = CountPostJobs();
    const bool progress = SubmitPostJobs(logical_index, true, true);
    if (CountPostJobs() <= post_jobs_before) {
      return progress;
    }
    return true;
  }

  bool ScheduleResidencyRecoveryForSourceIngress() {
    if (!ResidencyBackpressureRequiresProgress()) {
      return false;
    }

    // The measured Real post service is much cheaper than Generation. Prefer
    // a post that can directly free Residency before touching optional A/B
    // ownership or creating compound FinalOutput debt.
    const uint32_t direct_real =
        FindOldestDirectResidencyRecoveryRealPost();
    if (direct_real != UINT32_MAX &&
        SubmitResidencyRecoveryRealPost(direct_real)) {
      return true;
    }

    // Generation owns its A/B inputs until GPU completion. Once that optional
    // ownership is the shortest path to a physical Residency release, observe
    // it before parking more Reals in FinalOutput. This remains one driver
    // operation in the arbiter quantum.
    const uint32_t active_generation =
        FindOldestActiveResidencyRecoveryGeneration();
    if (active_generation != UINT32_MAX) {
      ++residency_recovery_generation_poll_total;
      const uint64_t releases_before =
          candidate_release_total.load(std::memory_order_relaxed);
      const DriverPollResult poll = PollGenerationJobs(active_generation);
      if (candidate_release_total.load(std::memory_order_relaxed) >
          releases_before) {
        ++residency_recovery_generation_physical_release_total;
      }
      return poll.progress;
    }

    const uint32_t waiting_generation =
        FindOldestWaitingResidencyRecoveryGeneration();
    if (waiting_generation != UINT32_MAX &&
        SyntheticChainSubmissionCapacityAvailable()) {
      ++residency_recovery_generation_attempt_total;
      LogicalOutput& synthetic = logical_outputs[waiting_generation];
      const uint64_t releases_before =
          candidate_release_total.load(std::memory_order_relaxed);

      const uint32_t jobs_before = CountGenerationJobs();
      const bool progress =
          SubmitWaitingGenerations(waiting_generation, true);
      if (CountGenerationJobs() > jobs_before) {
        ++residency_recovery_generation_submit_total;
      } else if (logical_outputs[waiting_generation].state ==
                 LogicalOutputState::kDropped) {
        ++residency_recovery_generation_drop_total;
      }
      if (candidate_release_total.load(std::memory_order_relaxed) >
          releases_before) {
        ++residency_recovery_generation_physical_release_total;
      }
      return progress;
    }

    const uint32_t logical_index =
        FindOldestCompoundResidencyRecoveryRealPost();
    if (logical_index == UINT32_MAX) {
      return false;
    }
    return SubmitResidencyRecoveryRealPost(logical_index);
  }

  DriverPollResult RunSourceIngressDriverOperation(
      uint32_t free_capture_slots) {
    DriverPollResult operation;
    const uint32_t used_capture_slots =
        kCapturePoolSize - free_capture_slots;
    // Turnover remains available from the first occupied Capture, but only a
    // genuinely low runway (one free slot) preempts an admitted Synthetic's
    // q0 commit. Earlier turnover participates in normal arbitration.
    const bool emergency = used_capture_slots >= kCapturePoolSize - 1;
    const bool proactive = used_capture_slots && !emergency;
    if (!emergency && !proactive) {
      return operation;
    }

    uint32_t oldest_capture = UINT32_MAX;
    uint64_t oldest_source = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < kCapturePoolSize; ++i) {
      const CaptureSlot& capture = capture_slots[i];
      const CaptureState state =
          capture.state.load(std::memory_order_acquire);
      if (state != CaptureState::kReadyWaitingResidency ||
          !capture.source_id) {
        continue;
      }
      if (capture.source_id < oldest_source) {
        oldest_capture = i;
        oldest_source = capture.source_id;
      }
    }
    if (proactive) {
      ++capture_proactive_turnover_attempt_total;
    } else {
      ++capture_emergency_progress_attempt_total;
    }
    if (oldest_capture != UINT32_MAX && CountFreeCandidateSlots()) {
      const uint32_t operations_before = arbiter_host_driver_ops_current;
      operation.progress = SubmitOneReadyCaptureTransfer();
      operation.invoked = arbiter_host_driver_ops_current != operations_before;
      if (operation.progress) {
        if (proactive) {
          ++capture_proactive_turnover_submit_total;
        } else {
          ++capture_emergency_progress_success_total;
        }
      }
      return operation;
    }

    // A transfer completion is part of Source ingress liveness, not optional
    // observation, once accepted Captures are consuming the reserve. Without
    // this path, a FinalReady head guard can suppress the normal Capture poll
    // indefinitely whenever there isn't also a ReadyWaitingResidency Capture
    // to submit in this exact cycle. Observe at most once, then return
    // authority to the ordered Pump like every other host-driver operation.
    if (emergency && CountCaptureTransfers()) {
      operation = PollCaptureTransfers();
      if (operation.progress) {
        ++capture_emergency_progress_success_total;
      }
      return operation;
    }

    if (oldest_capture == UINT32_MAX) {
      return operation;
    }

    if (ResidencyBackpressureRequiresProgress()) {
      const uint32_t operations_before = arbiter_host_driver_ops_current;
      operation.progress =
          ScheduleResidencyRecoveryForSourceIngress();
      operation.invoked = arbiter_host_driver_ops_current != operations_before;
      if (emergency && operation.progress) {
        ++capture_emergency_progress_success_total;
      }
    }
    return operation;
  }

  DriverPollResult PollPostJobs(
      uint32_t requested_logical_index = UINT32_MAX,
      bool only_if_observation_due = false) {
    DriverPollResult poll;
    PostJob* selected_job = nullptr;
    const uint64_t selection_now_ns = PresenterMonotonicTimeNs();
    const bool targeted = requested_logical_index < kLogicalOutputCapacity;
    if (targeted) {
      for (PostJob& job : post_jobs) {
        if (!job.active || job.logical_index != requested_logical_index) {
          continue;
        }
        selected_job = &job;
        break;
      }
      // Preserve the requested job as the sole authority for targeted polls.
      if (!selected_job) {
        return poll;
      }
    }
    if (!targeted) {
      uint64_t oldest_submit_ns = std::numeric_limits<uint64_t>::max();
      for (PostJob& job : post_jobs) {
        if (!job.active) {
          continue;
        }
        const uint64_t submit_ns =
            job.submit.submit_time_ns ? job.submit.submit_time_ns : 0;
        if (!selected_job || submit_ns < oldest_submit_ns) {
          selected_job = &job;
          oldest_submit_ns = submit_ns;
        }
      }
    }
    if (!selected_job) {
      return poll;
    }
    PostJob& job = *selected_job;
    poll.invoked = true;
    const bool used_timeline_fallback = job.timeline_fallback;
    PostCompletionEvidence completion_evidence =
        PostCompletionEvidence::kUnprovenFallback;
    if (used_timeline_fallback) {
      if (!vulkan_device->zerofg_teardown_idle())
        return poll;
      // Failure-only teardown path: sync_file export failed after Vulkan had
      // accepted the Post. The normal presentation path never executes this
      // driver query, but accepted ownership still needs a bounded completion
      // authority before resources are destroyed.
      uint64_t completed = 0;
      const bool teardown_idle = vulkan_device->zerofg_teardown_idle();
      const bool device_lost = vulkan_device->IsLost();
      if (!vulkan_device->zerofg_teardown_idle()) ++post_timeline_query_total;
      const uint64_t poll_begin_ns = PresenterMonotonicTimeNs();
      const VkResult timeline_result =
          teardown_idle
              ? (completed = job.submit.signal_value, VK_SUCCESS)
              : vulkan_device->vkGetSemaphoreCounterValue()(
                    vulkan_device->device(), job.submit.completion_semaphore,
                    &completed);
      active_post_poll_host_ns.Add(PresenterMonotonicTimeNs() - poll_begin_ns);
      ++active_post_poll_total;
      if (timeline_result != VK_SUCCESS) {
        if (job.logical_index < kLogicalOutputCapacity &&
            logical_outputs[job.logical_index].token == job.logical_token) {
          RecordPostCompletionReady(
              logical_outputs[job.logical_index], 0,
              PostCompletionEvidence::kObservationError);
        }
        job.next_observation_time_ns = SaturatingAddNs(
            selection_now_ns, kObservationDueRetryIntervalNs);
        return poll;
      }
      if (completed < job.submit.signal_value) {
        job.next_observation_time_ns =
            SaturatingAddNs(selection_now_ns, kReleaseFencePollIntervalNs);
        return poll;
      }
      completion_evidence =
          device_lost ? PostCompletionEvidence::kObservationError
                      : (teardown_idle
                             ? PostCompletionEvidence::kTeardownIdle
                             : PostCompletionEvidence::kTimeline);
      job.timeline_fallback = false;
    }
    bool post_fence_observed_signaled = false;
    if (!used_timeline_fallback) {
      ++post_sync_fd_poll_total;
      const bool teardown_idle = vulkan_device->zerofg_teardown_idle();
      bool post_fence_observation_error = false;
      if (job.completion_fence_fd >= 0 && !teardown_idle &&
          !IsReleaseFenceReady(job.completion_fence_fd,
                               &post_fence_observed_signaled,
                               &post_fence_observation_error)) {
        if (post_fence_observation_error &&
            job.logical_index < kLogicalOutputCapacity &&
            logical_outputs[job.logical_index].token == job.logical_token) {
          RecordPostCompletionReady(
              logical_outputs[job.logical_index], 0,
              PostCompletionEvidence::kObservationError);
        }
        if (job.candidate.kind == CandidateKind::kSynthetic) {
          ++split_apply_fence_wait_synthetic_total;
        } else {
          ++split_apply_fence_wait_real_total;
        }
        job.next_observation_time_ns =
            SaturatingAddNs(selection_now_ns, kReleaseFencePollIntervalNs);
        return poll;
      }
      if (job.completion_fence_fd >= 0) {
        close(job.completion_fence_fd);
        job.completion_fence_fd = -1;
      }
      completion_evidence =
          teardown_idle
              ? (vulkan_device->IsLost()
                     ? PostCompletionEvidence::kObservationError
                     : PostCompletionEvidence::kTeardownIdle)
              : (post_fence_observed_signaled
                     ? PostCompletionEvidence::kSyncFile
                     : (post_fence_observation_error
                            ? PostCompletionEvidence::kObservationError
                            : PostCompletionEvidence::kUnprovenFallback));
    }
    poll.progress = true;
    if (!used_timeline_fallback) {
      ++post_sync_fd_ready_total;
    }
    const uint64_t completion_time_ns = PresenterMonotonicTimeNs();
    if (job.candidate.kind == CandidateKind::kReal &&
        job.candidate.real_slot < kPoolSize) {
      const auto& observation = slots[job.candidate.real_slot].handoff_ready;
      if (observation) {
        const uint64_t ready = observation->Observe();
        // Both times are host first-observation bounds, not exact GPU times.
        // Sampling N may itself be its first successful poll in this cycle.
        const uint64_t post_observed_ns = PresenterMonotonicTimeNs();
        if (ready && ready <= post_observed_ns) {
          device_b_post_after_n_ready_ns.Add(post_observed_ns - ready);
          if (ready >= observation->publish_ns) {
            device_b_n_first_seen_ns.Add(ready - observation->publish_ns);
          }
          if (++device_b_post_observations % 120 == 0) {
            XELOGI(
                "ZeroFGDeviceB completion_observations={} "
                "N_first_seen_ready_ns={} "
                "source={} N_publish_to_first_seen_us_p90={} "
                "post_after_n_ready_us_p50/p90/max={}/{}/{} "
                "ingress_submit_host_us_p90={} domain=presenter_B",
                device_b_post_observations, ready, observation->source_id,
                device_b_n_first_seen_ns.Quantile(90, 100) / 1000,
                device_b_post_after_n_ready_ns.Quantile(50, 100) / 1000,
                device_b_post_after_n_ready_ns.Quantile(90, 100) / 1000,
                device_b_post_after_n_ready_ns.maximum() / 1000,
                ingress_submit_host_ns.Quantile(90, 100) / 1000);
          }
        }
      }
    }
    const uint64_t submit_to_ready_ns =
        job.submit.submit_time_ns &&
                completion_time_ns >= job.submit.submit_time_ns
            ? completion_time_ns - job.submit.submit_time_ns
            : 0;
    post_submit_to_ready_ns.Add(submit_to_ready_ns);
    post_gpu_completion_ns.Add(submit_to_ready_ns);
    post_process_release_callback(job.final_output_index);
    FinalOutputSlot& output = final_outputs[job.final_output_index];
    if (job.candidate.kind == CandidateKind::kSynthetic) {
      if (job.candidate.synthetic_index >= kSyntheticPoolSize) {
        Terminalize(TerminalReason::kInvalidHandoff,
                    job.candidate.source_id);
      } else {
        GenerationJob& generation =
            generation_jobs[job.candidate.synthetic_index];
        if (generation.active && generation.chained_to_post &&
            generation.logical_index == job.logical_index &&
            generation.logical_token == job.logical_token) {
          if (!used_timeline_fallback) {
            ObserveSyntheticChainResidence(
                generation.chain_submit_time_ns, completion_time_ns,
                generation.synthetic_chain_warmup_submission);
          }
          ReleaseRealSlot(generation.previous_real_slot);
          ReleaseRealSlot(generation.current_real_slot);
          generation.active = false;
          generation.chained_to_post = false;
          ++synthetic_generated_total;
          ++generation_to_post_chain_retired_total;
          // A Generation chained into Post never reaches the CPU completion
          // poll, so it counts toward the bootstrap here too; otherwise the
          // bootstrap (telemetry only) reports warmup forever.
          if (!generation_bootstrap_armed &&
              ++generation_bootstrap_success_total >=
                  kGenerationBootstrapSamples) {
            generation_bootstrap_armed = true;
          }
        }
        ReleaseSyntheticBackingOwnership(job.candidate.synthetic_index);
        synthetic_release_callback(job.candidate.synthetic_index);
        synthetic_pool_occupancy = CountGenerationJobs();
      }
    } else {
      const uint64_t release_total_before =
          candidate_release_total.load(std::memory_order_relaxed);
      ReleaseRealSlot(job.candidate.real_slot);
      const bool physically_released =
          candidate_release_total.load(std::memory_order_relaxed) >
          release_total_before;
      if (job.residency_recovery_post && physically_released) {
        ++residency_recovery_post_physical_release_total;
      }
      if (job.residency_recovery_early_transfer) {
        if (physically_released) {
          ++early_real_transfer_actual_release_total;
        } else if (job.residency_recovery_compound_transfer) {
          // The matching generation still owns the final Candidate ref.
          // Its completion is the second half of this bounded recovery
          // chain and will publish the physical release event.
          ++early_real_transfer_compound_pending_total;
        } else {
          // This should be prevented by release-capable admission. Keep the
          // event visible in case ownership changed unexpectedly after
          // submission rather than treating the post as a valid credit.
          ++early_real_transfer_no_release_total;
        }
      }
    }
    const uint32_t logical_index = job.logical_index;
    const uint64_t logical_token = job.logical_token;
    const uint32_t final_output_index = job.final_output_index;
    if (logical_index < kLogicalOutputCapacity &&
        logical_outputs[logical_index].token == logical_token) {
      // Preserve the lifecycle stamp and downstream bookkeeping in both B1
      // modes. B1 gates only controller scoring on the evidence class.
      RecordPostCompletionReady(logical_outputs[logical_index],
                                completion_time_ns, completion_evidence);
    }
    const bool release_output = job.release_output_on_completion;
    job.active = false;
    job.release_output_on_completion = false;
    if (release_output || logical_index >= kLogicalOutputCapacity ||
        (logical_outputs[logical_index].token != logical_token &&
         output.state.load(std::memory_order_acquire) !=
             FinalOutputState::kTransactionApplied)) {
      ReleaseFinalOutputSlotNow(final_output_index);
    }
    if (generation_service_gpu_ns.count() >= kGenerationBootstrapSamples &&
        post_service_gpu_ns.count() >= kGenerationBootstrapSamples &&
        source_period_ns) {
      const bool service_pressure =
          GenerationCostEstimateNs() + PostCostEstimateNs() * 2 >
          source_period_ns;
      if (service_pressure && !generation_unsustainable) {
        ++generation_unsustainable_total;
      }
      generation_unsustainable = service_pressure;
    }
    return poll;
  }

  DriverPollResult PollHeadCriticalGpuJob() {
    LogicalOutput* head = FindLogicalBySequence(next_apply_sequence);
    if (!head) {
      return {};
    }
    const uint32_t head_index = uint32_t(head - logical_outputs.data());
    DriverPollResult poll;
    if (head->state == LogicalOutputState::kWaitingGeneration &&
               LogicalHasJob(head_index, head->token)) {
      poll = PollGenerationJobs(head_index, true);
    } else {
      return poll;
    }
    if (poll.deferred_not_due) {
      ++head_observation_not_due_total;
    }
    return poll;
  }

  DriverPollResult PollOneNormalDriverObservation() {
    // Capture readiness is carried by GPU dependencies. Normal polling is
    // retained only for Generation jobs that require CPU-side retirement.
    return PollGenerationJobs(UINT32_MAX, true);
  }

  bool CaptureReclamationObservationRequired() const {
    if (!CountCaptureTransfers()) {
      return false;
    }
    const bool ingress_context_starved =
        !incoming_real.occupied() &&
        ingress_publication_pending.load(std::memory_order_acquire) &&
        FindFreeIngressContext() == UINT32_MAX;
    const bool residency_reclamation_needed =
        incoming_real.occupied() && CountCapturesWaitingForResidency() &&
        !CountFreeCandidateSlots();
    return ingress_context_starved || residency_reclamation_needed;
  }

  bool SubmitWaitingGenerations(uint32_t requested_logical_index = UINT32_MAX,
                                bool submit_one = false) {
    bool progress = false;
    for (uint32_t iteration = 0; iteration < kLogicalOutputCapacity;
         ++iteration) {
      uint32_t i = requested_logical_index;
      if (i >= kLogicalOutputCapacity ||
          logical_outputs[i].state !=
              LogicalOutputState::kWaitingGeneration ||
          LogicalHasJob(i, logical_outputs[i].token)) {
        i = FindOldestWaitingGeneration();
      }
      if (i == UINT32_MAX) {
        break;
      }
      LogicalOutput& logical = logical_outputs[i];
      if (!GenerationJitActionable(logical, PresenterMonotonicTimeNs())) {
        break;
      }
      const uint32_t before = CountGenerationJobs();
      if (!SubmitGenerationJob(i, logical.previous_real_slot,
                               logical.current_real_slot)) {
        return false;
      }
      const bool submitted = logical.state == LogicalOutputState::kDropped ||
                             CountGenerationJobs() != before;
      progress |= submitted;
      requested_logical_index = UINT32_MAX;
      if (submit_one || !submitted) {
        break;
      }
    }
    return progress;
  }

  bool ScheduleGpuWork() {
    bool progress = false;
    constexpr uint64_t kArbitrationMarginNs = 500000ull;
    // One arbiter quantum may start at most one non-preemptible host/GPU
    // operation. The outer presenter loop pumps the ordered head and refreshes
    // time before granting the next quantum.
    for (uint32_t iteration = 0; iteration < 1; ++iteration) {
      uint32_t post_index = UINT32_MAX;
      const bool generation_capacity_available =
          SyntheticChainSubmissionCapacityAvailable();
      if (!generation_capacity_available) {
        MarkSyntheticsWaitingForProductionCapacity();
      }
      uint32_t generation_index = generation_capacity_available
                                      ? FindOldestWaitingGeneration()
                                      : UINT32_MAX;
      LogicalOutput* head = FindLogicalBySequence(next_apply_sequence);
      bool actionable_head_selected = false;
      const uint64_t now_ns = PresenterMonotonicTimeNs();

      // The actionable logical head is the liveness authority. Once its work
      // has been submitted, independent work behind it may be arbitrated, but
      // a later Real post must not consume GPU/final-output capacity while an
      // unsubmitted head Synthetic is the only operation that can unblock the
      // ordered stream.
      if (head) {
        const uint32_t head_index = uint32_t(head - logical_outputs.data());
        if (head->state == LogicalOutputState::kWaitingPost &&
            PostJitActionable(*head, now_ns)) {
          post_index = head_index;
          actionable_head_selected = true;
        } else if (head->state == LogicalOutputState::kWaitingGeneration &&
                   !LogicalHasJob(head_index, head->token) &&
                   GenerationJitActionable(*head, now_ns)) {
          actionable_head_selected = true;
          // A full bounded generation pipeline is transient pressure, not a
          // reason to burn the head immediately. Completion polling and its
          // existing cutoff remain live while no later work bypasses it.
          generation_index = generation_capacity_available ? head_index
                                                            : UINT32_MAX;
          if (!generation_capacity_available) {
            ++split_s_waiting_for_physical_chain_total;
          }
        }
      }

      if (!actionable_head_selected) {
        const uint32_t oldest_post = FindOldestWaitingPost();
        if (oldest_post == UINT32_MAX) {
          // No post work is ready, so the oldest optional generation is the
          // only actionable GPU stage.
        } else if (generation_index == UINT32_MAX) {
          post_index = oldest_post;
        } else {
          const LogicalOutput& post = logical_outputs[oldest_post];
          const LogicalOutput& generation =
              logical_outputs[generation_index];
          if (generation.sequence_id < post.sequence_id) {
            // Preserve logical liveness across kinds: an earlier Synthetic
            // generation is not bypassed by a later Real post.
          } else if (post.candidate.kind == CandidateKind::kReal) {
            // An older accepted Real post may beat a newer optional
            // generation, but never an earlier already-generated Synthetic.
            post_index = oldest_post;
          } else {
            // The older Synthetic is already generated and has paid its main
            // cost. A newer generation may proceed only while the measured
            // cost model, including bounded work already resident, says it
            // cannot jeopardize either post deadline.
            const uint64_t now = PresenterMonotonicTimeNs();
            const uint64_t generation_service = GenerationCostEstimateNs();
            const uint64_t post_service = PostCostEstimateNs();
            // H13: the old model added GenerationResidenceEstimateNs(), which
            // is observed submit-to-CPU-observation wall clock and therefore
            // contains this scheduler's own observation delay. A late poll
            // inflated the residence, which inflated this block estimate,
            // which starved Generation further — a self-referential ratchet
            // that manufactured fake occupancy. Only bounded service cost of
            // the candidate plus the work already resident is causal here.
            // This stays a priority heuristic; it is never eligibility,
            // admission or drop authority.
            const uint64_t generation_block =
                generation_service +
                generation_service * CountGenerationJobs() +
                post_service * CountPostJobs() + kArbitrationMarginNs;
            const uint64_t post_completion =
                post_service + kArbitrationMarginNs;
            const uint64_t submit_slack =
                post.post_submit_soft_deadline_ns > now
                    ? post.post_submit_soft_deadline_ns - now
                    : 0;
            const uint64_t ready_slack =
                post.final_ready_soft_deadline_ns > now
                    ? post.final_ready_soft_deadline_ns - now
                    : 0;
            // A presentation-retired Post has no deadline left to be urgent
            // about; its expired soft deadlines would otherwise read as zero
            // slack and permanently pre-empt Generation.
            if (!post.synthetic_presentation_retired &&
                (submit_slack <= generation_block ||
                 ready_slack <= generation_block + post_completion)) {
              post_index = oldest_post;
              ++arbitration_post_urgent_total;
            }
          }
        }
      }

      // P3, arbiter half. Corrected twice, and the second correction is the
      // real defect: the dispatch below selects a Post, the submit refuses it
      // on FinalOutput funding, and the quantum is spent -- the loop breaks
      // without ever trying the Generation that was already selectable. A
      // blocked Post starves Production while the physical pool sits idle.
      //
      // The first version instead inverted the priority, pre-empting a Post to
      // start a Generation. That was wrong in both directions: a Real Post
      // frees Residency and a Synthetic Post's completion releases backing, so
      // BOTH reduce the accepted Real's debt, while a Generation is neutral on
      // the frontier -- it only trades an obligation for owned backing. It
      // regressed Rayman from about 25 s of production hold to 112.9 s and
      // pinned the plant to one chain cycle per Source period. Removed.
      //
      // What is left is not a priority at all: do not waste the quantum.
      if (SidecarProgressCritical() && generation_index == UINT32_MAX &&
          !CountGenerationJobs()) {
        // Progress-critical, chain idle, nothing selectable to start. If this
        // dominates, the stall is upstream of arbitration.
        ++arbitration_progress_no_candidate_total;
      }
      bool iteration_progress = false;
      if (post_index != UINT32_MAX) {
        iteration_progress = SubmitPostJobs(post_index, true);
        if (iteration_progress) {
          ++arbitration_post_selected_total;
        }
      }
      if (!iteration_progress && generation_index != UINT32_MAX) {
        // The Post either was not selected or could not be submitted. Either
        // way the quantum is still unspent, and an already-selected Generation
        // is real work. Candidate-neutral and no pre-emption: a Post that CAN
        // run still runs first.
        const bool post_was_blocked = post_index != UINT32_MAX;
        iteration_progress =
            SubmitWaitingGenerations(generation_index, true);
        if (iteration_progress) {
          ++arbitration_generation_selected_total;
          if (post_was_blocked) {
            ++arbitration_generation_after_blocked_post_total;
          }
        }
      }
      progress |= iteration_progress;
      if (!iteration_progress) {
        break;
      }
    }
    return progress;
  }

  // The presentation advance (1.0.1): the semantic target is unchanged (lattice,
  // order, commitments and the contract keep it), the physical presentation
  // goes this much earlier. Every dispatch decision and the egress's desired
  // time read the target through here.
  uint64_t PresentationTargetNs(const LogicalOutput& logical) const {
    const uint64_t target_ns = logical.candidate.target_time_ns;
    if (!presentation_advance_ns || !target_ns) {
      return target_ns;
    }
    return target_ns > presentation_advance_ns
               ? target_ns - presentation_advance_ns
               : 1;
  }

  // The drain of the latency well without skipping anything. Observed at
  // every applied output, R and S. The advance grows one panel refresh after
  // kPresentationAdvanceRiseStreak Reals in a row that still reach the panel
  // two refreshes or more past Better D's operating depth, with every output
  // in that streak, S included, FinalReady two refreshes or more before its
  // advanced planned dispatch. It shrinks one refresh after
  // kPresentationAdvanceLateStreak outputs in a row late at their advanced
  // dispatch, at once on a capacity-gate refusal, or after
  // kPresentationAdvanceRiseStreak Reals below the operating depth. At most one
  // move every kPresentationAdvanceEvery outputs, never past
  // kPresentationAdvanceMaxNs. A step changes only the next interval.
  //
  // S counts because an S needs its B captured: an advance that asks S before
  // it can exist hangs S obligations, the capacity gate then holds the next
  // Reals, and the phase it was draining comes back larger. 2026-10-08, Halo 3
  // hot at 15 fps: watching Reals only, the advance climbed to its 400 ms cap
  // while the gate went 0 -> 15515 refusals and the latency to ~690 ms.
  // Before that, an undamped form stepped every second output (576 up, 580
  // down in three minutes).
  void ObservePresentationAdvance(const LogicalOutput& logical,
                                  uint64_t planned_dispatch_ns) {
    ++presentation_advance_outputs_since;
    if (!kPresentationAdvanceEnabled || !stable_output_quantum_ns ||
        !main_surface_egress) {
      return;
    }
    const uint64_t refresh_ns = main_surface_egress->RefreshCycleNs();
    if (!refresh_ns || stable_output_quantum_ns <= refresh_ns + refresh_ns / 8) {
      // No vsync to gain inside one output quantum.
      return;
    }
    const int64_t band_ns = int64_t(refresh_ns) * 2;
    // How long before its (advanced) planned dispatch the output was
    // FinalReady; negative: it was late.
    const int64_t ready_slack_ns =
        logical.final_ready_time_ns
            ? PresenterSignedDeltaNs(planned_dispatch_ns,
                                     logical.final_ready_time_ns)
            : 0;
    const bool capacity_refused =
        split_preaccept_capacity_gate_total != presentation_advance_gate_seen;
    presentation_advance_gate_seen = split_preaccept_capacity_gate_total;
    const bool late = logical.final_ready_time_ns && ready_slack_ns < 0;
    presentation_advance_late_streak =
        late ? presentation_advance_late_streak + 1 : 0;
    if (ready_slack_ns < band_ns || capacity_refused) {
      presentation_advance_up_streak = 0;
    }
    const bool real = logical.candidate.kind == CandidateKind::kReal &&
                      logical.candidate.issue_time_ns && operating_latency_ns;
    if (real) {
      const int64_t excess_ns = PresenterSignedDeltaNs(
          PresentationTargetNs(logical),
          SaturatingAddNs(logical.candidate.issue_time_ns,
                          operating_latency_ns));
      presentation_advance_excess_max_ns =
          std::max(presentation_advance_excess_max_ns,
                   excess_ns > 0 ? uint64_t(excess_ns) : 0);
      if (excess_ns >= band_ns && ready_slack_ns >= band_ns &&
          !capacity_refused) {
        ++presentation_advance_up_streak;
      } else {
        presentation_advance_up_streak = 0;
      }
      presentation_advance_below_streak =
          excess_ns < 0 ? presentation_advance_below_streak + 1 : 0;
    }
    if (presentation_advance_ns && capacity_refused) {
      ++presentation_advance_capacity_down;
    } else if (presentation_advance_outputs_since < kPresentationAdvanceEvery) {
      return;
    }
    if (presentation_advance_ns &&
        (capacity_refused ||
         presentation_advance_late_streak >= kPresentationAdvanceLateStreak ||
         presentation_advance_below_streak >=
             kPresentationAdvanceRiseStreak)) {
      presentation_advance_ns = presentation_advance_ns > refresh_ns
                                    ? presentation_advance_ns - refresh_ns
                                    : 0;
      presentation_advance_outputs_since = 0;
      presentation_advance_late_streak = 0;
      presentation_advance_below_streak = 0;
      presentation_advance_up_streak = 0;
      ++presentation_advance_steps_down;
      return;
    }
    if (real &&
        presentation_advance_up_streak >= kPresentationAdvanceRiseStreak &&
        presentation_advance_ns + refresh_ns <= kPresentationAdvanceMaxNs) {
      presentation_advance_ns += refresh_ns;
      presentation_advance_outputs_since = 0;
      presentation_advance_up_streak = 0;
      ++presentation_advance_steps_up;
      presentation_advance_max_ns =
          std::max(presentation_advance_max_ns, presentation_advance_ns);
    }
  }

  uint64_t PlannedDispatchTimeNs(const LogicalOutput& logical) const {
    const uint64_t target_ns = PresentationTargetNs(logical);
    return target_ns > logical.dispatch_lead_ns
               ? target_ns - logical.dispatch_lead_ns
               : target_ns;
  }


  size_t BlockingHazardEvidenceSamples() const {
    return capture_transfer_submit_host_ns.count() + post_host_block_ns.count() +
           generation_host_block_ns.count() +
           active_capture_poll_host_ns.count() +
           active_generation_poll_host_ns.count() +
           active_post_poll_host_ns.count();
  }

  uint64_t BlockingHazardLeadNs() const {
    if (!stable_output_quantum_ns) {
      return kBlockingHazardFloorNs;
    }
    const uint64_t bootstrap_ns = std::max<uint64_t>(
        kBlockingHazardFloorNs,
        std::min<uint64_t>(kBlockingHazardBootstrapNs,
                           stable_output_quantum_ns / 2));
    if (BlockingHazardEvidenceSamples() < kBlockingHazardArmSamples) {
      return std::min(bootstrap_ns, stable_output_quantum_ns);
    }
    const uint64_t capture_submit_ns =
        capture_transfer_submit_host_ns.count()
            ? capture_transfer_submit_host_ns.Quantile(90, 100)
            : 0;
    const uint64_t post_ns = post_host_block_ns.count()
                                 ? post_host_block_ns.Quantile(90, 100)
                                 : 0;
    const uint64_t generation_ns =
        generation_host_block_ns.count()
            ? generation_host_block_ns.Quantile(90, 100)
            : 0;
    const uint64_t submit_risk_ns =
        std::max({capture_submit_ns, post_ns, generation_ns});
    // Poll APIs are non-waiting in Vulkan semantics, but runtime demonstrated
    // material host tails in the driver. Observe only active calls (never
    // empty polling cycles) and reserve for their P99 tail, still capped to one
    // semantic quantum so Source-first cannot turn an outlier into a long
    // output guard.
    const uint64_t capture_poll_ns =
        active_capture_poll_host_ns.count()
            ? active_capture_poll_host_ns.Quantile(99, 100)
            : 0;
    const uint64_t generation_poll_ns =
        active_generation_poll_host_ns.count()
            ? active_generation_poll_host_ns.Quantile(99, 100)
            : 0;
    const uint64_t post_poll_ns =
        active_post_poll_host_ns.count()
            ? active_post_poll_host_ns.Quantile(99, 100)
            : 0;
    const uint64_t poll_risk_ns =
        std::max({capture_poll_ns, generation_poll_ns, post_poll_ns});
    return std::clamp<uint64_t>(
        SaturatingAddNs(std::max(submit_risk_ns, poll_risk_ns),
                        kBlockingHazardMarginNs),
        std::min(kBlockingHazardFloorNs, stable_output_quantum_ns),
        stable_output_quantum_ns);
  }

  // Cold-wake latency of the first wake before a dispatch (guard-start
  // overshoot is measured only on deadline returns, so it is OS wake
  // latency, not busy time).
  uint64_t SplitColdWakeMarginNs() const {
    const auto& cold = presenter_wait_overshoot_ns[size_t(
        PresenterWakeReason::kGuardStart)];
    if (cold.count() < kBlockingHazardArmSamples) {
      return kBlockingHazardBootstrapNs;
    }
    return std::min(kSplitColdWakeMarginMaxNs,
                    SaturatingAddNs(cold.Quantile(90, 100),
                                    kSplitColdWakePadNs));
  }

  uint64_t BlockingHazardGuardStartNs(
      const LogicalOutput& logical) const {
    const uint64_t planned_dispatch_ns = PlannedDispatchTimeNs(logical);
    uint64_t hazard_lead_ns = BlockingHazardLeadNs();
    // Split pre-wake: the guard start also covers the cold-wake latency.
    hazard_lead_ns = std::max(hazard_lead_ns, SplitColdWakeMarginNs());
    return planned_dispatch_ns > hazard_lead_ns
               ? planned_dispatch_ns - hazard_lead_ns
               : 0;
  }

  const LogicalOutput* FindHeadAwaitingPlannedDispatch(uint64_t now_ns) const {
    const LogicalOutput* head = FindLogicalBySequence(next_apply_sequence);
    if (!head || head->state != LogicalOutputState::kFinalReady ||
        !head->candidate.target_time_ns) {
      return nullptr;
    }
    const uint64_t planned_dispatch_ns = PlannedDispatchTimeNs(*head);
    if (!planned_dispatch_ns || now_ns >= planned_dispatch_ns) {
      return nullptr;
    }
    const uint64_t guard_start_ns = BlockingHazardGuardStartNs(*head);
    if (now_ns < guard_start_ns) {
      return nullptr;
    }
    return head;
  }

  void ObserveHardReadyDispatchMiss(LogicalOutput& logical, uint64_t now_ns) {
    if (logical.candidate.kind != CandidateKind::kReal ||
        logical.state != LogicalOutputState::kFinalReady ||
        !logical.final_ready_time_ns ||
        !logical.candidate.target_time_ns) {
      return;
    }
    const uint64_t planned_dispatch_time_ns = PlannedDispatchTimeNs(logical);
    // This is deliberately narrower than a generic hard Real miss. The fully
    // rendered output must have been available before the planned host-dispatch
    // window, proving that GPU, post, funding and ordering readiness did not
    // cause the miss. ApplyFinalOutput has not started on this path, so the
    // sample can't duplicate a transaction-preparation-crossed-target miss.
    if (!planned_dispatch_time_ns ||
        logical.final_ready_time_ns > planned_dispatch_time_ns ||
        now_ns <= planned_dispatch_time_ns) {
      return;
    }

    const uint64_t dispatch_lateness_ns =
        now_ns - planned_dispatch_time_ns;
    hard_ready_dispatch_late_ns.Add(dispatch_lateness_ns);
    ++dispatch_lead_hard_ready_miss_sample_total;
    const bool use_current_blocking_operation =
        arbiter_last_blocking_operation != BlockingOperation::kNone;
    const BlockingOperation attributed_operation =
        use_current_blocking_operation
            ? arbiter_last_blocking_operation
            : previous_cycle_blocking_operation;
    const uint64_t attributed_begin_ns =
        use_current_blocking_operation
            ? arbiter_last_blocking_begin_ns
            : previous_cycle_blocking_begin_ns;
    const uint64_t attributed_end_ns =
        use_current_blocking_operation
            ? arbiter_last_blocking_end_ns
            : previous_cycle_blocking_end_ns;
    const size_t blocking_index = size_t(attributed_operation);
    ++hard_ready_miss_by_blocking_operation[blocking_index];
    const bool crossed_planned =
        attributed_begin_ns && attributed_end_ns &&
        attributed_begin_ns < planned_dispatch_time_ns &&
        attributed_end_ns >= planned_dispatch_time_ns;
    if (crossed_planned) {
      ++hard_ready_miss_crossed_by_blocking_operation[blocking_index];
    }
    ++hard_ready_miss_after_wake_reason[
        size_t(last_presenter_wake_reason)];
    if (hard_ready_miss_attribution_sample_total < 8) {
      ++hard_ready_miss_attribution_sample_total;
    }

    RefreshDispatchLeadEstimate();

    // E1 keeps an ordinary late FinalReady Real at its desired target. Only a
    // separate structural fallback may still use the existing Real-defer
    // operation, and that operation must not carry the disproven lead into
    // its next tick.
    logical.dispatch_lead_ns =
        std::max(logical.dispatch_lead_ns, DispatchLeadEstimateNs());
  }

  void PropagateLogicalTiming(uint32_t logical_index) {
    LogicalOutput& logical = logical_outputs[logical_index];
    ConfigureLogicalStaging(
        logical, logical.candidate.kind == CandidateKind::kSynthetic);
    if (logical.final_output_index < kFinalOutputPoolSize) {
      FinalOutputSlot& output = final_outputs[logical.final_output_index];
      const FinalOutputState state =
          output.state.load(std::memory_order_acquire);
      if (state == FinalOutputState::kPostProcessing ||
          state == FinalOutputState::kReady) {
        output.semantic_target_time_ns =
            logical.candidate.semantic_target_time_ns;
        output.target_time_ns = logical.candidate.target_time_ns;
        output.semantic_epoch_origin_ns =
            logical.candidate.semantic_epoch_origin_ns;
        output.semantic_output_quantum_ns =
            logical.candidate.semantic_output_quantum_ns;
        output.semantic_tick_index = logical.candidate.semantic_tick_index;
        output.final_ready_soft_deadline_ns =
            logical.final_ready_soft_deadline_ns;
      }
    }
    for (PostJob& job : post_jobs) {
      if (job.active && job.logical_index == logical_index &&
          job.logical_token == logical.token) {
        job.candidate.semantic_target_time_ns =
            logical.candidate.semantic_target_time_ns;
        job.candidate.target_time_ns = logical.candidate.target_time_ns;
        job.candidate.semantic_epoch_origin_ns =
            logical.candidate.semantic_epoch_origin_ns;
        job.candidate.semantic_output_quantum_ns =
            logical.candidate.semantic_output_quantum_ns;
        job.candidate.semantic_tick_index =
            logical.candidate.semantic_tick_index;
      }
    }
  }

  void AdvanceSemanticCursorToTick(const OutputCandidate& candidate,
                                   uint64_t next_tick_index) {
    if (!stable_latency_metronome_armed ||
        candidate.semantic_epoch_origin_ns != semantic_epoch_origin_ns ||
        candidate.semantic_output_quantum_ns != stable_output_quantum_ns ||
        next_semantic_tick_index >= next_tick_index) {
      return;
    }
    next_semantic_tick_index = next_tick_index;
    next_semantic_target_ns =
        SemanticTargetForTick(next_semantic_tick_index);
  }

  // Real first-commitment cause telemetry. Observation only: none of it is a
  // pacing, HOLD or commitment input.
  enum class OrderedBoundaryKind : uint8_t {
    kNone,
    kAppliedReal,
    kAppliedSynthetic,
    kExplicitSHold,
    kImplicitHoldBeforeCandidate,
  };
  struct OrderedBoundaryProvenance {
    OrderedBoundaryKind kind = OrderedBoundaryKind::kNone;
    uint64_t source_id = 0;
    uint64_t sequence_id = 0;
    uint64_t tick = 0;
    uint64_t target_ns = 0;
  };
  // The floors the ordinary projection compared.
  struct RealProjectionDiagnostics {
    uint64_t nominal_tick = 0;
    uint64_t ordered_tick = 0;  // Zero: no ordered-boundary floor.
    uint64_t now_tick = 0;      // Zero: no live floor.
    uint64_t assigned_tick = 0;
    // Every floor equal to assigned_tick: 1 nominal/eligibility, 2 ordered
    // boundary, 4 now/live. Ties are never broken.
    uint32_t winner_mask = 0;
  };
  static constexpr uint32_t kRealCommitRoutePairBoundary = 0;
  static constexpr uint32_t kRealCommitRouteProjection = 2;

  static const char* OrderedBoundaryKindName(OrderedBoundaryKind kind) {
    switch (kind) {
      case OrderedBoundaryKind::kAppliedReal:
        return "applied_real";
      case OrderedBoundaryKind::kAppliedSynthetic:
        return "applied_synthetic";
      case OrderedBoundaryKind::kExplicitSHold:
        return "explicit_s_hold";
      case OrderedBoundaryKind::kImplicitHoldBeforeCandidate:
        return "implicit_hold_before_candidate";
      default:
        return "none";
    }
  }

  static const char* RealCommitRouteName(uint32_t route) {
    return route == kRealCommitRoutePairBoundary ? "pair_boundary"
                                                 : "projection";
  }

  // Which update last carried the ordered boundary (the max of the applied
  // and accounted semantic targets) forward.
  void RecordOrderedBoundaryProvenance(OrderedBoundaryKind kind,
                                       uint64_t source_id,
                                       uint64_t sequence_id, uint64_t tick,
                                       uint64_t target_ns) {
    if (target_ns < ordered_boundary_provenance.target_ns) {
      return;
    }
    ordered_boundary_provenance = {kind, source_id, sequence_id, tick,
                                   target_ns};
  }

  // Decomposes a Real's future runway at the instant of its first
  // commitment: how long it was already FinalReady, which route and floor set
  // its tick, and what carried the ordered boundary.
  void ObserveRealCommitCause(
      const LogicalOutput& logical, uint64_t now_ns, uint32_t route,
      const RealProjectionDiagnostics& projection, uint64_t nominal_tick,
      uint64_t assigned_tick, uint64_t ordered_boundary_ns,
      const OrderedBoundaryProvenance& boundary, bool rolling_pair_head,
      bool rolling_pair_epoch_current, uint64_t pair_source_distance,
      uint64_t pair_lattice_distance, uint64_t pair_boundary_tick,
      uint64_t executive_debt_ticks) {
    const OutputCandidate& candidate = logical.candidate;
    const bool ready_at_commit =
        logical.final_ready_time_ns && now_ns >= logical.final_ready_time_ns;
    const uint64_t ready_age_ns =
        ready_at_commit ? now_ns - logical.final_ready_time_ns : 0;
    const int64_t commit_runway_ns =
        PresenterSignedDeltaNs(candidate.target_time_ns, now_ns);
    const uint64_t extra_ticks =
        assigned_tick > nominal_tick ? assigned_tick - nominal_tick : 0;
    const uint32_t mask = projection.winner_mask & 7u;
    if (route == kRealCommitRoutePairBoundary) {
      ++commit_route_pair_boundary_total;
    } else {
      ++commit_route_projection_total;
      ++commit_projection_mask_total[mask];
      commit_projection_mask_extra_tick_total[mask] = SaturatingAddNs(
          commit_projection_mask_extra_tick_total[mask], extra_ticks);
    }
    ++commit_boundary_kind_total[size_t(boundary.kind)];
    if (ready_at_commit) {
      ++commit_ready_at_commit_total;
      real_ready_age_at_commit_ns.Add(ready_age_ns);
    }
    if (commit_runway_ns > 0) {
      PresenterSampleWindow<128>& runway =
          route == kRealCommitRoutePairBoundary
              ? real_commit_runway_pair_ns
              : real_commit_runway_projection_ns;
      runway.Add(uint64_t(commit_runway_ns));
    } else {
      // Already due or late at its first commitment: kept visible.
      ++real_commit_late_total;
      real_commit_lateness_ns.Add(uint64_t(-commit_runway_ns));
    }
    ++real_commit_observed_total;
  }

  void AccountSemanticHoldsBefore(const OutputCandidate& candidate,
                                  uint64_t exclusive_tick_index) {
    const uint64_t origin_ns = candidate.semantic_epoch_origin_ns;
    const uint64_t quantum_ns = candidate.semantic_output_quantum_ns;
    if (!origin_ns || !quantum_ns || !exclusive_tick_index) {
      return;
    }
    // The candidate may be a Real whose immediately preceding optional S was
    // never represented by a LogicalOutput. Start at the first unresolved
    // tick in the epoch rather than at the candidate's own nominal tick so
    // those implicit HOLD opportunities remain observable.
    uint64_t first_tick = 0;
    if (last_accounted_semantic_target_ns >= origin_ns) {
      const uint64_t accounted_tick =
          (last_accounted_semantic_target_ns - origin_ns) / quantum_ns;
      first_tick = std::max(first_tick, SaturatingAddNs(accounted_tick, 1));
    }
    if (first_tick >= exclusive_tick_index) {
      return;
    }
    semantic_hold_total = SaturatingAddNs(
        semantic_hold_total, exclusive_tick_index - first_tick);
    const uint64_t last_hold_tick = exclusive_tick_index - 1;
    last_accounted_semantic_target_ns = std::max(
        last_accounted_semantic_target_ns,
        SaturatingAddNs(origin_ns,
                        SaturatingMultiplyNs(quantum_ns, last_hold_tick)));
    RecordOrderedBoundaryProvenance(
        OrderedBoundaryKind::kImplicitHoldBeforeCandidate, candidate.source_id,
        candidate.sequence_id, last_hold_tick,
        SaturatingAddNs(origin_ns,
                        SaturatingMultiplyNs(quantum_ns, last_hold_tick)));
    AdvanceSemanticCursorToTick(candidate, exclusive_tick_index);
  }

  bool ProjectPendingRealSemanticTiming(
      const OutputCandidate& candidate, uint64_t now_ns,
      uint64_t ordered_boundary_ns, uint64_t& nominal_target_out,
      uint64_t& assigned_target_out, uint64_t& assigned_tick_out,
      RealProjectionDiagnostics* diagnostics = nullptr) const {
    nominal_target_out = 0;
    assigned_target_out = 0;
    assigned_tick_out = 0;
    if (candidate.kind != CandidateKind::kReal ||
        candidate.target_time_ns || !candidate.earliest_eligible_time_ns ||
        !stable_latency_metronome_armed || !semantic_epoch_origin_ns ||
        !stable_output_quantum_ns) {
      return false;
    }

    uint64_t nominal_tick = 0;
    if (candidate.earliest_eligible_time_ns > semantic_epoch_origin_ns) {
      const uint64_t eligibility_distance_ns =
          candidate.earliest_eligible_time_ns - semantic_epoch_origin_ns;
      nominal_tick =
          eligibility_distance_ns / stable_output_quantum_ns +
          (eligibility_distance_ns % stable_output_quantum_ns ? 1 : 0);
    }
    uint64_t assigned_tick = nominal_tick;
    if (ordered_boundary_ns >= semantic_epoch_origin_ns) {
      assigned_tick = std::max(
          assigned_tick,
          SaturatingAddNs(
              (ordered_boundary_ns - semantic_epoch_origin_ns) /
                  stable_output_quantum_ns,
              1));
    }
    if (now_ns >= semantic_epoch_origin_ns) {
      assigned_tick = std::max(
          assigned_tick,
          SaturatingAddNs(
              (now_ns - semantic_epoch_origin_ns) /
                  stable_output_quantum_ns,
              1));
    }
    if (diagnostics) {
      // Observation only: the same floors as above, never a decision input.
      diagnostics->nominal_tick = nominal_tick;
      diagnostics->ordered_tick =
          ordered_boundary_ns >= semantic_epoch_origin_ns
              ? SaturatingAddNs((ordered_boundary_ns -
                                 semantic_epoch_origin_ns) /
                                    stable_output_quantum_ns,
                                1)
              : 0;
      diagnostics->now_tick =
          now_ns >= semantic_epoch_origin_ns
              ? SaturatingAddNs((now_ns - semantic_epoch_origin_ns) /
                                    stable_output_quantum_ns,
                                1)
              : 0;
      diagnostics->assigned_tick = assigned_tick;
      diagnostics->winner_mask =
          (nominal_tick == assigned_tick ? 1u : 0u) |
          (diagnostics->ordered_tick &&
                   diagnostics->ordered_tick == assigned_tick
               ? 2u
               : 0u) |
          (diagnostics->now_tick && diagnostics->now_tick == assigned_tick
               ? 4u
               : 0u);
    }
    const uint64_t nominal_target_ns = SemanticTargetForTick(nominal_tick);
    const uint64_t assigned_target_ns = SemanticTargetForTick(assigned_tick);
    if (!assigned_target_ns ||
        assigned_target_ns == std::numeric_limits<uint64_t>::max() ||
        !nominal_target_ns ||
        nominal_target_ns == std::numeric_limits<uint64_t>::max() ||
        nominal_target_ns < candidate.earliest_eligible_time_ns ||
        assigned_target_ns < nominal_target_ns ||
        assigned_target_ns <= now_ns ||
        (ordered_boundary_ns && assigned_target_ns <= ordered_boundary_ns)) {
      return false;
    }

    nominal_target_out = nominal_target_ns;
    assigned_target_out = assigned_target_ns;
    assigned_tick_out = assigned_tick;
    return true;
  }

  bool AnchorSyntheticPairsToCommittedReal(
      const OutputCandidate& committed_a) {
    // TransitionRealOnly has no S to carry the pair identity, so anchor its B
    // directly when A receives the ordinary first commitment. An already
    // committed B is immutable; an existing eligibility floor is only raised,
    // never pulled backward to manufacture planned space.
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& real_b = logical_outputs[i];
      if (real_b.state == LogicalOutputState::kFree ||
          real_b.state == LogicalOutputState::kDropped ||
          real_b.candidate.kind != CandidateKind::kReal ||
          real_b.candidate.pair_contract != PairContract::kTransitionRealOnly ||
          real_b.candidate.pair_a_source_id != committed_a.source_id ||
          real_b.candidate.target_time_ns) {
        continue;
      }
      uint64_t source_distance = 0;
      if (!AcceptedPairSourceDistance(committed_a.source_id,
                                      real_b.candidate.source_id,
                                      source_distance)) {
        continue;
      }
      uint64_t lattice_distance = real_b.candidate.pair_lattice_distance;
      if (!lattice_distance) {
        lattice_distance = DecidePairLatticeDistance(
            committed_a, source_distance, real_b.candidate.issue_time_ns);
      }
      uint64_t unused_s_semantic_ns = 0;
      uint64_t planned_b_floor_ns = 0;
      uint64_t unused_s_target_ns = 0;
      uint64_t unused_s_tick = 0;
      if (!lattice_distance || lattice_distance > source_distance ||
          !ComputePairTimingFromCommittedReal(
              committed_a, lattice_distance, unused_s_semantic_ns,
              planned_b_floor_ns, unused_s_target_ns, unused_s_tick, true)) {
        continue;
      }
      real_b.candidate.pair_lattice_distance = lattice_distance;
      const uint64_t previous_floor =
          real_b.candidate.earliest_eligible_time_ns;
      real_b.candidate.transition_planned_floor_was_blocked =
          previous_floor > planned_b_floor_ns;
      real_b.candidate.earliest_eligible_time_ns =
          std::max(previous_floor, planned_b_floor_ns);
      PropagateLogicalTiming(i);
    }
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& synthetic = logical_outputs[i];
      if (synthetic.state == LogicalOutputState::kFree ||
          synthetic.state == LogicalOutputState::kDropped ||
          synthetic.candidate.kind != CandidateKind::kSynthetic ||
          synthetic.candidate.pair_a_source_id != committed_a.source_id) {
        continue;
      }
      LogicalOutput* real_b = nullptr;
      for (LogicalOutput& logical : logical_outputs) {
        if (logical.state != LogicalOutputState::kFree &&
            logical.state != LogicalOutputState::kDropped &&
            logical.candidate.kind == CandidateKind::kReal &&
            logical.candidate.source_id == synthetic.candidate.source_id) {
          real_b = &logical;
          break;
        }
      }
      uint64_t synthetic_semantic_ns = 0;
      uint64_t real_earliest_ns = 0;
      uint64_t synthetic_target_ns = 0;
      uint64_t synthetic_tick = 0;
      // The loop filter already bound S to this A through pair_a_source_id,
      // and B is S's own source_id. Accepted adjacency needs only strict
      // order. The lattice distance is decided once, the first time the pair
      // is placed; a later re-anchor (A deferred) keeps it.
      uint64_t source_distance = 0;
      uint64_t lattice_distance = 0;
      if (real_b && AcceptedPairSourceDistance(committed_a.source_id,
                                               synthetic.candidate.source_id,
                                               source_distance)) {
        lattice_distance = synthetic.candidate.pair_lattice_distance;
        if (!lattice_distance) {
          lattice_distance = real_b->candidate.pair_lattice_distance;
        }
        if (!lattice_distance) {
          lattice_distance = DecidePairLatticeDistance(
              committed_a, source_distance, real_b->candidate.issue_time_ns);
        }
      }
      if (!real_b || !lattice_distance || lattice_distance > source_distance ||
          !ComputePairTimingFromCommittedReal(
              committed_a, lattice_distance, synthetic_semantic_ns,
              real_earliest_ns, synthetic_target_ns, synthetic_tick)) {
        ++ordering_error_total;
        Terminalize(TerminalReason::kOrderingFailure,
                    synthetic.candidate.source_id);
        return false;
      }
      synthetic.candidate.pair_lattice_distance = lattice_distance;
      real_b->candidate.pair_lattice_distance = lattice_distance;
      synthetic.candidate.semantic_target_time_ns = synthetic_semantic_ns;
      synthetic.candidate.target_time_ns = synthetic_target_ns;
      synthetic.candidate.semantic_epoch_origin_ns =
          committed_a.semantic_epoch_origin_ns;
      synthetic.candidate.semantic_output_quantum_ns =
          committed_a.semantic_output_quantum_ns;
      synthetic.candidate.semantic_tick_index = synthetic_tick;
      synthetic.synthetic_pair_safe_deadline_ns =
          SyntheticPairSafeDeadlineFromReal(real_earliest_ns,
                                             dispatch_lead_ns);
      real_b->candidate.earliest_eligible_time_ns =
          std::max(real_b->candidate.earliest_eligible_time_ns,
                   real_earliest_ns);
      last_semantic_target_ns =
          std::max(last_semantic_target_ns, synthetic_target_ns);
      // H12: the anchor may legitimately arrive at any live production stage.
      // Record which one, so a run can prove production is no longer gated by
      // the semantic contract. No image is regenerated; PropagateLogicalTiming
      // below pushes the new timing into the live FinalOutput and PostJob.
      if (synthetic.synthetic_unanchored_at_admit &&
          !synthetic.synthetic_anchor_stage_recorded) {
        synthetic.synthetic_anchor_stage_recorded = true;
        switch (synthetic.state) {
          case LogicalOutputState::kWaitingGeneration:
            ++synthetic_anchored_before_generation_total;
            break;
          case LogicalOutputState::kWaitingPost:
            ++synthetic_anchored_after_generation_total;
            break;
          case LogicalOutputState::kPostSubmitted:
            ++synthetic_anchored_after_post_total;
            break;
          case LogicalOutputState::kFinalReady:
            ++synthetic_anchored_after_final_ready_total;
            break;
          default:
            break;
        }
        const uint64_t anchor_now_ns = PresenterMonotonicTimeNs();
        if (synthetic.synthetic_admit_time_ns &&
            anchor_now_ns > synthetic.synthetic_admit_time_ns) {
          synthetic_anchor_latency_ns.Add(
              anchor_now_ns - synthetic.synthetic_admit_time_ns);
        }
      }
      PropagateLogicalTiming(i);
      // H12 deferred latency feedback: an S that reached FinalReady before
      // this anchor is judged now, its recorded FinalReady time against the
      // soft deadline PropagateLogicalTiming() just derived.
      if (synthetic.state == LogicalOutputState::kFinalReady &&
          !synthetic.latency_feedback_scored &&
          !synthetic.synthetic_presentation_retired &&
          synthetic.final_ready_time_ns &&
          synthetic.final_ready_soft_deadline_ns) {
        // If physical completion preceded this pair-local anchor, judge the
        // proof timestamp now. A lifecycle-only fallback/error observation
        // is retained for output bookkeeping but is not a B1 success.
        MaybeScoreSyntheticPostCompletionLatency(synthetic, true);
      }
      // B3a: the pair is placed now, so a C_pair sample taken before it can
      // finally be judged - with its lattice distance resolved, and with how
      // long it had to wait for this contract.
      ResolveDeferredBetterDSample(synthetic);
    }
    return true;
  }

  // H12 lifecycle reclamation. An unanchored S stays valid for as long as its
  // A commitment is still causally resolvable. It becomes structurally
  // impossible only when A holds no semantic commitment, is not the last
  // applied commitment, and no longer exists as a live accepted Real — that is,
  // when A's identity has been irreversibly passed by the accepted-Real
  // stream. This is derived from accepted-Real identity, never from elapsed
  // time: there is deliberately no timeout, no Source-period budget and no
  // publication-count heuristic here.
  bool SyntheticPairAnchorStructurallyImpossible(
      const LogicalOutput& synthetic) const {
    if (synthetic.candidate.kind != CandidateKind::kSynthetic ||
        synthetic.state == LogicalOutputState::kFree ||
        synthetic.state == LogicalOutputState::kDropped ||
        synthetic.candidate.target_time_ns) {
      return false;
    }
    const uint64_t pair_a_source_id = synthetic.candidate.pair_a_source_id;
    if (!pair_a_source_id) {
      return false;
    }
    OutputCandidate committed_a;
    if (FindRealSemanticCommitment(pair_a_source_id, committed_a)) {
      // The anchor is still reachable; CommitHeadRealToSemanticTick() or the
      // already applied commitment will resolve this pair.
      return false;
    }
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kReal &&
          logical.candidate.source_id == pair_a_source_id) {
        // A is still a live accepted Real and can still be committed.
        return false;
      }
    }
    return true;
  }

  void ReclaimStructurallyOrphanedSynthetics() {
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      if (!SyntheticPairAnchorStructurallyImpossible(logical_outputs[i])) {
        continue;
      }
      if (!shutdown_requested.load(std::memory_order_acquire) &&
          !detach_requested.load(std::memory_order_acquire)) {
        ++split_forbidden_s_drop_total;
        XELOGE("ZeroFGSplitStructuralOrphan source={} action=fail_open",
               logical_outputs[i].candidate.source_id);
        Terminalize(TerminalReason::kOrderingFailure,
                    logical_outputs[i].candidate.source_id);
        return;
      }
      ++synthetic_structural_orphan_reclaimed_total;
      // DropSyntheticLogical() is the existing bounded ownership retirement
      // path: it releases the Synthetic image, the A/B Real slot references and
      // a Ready FinalOutput, and leaves already submitted Generation/Post GPU
      // work to retire asynchronously through its job/token model rather than
      // double-releasing or waiting Source.
      DropSyntheticLogical(i, SyntheticDropReason::kStructuralOrphan);
    }
  }

  bool CommitHeadRealToSemanticTick(uint32_t logical_index,
                                    uint64_t now_ns) {
    if (logical_index >= kLogicalOutputCapacity) {
      return false;
    }
    LogicalOutput& logical = logical_outputs[logical_index];
    OutputCandidate& candidate = logical.candidate;
    if (candidate.kind != CandidateKind::kReal ||
        candidate.target_time_ns || !candidate.earliest_eligible_time_ns ||
        !stable_latency_metronome_armed || !semantic_epoch_origin_ns ||
        !stable_output_quantum_ns) {
      return false;
    }

    const uint64_t ordered_boundary_ns =
        std::max(last_accounted_semantic_target_ns,
                 last_applied_semantic_target_ns);
    // Commitment-cause telemetry (observation only).
    const OrderedBoundaryProvenance boundary_provenance =
        ordered_boundary_provenance;
    uint32_t commit_route = kRealCommitRouteProjection;
    uint64_t trace_nominal_tick = 0;
    RealProjectionDiagnostics projection_diagnostics;
    uint64_t nominal_target_ns = 0;
    uint64_t assigned_target_ns = 0;
    uint64_t assigned_tick = 0;
    bool pair_local_head_commit = false;
    bool rolling_pair_head = false;
    bool rolling_pair_epoch_current = false;
    uint64_t pair_source_distance = 0;
    uint64_t pair_lattice_distance = 0;
    uint64_t pair_boundary_tick = 0;
    uint64_t pair_normal_boundary_tick = 0;
    // The last applied Real is this head's accepted predecessor: accepted
    // Reals are never dropped and apply strictly in order. The pair boundary
    // lies 2*d_lat ticks after it. A consecutive head is always a rolling pair
    // (d_lat = 1); a gap head is one only if its pair was placed with an S
    // (admission or the S anchor recorded d_lat). A gap B without S takes the
    // ordinary commitment below, exactly as before the accepted-gap reopen.
    if (candidate.source_id &&
        AcceptedPairSourceDistance(last_applied_real_commitment.source_id,
                                   candidate.source_id,
                                   pair_source_distance)) {
      pair_lattice_distance =
          pair_source_distance == 1 ? 1 : candidate.pair_lattice_distance;
    }
    if (pair_lattice_distance) {
      rolling_pair_head = true;
      const bool transition_real_only =
          candidate.pair_contract == PairContract::kTransitionRealOnly;
      uint64_t synthetic_semantic_ns = 0;
      uint64_t pair_boundary_ns = 0;
      uint64_t synthetic_target_ns = 0;
      uint64_t synthetic_tick = 0;
      const bool pair_timing_valid =
          pair_lattice_distance <= pair_source_distance &&
          ComputePairTimingFromCommittedReal(
              last_applied_real_commitment, pair_lattice_distance,
              synthetic_semantic_ns, pair_boundary_ns, synthetic_target_ns,
              synthetic_tick, transition_real_only);
      rolling_pair_epoch_current =
          pair_timing_valid &&
          last_applied_real_commitment.semantic_epoch_origin_ns ==
              semantic_epoch_origin_ns &&
          last_applied_real_commitment.semantic_output_quantum_ns ==
              stable_output_quantum_ns;
      pair_boundary_tick =
          pair_timing_valid
              ? SaturatingAddNs(
                    last_applied_real_commitment.semantic_tick_index,
                    SaturatingMultiplyNs(
                        pair_lattice_distance,
                        transition_real_only ? 1 : 2))
              : 0;
      pair_normal_boundary_tick =
          pair_timing_valid
              ? SaturatingAddNs(
                    last_applied_real_commitment.semantic_tick_index,
                    SaturatingMultiplyNs(pair_lattice_distance, 2))
              : 0;
      // H14.1: the pair boundary is allowed to become B's first commitment
      // only here, while B is the ordered head, and only while A and the
      // global future cursor still describe the same epoch. The exact
      // earliest floor proves this is the A/S/B boundary rather than a deeper
      // acceptance projection. A future-only reanchor deliberately fails this
      // test and falls through to ordinary current-epoch shallow commitment.
      if (!pair_local_head_commit && rolling_pair_epoch_current &&
          pair_boundary_ns &&
          candidate.earliest_eligible_time_ns == pair_boundary_ns &&
          (!ordered_boundary_ns || pair_boundary_ns > ordered_boundary_ns) &&
          pair_boundary_tick != std::numeric_limits<uint64_t>::max() &&
          SemanticTargetForTick(pair_boundary_tick) == pair_boundary_ns) {
        nominal_target_ns = pair_boundary_ns;
        assigned_target_ns = pair_boundary_ns;
        assigned_tick = pair_boundary_tick;
        pair_local_head_commit = true;
        commit_route = kRealCommitRoutePairBoundary;
        trace_nominal_tick = pair_boundary_tick;
        ObservePairBoundaryPhaseExcess(pair_source_distance, pair_boundary_ns,
                                       candidate.issue_time_ns,
                                       stable_output_quantum_ns);
      }
    }
    if (!pair_local_head_commit) {
      if (!ProjectPendingRealSemanticTiming(
              candidate, now_ns, ordered_boundary_ns, nominal_target_ns,
              assigned_target_ns, assigned_tick, &projection_diagnostics)) {
        ++semantic_target_order_violation_total;
        return false;
      }
      trace_nominal_tick = projection_diagnostics.nominal_tick;
    }

    const uint64_t previous_debt_ticks = CurrentRealExecutiveDebtTicks();
    candidate.semantic_target_time_ns = nominal_target_ns;
    candidate.semantic_epoch_origin_ns = semantic_epoch_origin_ns;
    candidate.semantic_output_quantum_ns = stable_output_quantum_ns;
    candidate.semantic_tick_index = assigned_tick;
    candidate.target_time_ns = assigned_target_ns;
    last_semantic_target_ns =
        std::max(last_semantic_target_ns, assigned_target_ns);
    logical.dispatch_lead_ns =
        std::max(logical.dispatch_lead_ns, DispatchLeadEstimateNs());
    UpdateSyntheticPairSafeDeadlineForReal(
        candidate.source_id, candidate.target_time_ns, logical.dispatch_lead_ns);
    PropagateLogicalTiming(logical_index);
    if (!AnchorSyntheticPairsToCommittedReal(candidate)) {
      return false;
    }
    const uint64_t debt_ticks = CandidateExecutiveDebtTicks(candidate);
    ObserveNewRealExecutiveDebt(previous_debt_ticks, debt_ticks);
    ++shallow_real_commit_total;
    if (pair_local_head_commit) {
      ++h14_b_head_pair_commit_total;
    } else if (rolling_pair_head) {
      ++h14_b_head_current_epoch_commit_total;
      if (!rolling_pair_epoch_current) {
        ++h14_b_head_reanchor_cutover_total;
      }
    }
    if (rolling_pair_head && pair_source_distance > 1) {
      ++accepted_gap_real_scaled_commit_total;
    }
    if (candidate.pair_contract == PairContract::kNormal &&
        candidate.pair_lattice_distance && rolling_pair_epoch_current &&
        pair_boundary_tick && assigned_tick < pair_boundary_tick) {
      // Must stay zero: a same-epoch head whose pair was placed with an S is
      // never committed inside its lattice distance. A B without S follows
      // the ordinary cursor placement and may legitimately land earlier.
      ++pair_distance_compressed_real_total;
      ++h1c_pair_compressed_normal_total;
    }
    if (debt_ticks) {
      ++shallow_real_commit_backlog_total;
      shallow_real_commit_backlog_tick_total = SaturatingAddNs(
          shallow_real_commit_backlog_tick_total, debt_ticks);
    } else {
      ++shallow_real_commit_nominal_total;
    }
    ObserveRealCommitCause(logical, now_ns, commit_route,
                           projection_diagnostics, trace_nominal_tick,
                           assigned_tick, ordered_boundary_ns,
                           boundary_provenance, rolling_pair_head,
                           rolling_pair_epoch_current, pair_source_distance,
                           pair_lattice_distance, pair_boundary_tick,
                           debt_ticks);
    if (candidate.pair_contract == PairContract::kTransitionRealOnly &&
        rolling_pair_epoch_current && pair_boundary_tick &&
        pair_normal_boundary_tick) {
      const uint64_t saved_ticks =
          pair_normal_boundary_tick > assigned_tick
              ? pair_normal_boundary_tick - assigned_tick
              : 0;
      h1c_planned_ticks_saved_total = SaturatingAddNs(
          h1c_planned_ticks_saved_total, saved_ticks);
      if (assigned_tick == pair_boundary_tick) {
        ++h1c_short_boundary_commit_total;
      }
      if (candidate.transition_planned_floor_was_blocked &&
          assigned_tick > pair_boundary_tick) {
        ++h1c_pair_planned_floor_blocked_total;
      }
      if (h1c_commit_trace_total < 16) {
        ++h1c_commit_trace_total;
      }
      candidate.transition_planned_floor_was_blocked = false;
    }
    ObservePlannedSpaceRealCommit(logical);
    return true;
  }

  bool DeferRealToNextSemanticTick(uint32_t logical_index,
                                   uint64_t now_ns) {
    LogicalOutput& logical = logical_outputs[logical_index];
    OutputCandidate& candidate = logical.candidate;
    if (candidate.kind != CandidateKind::kReal ||
        !candidate.target_time_ns || !candidate.semantic_epoch_origin_ns ||
        !candidate.semantic_output_quantum_ns) {
      return false;
    }
    const uint64_t desired_target_ns = std::max(
        SaturatingAddNs(now_ns, logical.dispatch_lead_ns),
        last_accounted_semantic_target_ns
            ? SaturatingAddNs(last_accounted_semantic_target_ns,
                              candidate.semantic_output_quantum_ns)
            : 0);
    uint64_t target_tick = candidate.semantic_tick_index + 1;
    if (desired_target_ns > candidate.semantic_epoch_origin_ns) {
      const uint64_t distance_ns =
          desired_target_ns - candidate.semantic_epoch_origin_ns;
      const uint64_t rounded_tick =
          distance_ns / candidate.semantic_output_quantum_ns +
          (distance_ns % candidate.semantic_output_quantum_ns ? 1 : 0);
      target_tick = std::max(target_tick, rounded_tick);
    }
    const uint64_t target_ns = SaturatingAddNs(
        candidate.semantic_epoch_origin_ns,
        SaturatingMultiplyNs(candidate.semantic_output_quantum_ns,
                             target_tick));
    if (!target_ns || target_ns == std::numeric_limits<uint64_t>::max()) {
      return false;
    }
    AccountSemanticHoldsBefore(candidate, target_tick);
    semantic_real_deferred_tick_total = SaturatingAddNs(
        semantic_real_deferred_tick_total,
        target_tick - candidate.semantic_tick_index);
    ++semantic_real_deferred_total;
    candidate.semantic_tick_index = target_tick;
    candidate.target_time_ns = target_ns;
    UpdateSyntheticPairSafeDeadlineForReal(
        candidate.source_id, candidate.target_time_ns, logical.dispatch_lead_ns);
    real_exec_debt_high_water = std::max(
        real_exec_debt_high_water,
        CandidateExecutiveDebtTicks(candidate));
    PropagateLogicalTiming(logical_index);
    if (!AnchorSyntheticPairsToCommittedReal(candidate)) {
      return false;
    }
    return true;
  }

  // In the split presenter, a Post submission is logically FinalReady before
  // its GPU work is necessarily complete. Do not hand that FinalOutput to the
  // egress until the already-exported completion fence is observed ready. This is a non-blocking poll only; the active PostJob's normal wake
  // schedule supplies the retry.
  bool PostCompletionReadyForApply(uint32_t logical_index,
                                   LogicalOutput& logical,
                                   uint64_t now_ns) {
    for (PostJob& job : post_jobs) {
      if (!job.active || job.logical_index != logical_index ||
          job.logical_token != logical.token) {
        continue;
      }
      if (job.timeline_fallback) {
        // This is the failure-only accepted-submit fallback. It has no
        // CPU-visible fence; preserve the existing fallback behavior rather
        // than introducing a new driver query or an unbounded logical stall.
        RecordPostCompletionReady(
            logical, 0, PostCompletionEvidence::kUnprovenFallback);
        return true;
      }
      if (job.completion_fence_fd < 0) {
        RecordPostCompletionReady(
            logical, 0, PostCompletionEvidence::kUnprovenFallback);
        return true;
      }
      if (job.next_observation_time_ns > now_ns) {
        // PollPostJobs owns the same bounded observation cadence. Avoid a
        // second poll when the presenter was woken for an unrelated event
        // before that cadence became due, but keep the head retry bounded to
        // the short fence interval rather than inheriting a long stage
        // estimate.
        job.next_observation_time_ns = std::min(
            job.next_observation_time_ns,
            SaturatingAddNs(now_ns, kReleaseFencePollIntervalNs));
        return false;
      }
      bool post_fence_observed_signaled = false;
      bool post_fence_observation_error = false;
      if (!IsReleaseFenceReady(job.completion_fence_fd,
                               &post_fence_observed_signaled,
                               &post_fence_observation_error)) {
        if (post_fence_observation_error) {
          RecordPostCompletionReady(
              logical, 0, PostCompletionEvidence::kObservationError);
        }
        job.next_observation_time_ns =
            SaturatingAddNs(now_ns, kReleaseFencePollIntervalNs);
        if (logical.candidate.kind == CandidateKind::kSynthetic) {
          ++split_apply_fence_wait_synthetic_total;
        } else {
          ++split_apply_fence_wait_real_total;
        }
        return false;
      }
      // Keep lifecycle timestamping in both modes. An error-only readiness
      // event preserves the historical apply path but cannot prove completion
      // for B1's controller judgement.
      RecordPostCompletionReady(
          logical, now_ns,
          post_fence_observed_signaled
              ? PostCompletionEvidence::kSyncFile
              : PostCompletionEvidence::kObservationError);
      return true;
    }
    return true;
  }

  // Phase 0 sensors 3 and 8. Recording only: nothing here feeds the decay,
  // the attack, or any gate. Phase 1 changes what the judgement reads.
  void ObservePostCompletionEvidence(LogicalOutput& logical,
                                     uint64_t completion_ns) {
    // Sensor 3: how optimistic the submit stamp is, and how many frames the
    // decay counted as successes that a completion deadline would have
    // failed. That difference is blocker #21's false-success rate.
    if (logical.final_ready_time_ns &&
        completion_ns > logical.final_ready_time_ns) {
      post_submit_to_completion_ns.Add(completion_ns -
                                       logical.final_ready_time_ns);
    }
    if (logical.final_ready_soft_deadline_ns) {
      ++final_ready_completion_judged_total;
      const bool submit_passed =
          logical.final_ready_time_ns <
          logical.final_ready_soft_deadline_ns;
      const bool completion_passed =
          completion_ns < logical.final_ready_soft_deadline_ns;
      if (submit_passed && !completion_passed) {
        ++final_ready_false_success_total;
      }
    }
    // Sensor 8: the raw causal term of C_pair, A issue -> S physically ready,
    // as ONE end-to-end sample. Splitting it into an AB term plus a chain term
    // would lose the correlation between them, which is the defect that sank
    // the summed-percentile formulation.
    //
    // Raw on purpose. The provenance subtractions it still needs (our own
    // holds, BP_policy, presentation residence) only become available after
    // phase 2, so synthesizing a net sample now would launder contamination
    // into a number that looks clean. B issue -> S ready is recorded beside
    // it, so the AB spacing reads as their difference and is never estimated
    // on its own.
    if (logical.candidate.kind == CandidateKind::kSynthetic) {
      if (logical.candidate.pair_a_issue_time_ns &&
          completion_ns > logical.candidate.pair_a_issue_time_ns) {
        const uint64_t raw_ns =
            completion_ns - logical.candidate.pair_a_issue_time_ns;
        pair_causal_raw_ns.Add(raw_ns);
        ++pair_causal_sample_total;
        // Commit B1. C_pair_clean_i = C_pair_raw_i - policy_wait_i, removed
        // from THIS sample and never as a quantile of one window minus a
        // quantile of another - that summed-percentile form is exactly what
        // sank the previous formulation, because it loses the correlation
        // between the terms. Only the sidecar residence of this pair's own
        // Real is removed so far; BP and queue residence are not yet
        // attributable per sample and stay inside the number, which the log
        // line states rather than implies.
        // Commit B2. Two removals, both sample-aligned and both proven ours:
        // the sidecar hold of this pair's A (B1), and the funding this S's own
        // Post suffered while returning our own pacing holds would have
        // funded it (B2c's PolicyButFor). They are disjoint in time - the
        // hold ends before S is materialized, the funding wait starts after
        // its Generation - so they add and cannot double-count. Nothing else
        // is removed: the deep state is not decomposed any further, it is
        // censored where D learns (§5.14).
        const uint64_t policy_wait_ns = logical.candidate.pair_a_policy_wait_ns;
        const uint64_t funding_policy_ns = logical.funding_policy_wait_ns;
        const uint64_t removed_ns =
            SaturatingAddNs(policy_wait_ns, funding_policy_ns);
        const uint64_t clean_ns = raw_ns > removed_ns ? raw_ns - removed_ns : 0;
        pair_causal_clean_ns.Add(clean_ns);
        ObserveBetterDSample(logical, clean_ns, completion_ns);
        if (funding_policy_ns) {
          pair_funding_policy_wait_ns.Add(funding_policy_ns);
          ++pair_causal_clean_funding_subtracted_total;
        }
        if (policy_wait_ns) {
          // Conditional on having waited at all, so these quantiles are the
          // distribution of NON-ZERO holds. subtracted=N/total is printed
          // beside them and is what makes them readable.
          pair_policy_wait_ns.Add(policy_wait_ns);
          ++pair_causal_clean_subtracted_total;
        }
      }
      if (logical.candidate.issue_time_ns &&
          completion_ns > logical.candidate.issue_time_ns) {
        // candidate.issue_time_ns on an S is B's issue, not A's.
        pair_chain_raw_ns.Add(completion_ns - logical.candidate.issue_time_ns);
      }
    }
  }

  uint64_t PhysicalPresentNotBeforeNs(const LogicalOutput& logical) const {
    uint64_t present_not_before_ns = PresentationTargetNs(logical);
    if (logical.post_completion_ready_time_ns) {
      present_not_before_ns = std::max(
          present_not_before_ns,
          SaturatingAddNs(logical.post_completion_ready_time_ns,
                          logical.dispatch_lead_ns));
    }
    return present_not_before_ns;
  }

  // Main Surface Authority apply: hand the FinalOutput to the egress ring.
  // Never blocks. Returns false, with the slot untouched, only if B stopped
  // producing or the ring filled since the checks in ApplyFinalOutput.
  bool EnqueueMainSurfaceOutput(const LogicalOutput& logical,
                                FinalOutputSlot& output,
                                uint64_t present_not_before_ns,
                                uint64_t& apply_time_ns_out) {
    ZeroFGMainSurfaceEgress::Request request;
    request.slot = logical.final_output_index;
    request.sequence_id = logical.sequence_id;
    request.synthetic = logical.candidate.kind == CandidateKind::kSynthetic;
    request.desired_present_ns = present_not_before_ns;
    request.output_quantum_ns = logical.candidate.semantic_output_quantum_ns;
    request.image = output.image;
    request.post_semaphore = output.post_completion_semaphore;
    request.post_value = output.post_completion_value;
    const uint64_t apply_time_ns = PresenterMonotonicTimeNs();
    request.apply_time_ns = apply_time_ns;
    const uint64_t previous_sequence_id = output.sequence_id;
    output.sequence_id = logical.sequence_id;
    output.state.store(FinalOutputState::kTransactionApplied,
                       std::memory_order_release);
    const ZeroFGMainSurfaceEgress::EnqueueResult enqueue_result =
        main_surface_egress
            ? main_surface_egress->TryEnqueue(request)
            : ZeroFGMainSurfaceEgress::EnqueueResult::kNotProducing;
    if (enqueue_result != ZeroFGMainSurfaceEgress::EnqueueResult::kQueued) {
      output.sequence_id = previous_sequence_id;
      output.state.store(FinalOutputState::kReady, std::memory_order_release);
      if (enqueue_result == ZeroFGMainSurfaceEgress::EnqueueResult::kFull) {
        ++main_surface_ring_full_total;
      } else {
        NoteMainSurfaceWaitingForB(logical.sequence_id);
      }
      return false;
    }
    if (main_surface_waiting_for_b_begin_ns) {
      // Presentation opens: the first output B accepted after a wait.
      const uint64_t waited_ns =
          apply_time_ns > main_surface_waiting_for_b_begin_ns
              ? apply_time_ns - main_surface_waiting_for_b_begin_ns
              : 0;
      main_surface_waiting_for_b_max_ns =
          std::max(main_surface_waiting_for_b_max_ns, waited_ns);
      main_surface_waiting_for_b_begin_ns = 0;
      XELOGI(
          "ZeroFGMainSurface presentation_open waited_for_B_us={} "
          "deferred_outputs={}",
          waited_ns / 1000, main_surface_waiting_for_b_total);
    }
    output.dispatch_lead_ns = logical.dispatch_lead_ns;
    output.final_ready_time_ns = logical.final_ready_time_ns;
    output.final_ready_soft_deadline_ns = logical.final_ready_soft_deadline_ns;
    output.semantic_forward_skipped = logical.semantic_forward_skipped;
    ++main_surface_applied_total;
    apply_time_ns_out = apply_time_ns;
    return true;
  }

  ApplyFinalOutputResult ApplyFinalOutput(LogicalOutput& logical) {
    if (logical.synthetic_presentation_retired) {
      // H13 hard safety: a presentation-retired S has already committed its
      // semantic HOLD and the ordered executive has advanced past it. It must
      // never reach the swapchain. Reaching here means the
      // presentation/production split was violated somewhere upstream.
      ++presentation_retired_reached_apply_total;
      XELOGE("ZeroFGPresentationRetiredApply source={} sequence={}",
             logical.candidate.source_id, logical.sequence_id);
      return ApplyFinalOutputResult::kFailure;
    }
    if (logical.final_output_index >= kFinalOutputPoolSize) {
      return ApplyFinalOutputResult::kFailure;
    }
    FinalOutputSlot& output = final_outputs[logical.final_output_index];
    if (output.state.load(std::memory_order_acquire) !=
        FinalOutputState::kReady) {
      return ApplyFinalOutputResult::kFailure;
    }
    if (stable_latency_metronome_armed &&
        logical.candidate.kind == CandidateKind::kReal &&
        !logical.candidate.target_time_ns) {
      ++armed_real_apply_without_target;
      XELOGE("ZeroFGArmedRealWithoutTarget source={} sequence={}",
             logical.candidate.source_id, logical.sequence_id);
      ++ordering_error_total;
      Terminalize(TerminalReason::kOrderingFailure, logical.candidate.source_id);
      return ApplyFinalOutputResult::kFailure;
    }
    const uint64_t previous_sequence =
        last_applied_logical_sequence.load(std::memory_order_acquire);
    const uint64_t previous_target = last_applied_semantic_target_ns;
    const uint64_t previous_source =
        last_applied_source_id.load(std::memory_order_acquire);
    const bool sequence_bad =
        previous_sequence && logical.sequence_id <= previous_sequence;
    const bool target_bad =
        logical.candidate.target_time_ns && previous_target &&
        logical.candidate.target_time_ns <= previous_target;
    // Accepted adjacency, not numeric +1: S must follow exactly the Real
    // applied before it, and sit its pair's lattice distance d_lat ticks after
    // that Real, with 1 <= d_lat <= the IssueSwap distance.
    uint64_t synthetic_source_distance = 0;
    const uint64_t synthetic_pair_distance =
        logical.candidate.pair_lattice_distance;
    const bool synthetic_pair_distance_valid =
        logical.candidate.kind == CandidateKind::kSynthetic &&
        AcceptedPairSourceDistance(logical.candidate.pair_a_source_id,
                                   logical.candidate.source_id,
                                   synthetic_source_distance) &&
        synthetic_pair_distance &&
        synthetic_pair_distance <= synthetic_source_distance;
    const bool pair_bad =
        logical.candidate.kind == CandidateKind::kSynthetic &&
        (!synthetic_pair_distance_valid ||
         previous_source != logical.candidate.pair_a_source_id);
    const bool real_order_bad =
        logical.candidate.kind == CandidateKind::kReal && previous_source &&
        logical.candidate.source_id <= previous_source;
    const uint64_t synthetic_pair_quantum_ns =
        logical.candidate.semantic_output_quantum_ns;
    const bool synthetic_pair_target_bad =
        logical.candidate.kind == CandidateKind::kSynthetic &&
        (!synthetic_pair_distance_valid ||
         last_applied_real_commitment.source_id !=
             logical.candidate.pair_a_source_id ||
         !last_applied_real_commitment.target_time_ns ||
         !synthetic_pair_quantum_ns ||
         synthetic_pair_quantum_ns >
             std::numeric_limits<uint64_t>::max() / synthetic_pair_distance ||
         last_applied_real_commitment.target_time_ns >
             std::numeric_limits<uint64_t>::max() -
                 synthetic_pair_quantum_ns * synthetic_pair_distance ||
         logical.candidate.target_time_ns !=
             last_applied_real_commitment.target_time_ns +
                 synthetic_pair_quantum_ns * synthetic_pair_distance);
    const bool real_pair_floor_bad =
        logical.candidate.kind == CandidateKind::kReal &&
        last_applied_synthetic_commitment.source_id ==
            logical.candidate.source_id &&
        last_applied_synthetic_commitment.target_time_ns &&
        last_applied_synthetic_commitment.semantic_output_quantum_ns &&
        (last_applied_synthetic_commitment.target_time_ns >
             std::numeric_limits<uint64_t>::max() -
                 last_applied_synthetic_commitment.semantic_output_quantum_ns ||
         logical.candidate.target_time_ns <
             last_applied_synthetic_commitment.target_time_ns +
                 last_applied_synthetic_commitment
                     .semantic_output_quantum_ns);
    if (sequence_bad || target_bad || pair_bad || real_order_bad ||
        synthetic_pair_target_bad || real_pair_floor_bad) {
      ++ordering_error_total;
      XELOGE(
          "ZeroFGPhaseCOrderingPreflight sequence/target/pair/real/"
          "s_pair_target/r_pair_floor={}/{}/{}/{}/{}/{} "
          "current sequence/source/pair/target={}/{}/{}/{} previous sequence/"
          "source/target={}/{}/{}",
          sequence_bad, target_bad, pair_bad, real_order_bad,
          synthetic_pair_target_bad, real_pair_floor_bad,
          logical.sequence_id, logical.candidate.source_id,
          logical.candidate.pair_a_source_id,
          logical.candidate.target_time_ns, previous_sequence, previous_source,
          previous_target);
      Terminalize(TerminalReason::kOrderingFailure, output.source_id);
      return ApplyFinalOutputResult::kFailure;
    }
    if (main_surface_producer.state() !=
        ZeroFGMainSurfaceProducer::State::kB) {
      // Main Surface Authority: Presentation opens only once B produces. While
      // A still holds the Surface, or in the handoff gap, the ordered head
      // stays FinalReady: nothing is applied, counted or advanced, and
      // Production continues. Not ring pressure.
      NoteMainSurfaceWaitingForB(logical.sequence_id);
      return ApplyFinalOutputResult::kNotReady;
    }
    if (!main_surface_egress || !main_surface_egress->HasCapacity()) {
      // Main Surface Authority: the pump->egress ring is bounded by the
      // FinalOutput pool. Full is physical backpressure, not a drop: the
      // ordered head stays FinalReady and is retried; nothing is skipped,
      // replayed or compressed.
      ++main_surface_ring_full_total;
      return ApplyFinalOutputResult::kNotReady;
    }
    const uint64_t transaction_prepare_begin_ns = PresenterMonotonicTimeNs();
    const uint64_t present_not_before_ns =
        PhysicalPresentNotBeforeNs(logical);
    const uint64_t planned_dispatch_time_ns =
        present_not_before_ns > logical.dispatch_lead_ns
            ? present_not_before_ns - logical.dispatch_lead_ns
            : present_not_before_ns;
    if (planned_dispatch_time_ns &&
        transaction_prepare_begin_ns < planned_dispatch_time_ns) {
      ++apply_before_planned_violation_total;
    }
    ObservePresentationAdvance(logical, planned_dispatch_time_ns);
    dispatch_start_late_ns.Add(
        planned_dispatch_time_ns &&
                transaction_prepare_begin_ns > planned_dispatch_time_ns
            ? transaction_prepare_begin_ns - planned_dispatch_time_ns
            : 0);
    uint64_t apply_time_ns = 0;
    uint64_t apply_return_time_ns = 0;
    // Main Surface Authority: the apply is the hand-off to the egress
    // ring; the FIFO shows the output at the first vertical blank at or
    // after present_not_before_ns.
    if (!EnqueueMainSurfaceOutput(logical, output, present_not_before_ns,
                                  apply_time_ns)) {
      return ApplyFinalOutputResult::kNotReady;
    }
    apply_return_time_ns = PresenterMonotonicTimeNs();
    if (output.residency_recovery_debt.exchange(false,
                                                std::memory_order_acq_rel)) {
      ++residency_recovery_debt_discharged_total;
    }
    output.apply_time_ns = apply_time_ns;
    if (present_not_before_ns) {
      // Blockers #1 and #6. apply_time_ns is when WE submitted, and
      // present_not_before_ns is a floor WE computed; neither is a
      // presentation, and neither raises the next floor (the inherited tick).
      // Main Surface Authority has no ratchet at all: FIFO already allows one
      // image per vertical blank, and actualPresentTime is only observed.
      if (present_not_before_ns > logical.candidate.target_time_ns) {
        const uint64_t push_ns =
            present_not_before_ns - logical.candidate.target_time_ns;
        ++split_present_spacing_push_total;
        split_present_spacing_push_max_ns =
            std::max(split_present_spacing_push_max_ns, push_ns);
      }
    }
    applied_timestamps.Add(apply_time_ns);
    xe::RecordOutputPresent();
    if (logical.candidate.kind == CandidateKind::kSynthetic) {
      if (logical.synthetic_pair_safe_deadline_ns &&
          apply_time_ns >= logical.synthetic_pair_safe_deadline_ns) {
        ++h14_s_applied_after_old_bsafe_total;
      }
      ++synthetic_applied_total;
      if (logical.synthetic_late_salvage_started) {
        ++late_salvage_applied_total;
      }
      if (logical.synthetic_chain_warmup_submission) {
        ++synthetic_chain_warmup_applied_total;
      }
      if (logical.stable_pair) {
        ++stable_synthetic_applied_total;
      }
    } else {
      if (previous_sequence &&
          logical.sequence_id == previous_sequence + 1 &&
          last_applied_synthetic_commitment.source_id ==
              logical.candidate.source_id) {
        ++h14_b_applied_after_ready_s_total;
        h14_b_apply_lateness_after_ready_s_ns.Add(
            logical.candidate.target_time_ns &&
                    apply_time_ns > logical.candidate.target_time_ns
                ? apply_time_ns - logical.candidate.target_time_ns
                : 0);
      }
      ++real_applied_total;
    }
    if (planned_dispatch_time_ns &&
        apply_time_ns > planned_dispatch_time_ns) {
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        ++soft_dispatch_miss_salvaged_synthetic_total;
      } else {
        ++soft_dispatch_miss_salvaged_real_total;
      }
    }
    output.actual_apply_lead_ns =
        output.target_time_ns > apply_time_ns
            ? output.target_time_ns - apply_time_ns
            : 0;
    output.ordered_ready_wait_ns =
        output.final_ready_time_ns && apply_time_ns >= output.final_ready_time_ns
            ? apply_time_ns - output.final_ready_time_ns
            : 0;
    transaction_prepare_ns.Add(apply_time_ns - transaction_prepare_begin_ns);
    surface_apply_host_ns.Add(apply_return_time_ns - apply_time_ns);
    if (planned_dispatch_time_ns) {
      const uint64_t phase_error_ns =
          apply_time_ns > planned_dispatch_time_ns
              ? apply_time_ns - planned_dispatch_time_ns
              : planned_dispatch_time_ns - apply_time_ns;
      apply_call_phase_error_ns.Add(phase_error_ns);
      if (apply_time_ns >= planned_dispatch_time_ns) {
        apply_call_phase_late_ns.Add(apply_time_ns - planned_dispatch_time_ns);
      } else {
        apply_call_phase_early_ns.Add(planned_dispatch_time_ns - apply_time_ns);
      }
    }
    actual_apply_lead_ns.Add(output.actual_apply_lead_ns);
    ordered_ready_wait_ns.Add(output.ordered_ready_wait_ns);
    RefreshDispatchLeadEstimate();
    ++transaction_applied_total;
    if (logical.candidate.publish_time_ns &&
        output.apply_time_ns >= logical.candidate.publish_time_ns) {
      publish_to_apply_ns.Add(output.apply_time_ns -
                              logical.candidate.publish_time_ns);
    }
    if (logical.candidate.ready_time_ns &&
        output.apply_time_ns >= logical.candidate.ready_time_ns) {
      ready_to_apply_ns.Add(output.apply_time_ns -
                            logical.candidate.ready_time_ns);
    }
    // Logical sequence includes optional Synthetic positions. PumpLogicalOrder
    // consumes every position in order, including explicit drops, so applied
    // transactions may legitimately have gaps. They must only be strictly
    // monotonic here; duplicates and regressions remain fatal.
    last_applied_logical_sequence.store(logical.sequence_id,
                                        std::memory_order_release);
    if (logical.candidate.kind == CandidateKind::kReal) {
      last_applied_source_id.store(logical.candidate.source_id,
                                   std::memory_order_release);
      last_applied_real_commitment = logical.candidate;
    } else {
      last_applied_synthetic_commitment = logical.candidate;
      if (SyntheticPairSourceDistance(logical.candidate) > 1) {
        ++accepted_gap_synthetic_applied_total;
      }
    }
    if (logical.candidate.target_time_ns) {
      AccountSemanticHoldsBefore(logical.candidate,
                                 logical.candidate.semantic_tick_index);
      last_applied_semantic_target_ns = logical.candidate.target_time_ns;
      last_accounted_semantic_target_ns = std::max(
          last_accounted_semantic_target_ns,
          logical.candidate.target_time_ns);
      RecordOrderedBoundaryProvenance(
          logical.candidate.kind == CandidateKind::kReal
              ? OrderedBoundaryKind::kAppliedReal
              : OrderedBoundaryKind::kAppliedSynthetic,
          logical.candidate.source_id, logical.sequence_id,
          logical.candidate.semantic_tick_index,
          logical.candidate.target_time_ns);
      AdvanceSemanticCursorToTick(
          logical.candidate,
          SaturatingAddNs(logical.candidate.semantic_tick_index, 1));
      if (logical.candidate.kind == CandidateKind::kSynthetic) {
        ++semantic_synthetic_tick_total;
      } else {
        ++semantic_real_tick_total;
      }
      if (semantic_apply_diagnostic_total < 8) {
        ++semantic_apply_diagnostic_total;
      }
    }
    last_transaction_output_slot = logical.final_output_index;
    transaction_in_flight_high_water = std::max(
        transaction_in_flight_high_water, CountTransactionsInFlight());
    return ApplyFinalOutputResult::kApplied;
  }

  uint32_t CountTransactionsInFlight() const {
    uint32_t count = 0;
    for (const FinalOutputSlot& output : final_outputs) {
      const FinalOutputState state =
          output.state.load(std::memory_order_acquire);
      count += state == FinalOutputState::kTransactionApplied;
    }
    return count;
  }

  bool PumpLogicalOrder() {
    if (terminal_reason.load(std::memory_order_acquire) !=
        TerminalReason::kNone) {
      return false;
    }
    bool progress = false;
    bool late_real_applied_this_pump = false;
    // E1F-SHADOW pump-local observational state. It must not persist beyond this
    // PumpLogicalOrder() invocation: shadow origin is the E1 late Real of the
    // current pump only, not a new policy authority.
    bool e1f_same_pump_adjacency_counted = false;
    const uint64_t pump_now_ns = PresenterMonotonicTimeNs();
    const bool semantic_cursor_late =
        stable_latency_metronome_armed && next_semantic_target_ns &&
        next_semantic_target_ns <= pump_now_ns;
    if (semantic_cursor_late) {
      semantic_cursor_late_streak =
          SaturatingAddNs(semantic_cursor_late_streak, 1);
      semantic_cursor_late_streak_max =
          std::max(semantic_cursor_late_streak_max,
                   semantic_cursor_late_streak);
    } else {
      semantic_cursor_late_streak = 0;
    }
    const uint64_t late_ready_not_head =
        CountLateReadyRealsBehindHead(pump_now_ns);
    real_late_blocked_not_head_total += late_ready_not_head;
    real_late_blocked_total += late_ready_not_head;
    for (;;) {
      LogicalOutput* logical = FindLogicalBySequence(next_apply_sequence);
      if (!logical) {
        break;
      }
      if (logical->synthetic_presentation_retired) {
        // H13 hard safety: a retired presentation position was already
        // consumed, so it must never be the ordered head again. If it is, the
        // executive would stall behind a ticket that owes no presentation and
        // the sovereign Real behind it could never advance.
        ++presentation_retired_blocked_real_total;
        XELOGE("ZeroFGPresentationRetiredHead source={} sequence={} next={}",
               logical->candidate.source_id, logical->sequence_id,
               next_apply_sequence);
        break;
      }
      const uint32_t logical_index =
          uint32_t(logical - logical_outputs.data());
      uint64_t now = PresenterMonotonicTimeNs();
      if (logical->candidate.kind == CandidateKind::kSynthetic) {
        ObserveSyntheticNominalLateness(*logical, now);
      }
      if (logical->candidate.kind == CandidateKind::kReal &&
          !logical->candidate.target_time_ns &&
          logical->candidate.earliest_eligible_time_ns) {
        // E1F-SHADOW: a next ordered Real without a target has reached the
        // single shallow-commitment point inside the same PumpLogicalOrder()
        // invocation in which an E1 late Real already applied. Observation
        // only: the CommitHeadRealToSemanticTick() below is not blocked,
        // retargeted, reanchored, nor is the cursor touched. Count only the
        // first such next Real per pump. The old-cursor question is re-decided
        // here against the live cursor, not from the post-A result: between A
        // and B the Synthetic S_AB may have been applied or accounted as
        // HOLD/drop and advanced next_semantic_target_ns, so the post-A
        // mismatch must not be reused. late_real_applied_this_pump implies the
        // metronome is armed and the executive is running, so the recompute
        // inputs are valid here.
        if (late_real_applied_this_pump && !e1f_same_pump_adjacency_counted) {
          e1f_same_pump_adjacency_counted = true;
          ++e1f_next_real_commit_same_pump_total;
          const uint64_t e1f_commit_boundary_first_target_ns =
              ComputeE1FShadowFirstTargetNs(PresenterMonotonicTimeNs());
          const uint64_t e1f_commit_boundary_impulse_value_ns =
              e1f_commit_boundary_first_target_ns > next_semantic_target_ns
                  ? e1f_commit_boundary_first_target_ns - next_semantic_target_ns
                  : 0;
          // Recorded for every first same-pump adjacency, including a no-op
          // (zero) when the cursor is no longer behind the safe floor at this
          // boundary. This keeps the window's population equal to
          // e1f_next_real_commit_same_pump_total so the quantiles describe the
          // full adjacency set, not only the still-behind subset.
          e1f_commit_boundary_impulse_ns.Add(
              e1f_commit_boundary_impulse_value_ns);
          if (e1f_commit_boundary_first_target_ns > next_semantic_target_ns) {
            ++e1f_same_pump_commit_would_use_old_cursor_total;
          }
        }
        // Physical acceptance may be deep to preserve Source sovereignty.
        // Executive commitment is deliberately shallow: only the ordered
        // head receives a concrete ordinary lattice opportunity.
        ObserveUnifiedRunwayShadow(now);
        ObserveUnifiedRunwayShadowCommitBoundary(*logical, now);
        if (e2b.pending) {
          // E2B-R is DEMOTED to shadow by H12. Its safety and robust-P
          // mechanisms both passed, but runtime efficacy failed: the reform
          // applied correctly and Synthetic stayed dead, falsifying
          // micro-frequency mismatch as a sufficient cause for the round9
          // collapse. A disproven canary must not keep behavioral authority
          // over the production clock, so the armed witness is retired here as
          // counterfactual telemetry and no reform is performed. The robust
          // Source-rate/reanchor machinery for genuine operating-point changes
          // is untouched.
          e2b.pending = false;
          ++e2b.would_apply_total;
        }
        if (!CommitHeadRealToSemanticTick(logical_index, now)) {
          Terminalize(TerminalReason::kOrderingFailure,
                      logical->candidate.source_id);
          return false;
        }
        now = PresenterMonotonicTimeNs();
        progress = true;
      }
      // H14: B is a factual boundary, not a predicted competitor. A ready S is
      // always serialized before B. An unfinished S yields its ordered position
      // only when its sovereign B is physically FinalReady and its factual
      // pair-local eligibility boundary is due. B remains targetless while it
      // is behind S; dispatch_lead and the historical pair-safe deadline have
      // no authority here. Production stays live through the H13 retirement
      // path after the semantic HOLD.
      if (logical->candidate.kind == CandidateKind::kSynthetic &&
          logical->state != LogicalOutputState::kDropped &&
          !logical->synthetic_presentation_retired) {
        const LogicalOutput* real_b =
            FindRollingPairBoundaryReal(*logical);
        const bool b_due_and_ready =
            real_b && real_b->candidate.earliest_eligible_time_ns &&
            now >= real_b->candidate.earliest_eligible_time_ns &&
            real_b->state == LogicalOutputState::kFinalReady;
        if (b_due_and_ready) {
          if (logical->state == LogicalOutputState::kFinalReady) {
            ++h14_b_due_ready_s_ready_total;
          } else {
            // Split Always-S has no H14 presentation retirement authority:
            // B remains behind the owed S until S is actually FinalReady.
            ++split_b_ready_waiting_for_s_total;
            break;
            ++h14_b_due_ready_s_not_ready_hold_total;
            RetireSyntheticPresentation(logical_index, now);
            progress = true;
            continue;
          }
        }
      }
      if (logical->state == LogicalOutputState::kDropped) {
        if (logical->candidate.kind == CandidateKind::kSynthetic &&
            logical->candidate.target_time_ns &&
            logical->candidate.target_time_ns >
                last_accounted_semantic_target_ns) {
          AccountSemanticHoldsBefore(
              logical->candidate,
              logical->candidate.semantic_tick_index);
          ++semantic_hold_total;
          last_accounted_semantic_target_ns =
              logical->candidate.target_time_ns;
          RecordOrderedBoundaryProvenance(
              OrderedBoundaryKind::kExplicitSHold,
              logical->candidate.source_id, logical->sequence_id,
              logical->candidate.semantic_tick_index,
              logical->candidate.target_time_ns);
          AdvanceSemanticCursorToTick(
              logical->candidate,
              SaturatingAddNs(logical->candidate.semantic_tick_index, 1));
        }
        ++next_apply_sequence;
        // A dropped S is an explicit HOLD in the semantic clock. The
        // following sovereign Real retains its frozen target; pulling it
        // forward would compress O and make feasibility jitter move Tn.
        if (!LogicalHasJob(logical_index, logical->token)) {
          *logical = {};
        }
        progress = true;
        continue;
      }
      const bool real_target_expired =
          logical->candidate.kind == CandidateKind::kReal &&
          logical->candidate.target_time_ns &&
          now >= logical->candidate.target_time_ns;
      const bool real_target_order_blocked =
          logical->candidate.kind == CandidateKind::kReal &&
          logical->candidate.target_time_ns &&
          last_accounted_semantic_target_ns &&
          logical->candidate.target_time_ns <=
              last_accounted_semantic_target_ns;
      const bool real_late_eligible =
          real_target_expired &&
          logical->state == LogicalOutputState::kFinalReady &&
          !real_target_order_blocked;
      if (logical->candidate.kind == CandidateKind::kReal &&
          logical->candidate.target_time_ns &&
          (real_target_expired || real_target_order_blocked)) {
        const bool late_salvage_live = CountLiveLateSyntheticSalvage() != 0;
        if (real_target_expired) {
          if (late_salvage_live) {
            ++real_hard_miss_while_late_salvage_total;
          }
          ObserveHardReadyDispatchMiss(*logical, now);
          RecordHardTargetMiss(*logical, now);
        }
        if (real_late_eligible) {
          ++real_late_eligible_total;
          ++real_old_policy_would_defer_total;
          if (late_real_applied_this_pump) {
            ++real_late_blocked_total;
            ++real_multi_late_apply_same_pump_total;
            break;
          }
        } else {
          ++real_late_blocked_total;
          if (real_target_order_blocked) {
            ++real_late_blocked_order_total;
          } else {
            ++real_late_blocked_structural_total;
          }
        }
        if (!real_late_eligible) {
          if (real_target_order_blocked) {
            ++split_live_real_retarget_total;
            XELOGE("ZeroFGSplitRealTargetOrderBlocked source={} sequence={} "
                   "action=fail_open",
                   logical->candidate.source_id, logical->sequence_id);
            Terminalize(TerminalReason::kOrderingFailure,
                        logical->candidate.source_id);
          }
          // A targetless/unfinished Real remains sovereign and waits for
          // physical completion. Split never rewrites a live target to hide
          // plant lateness or create catch-up ticks.
          break;
          if (!DeferRealToNextSemanticTick(logical_index, now)) {
            break;
          }
        }
        if (!real_late_eligible && late_salvage_live) {
          ++real_defer_while_late_salvage_total;
        }
        if (!real_late_eligible) {
          progress = true;
          continue;
        }
      }
      if (logical->candidate.kind == CandidateKind::kSynthetic &&
          !logical->candidate.target_time_ns) {
        // H12 MANDATORY: presentation authority begins only after the
        // pair-local anchor exists. An unanchored S may be fully produced but
        // must never reach ApplyFinalOutput(), never reach the swapchain,
        // never be treated as planned_dispatch=0 and never count
        // as a semantic HOLD. It waits here; ordering preflight downstream
        // assumes semantic fields are present. Structural orphans are retired
        // causally by ReclaimStructurallyOrphanedSynthetics(), not by timeout.
        ++synthetic_apply_blocked_unanchored_total;
        break;
      }
      if (logical->state != LogicalOutputState::kFinalReady) {
        break;
      }
      if (logical->candidate.target_time_ns &&
          now < PlannedDispatchTimeNs(*logical)) {
        break;
      }
      if (!PostCompletionReadyForApply(logical_index, *logical,
                                       PresenterMonotonicTimeNs())) {
        // The ordered head remains FinalReady and is retried by the active
        // PostJob's existing one-millisecond observation wake. No apply,
        // sequence advance, drop or semantic retarget occurs here.
        break;
      }
      const ApplyFinalOutputResult apply_result = ApplyFinalOutput(*logical);
      if (apply_result == ApplyFinalOutputResult::kNotReady) {
        // Egress not ready (B not producing yet, or the ring is full): the
        // ordered head stays in place and is retried, never skipped.
        break;
      }
      if (apply_result == ApplyFinalOutputResult::kFailure) {
        return false;
      }
      if (logical->candidate.kind == CandidateKind::kReal) {
        if (real_late_eligible) {
          ++real_late_applied_total;
          late_real_applied_this_pump = true;
          // E1F-SHADOW: the E1 late-apply has actually completed (ApplyFinalOutput
          // returned kApplied for an ordered FinalReady Real past its target).
          // Measure, without touching any pacing state, whether the future
          // semantic cursor now trails the causally-safe floor and by how much.
          if (stable_latency_metronome_armed && stable_output_quantum_ns &&
              next_semantic_target_ns) {
            const uint64_t e1f_shadow_now_ns = PresenterMonotonicTimeNs();
            const uint64_t e1f_shadow_first_target_ns =
                ComputeE1FShadowFirstTargetNs(e1f_shadow_now_ns);
            ++e1f_shadow_request_total;
            if (e1f_shadow_first_target_ns > next_semantic_target_ns) {
              ++e1f_shadow_would_move_total;
              e1f_shadow_impulse_ns.Add(e1f_shadow_first_target_ns -
                                        next_semantic_target_ns);
            } else {
              ++e1f_shadow_noop_total;
            }
          }
        }
        ObserveNewRealExecutiveDebt(
            CandidateExecutiveDebtTicks(logical->candidate), 0);
      }
      *logical = {};
      ++next_apply_sequence;
      progress = true;
    }
    return progress;
  }

  bool PumpLogicalOrderObserved() {
    const uint64_t begin_ns = PresenterMonotonicTimeNs();
    const bool progress = PumpLogicalOrder();
    const uint64_t duration_ns = PresenterMonotonicTimeNs() - begin_ns;
    pump_logical_order_ns.Add(duration_ns);
    pump_logical_order_cpu_total_ns =
        SaturatingAddNs(pump_logical_order_cpu_total_ns, duration_ns);
    ++pump_logical_order_call_total;
    return progress;
  }

  // H13: sweep for production tickets whose presentation window closed while
  // they were still producing and which have since reached FinalReady. Run
  // once per cycle so the submit/poll paths keep their own bookkeeping intact.
  void RetireCompletedPresentationRetiredSynthetics() {
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      FinalizeRetiredSyntheticProduction(i);
    }
  }

  bool CleanupDroppedLogicalOutputs() {
    bool progress = false;
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& logical = logical_outputs[i];
      if (logical.state == LogicalOutputState::kDropped &&
          logical.sequence_id < next_apply_sequence &&
          !LogicalHasJob(i, logical.token)) {
        logical = {};
        progress = true;
      }
    }
    return progress;
  }

  // Split sidecar. incoming_real (depth 1) is an accepted Real not yet
  // materialized as S+B. It waits only for materialization capacity (room
  // for the Real and its mandatory Synthetic: the former pre-acceptance
  // gate), never for an anchor or the metronome, and is promoted the moment
  // that capacity returns. The wait keeps identity and the original
  // issue/publish timestamps, creates no S, moves no cursor and teaches
  // neither P nor D. Live unanchored Synthetics are counted for telemetry
  // only: an unanchored S is the normal H12 state, not a hold reason.
  uint32_t CountLiveUnanchoredSynthetics() const {
    uint32_t count = 0;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state != LogicalOutputState::kFree &&
          logical.state != LogicalOutputState::kDropped &&
          logical.candidate.kind == CandidateKind::kSynthetic &&
          !logical.candidate.target_time_ns) {
        ++count;
      }
    }
    return count;
  }

  // Bound B, the Production Frontier: physical backing owned plus admitted S
  // obligations that do not own backing yet. An S leaves the frontier when its
  // backing is released at Post COMPLETION -- not at apply, which is what the
  // logical ticket cap waited for and is the whole of blocker #20.
  // An S owns physical backing from the moment Generation is submitted. Its
  // candidate.synthetic_index is NOT assigned until that Generation completes,
  // so during an active Generation the link must come from the job, not from
  // the candidate. Reading only the candidate made the same S count once as
  // owned backing and again as an unbacked obligation, and the frontier hit
  // its bound with the physical pool half idle.
  bool LogicalOwnsSyntheticBacking(uint32_t logical_index,
                                   const LogicalOutput& logical) const {
    if (LogicalHasJob(logical_index, logical.token)) {
      return true;
    }
    const uint32_t index = logical.candidate.synthetic_index;
    return index < kSyntheticPoolSize &&
           ((synthetic_backing_owned_mask >> index) & 1u);
  }

  uint32_t CountSyntheticProductionFrontier() const {
    // Owned backing is exact and counts once, whoever holds it.
    uint32_t frontier = CountSyntheticBackingOwned();
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      const LogicalOutput& logical = logical_outputs[i];
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped ||
          logical.candidate.kind != CandidateKind::kSynthetic) {
        continue;
      }
      if (logical.state == LogicalOutputState::kFinalReady) {
        continue;  // Past Post: presentation ownership only, no credit.
      }
      if (LogicalOwnsSyntheticBacking(i, logical)) {
        continue;  // Already in the owned count above.
      }
      ++frontier;  // An admitted obligation with no backing yet.
    }
    return frontier;
  }

  // Room to materialize the next Real and its mandatory Synthetic.
  bool SplitMaterializationCapacityAvailable() const {
    if (kLogicalOutputCapacity - CountLogicalOutputs() < 2) {
      return false;
    }
    // NOTE: kLogicalSyntheticCapacity is the same numeral as the frontier
    // bound only because both derive from kSyntheticPoolSize + 1. The
    // quantities differ exactly where it matters -- a FinalReady S counts
    // against the old cap and not against the frontier -- so a runtime
    // reading that "the number did not change" is meaningless here.
    return CountSyntheticProductionFrontier() < kLogicalSyntheticCapacity;
  }

  // Phase 0 sensor 7. When the Synthetic tickets are at the cap, the decisive
  // question is what those tickets are doing: a ticket whose Post completed is
  // holding production credit purely for presentation, which is the hold that
  // must reach zero after phase 2. Read-only.
  SidecarHoldCause ClassifySidecarHold(bool residency_ready) const {
    if (!incoming_real.occupied()) {
      return SidecarHoldCause::kNone;
    }
    if (!residency_ready) {
      return SidecarHoldCause::kAwaitingResidency;
    }
    if (SplitMaterializationCapacityAvailable()) {
      return SidecarHoldCause::kNone;
    }
    // Ask which clause actually refused. The first version inferred it from
    // the live Synthetic ticket count, which the frontier deliberately stops
    // bounding -- so once tickets exceeded the old cap every hold was
    // misreported as presentation residence, including the legitimate ones.
    if (kLogicalOutputCapacity - CountLogicalOutputs() < 2) {
      return SidecarHoldCause::kLogicalIdentity;
    }
    // The frontier refused. By construction an S past Post is not in the
    // frontier, so this is production credit and nothing else. Presentation
    // residence is structurally unreachable here, which is exactly what the
    // phase 2 gate asserts.
    return SidecarHoldCause::kProductionFrontier;
  }

  // Attribution runs at pump granularity: close the open interval against the
  // previous cause, then reopen against the new one.
  void AccrueSidecarHold(SidecarHoldCause cause) {
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (split_sidecar_hold_cause != SidecarHoldCause::kNone &&
        split_sidecar_hold_cause_since_ns &&
        now_ns > split_sidecar_hold_cause_since_ns) {
      split_sidecar_hold_cause_ns[size_t(split_sidecar_hold_cause)] +=
          now_ns - split_sidecar_hold_cause_since_ns;
    }
    if (cause != split_sidecar_hold_cause) {
      if (cause != SidecarHoldCause::kNone) {
        ++split_sidecar_hold_cause_total[size_t(cause)];
      }
      split_sidecar_hold_cause = cause;
    }
    split_sidecar_hold_cause_since_ns =
        cause == SidecarHoldCause::kNone ? 0 : now_ns;
  }

  // Phase 0 sensor 1. Must run before every incoming_real.Reset() that ends a
  // live occupancy, so the residence sample and the exit reason are recorded
  // against the Real that actually held the sidecar.
  void FinishSidecarResidence(SidecarExitReason reason) {
    if (!incoming_real.occupied()) {
      return;
    }
    AccrueSidecarHold(SidecarHoldCause::kNone);
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (incoming_real.armed_ns && now_ns > incoming_real.armed_ns) {
      split_sidecar_residence_ns.Add(now_ns - incoming_real.armed_ns);
    }
    ++split_sidecar_exit_total[size_t(reason)];
  }

  // Phase 0 sensor 2. Ownership is acquired when the adapter hands us a
  // Synthetic index and released only when synthetic_release_callback runs,
  // which is exactly what clears the adapter's output_busy.
  uint32_t CountSyntheticBackingOwned() const {
    uint32_t count = 0;
    for (uint32_t i = 0; i < kSyntheticPoolSize; ++i) {
      count += (synthetic_backing_owned_mask >> i) & 1u;
    }
    return count;
  }

  void AcquireSyntheticBackingOwnership(uint32_t index) {
    if (index >= kSyntheticPoolSize) {
      return;
    }
    synthetic_backing_owned_mask |= 1u << index;
    synthetic_backing_owned_high_water = std::max(
        synthetic_backing_owned_high_water, CountSyntheticBackingOwned());
  }

  void ReleaseSyntheticBackingOwnership(uint32_t index) {
    if (index >= kSyntheticPoolSize) {
      return;
    }
    synthetic_backing_owned_mask &= ~(1u << index);
  }

  // Blocker #22 read directly: the scheduler believes backing is free that is
  // still physically owned, so it spends a host quantum on a Generation the
  // adapter must refuse.
  void ObserveSyntheticBackingAuthority() {
    const uint32_t owned = CountSyntheticBackingOwned();
    const uint32_t jobs = CountGenerationJobs();
    if (owned > jobs) {
      ++synthetic_backing_false_free_total;
      synthetic_backing_false_free_high_water = std::max(
          synthetic_backing_false_free_high_water, owned - jobs);
    }
  }

  // Phase 0 sensor 9. How much of the egress capacity Synthetic presentation
  // residence is holding: the quantity P2 must bound, measured before the fix
  // so the accident of 4 < 5 is visible rather than assumed.
  uint32_t CountFinalOutputsOwnedBySynthetic() const {
    // Physical ownership, not logical tickets: P2 bounds who holds the egress
    // slots. A slot whose logical is already gone (orphan Ready, release
    // pending) still occupies a Real-progress lane.
    uint32_t count = 0;
    for (uint32_t i = 0; i < kFinalOutputPoolSize; ++i) {
      if (final_outputs[i].state.load(std::memory_order_acquire) !=
              FinalOutputState::kFree &&
          final_outputs[i].candidate_kind == CandidateKind::kSynthetic) {
        ++count;
      }
    }
    return count;
  }

  void ObservePoolOccupancy() {
    uint32_t capture_used = 0;
    for (uint32_t i = 0; i < kCapturePoolSize; ++i) {
      capture_used += capture_slots[i].state.load(std::memory_order_acquire) !=
                      CaptureState::kFree;
    }
    pool_high_water_capture = std::max(pool_high_water_capture, capture_used);
    pool_high_water_residency =
        std::max(pool_high_water_residency, CountUsedSlots());
    pool_high_water_final_output =
        std::max(pool_high_water_final_output, CountUsedFinalOutputs());
    pool_high_water_logical_synthetic =
        std::max(pool_high_water_logical_synthetic,
                 CountLogicalOutputs(CandidateKind::kSynthetic));
    if (CountSyntheticProductionFrontier() >= kLogicalSyntheticCapacity &&
        !CountSyntheticBackingOwned() && !CountGenerationJobs()) {
      // Admitted S obligations fill the frontier while no physical backing is
      // in use at all: the credit is spoken for and the hardware is idle.
      // This is the P3 signal -- capacity that exists and is not being spent.
      ++frontier_saturated_backing_idle_total;
    }
    logical_synthetic_ticket_high_water =
        std::max(logical_synthetic_ticket_high_water,
                 CountLogicalOutputs(CandidateKind::kSynthetic));
    const uint32_t synthetic_egress = CountFinalOutputsOwnedBySynthetic();
    pool_high_water_final_output_synthetic =
        std::max(pool_high_water_final_output_synthetic, synthetic_egress);
    if (synthetic_egress >= kFinalOutputPoolSize) {
      // P2 violated: no Real-progress lane is free of Synthetic ownership.
      ++final_output_synthetic_owns_all_total;
    } else if (synthetic_egress + 1 == kFinalOutputPoolSize) {
      // Exactly one lane left. Today this is the accident that keeps Real
      // egress alive, and it is the margin removing the S cap would destroy.
      ++final_output_synthetic_owns_all_but_one_total;
    }
  }

  // P3. An occupied sidecar whose promotion is currently refused has a
  // progress claim: the chain that would release it is Source-critical, and
  // discretionary suppression may not stand in front of it. Read-only.
  bool SidecarProgressCritical() const {
    return incoming_real.occupied() &&
           !SplitMaterializationCapacityAvailable();
  }

  bool SplitSidecarPromotionHeld() {
    if (SplitMaterializationCapacityAvailable()) {
      return false;
    }
    ++split_preaccept_capacity_gate_total;
    const uint64_t now_ns = PresenterMonotonicTimeNs();
    if (!split_sidecar_hold_begin_ns) {
      split_sidecar_hold_begin_ns = now_ns;
      split_sidecar_hold_alarm_raised = false;
      ++split_sidecar_hold_total;
      if (CountLogicalOutputs(CandidateKind::kSynthetic) >=
          kLogicalSyntheticCapacity) {
        ++split_sidecar_hold_s_capacity_total;
      }
      ObserveSplitSidecarHoldStart(now_ns);
    }
    if (!split_sidecar_hold_alarm_raised &&
        now_ns - split_sidecar_hold_begin_ns >= kSplitSidecarHoldAlarmNs) {
      split_sidecar_hold_alarm_raised = true;
      if (!split_sidecar_hold_timeout_total++) {
        XELOGW(
            "ZeroFGSplitSidecar hold_over_1s source={} logical={} S={} "
            "action=keep_holding",
            incoming_real.source_id, CountLogicalOutputs(),
            CountLogicalOutputs(CandidateKind::kSynthetic));
      }
    }
    return true;
  }

  // What fills the Synthetic capacity when a Real starts waiting in the
  // sidecar: production (Generation/Post) or presentation residence
  // (FinalReady S waiting for its tick). Telemetry only.
  void ObserveSplitSidecarHoldStart(uint64_t now_ns) {
    uint32_t generation = 0;
    uint32_t post = 0;
    uint32_t submitted = 0;
    uint32_t ready = 0;
    uint32_t anchored = 0;
    uint64_t nearest_ticks = 0;
    uint64_t farthest_ticks = 0;
    uint64_t latest_real_sequence = 0;
    bool predecessor_committed = false;
    for (const LogicalOutput& logical : logical_outputs) {
      if (logical.state == LogicalOutputState::kFree ||
          logical.state == LogicalOutputState::kDropped) {
        continue;
      }
      if (logical.candidate.kind == CandidateKind::kReal) {
        if (logical.sequence_id >= latest_real_sequence) {
          latest_real_sequence = logical.sequence_id;
          predecessor_committed = logical.candidate.target_time_ns != 0;
        }
        continue;
      }
      switch (logical.state) {
        case LogicalOutputState::kWaitingGeneration:
          ++generation;
          break;
        case LogicalOutputState::kWaitingPost:
          ++post;
          break;
        case LogicalOutputState::kPostSubmitted:
          ++submitted;
          break;
        case LogicalOutputState::kFinalReady:
          ++ready;
          break;
        default:
          break;
      }
      if (logical.candidate.target_time_ns && stable_output_quantum_ns) {
        const uint64_t ticks =
            logical.candidate.target_time_ns > now_ns
                ? (logical.candidate.target_time_ns - now_ns) /
                      stable_output_quantum_ns
                : 0;
        nearest_ticks = anchored ? std::min(nearest_ticks, ticks) : ticks;
        farthest_ticks = std::max(farthest_ticks, ticks);
        ++anchored;
      }
    }
    split_sidecar_hold_s_generation_total += generation;
    split_sidecar_hold_s_post_total += post;
    split_sidecar_hold_s_submitted_total += submitted;
    split_sidecar_hold_s_ready_total += ready;
    split_sidecar_hold_s_anchored_total += anchored;
    if (predecessor_committed) {
      ++split_sidecar_hold_predecessor_committed_total;
    }
    const uint64_t sample = split_sidecar_hold_total;
    if (sample > 8 && sample % 64) {
      return;
    }
  }

  // The gate let this Real through; no timing has been computed yet, so a
  // pending reanchor may still consume its promotion time.
  void ObserveSplitSidecarPromotionStart() {
    if (preadmitted) {
      ++split_sidecar_promote_with_preadmitted_total;
    }
    if (split_sidecar_hold_begin_ns &&
        (source_phase_reanchor_pending ||
         source_rate_transition_reanchor_pending ||
         latency_depth_reanchor_pending)) {
      ++split_sidecar_promote_reanchor_pending_total;
    }
  }

  void ObserveSplitSidecarPromotionDone() {
    if (split_sidecar_hold_begin_ns) {
      split_sidecar_hold_ns.Add(PresenterMonotonicTimeNs() -
                                split_sidecar_hold_begin_ns);
      split_sidecar_hold_begin_ns = 0;
    }
    // The <= 1 invariant is the armed Always-S contract; warmup pairs made
    // before arming are not held by the gate.
    if (SplitAlwaysSArmed()) {
      split_live_unanchored_synthetic_high_water =
          std::max(split_live_unanchored_synthetic_high_water,
                   CountLiveUnanchoredSynthetics());
    }
  }

  void ObserveSourceReanchorAnchor(bool live_anchor_wins) {
    if (!live_anchor_wins) {
      return;
    }
    ++source_reanchor_live_anchor_won_total;
    if (split_sidecar_hold_begin_ns) {
      ++source_reanchor_live_anchor_won_held_total;
      // Blocker #11. The live anchor won because our own sidecar hold pushed
      // the issue anchor into the past, so the epoch origin it produces is
      // ours. Phase debt measured against that origin is our promotion delay
      // read back as Source drift, and that reading is what later asks for a
      // reanchor - a loop with our hold on both ends.
      //
      // The anchor itself stands: an epoch cannot begin in the past. Only the
      // attribution is refused.
      ResetPhaseDebt();
      ++source_phase_debt_ours_reset_total;
    }
  }

  // Ingress promotion, then the ordered Pump. When that Pump frees the
  // capacity the sidecar was waiting for, promote once more in the same pump
  // instead of waiting for the next wake (bounded: one extra pass).
  bool ProcessIncomingRealAndPump() {
    bool progress = ProcessIncomingReal();
    progress |= PumpLogicalOrderObserved();
    if (split_sidecar_hold_begin_ns && incoming_real.occupied() &&
        SplitMaterializationCapacityAvailable() && ProcessIncomingReal()) {
      ++split_sidecar_same_pump_promotion_total;
      progress = true;
      progress |= PumpLogicalOrderObserved();
    }
    return progress;
  }

  uint32_t FindIncomingResidencySlot() const {
    if (!incoming_real.occupied()) {
      return UINT32_MAX;
    }
    for (uint32_t i = 0; i < kPoolSize; ++i) {
      const Slot& slot = slots[i];
      const SlotState state = slot.state.load(std::memory_order_acquire);
      if ((state != SlotState::kTransferSubmitted &&
           state != SlotState::kReady) ||
          slot.runtime_admitted ||
          slot.source_id.load(std::memory_order_relaxed) !=
              incoming_real.source_id) {
        continue;
      }
      return i;
    }
    return UINT32_MAX;
  }

  bool EnqueueRealOutput(uint32_t slot_index,
                         uint64_t earliest_eligible_time_ns,
                         uint64_t pair_lattice_distance,
                         bool semantic_forward_skipped, bool stable_pair,
                         uint64_t pair_a_source_id = 0,
                         PairContract pair_contract = PairContract::kNormal) {
    if (stable_latency_metronome_armed && !earliest_eligible_time_ns) {
      ++armed_real_without_eligibility;
      XELOGE("ZeroFGArmedRealWithoutEligibility source={}",
             slots[slot_index].source_id.load(std::memory_order_relaxed));
      Terminalize(TerminalReason::kOrderingFailure,
                  slots[slot_index].source_id.load(std::memory_order_relaxed));
      return false;
    }
    OutputCandidate real =
        MakeRealCandidate(slot_index, earliest_eligible_time_ns);
    real.pair_lattice_distance = pair_lattice_distance;
    real.pair_a_source_id = pair_a_source_id;
    real.pair_contract = pair_contract;
    RetainRealSlot(slot_index);
    const uint32_t logical = AllocateLogicalOutput(
        real, LogicalOutputState::kWaitingPost);
    if (logical == UINT32_MAX) {
      ReleaseRealSlot(slot_index);
      RefreshPipelineSnapshot();
      Terminalize(TerminalReason::kLogicalOutputCapacityExhausted,
                  real.source_id);
      return false;
    }
    LogicalOutput& output = logical_outputs[logical];
    output.stable_pair = stable_pair;
    output.semantic_forward_skipped = semantic_forward_skipped;
    output.post_residence_budget_ns = PostResidenceEstimateNs();
    ConfigureLogicalStaging(output, false);
    ObservePlannedSpaceRealEnqueue(output);
    return true;
  }

  bool ProcessIncomingReal() {
    if (!accepting.load(std::memory_order_acquire) ||
        terminal_reason.load(std::memory_order_acquire) != TerminalReason::kNone) {
      return false;
    }
    bool progress = false;
    bool split_always_s = SplitAlwaysSArmed();
    for (;;) {
      const uint32_t selected = FindIncomingResidencySlot();
      if (selected == UINT32_MAX) {
        AccrueSidecarHold(ClassifySidecarHold(false));
        break;
      }
      // Sidecar gate before any per-promotion telemetry or state change: a
      // held Real is re-examined on every pass.
      if (SplitSidecarPromotionHeld()) {
        AccrueSidecarHold(ClassifySidecarHold(true));
        break;
      }
      AccrueSidecarHold(SidecarHoldCause::kNone);
      ObserveSplitSidecarPromotionStart();
      Slot& current = slots[selected];
      const uint64_t issue_ns =
          current.issue_time_ns.load(std::memory_order_relaxed);
      const uint64_t current_source_id =
          current.source_id.load(std::memory_order_relaxed);
      const uint64_t source_issue_sequence =
          last_processed_source_issue_period_sequence;
      UpdateH1cEpisode(PresenterMonotonicTimeNs(), source_issue_sequence);
      const uint64_t publish_ns =
          current.publish_time_ns.load(std::memory_order_relaxed);
      if (issue_ns && publish_ns >= issue_ns) {
        source_issue_to_publish_ns.Add(publish_ns - issue_ns);
      }
      source_protection_pair_floor =
          current.source_id.load(std::memory_order_relaxed);
      RetainRealSlot(selected);  // History ownership.
      uint64_t synthetic_semantic_ns = 0;
      uint64_t real_earliest_eligible_ns = 0;
      uint64_t synthetic_target_ns = 0;
      uint64_t synthetic_tick_index = 0;
      uint64_t synthetic_epoch_origin_ns = 0;
      uint64_t synthetic_quantum_ns = 0;
      uint64_t pair_a_source_id = 0;
      uint64_t pair_a_issue_time_ns = 0;
      uint64_t pair_a_policy_wait_ns = 0;
      uint64_t pair_source_distance = 0;
      uint64_t pair_lattice_distance = 0;
      bool synthetic_eligible = false;
      bool stable_pair = false;
      bool pair_targets_valid = false;
      bool semantic_forward_skipped = false;
      bool h1c_transition_real_only = false;
      bool h1c_postconfirm_quarantined = false;
      uint32_t synthetic_logical = UINT32_MAX;
      if (history_slot >= kPoolSize) {
        RecordSyntheticDrop(SyntheticDropReason::kNoHistory);
      } else {
        const Slot& previous = slots[history_slot];
        const uint64_t previous_source =
            previous.source_id.load(std::memory_order_relaxed);
        const uint64_t current_source =
            current.source_id.load(std::memory_order_relaxed);
        const bool generation_compatible =
            previous.generation.load(std::memory_order_relaxed) ==
            current.generation.load(std::memory_order_relaxed);
        const bool extent_compatible =
            previous.storage_content_extent.width ==
                current.storage_content_extent.width &&
            previous.storage_content_extent.height ==
                current.storage_content_extent.height &&
            previous.source_content_extent.width ==
                current.source_content_extent.width &&
            previous.source_content_extent.height ==
                current.source_content_extent.height;
        if (!generation_compatible || !extent_compatible) {
          // A lifecycle or geometry change between accepted Reals is a real
          // structural discontinuity: no A/B pair can span it.
          RecordSyntheticDrop(SyntheticDropReason::kDiscontinuity);
          ReleaseRealSlot(selected);  // Undo provisional History ownership.
          FailStructuralTimingEpoch(
              !generation_compatible ? "generation_change" : "extent_change",
              current_source);
          return false;
        }
        if (!previous_source) {
          // No valid accepted predecessor identity; B is published alone.
          RecordSyntheticDrop(SyntheticDropReason::kNoHistory);
        } else if (!AcceptedPairSourceDistance(previous_source,
                                               current_source,
                                               pair_source_distance)) {
          // Ingress already rejects duplicate or regressed accepted IDs, so a
          // non-increasing pair here is an accepted-Real ordering violation.
          ++ordering_error_total;
          ReleaseRealSlot(selected);  // Undo provisional History ownership.
          Terminalize(TerminalReason::kOrderingFailure, current_source);
          return false;
        } else {
          // Accepted adjacency chooses the pair: history/current are
          // consecutive in the accepted Real stream by construction. A
          // source-id gap only means the latest-wins mailbox never delivered
          // the intermediate IssueSwaps; they were never accepted and carry no
          // sovereignty. The gap no longer suppresses S; the pair is placed at
          // its phase-bounded lattice distance instead.
          pair_a_source_id = previous_source;
          pair_a_issue_time_ns =
              previous.issue_time_ns.load(std::memory_order_relaxed);
          pair_a_policy_wait_ns =
              previous.policy_wait_ns.load(std::memory_order_relaxed);
          if (pair_source_distance > 1) {
            ++accepted_gap_observation_total;
            ++accepted_gap_pair_distance_total[size_t(
                std::min<uint64_t>(pair_source_distance, 4) - 2)];
            // Burst witness: the d IssueSwaps took less Source time than
            // their count implies.
            const uint64_t a_issue_ns =
                previous.issue_time_ns.load(std::memory_order_relaxed);
            const uint64_t count_span_ns =
                SaturatingMultiplyNs(source_period_ns, pair_source_distance);
            if (a_issue_ns && issue_ns > a_issue_ns &&
                SaturatingAddNs(issue_ns - a_issue_ns, source_period_ns / 2) <
                    count_span_ns) {
              ++accepted_gap_issue_bunched_total;
              accepted_gap_issue_bunch_max_ns =
                  std::max(accepted_gap_issue_bunch_max_ns,
                           count_span_ns - (issue_ns - a_issue_ns));
            }
          }
          ++pair_opportunity_total;
          const bool valid_for_s_execution =
              IsValidSyntheticExecutionState();
          stable_pair = IsStablePacingState();
          if (stable_pair) {
            ++stable_pair_opportunity_total;
          }
          if (!valid_for_s_execution) {
            RecordSyntheticDrop(SyntheticDropReason::kPeriodNotArmed);
          } else if (!SourceProtectionAllowsSynthetic()) {
            RecordSyntheticDrop(SyntheticDropReason::kSourceProtection);
          } else if (accepting.load(std::memory_order_acquire)) {
            synthetic_eligible = true;
          }
        }
      }

      if (source_period_ns) {
        // A pair, consecutive or gap, is placed from A only when its S is
        // eligible. A Real without S keeps the ordinary cursor placement, the
        // lattice's only drain: holding no-S pairs at pair spacing removed it
        // and let latency ratchet (2026-09-10 runtime).
        pair_targets_valid = ComputeAcceptedRealTiming(
            current, synthetic_eligible ? pair_a_source_id : 0,
            synthetic_semantic_ns, real_earliest_eligible_ns,
            synthetic_target_ns, synthetic_tick_index,
            synthetic_epoch_origin_ns, synthetic_quantum_ns,
            pair_lattice_distance, semantic_forward_skipped);
      }
      // EnsureStableLatencyMetronome() may complete the qualification while
      // computing this pair. Re-evaluate before applying any shared-era
      // capacity veto so the first post-warmup pair is mandatory as well.
      split_always_s = SplitAlwaysSArmed();
      if (synthetic_eligible && pair_targets_valid && pair_a_source_id &&
          H1cMayCreateTransitionRealOnlyPair(PresenterMonotonicTimeNs(),
              last_processed_source_issue_period_sequence)) {
        // ComputeAcceptedRealTiming above already ran the metronome and
        // Better-D/phase observers for this Source issue. Re-running it for
        // the H1c projection would create a second policy observation. The
        // pair distance is already fixed when A is committed; use the pure
        // pair projection in that case. If A is still uncommitted, its later
        // ordinary commitment will anchor this explicit pair contract.
        OutputCandidate committed_a;
        if (pair_a_source_id &&
            FindRealSemanticCommitment(pair_a_source_id, committed_a)) {
          uint64_t unused_s_semantic_ns = 0;
          uint64_t planned_b_floor_ns = 0;
          uint64_t unused_s_target_ns = 0;
          uint64_t unused_s_tick = 0;
          if (!pair_lattice_distance ||
              !ComputePairTimingFromCommittedReal(
                  committed_a, pair_lattice_distance,
                  unused_s_semantic_ns, planned_b_floor_ns,
                  unused_s_target_ns, unused_s_tick, true)) {
            // Keep the ordinary admitted pair if its pair-local projection
            // cannot be proven; never turn a failed calculation into an S
            // omission.
          } else {
            h1c_transition_real_only = true;
            synthetic_eligible = false;
            synthetic_semantic_ns = 0;
            synthetic_target_ns = 0;
            synthetic_tick_index = 0;
            real_earliest_eligible_ns = planned_b_floor_ns;
          }
        } else {
          h1c_transition_real_only = true;
          synthetic_eligible = false;
          synthetic_semantic_ns = 0;
          synthetic_target_ns = 0;
          synthetic_tick_index = 0;
          // Keep the normal unanchored provisional floor. The first
          // commitment of pair A will replace/raise it using A+ld*O.
        }
        if (h1c_transition_real_only) {
          ++h1c_transition_real_only_pair_total;
          ++h1c_episode_real_only_pairs;
        }
      } else if (synthetic_eligible &&
                 transition_planned_space_state ==
                     TransitionPlannedSpaceState::kPostConfirmDrain) {
        UpdateH1cEpisode(PresenterMonotonicTimeNs(),
                         last_processed_source_issue_period_sequence);
        if (transition_planned_space_state ==
            TransitionPlannedSpaceState::kPostConfirmDrain) {
          h1c_postconfirm_quarantined = true;
          synthetic_eligible = false;
          ++h1c_post_confirm_quarantine_total;
          ++h1c_episode_quarantined_pairs;
          // This arm is ordinary targetless Real placement, not the
          // pre-confirm TransitionRealOnly A+ld*O contract. The paired
          // calculation above already ran timing observers; mirror its
          // side-effect-free no-pair projection from the current cursor.
          synthetic_semantic_ns = 0;
          synthetic_target_ns = 0;
          synthetic_tick_index = 0;
          pair_lattice_distance = 0;
          real_earliest_eligible_ns = SemanticTargetForTick(
              SaturatingAddNs(next_semantic_tick_index, 1));
        }
      }
      if (pair_source_distance > 1 && pair_targets_valid &&
          synthetic_target_ns) {
        // How far a gap B's floor sits ahead of its own acceptance. This
        // includes the operating latency D; b_phase_excess isolates the part
        // beyond B's Source phase.
        const uint64_t now_ns = PresenterMonotonicTimeNs();
        if (real_earliest_eligible_ns > now_ns) {
          accepted_gap_real_floor_lead_max_ns =
              std::max(accepted_gap_real_floor_lead_max_ns,
                       real_earliest_eligible_ns - now_ns);
        }
      }
      if (synthetic_eligible && !pair_targets_valid) {
        synthetic_eligible = false;
        RecordSyntheticDrop(SyntheticDropReason::kPeriodNotArmed);
      }
      if (synthetic_eligible && !split_always_s &&
          !SyntheticRetentionHasPhysicalRunway()) {
        // A committed Capture is already physically stranded. Optional S must
        // not acquire new A/B ownership until the concrete destination exists.
        synthetic_eligible = false;
        ++synthetic_residency_active_pressure_drop_total;
        RecordSyntheticDrop(SyntheticDropReason::kResidencyPressure);
      }
      if (synthetic_eligible && !split_always_s &&
          !SyntheticChainSubmissionCapacityAvailable()) {
        if (CountLiveLateSyntheticSalvage()) {
          ++new_s_blocked_no_s_slot_with_salvage_total;
        }
        synthetic_eligible = false;
        RecordSyntheticDrop(SyntheticDropReason::kNoSyntheticSlot);
      }
      if (synthetic_eligible && synthetic_target_ns &&
          SyntheticChainEstimatorArmed()) {
        const uint64_t now_ns = PresenterMonotonicTimeNs();
        const uint64_t remaining_ns =
            synthetic_target_ns > now_ns ? synthetic_target_ns - now_ns : 0;
        if (SyntheticChainResidenceEstimateNs() > remaining_ns) {
          // Counterfactual only: once identity, timing and physical admission
          // are valid, measured residence does not decide whether S may exist.
          ++synthetic_chain_predicted_bypassed_total;
        }
      }
      if (synthetic_eligible && !split_always_s &&
          !HasLogicalCapacityForSynthetic()) {
        if (CountLiveLateSyntheticSalvage()) {
          ++new_s_blocked_no_logical_with_salvage_total;
        }
        synthetic_eligible = false;
        RecordSyntheticDrop(SyntheticDropReason::kNoLogicalCapacity);
      }
      if (synthetic_eligible) {
        if (CountFreeCandidateSlots() <= kCandidateReserveTarget) {
          ++synthetic_residency_eager_admission_total;
        } else {
          ++synthetic_residency_surplus_admission_total;
        }
      }
      if (synthetic_eligible) {
        const uint64_t now = PresenterMonotonicTimeNs();
        last_synthetic_slack_ns =
            synthetic_target_ns > now ? synthetic_target_ns - now : 0;
        last_real_slack_ns =
            real_earliest_eligible_ns > now
                ? real_earliest_eligible_ns - now
                : 0;
        OutputCandidate synthetic;
        synthetic.kind = CandidateKind::kSynthetic;
        synthetic.source_id =
            current.source_id.load(std::memory_order_relaxed);
        synthetic.pair_a_source_id =
            pair_a_source_id;
        synthetic.pair_a_issue_time_ns = pair_a_issue_time_ns;
        synthetic.pair_a_policy_wait_ns = pair_a_policy_wait_ns;
        synthetic.pair_lattice_distance = pair_lattice_distance;
        synthetic.issue_time_ns =
            current.issue_time_ns.load(std::memory_order_relaxed);
        synthetic.publish_time_ns =
            current.publish_time_ns.load(std::memory_order_relaxed);
        synthetic.content_extent = current.source_content_extent;
        synthetic.storage_extent = current.storage_content_extent;
        synthetic.display_aspect_ratio_x = current.display_aspect_ratio_x;
        synthetic.display_aspect_ratio_y = current.display_aspect_ratio_y;
        synthetic.is_8bpc = current.is_8bpc;
        synthetic.semantic_target_time_ns = synthetic_semantic_ns;
        synthetic.target_time_ns = synthetic_target_ns;
        synthetic.semantic_epoch_origin_ns = synthetic_epoch_origin_ns;
        synthetic.semantic_output_quantum_ns = synthetic_quantum_ns;
        synthetic.semantic_tick_index = synthetic_tick_index;
        const uint64_t generation_residence =
            GenerationResidenceEstimateNs();
        const uint64_t post_residence = PostResidenceEstimateNs();
        // H12: identity and bounded physical capacity decide whether S exists.
        // The semantic anchor decides only when it may be presented, so an
        // unanchored S still enters production immediately.
        const uint32_t frontier_before_create =
            CountSyntheticProductionFrontier();
        const uint32_t logical = AllocateLogicalOutput(
            synthetic, LogicalOutputState::kWaitingGeneration);
        if (logical == UINT32_MAX) {
          if (split_always_s) {
            ++split_forbidden_s_drop_total;
            Terminalize(TerminalReason::kLogicalOutputCapacityExhausted,
                        synthetic.source_id);
            return false;
          }
          synthetic_eligible = false;
          RecordSyntheticDrop(SyntheticDropReason::kNoSyntheticSlot);
        } else {
          if (split_always_s) {
            ++split_s_obligation_created_total;
            split_s_obligation_high_water = std::max(
                split_s_obligation_high_water,
                uint64_t(CountLogicalOutputs(CandidateKind::kSynthetic)));
          }
          synthetic_logical = logical;
          if (pair_source_distance > 1) {
            ++accepted_gap_synthetic_admitted_total;
          }
          if (synthetic_target_ns) {
            last_semantic_target_ns =
                std::max(last_semantic_target_ns, synthetic_target_ns);
          }
          LogicalOutput& output = logical_outputs[logical];
          output.stable_pair = stable_pair;
          output.semantic_forward_skipped = semantic_forward_skipped;
          output.post_residence_budget_ns = post_residence;
          output.generation_residence_budget_ns = generation_residence;
          output.synthetic_admit_time_ns = PresenterMonotonicTimeNs();
          output.synthetic_unanchored_at_admit = !synthetic_target_ns;
          if (synthetic_target_ns) {
            ++synthetic_admitted_anchored_total;
            // A concrete pair-local anchor already exists, so retain the old
            // B-safe projection as counterfactual telemetry only. H14 never
            // uses it to decide whether S may be presented.
            output.synthetic_pair_safe_deadline_ns =
                SyntheticPairSafeDeadlineFromReal(real_earliest_eligible_ns,
                                                  dispatch_lead_ns);
            ConfigureLogicalStaging(output, true);
          } else {
            ++synthetic_admitted_unanchored_total;
            // H12 MANDATORY: A has no commitment yet, so real_earliest_eligible
            // here is only a provisional global-cursor projection. Deriving a
            // pair-safe observation from it would not describe a real pair.
            // Only AnchorSyntheticPairsToCommittedReal() may establish this
            // historical value from the real pair-local anchor.
            output.synthetic_pair_safe_deadline_ns = 0;
          }
          output.previous_real_slot = history_slot;
          output.current_real_slot = selected;
          RetainRealSlot(history_slot);
          RetainRealSlot(selected);
          ObservePlannedSpaceSyntheticOpportunity(
              output, real_earliest_eligible_ns, frontier_before_create);
        }
      }
      if (stable_pair && synthetic_logical == UINT32_MAX &&
          !h1c_transition_real_only && !h1c_postconfirm_quarantined) {
        ++stable_synthetic_dropped_total;
      }
      const uint64_t nominal_cursor_tick_before_real_accept =
          next_semantic_tick_index;
      const uint64_t nominal_cursor_target_before_real_accept =
          next_semantic_target_ns;
      if (!EnqueueRealOutput(
              selected,
              pair_targets_valid ? real_earliest_eligible_ns : 0,
              pair_targets_valid ? pair_lattice_distance : 0,
              pair_targets_valid && semantic_forward_skipped, stable_pair,
              h1c_transition_real_only ? pair_a_source_id : 0,
              h1c_transition_real_only ? PairContract::kTransitionRealOnly
                                       : PairContract::kNormal)) {
        return false;
      }
      if (next_semantic_tick_index !=
              nominal_cursor_tick_before_real_accept ||
          next_semantic_target_ns !=
              nominal_cursor_target_before_real_accept) {
        ++physical_real_acceptance_cursor_advance_total;
      }
      current.runtime_admitted = true;
      // Commit B1. The sidecar is depth 1, so this hold belongs to exactly
      // this Real. Read before ObserveSplitSidecarPromotionDone() closes it;
      // same clock, microseconds apart, and no ordering is disturbed. Written
      // unconditionally: an unheld promotion is a zero-length hold, and
      // skipping the write is what let a recycled slot keep a stale one.
      {
        const uint64_t now_hold_ns = PresenterMonotonicTimeNs();
        current.policy_wait_ns.store(
            split_sidecar_hold_begin_ns &&
                    now_hold_ns > split_sidecar_hold_begin_ns
                ? now_hold_ns - split_sidecar_hold_begin_ns
                : 0,
            std::memory_order_relaxed);
      }
      ObserveSplitSidecarPromotionDone();
      ++incoming_real_admit_total;
      FinishSidecarResidence(SidecarExitReason::kPromoted);
      incoming_real.Reset();
      ++incoming_real_release_total;
      if (history_slot < kPoolSize) {
        ReleaseRealSlot(history_slot);
      }
      history_slot = selected;
      progress = true;
    }
    return progress;
  }

  void BeginAsyncDrainForShutdown() {
    FinishPlannedSpaceTrace("shutdown");
    vulkan_device->DrainZeroFGForTeardown();
    // Split sidecar: an accepted Real not yet materialized has no logical
    // entry, and the drain exit never waits for it. Discard it explicitly;
    // its Residency slot was never retained, so nothing is released here.
    if (incoming_real.occupied()) {
      ++split_sidecar_drain_discard_total;
      // Phase 0 sensor 1 / sensor 6: this is the accepted-Real destruction
      // counter. Blocker #23 is closed when kLifecycleDrain stays at zero.
      FinishSidecarResidence(SidecarExitReason::kLifecycleDrain);
      incoming_real.Reset();
    }
    split_sidecar_hold_begin_ns = 0;
    for (uint32_t i = 0; i < kLogicalOutputCapacity; ++i) {
      LogicalOutput& logical = logical_outputs[i];
      // H12: kWaitingPairAnchor is unreachable; retained only for
      // telemetry/state-index stability. Not production authority.
      if (logical.state == LogicalOutputState::kWaitingPairAnchor ||
          logical.state == LogicalOutputState::kWaitingGeneration) {
        DropSyntheticLogical(i, SyntheticDropReason::kLifecycleEpoch);
      } else if (logical.state == LogicalOutputState::kWaitingPost) {
        if (logical.candidate.kind == CandidateKind::kSynthetic &&
            !LogicalHasJob(i, logical.token)) {
          ReleaseSyntheticBackingOwnership(logical.candidate.synthetic_index);
          synthetic_release_callback(logical.candidate.synthetic_index);
        } else if (logical.candidate.kind == CandidateKind::kReal) {
          ReleaseRealSlot(logical.candidate.real_slot);
        }
        logical.state = LogicalOutputState::kDropped;
      } else if (logical.state == LogicalOutputState::kFinalReady) {
        if (logical.final_output_index < kFinalOutputPoolSize) {
          ReleaseFinalOutputSlot(logical.final_output_index);
          logical.final_output_index = UINT32_MAX;
        }
        logical.state = LogicalOutputState::kDropped;
      } else if (logical.state == LogicalOutputState::kPostSubmitted) {
        logical.state = LogicalOutputState::kDropped;
      }
    }
  }

  void DetachSurfaceForFailOpen() {
    // Main Surface Authority handback, in this order: admission is already
    // closed; the egress stops (bounded join, queue idle) and B's swapchain
    // is destroyed, moving the producer to handback_gap; only then is
    // authority released. A recreates its swapchain on the UI thread, never
    // here, and an MSA session that failed stays on A.
    StopMainSurfaceEgress(true);
    final_output_authority.store(false, std::memory_order_release);
    if (terminal_reason.load(std::memory_order_acquire) !=
        TerminalReason::kNone) {
      if (!main_surface_refused.exchange(true, std::memory_order_acq_rel)) {
        if (!std::strcmp(main_surface_refusal_reason.load(
                             std::memory_order_acquire),
                         "none")) {
          main_surface_refusal_reason.store(
              TerminalReasonName(
                  terminal_reason.load(std::memory_order_acquire)),
              std::memory_order_release);
        }
      }
      XELOGW(
          "ZeroFGMainSurface handback reason={} producer={} "
          "action=ui_reconnect_native_A session_refused=true",
          main_surface_refusal_reason.load(std::memory_order_acquire),
          ZeroFGMainSurfaceProducer::Name(main_surface_producer.state()));
      RequestMainSurfaceUIPaint();
    }
  }

  void ApplyThreadPriority() {
    // ZeroFG is cheap on the CPU and almost all GPU. When elevated, the
    // presenter takes the highest nice the process may set, falling back
    // until one is accepted, so a timed wake is not lost to the Source's
    // threads.
    static constexpr int kElevatedNiceLadder[] = {-20, -19, -16, -12,
                                                  -8,  -4,  -2};
    int requested_nice = 0;
    errno = 0;
    int set_result = 0;
    if (elevated_presenter_priority) {
      set_result = -1;
      for (const int nice_value : kElevatedNiceLadder) {
        requested_nice = nice_value;
        if (setpriority(PRIO_PROCESS, gettid(), nice_value) == 0) {
          set_result = 0;
          break;
        }
      }
    }
    errno = 0;
    const int effective_nice = getpriority(PRIO_PROCESS, gettid());
    const int priority_errno = errno;
    presenter_priority_effective.store(effective_nice,
                                       std::memory_order_relaxed);
    presenter_priority_set_succeeded.store(
        !elevated_presenter_priority || set_result == 0,
        std::memory_order_relaxed);
    XELOGI(
        "ZeroFGC0: presenter priority requested_nice={} effective_nice={} "
        "set_ok={} query_errno={} (SCHED_FIFO is never requested)",
        requested_nice, effective_nice,
        !elevated_presenter_priority || set_result == 0, priority_errno);
  }

  static uint64_t CpuClockNs(clockid_t clock) {
    timespec ts = {};
    if (clock_gettime(clock, &ts) != 0) {
      return 0;
    }
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
  }

  static uint64_t SafeDeltaNs(uint64_t now, uint64_t last) {
    return now >= last ? now - last : now;
  }

  // Presenter-thread and whole-process CPU against wall time since the last
  // summary. LogSummary() runs on the presenter thread, so the thread clock is
  // the presenter's own CPU. Process CPU also covers guest/recompiler and
  // in-process driver threads, but not SurfaceFlinger.
  void LogPresenterCpu() {
    const uint64_t wall_ns = PresenterMonotonicTimeNs();
    const uint64_t thread_ns = CpuClockNs(CLOCK_THREAD_CPUTIME_ID);
    const uint64_t process_ns = CpuClockNs(CLOCK_PROCESS_CPUTIME_ID);
    const uint64_t applied =
        transaction_applied_total.load(std::memory_order_relaxed);
    const uint64_t wall_delta = SafeDeltaNs(wall_ns, cpu_summary_last_wall_ns);
    const uint64_t thread_delta =
        SafeDeltaNs(thread_ns, cpu_summary_last_thread_ns);
    const uint64_t process_delta =
        SafeDeltaNs(process_ns, cpu_summary_last_process_ns);
    const uint64_t cycles_delta =
        SafeDeltaNs(presenter_arbiter_cycle_total, cpu_summary_last_cycles);
    const uint64_t zero_progress_delta = SafeDeltaNs(
        presenter_zero_progress_cycle_total, cpu_summary_last_zero_progress);
    const uint64_t wait_calls_delta =
        SafeDeltaNs(presenter_wait_call_total, cpu_summary_last_wait_calls);
    const uint64_t wait_blocked_delta = SafeDeltaNs(
        presenter_wait_blocked_ns_total, cpu_summary_last_wait_blocked_ns);
    const uint64_t applied_delta =
        SafeDeltaNs(applied, cpu_summary_last_applied);
    const bool have_window = cpu_summary_last_wall_ns != 0 && wall_delta != 0;
    XELOGI(
        "ZeroFGPresenterCpu window_ms={} thread_cpu_ms total/delta={}/{} "
        "thread_pct_x10={} process_cpu_ms total/delta={}/{} "
        "process_cores_x100={} wait_blocked_pct_x10={} cycles_delta={} "
        "zero_progress total/delta={}/{} wait_calls_delta={} "
        "applied_delta={} thread/process_us_per_output={}/{}",
        wall_delta / 1000000, thread_ns / 1000000, thread_delta / 1000000,
        have_window ? thread_delta * 1000 / wall_delta : 0,
        process_ns / 1000000, process_delta / 1000000,
        have_window ? process_delta * 100 / wall_delta : 0,
        have_window ? wait_blocked_delta * 1000 / wall_delta : 0,
        cycles_delta, presenter_zero_progress_cycle_total,
        zero_progress_delta, wait_calls_delta, applied_delta,
        applied_delta ? thread_delta / 1000 / applied_delta : 0,
        applied_delta ? process_delta / 1000 / applied_delta : 0);
    cpu_summary_last_wall_ns = wall_ns;
    cpu_summary_last_thread_ns = thread_ns;
    cpu_summary_last_process_ns = process_ns;
    cpu_summary_last_cycles = presenter_arbiter_cycle_total;
    cpu_summary_last_zero_progress = presenter_zero_progress_cycle_total;
    cpu_summary_last_wait_calls = presenter_wait_call_total;
    cpu_summary_last_wait_blocked_ns = presenter_wait_blocked_ns_total;
    cpu_summary_last_applied = applied;
  }

  void LogSummary() {
    XELOGI(
        "ZeroFGPresentationAdvance enabled={} advance_us={} max_us={} "
        "steps_up/down={}/{} capacity_down={} excess_max_us={} refresh_us={}",
        kPresentationAdvanceEnabled, presentation_advance_ns / 1000,
        presentation_advance_max_ns / 1000, presentation_advance_steps_up,
        presentation_advance_steps_down, presentation_advance_capacity_down,
        presentation_advance_excess_max_ns / 1000,
        main_surface_egress ? main_surface_egress->RefreshCycleNs() / 1000
                            : 0);
    XELOGI(
        "ZeroFGDeviceB domain=presenter_B Source_q0_counts_exclude_B=true "
        "ingress_queue_busy_retry={} submit_after_teardown_idle={} "
        "ingress_queue/submit_us_p90={}/{} "
        "capture_queue/submit/first_seen_ready_us_p90={}/{}/{} "
        "generation_queue/submit/observed_completion_us_p90={}/{}/{} "
        "post_queue/submit/observed_completion_us_p90={}/{}/{} "
        "N_ready_samples={} N_publish_to_first_seen_us_p90={} "
        "post_after_n_ready_us_p50/p90/max={}/{}/{} lost={}",
        device_b_ingress_queue_busy, device_b_submit_after_teardown_idle,
        device_b_ingress_queue_host_ns.Quantile(90, 100) / 1000,
        ingress_submit_host_ns.Quantile(90, 100) / 1000,
        capture_transfer_queue_wait_ns.Quantile(90, 100) / 1000,
        capture_transfer_submit_host_ns.Quantile(90, 100) / 1000,
        capture_first_seen_ready_ns.Quantile(90, 100) / 1000,
        generation_queue_wait_ns.Quantile(90, 100) / 1000,
        generation_submit_host_ns.Quantile(90, 100) / 1000,
        generation_submit_to_ready_ns.Quantile(90, 100) / 1000,
        post_queue_wait_ns.Quantile(90, 100) / 1000,
        post_submit_host_ns.Quantile(90, 100) / 1000,
        post_submit_to_ready_ns.Quantile(90, 100) / 1000,
        device_b_post_observations,
        device_b_n_first_seen_ns.Quantile(90, 100) / 1000,
        device_b_post_after_n_ready_ns.Quantile(50, 100) / 1000,
        device_b_post_after_n_ready_ns.Quantile(90, 100) / 1000,
        device_b_post_after_n_ready_ns.maximum() / 1000,
        vulkan_device->IsLost());
    // Per-window pool occupancy high water.
    pool_high_water_capture = 0;
    pool_high_water_residency = 0;
    pool_high_water_final_output = 0;
    pool_high_water_final_output_synthetic = 0;
    pool_high_water_logical_synthetic = 0;
    uint32_t recent_bp_intervals = 0;
    uint32_t recent_longest_clean_run = 0;
    SourceRecentIssueWindow(recent_bp_intervals, recent_longest_clean_run);
    XELOGI(
        "ZeroFGSourceRate P_us={} clean_run_p50/p90/max={}/{}/{} "
        "clean_runs_ge{}={} recent bp/longest_clean={}/{} unobservable={} "
        "probe state={} "
        "attempt/confirm/no_change/abort={}/{}/{}/{} drops={} "
        "last_candidate_us={}",
        source_period_ns / 1000,
        source_issue_clean_run_samples.Quantile(50, 100),
        source_issue_clean_run_samples.Quantile(90, 100),
        source_issue_clean_run_max,
        uint32_t(kSourceTransitionCleanRunIntervals),
        source_issue_clean_run_learnable_total, recent_bp_intervals,
        recent_longest_clean_run, SourceRateUnobservable(),
        uint32_t(source_rate_probe_state), source_rate_probe_attempt_total,
        source_rate_probe_confirm_total, source_rate_probe_no_change_total,
        source_rate_probe_abort_total, source_rate_probe_drop_total,
        source_rate_probe_last_candidate_ns / 1000);
    LogPresenterCpu();
    std::array<uint32_t, size_t(SlotState::kCount)> state_counts = {};
    for (const Slot& slot : slots) {
      ++state_counts[size_t(slot.state.load(std::memory_order_acquire))];
    }
    std::array<uint32_t, size_t(CaptureState::kCount)> capture_counts = {};
    for (const CaptureSlot& capture : capture_slots) {
      ++capture_counts[
          size_t(capture.state.load(std::memory_order_acquire))];
    }
    const xe::SourceBoundarySnapshot source_snapshot =
        xe::GetSourceBoundarySnapshot();
    const uint64_t source_queue_wait_samples =
        source_snapshot.totals.source_queue_lock_wait_count;
    const uint64_t source_queue_wait_total_ns =
        source_snapshot.totals.source_queue_lock_wait_total_ns;
    const uint64_t previous_samples = source_queue_wait_last_samples.exchange(
        source_queue_wait_samples, std::memory_order_relaxed);
    const uint64_t previous_total_ns =
        source_queue_wait_last_total_ns.exchange(source_queue_wait_total_ns,
                                                  std::memory_order_relaxed);
    const uint64_t delta_samples =
        source_queue_wait_samples >= previous_samples
            ? source_queue_wait_samples - previous_samples
            : 0;
    const uint64_t delta_total_ns =
        source_queue_wait_total_ns >= previous_total_ns
            ? source_queue_wait_total_ns - previous_total_ns
            : 0;
    const uint64_t interval_average_ns =
        delta_samples ? delta_total_ns / delta_samples : 0;
    XELOGI(
        "ZeroFGC0Summary generation={} accepting={} residency={}/{} high_water={} state="
        "free/transfer/ready={}/{}/{} source reserve/submit/publish/handoff/"
        "miss={}/{}/{}/{}/{} presenter consumed/applied/release="
        "{}/{}/{} ready_wait_us p50/p90/p99/max={}/{}/{}/{} "
        "release_us p50/p90/p99/max={}/{}/{}/{} applied_fps={:.2f} "
        "post passes={} submits={} final_pool={}/{} high_water={} "
        "post_queue_wait_us p90/max={}/{} post_submit_host_us p90/max={}/{} "
        "post_gpu_us p50/p90/p99/max={}/{}/{}/{} "
        "publish_apply_us p90={} "
        "ready_apply_us p90={} "
        "source_queue_wait samples/delta/total_us/delta_us/avg_us/max_us="
        "{}/{}/{}/{}/{}/{} "
        "ordering_errors={} fail_open={} "
        "last accepted/applied/released={} / {} / {} "
        "source_timeline_cpu_poll={}",
        generation, accepting.load(std::memory_order_relaxed), CountUsedSlots(),
        kPoolSize,
        pool_high_water.load(std::memory_order_relaxed),
        state_counts[size_t(SlotState::kFree)],
        state_counts[size_t(SlotState::kTransferSubmitted)],
        state_counts[size_t(SlotState::kReady)],
        source_reserved_total.load(std::memory_order_relaxed),
        source_submitted_total.load(std::memory_order_relaxed),
        source_published_total.load(std::memory_order_relaxed),
        source_handoff_total.load(std::memory_order_relaxed),
        source_reserve_miss_total.load(std::memory_order_relaxed),
        handoffs_consumed_total.load(std::memory_order_relaxed),
        transaction_applied_total.load(std::memory_order_relaxed),
        release_event_total.load(std::memory_order_relaxed),
        readiness_wait_ns.Quantile(50, 100) / 1000,
        readiness_wait_ns.Quantile(90, 100) / 1000,
        readiness_wait_ns.Quantile(99, 100) / 1000,
        readiness_wait_ns.maximum() / 1000,
        release_latency_ns.Quantile(50, 100) / 1000,
        release_latency_ns.Quantile(90, 100) / 1000,
        release_latency_ns.Quantile(99, 100) / 1000,
        release_latency_ns.maximum() / 1000,
        applied_timestamps.RateHz(),
        post_effect_count, post_submit_total.load(std::memory_order_relaxed),
        CountUsedFinalOutputs(), kFinalOutputPoolSize,
        final_output_high_water.load(std::memory_order_relaxed),
        post_queue_wait_ns.Quantile(90, 100) / 1000,
        post_queue_wait_ns.maximum() / 1000,
        post_submit_host_ns.Quantile(90, 100) / 1000,
        post_submit_host_ns.maximum() / 1000,
        post_gpu_completion_ns.Quantile(50, 100) / 1000,
        post_gpu_completion_ns.Quantile(90, 100) / 1000,
        post_gpu_completion_ns.Quantile(99, 100) / 1000,
        post_gpu_completion_ns.maximum() / 1000,
        publish_to_apply_ns.Quantile(90, 100) / 1000,
        ready_to_apply_ns.Quantile(90, 100) / 1000,
        source_queue_wait_samples, delta_samples,
        source_queue_wait_total_ns / 1000, delta_total_ns / 1000,
        interval_average_ns / 1000,
        source_snapshot.totals.source_queue_lock_wait_max_ns / 1000,
        ordering_error_total.load(std::memory_order_relaxed),
        fail_open_total.load(std::memory_order_relaxed),
        last_accepted_source_id.load(std::memory_order_relaxed),
        last_applied_source_id.load(std::memory_order_relaxed),
        last_released_source_id.load(std::memory_order_relaxed),
        source_timeline_cpu_poll_total.load(std::memory_order_relaxed));
    XELOGI(
        "ZeroFGPhaseCInvariants accepted_real_drop/apply_before_planned/"
        "multi_blocking_submit/multi_host_driver/first_pump_before_driver/"
        "future_cursor_by_real_defer/"
        "source_phase_reanchor_live_real_invalidation/"
        "shallow_real_commitment/pending_real_nominal/"
        "real_accept_cursor="
        "{}/{}/{}/{}/{}/{}/{}/{}/{}/{}",
        accepted_real_drop_violation_total,
        apply_before_planned_violation_total,
        multi_blocking_submit_quantum_violation_total,
        multi_host_driver_op_quantum_violation_total,
        first_pump_before_driver_violation_total,
        future_cursor_advanced_by_real_defer_total,
        source_phase_reanchor_live_real_invalidation_total,
        shallow_real_commitment_violation_total,
        pending_real_nominal_violation_total,
        physical_real_acceptance_cursor_advance_total);
    LogMainSurface("summary", main_surface_egress.get());
    XELOGI(
        "ZeroFGProfile mode={} "
        "readback_attempt/success/not_ready/error={}/{}/{}/{} "
        "service_gpu_us_p50/p90/p99/max={}/{}/{}/{}",
        ZeroFGSelectionName(), profile_readback_attempt_total,
        profile_readback_success_total, profile_readback_not_ready_total,
        profile_readback_error_total,
        generation_service_gpu_ns.Quantile(50, 100) / 1000,
        generation_service_gpu_ns.Quantile(90, 100) / 1000,
        generation_service_gpu_ns.Quantile(99, 100) / 1000,
        generation_service_gpu_ns.maximum() / 1000);
  }

  PresenterWakePlan NextPresenterWakePlan(uint64_t now_ns) {
    PresenterWakePlan plan;
    // Phase 0 sensor 6: kGpuPoll is six mechanisms in one bucket (D8-1).
    // Carry which one actually won the deadline.
    GpuPollSource poll_source = GpuPollSource::kNone;
    const auto consider = [now_ns, &plan, &poll_source](
                              uint64_t candidate_ns,
                              PresenterWakeReason reason,
                              GpuPollSource source = GpuPollSource::kNone) {
      if (candidate_ns > now_ns && candidate_ns < plan.deadline_ns) {
        plan.deadline_ns = candidate_ns;
        plan.reason = reason;
        poll_source = source;
      }
    };

    if (ingress_publication_pending.load(std::memory_order_acquire) &&
        ingress_retry_due_ns) {
      consider(ingress_retry_due_ns > now_ns
                   ? ingress_retry_due_ns
                   : SaturatingAddNs(now_ns, kOwnershipPollIntervalNs),
               PresenterWakeReason::kGpuPoll, GpuPollSource::kIngressRetry);
    }
    const bool source_reserve_emergency = CountSourceIngressSlots() >= 2;
    // A drain waits for every Capture transfer: never sleep without a poll
    // deadline while one remains.
    if (CountCaptureTransfers() &&
        (CaptureReclamationObservationRequired() ||
         source_reserve_emergency || presenter_draining)) {
      consider(SaturatingAddNs(now_ns, kOwnershipPollIntervalNs),
               PresenterWakeReason::kGpuPoll, GpuPollSource::kCaptureReclaim);
    }
    for (const GenerationJob& job : generation_jobs) {
      if (GenerationNeedsCpuObservation(job)) {
        if (job.next_observation_time_ns <= now_ns) {
          ++observation_bounded_retry_wake_total;
        }
        consider(job.next_observation_time_ns > now_ns
                     ? job.next_observation_time_ns
                     : SaturatingAddNs(now_ns,
                                       kObservationDueRetryIntervalNs),
                 PresenterWakeReason::kGpuPoll,
                 GpuPollSource::kGenerationObservation);
      }
    }
    for (const PostJob& job : post_jobs) {
      if (job.active) {
        // This is a native sync_file readiness check, not a Vulkan driver
        // observation. Keep retirement prompt even inside the head guard.
        consider(job.next_observation_time_ns > now_ns
                     ? job.next_observation_time_ns
                     : SaturatingAddNs(now_ns, kReleaseFencePollIntervalNs),
                 PresenterWakeReason::kGpuPoll, GpuPollSource::kPostFence);
      }
    }
    // Main Surface Authority: egress completions wake the presenter by
    // themselves; only the A-retirement request and its watchdog need a
    // deadline while the Surface is changing hands.
    if (main_surface_handoff_request_begin_ns ||
        main_surface_gap_begin_ns || main_surface_outdated_begin_ns) {
      consider(main_surface_handoff_next_request_ns > now_ns
                   ? main_surface_handoff_next_request_ns
                   : SaturatingAddNs(now_ns, kMainSurfaceUIRequestIntervalNs),
               PresenterWakeReason::kGpuPoll);
    }
    const LogicalOutput* head = FindLogicalBySequence(next_apply_sequence);
    if (head && head->candidate.target_time_ns) {
      if (head->state == LogicalOutputState::kFinalReady) {
        consider(BlockingHazardGuardStartNs(*head),
                 PresenterWakeReason::kGuardStart);
      }
      consider(PlannedDispatchTimeNs(*head),
               PresenterWakeReason::kPlannedDispatch);
    }
    // H12: for Synthetic these are wake hints, not execution permission.
    // consider() ignores zero and past values, so an unanchored S — whose
    // target, not_before and soft deadlines are all zero — invents no semantic
    // wake of its own. That is correct: it is immediately actionable, so
    // current Source/GPU events drive its progress, and if Source stops there
    // is no new work until the next publication wakes the presenter. For an
    // already anchored S these hints may only cause an extra wake; they can
    // never delay work. PresenterWakeReason telemetry measures their cost.
    for (const LogicalOutput& logical : logical_outputs) {
      consider(logical.candidate.target_time_ns,
               PresenterWakeReason::kSemanticDeadline);
      switch (logical.state) {
        case LogicalOutputState::kWaitingPairAnchor:
          // H12: unreachable; retained only for telemetry/state-index
          // stability. Not production authority.
          break;
        case LogicalOutputState::kWaitingGeneration:
          consider(logical.generation_submit_not_before_ns,
                   PresenterWakeReason::kSemanticDeadline);
          consider(logical.generation_submit_soft_deadline_ns,
                   PresenterWakeReason::kSemanticDeadline);
          consider(logical.post_submit_soft_deadline_ns,
                   PresenterWakeReason::kSemanticDeadline);
          break;
        case LogicalOutputState::kWaitingPost:
          consider(logical.post_submit_not_before_ns,
                   PresenterWakeReason::kSemanticDeadline);
          consider(logical.post_submit_soft_deadline_ns,
                   PresenterWakeReason::kSemanticDeadline);
          break;
        case LogicalOutputState::kPostSubmitted:
          consider(logical.final_ready_soft_deadline_ns,
                   PresenterWakeReason::kSemanticDeadline);
          break;
        case LogicalOutputState::kFinalReady:
          consider(PlannedDispatchTimeNs(logical),
                   PresenterWakeReason::kSemanticDeadline);
          break;
        case LogicalOutputState::kFree:
        case LogicalOutputState::kDropped:
        case LogicalOutputState::kCount:
          break;
      }
    }
    // Phase 0 sensor 6: attribute the poll cadence to the mechanism that
    // actually set the deadline, so a cold-wake margin is never again sized
    // from an average of six.
    if (plan.reason == PresenterWakeReason::kGpuPoll) {
      ++gpu_poll_source_total[size_t(poll_source)];
    }
    return plan;
  }

  void PresenterThreadMain() {
    presenter_thread_started.store(true, std::memory_order_release);
    ApplyThreadPriority();
    bool draining = false;
    presenter_draining = false;
    for (;;) {
      const uint64_t cycle_entry_ns = PresenterMonotonicTimeNs();
      previous_cycle_blocking_operation = arbiter_last_blocking_operation;
      previous_cycle_blocking_begin_ns = arbiter_last_blocking_begin_ns;
      previous_cycle_blocking_end_ns = arbiter_last_blocking_end_ns;
      ++presenter_arbiter_cycle_total;
      arbiter_blocking_submits_current = 0;
      arbiter_last_blocking_operation = BlockingOperation::kNone;
      arbiter_last_blocking_begin_ns = 0;
      arbiter_last_blocking_end_ns = 0;
      std::shared_ptr<EventBridge> cycle_bridge = bridge;
      if (!cycle_bridge) {
        break;
      }
      if (vulkan_device->IsLost() &&
          terminal_reason.load(std::memory_order_acquire) ==
              TerminalReason::kNone) {
        XELOGE(
            "ZeroFGDeviceB lost reason=Vulkan_device_lost "
            "Source_device_A_untouched=true");
        Terminalize(TerminalReason::kInvalidHandoff,
                    last_accepted_source_id.load());
      }
      ObserveMainSurfaceAuthority();
      const bool stopping =
          shutdown_requested.load(std::memory_order_acquire) ||
          terminal_reason.load(std::memory_order_acquire) !=
              TerminalReason::kNone ||
          (!accepting.load(std::memory_order_acquire) &&
           detach_requested.load(std::memory_order_acquire));
      if (stopping && !draining) {
        draining = true;
        presenter_draining = true;
        BeginAsyncDrainForShutdown();
      }
      bool progress = false;
      arbiter_host_driver_ops_current = 0;
      arbiter_first_pump_completed = draining;
      if (!draining) {
        if (accepting.load(std::memory_order_acquire)) {
          CheckSourceContract();
        }
        // Literal first authority: no Vulkan observation or submission is
        // allowed before the ordered head has been inspected. A Vulkan "poll"
        // is non-waiting semantically, but its host call is not latency-bounded
        // on every Android driver.
        pre_first_pump_ns.Add(PresenterMonotonicTimeNs() - cycle_entry_ns);
        progress |= PumpLogicalOrderObserved();
        arbiter_first_pump_completed = true;
        previous_cycle_blocking_operation = BlockingOperation::kNone;
        previous_cycle_blocking_begin_ns = 0;
        previous_cycle_blocking_end_ns = 0;
      }

      // Egress completions are deliberately after the literal first Pump:
      // they may not precede an already-FinalReady sovereign head.
      progress |= ProcessMainSurfaceCompletions();
      // Post progress is signaled by an Android sync_file. Polling this fd is
      // CPU/kernel bookkeeping and does not consume the single Vulkan
      // host-driver operation allowed in the arbiter quantum.
      progress |= PollPostJobs().progress;
      // q0 lineage telemetry: observation only, never progress or a wake.
      if (!draining) {
        for (auto& context : ingress_copy_contexts) {
          if (context.handoff_ready) context.handoff_ready->Observe();
        }
        ObserveCaptureTransfersForTelemetry();
      }
      progress |= RecoverOrphanReadyFinalOutputs();
      RetireCompletedPresentationRetiredSynthetics();
      progress |= CleanupDroppedLogicalOutputs();

      if (!draining) {
        ConsumeSourceIssuePeriods();
        progress |= ProcessIncomingRealAndPump();
      }

      const LogicalOutput* guard_head =
          !draining ? FindHeadAwaitingPlannedDispatch(PresenterMonotonicTimeNs())
                    : nullptr;

      bool host_driver_operation_invoked = false;
      bool host_driver_authority_returned = false;
      bool source_ingress_progress = false;
      const uint32_t capture_free_before =
          kCapturePoolSize - CountSourceIngressSlots();
      const uint32_t capture_used_before =
          kCapturePoolSize - capture_free_before;
      const bool capture_turnover_proactive =
          capture_used_before != 0;

      if (draining) {
        // The drain exits only when no Capture transfer remains, and normal
        // polling no longer observes them: poll them here first (after the
        // teardown drain B is idle and every transfer reads complete). Without
        // it a fail-open with a Real in flight never reached the handback, and
        // the screen froze on B's last frame while the game ran (Halo 3, a
        // Resize refusal at the first Generation, 2026-10-08).
        DriverPollResult poll = PollCaptureTransfers();
        if (!poll.invoked) {
          poll = PollOneNormalDriverObservation();
        }
        host_driver_operation_invoked = poll.invoked;
        progress |= poll.progress;
      } else {
        // The ordered head has already received literal CPU authority above.
        // Capture turnover participates only after admitted Synthetic work has
        // had the normal arbiter opportunity; low runway is not a policy
        // preemption.
        DriverPollResult poll;

        // Capture completion is not a readiness prerequisite for the
        // executive. Observe it only when physical reclamation is required,
        // and yield the quantum before attempting a subsequent ingress.
        if (!host_driver_operation_invoked &&
            CaptureReclamationObservationRequired()) {
          poll = PollCaptureTransfers();
          if (poll.invoked) {
            ++poll_allowed_source_critical_total;
            host_driver_operation_invoked = true;
            progress |= poll.progress;
            source_ingress_progress |= poll.progress;
          }
        }

        // If Source did not need this driver quantum, observe the ordered head
        // before any normal work. Authority returns to Pump immediately.
        if (!host_driver_operation_invoked) {
          poll = PollHeadCriticalGpuJob();
          if (poll.invoked) {
            ++poll_allowed_head_critical_total;
            host_driver_operation_invoked = true;
            progress |= poll.progress;
          }
        }

        if (host_driver_operation_invoked) {
          progress |= ProcessIncomingRealAndPump();
          host_driver_authority_returned = true;
          guard_head =
              FindHeadAwaitingPlannedDispatch(PresenterMonotonicTimeNs());
        }

        if (guard_head && !SourceRateProbeActive() &&
            !SidecarProgressCritical()) {
          // Split pre-wake quiet window: a FinalReady head is due within the
          // cold-wake margin. Start no discretionary host-driver work
          // (ingress, polls, GPU submits) until its dispatch. The head needs
          // none, and the next outputs keep their production lead. An open
          // rate probe is the exception: holding ingress for the margin would
          // put the Source back into the wait the probe exists to remove.
          ++split_quiet_window_cycle_total;
        } else {
          // P3: count the cycles where an accepted Real's progress claim beat
          // the quiet window, so the exception is measured and not assumed.
          if (guard_head && !SourceRateProbeActive() &&
              SidecarProgressCritical()) {
            ++split_quiet_window_progress_override_total;
          }
          bool normal_poll_attempted = false;
          // H12: `synthetic_commit_pending` used to yield the host-driver turn
          // to imminent JIT-urgent Synthetic work. With work-conserving
          // production, pending Synthetic work is the steady state rather than
          // a scarcity signal, so that predicate would have suppressed
          // TryIngressLatestPublication() almost permanently and starved
          // Source ingress — forbidden by Source content priority. Synthetic
          // still competes for the same shared turn through ScheduleGpuWork()
          // below; it simply no longer pre-empts Source ingress. No new
          // controller is introduced: the existing bounded round-robin
          // (host_driver_operation_invoked / normal_driver_turn_prefers_poll)
          // remains the sole arbiter.
          if (!host_driver_operation_invoked) {
            poll = TryIngressLatestPublication();
            if (poll.invoked) {
              host_driver_operation_invoked = true;
              normal_driver_turn_prefers_poll = true;
              progress |= poll.progress;
              source_ingress_progress |= poll.progress;
            } else {
              progress |= poll.progress;
            }
          }
          if (!host_driver_operation_invoked &&
              normal_driver_turn_prefers_poll) {
            poll = PollOneNormalDriverObservation();
            normal_poll_attempted = poll.invoked;
            if (poll.invoked) {
              host_driver_operation_invoked = true;
              normal_driver_turn_prefers_poll = false;
              progress |= poll.progress;
            }
          }
          if (!host_driver_operation_invoked) {
            progress |= ScheduleGpuWork();
            host_driver_operation_invoked =
                arbiter_host_driver_ops_current != 0;
            if (host_driver_operation_invoked) {
              normal_driver_turn_prefers_poll = true;
            }
          }
          if (!host_driver_operation_invoked && capture_turnover_proactive) {
            poll = RunSourceIngressDriverOperation(capture_free_before);
            if (poll.invoked) {
              host_driver_operation_invoked = true;
              progress |= poll.progress;
              source_ingress_progress |= poll.progress;
            }
          }
          if (!host_driver_operation_invoked && !normal_poll_attempted) {
            poll = PollOneNormalDriverObservation();
            if (poll.invoked) {
              host_driver_operation_invoked = true;
              normal_driver_turn_prefers_poll = false;
              progress |= poll.progress;
            }
          }
        }

        // Every actual host-driver operation yields directly back to ordered
        // authority with fresh time. No second poll or submit is permitted in
        // this arbiter quantum.
        if (host_driver_operation_invoked) {
          if (!host_driver_authority_returned) {
            progress |= ProcessIncomingRealAndPump();
          }
          // The operation has now been causally observed by Pump. Do not carry
          // it into the next cycle and misattribute a later timed-wait miss.
          arbiter_last_blocking_operation = BlockingOperation::kNone;
          arbiter_last_blocking_begin_ns = 0;
          arbiter_last_blocking_end_ns = 0;
        }
      }
      ReclaimStructurallyOrphanedSynthetics();
      RefreshPipelineSnapshot();
      if (!draining) {
        ObserveResidualPhaseShadowEpoch(PresenterMonotonicTimeNs());
        ObserveUnifiedRunwayShadow(PresenterMonotonicTimeNs());
      }
      if (draining && !CountCaptureTransfers() && !CountGenerationJobs() &&
          !CountPostJobs()) {
        break;
      }
      const uint64_t applied_total_for_summary =
          transaction_applied_total.load(std::memory_order_relaxed);
      if (applied_total_for_summary &&
          applied_total_for_summary >= last_summary_transaction + 300) {
        last_summary_transaction = applied_total_for_summary;
        LogSummary();
      }
      if (!progress) {
        ++presenter_zero_progress_cycle_total;
      }
      if (!progress) {
        std::shared_ptr<EventBridge> local_bridge = cycle_bridge;
        const PresenterWakePlan wake_plan =
            NextPresenterWakePlan(PresenterMonotonicTimeNs());
        requested_wake_deadline_ns =
            wake_plan.deadline_ns == std::numeric_limits<uint64_t>::max()
                ? 0
                : wake_plan.deadline_ns;
        ++presenter_wait_requested_by_reason[size_t(wake_plan.reason)];
        // PresenterMonotonicTimeNs and std::chrono::steady_clock share the Android
        // CLOCK_MONOTONIC timebase used by desiredPresentTime. This wait is an
        // absolute wake deadline; it is never the authority that advances Tn.
        // The counting semaphore makes producer wakeups persistent without
        // requiring Source to acquire the presenter mutex.
        bool producer_event_wake = false;
        const uint64_t wait_begin_ns = PresenterMonotonicTimeNs();
        if (wake_plan.deadline_ns ==
            std::numeric_limits<uint64_t>::max()) {
          local_bridge->presenter_wake.acquire();
          producer_event_wake = true;
        } else {
          producer_event_wake = local_bridge->presenter_wake.try_acquire_until(
              std::chrono::steady_clock::time_point(
                  std::chrono::nanoseconds(wake_plan.deadline_ns)));
        }
        actual_wait_return_ns = PresenterMonotonicTimeNs();
        ++presenter_wait_call_total;
        presenter_wait_blocked_ns_total = SaturatingAddNs(
            presenter_wait_blocked_ns_total,
            actual_wait_return_ns > wait_begin_ns
                ? actual_wait_return_ns - wait_begin_ns
                : 0);
        if (producer_event_wake) {
          last_presenter_wake_reason = PresenterWakeReason::kProducerEvent;
          ++presenter_wait_event_wake_total;
        } else {
          last_presenter_wake_reason = wake_plan.reason;
          ++presenter_wait_deadline_return_by_reason[
              size_t(wake_plan.reason)];
          presenter_wait_overshoot_ns[size_t(wake_plan.reason)].Add(
              actual_wait_return_ns > wake_plan.deadline_ns
                  ? actual_wait_return_ns - wake_plan.deadline_ns
                  : 0);
        }
        // Coalesce bursts. State is authoritative, so one pump can consume all
        // currently published work while future releases remain latched.
        while (local_bridge->presenter_wake.try_acquire()) {
        }
      }
    }
    if (history_slot < kPoolSize) {
      ReleaseRealSlot(history_slot);
      history_slot = UINT32_MAX;
    }
    // The handoff outlives this thread: never leave it a held lease.
    ClearPreadmission();
    // Runtime failure stops admission immediately. Already-submitted GPU and
    // compositor ownership remains protected by the bounded job/output state;
    // lifecycle teardown destroys it only after the owner thread has stopped.
    DetachSurfaceForFailOpen();
    presenter_thread_started.store(false, std::memory_order_release);
    std::shared_ptr<EventBridge> local_bridge = bridge;
    SignalPresenterWake(local_bridge, true);
  }

  VulkanDevice* vulkan_device = nullptr;
  const bool elevated_presenter_priority = false;
  const uint64_t synthetic_cost_seed_ns = 0;
  IngressSourceAcquireCallback ingress_source_acquire_callback;
  PaintConfigProvider paint_config_provider;
  GenerationCallback generation_callback;
  GenerationPollCallback generation_poll_callback;
  SyntheticReleaseCallback synthetic_release_callback;
  GenerationShutdownCallback generation_shutdown_callback;
  PostProcessCallback post_process_callback;
  PostProcessReleaseCallback post_process_release_callback;
  PostProcessShutdownCallback post_process_shutdown_callback;
  PresenterDeviceDrainCallback presenter_device_drain_callback;
  VkSemaphore timeline = VK_NULL_HANDLE;
  std::atomic<uint64_t> next_timeline_value{1};
  VkSemaphore capture_transfer_timeline = VK_NULL_HANDLE;
  uint64_t next_capture_transfer_value = 1;
  uint64_t completed_capture_transfer_value = 0;
  uint64_t device_b_post_observations = 0;
  PresenterSampleWindow<128> device_b_post_after_n_ready_ns;
  PresenterSampleWindow<128> device_b_ingress_queue_host_ns;
  uint64_t device_b_ingress_queue_busy = 0;
  uint64_t device_b_submit_after_teardown_idle = 0;
  PresenterSampleWindow<128> device_b_n_first_seen_ns;
  // Capture observation (telemetry only).
  PresenterSampleWindow<128> capture_first_seen_ready_ns;
  PresenterSampleWindow<128> capture_last_seen_pending_ns;
  PresenterSampleWindow<128> capture_observation_uncertainty_ns;
  PresenterSampleWindow<128> capture_reclaim_after_first_seen_ns;
  uint64_t capture_observation_poll_total = 0;
  uint64_t capture_observation_ready_total = 0;
  uint64_t capture_observation_error_total = 0;
  uint64_t capture_observation_last_poll_ns = 0;

  std::array<CaptureSlot, kCapturePoolSize> capture_slots;
  std::array<IngressCopyContext, kCapturePoolSize> ingress_copy_contexts;
  std::array<Slot, kPoolSize> slots;
  std::array<FinalOutputSlot, kFinalOutputPoolSize> final_outputs;
  std::array<LogicalOutput, kLogicalOutputCapacity> logical_outputs;
  std::array<GenerationJob, kSyntheticPoolSize> generation_jobs;
  std::array<PostJob, kFinalOutputPoolSize> post_jobs;
  VkExtent2D extent = {};
  VkFormat final_output_format = VK_FORMAT_UNDEFINED;
  uint64_t generation = 0;

  bool surface_connected = false;
  // Main Surface Authority (MSA): refused for the rest of the session after a
  // terminal handback; the normal presenter then keeps the Surface.
  std::atomic<bool> main_surface_refused{false};
  std::atomic<const char*> main_surface_refusal_reason{"none"};
  // Survives connections: who produces into the SurfaceView's Surface.
  ZeroFGMainSurfaceProducer main_surface_producer;
  // Borrowed from the normal presenter for one connection; never destroyed
  // here.
  VkSurfaceKHR main_surface = VK_NULL_HANDLE;
  // One per connection. Owned by the presenter thread while it runs; the UI
  // thread only notifies it, under main_surface_egress_mutex.
  std::unique_ptr<ZeroFGMainSurfaceEgress> main_surface_egress;
  std::mutex main_surface_egress_mutex;
  std::atomic<bool> main_surface_egress_live{false};
  std::atomic<bool> main_surface_reconnect_requested{false};
  std::function<void()> main_surface_ui_request;
  std::mutex main_surface_ui_request_mutex;
  uint64_t main_surface_handoff_request_begin_ns = 0;
  uint64_t main_surface_handoff_next_request_ns = 0;
  uint64_t main_surface_gap_begin_ns = 0;
  uint64_t main_surface_outdated_begin_ns = 0;
  uint64_t main_surface_outdated_next_request_ns = 0;
  uint64_t main_surface_applied_total = 0;
  uint64_t main_surface_ring_full_total = 0;
  // Ordered heads that found Presentation closed before B produced, and the
  // longest such wait (first deferral to the first output B accepted).
  uint64_t main_surface_waiting_for_b_total = 0;
  uint64_t main_surface_waiting_for_b_last_sequence = 0;
  uint64_t main_surface_waiting_for_b_begin_ns = 0;
  uint64_t main_surface_waiting_for_b_max_ns = 0;
  uint64_t main_surface_released_total = 0;
  uint64_t main_surface_abandoned_total = 0;
  uint64_t main_surface_completion_stale_total = 0;
  std::shared_ptr<EventBridge> bridge;
  std::thread presenter_thread;

  std::atomic<bool> accepting{false};
  std::atomic<bool> final_output_authority{false};
  std::atomic<bool> source_activation_armed{false};
  std::atomic<uint64_t> source_qualification_start_issue_total{0};
  std::atomic<uint64_t> qualified_native_source_period_ns{0};
  std::atomic<uint64_t> presenter_authority_activation_source_id{0};
  std::atomic<uint64_t> presenter_authority_activation_time_ns{0};
  std::atomic<uint32_t> source_bootstrap_real_only_remaining{0};
  std::atomic<uint64_t> source_native_qualification_attempt_total{0};
  std::atomic<uint64_t> source_native_qualification_success_total{0};
  std::atomic<bool> shutdown_requested{false};
  std::atomic<bool> detach_requested{false};
  std::atomic<bool> presenter_thread_started{false};
  std::atomic<TerminalReason> terminal_reason{TerminalReason::kNone};
  std::atomic<bool> terminal_logged{false};

  std::atomic<bool> ingress_publication_pending{false};
  uint64_t ingress_retry_due_ns = 0;
  // Split physical pre-admission (depth 1): a publication taken off the
  // handoff, lease held, that the Always-S gate has not admitted yet. It is
  // not a Real: no acceptance, S obligation, cursor or commitment.
  IngressSourcePublication preadmitted_source;
  bool preadmitted = false;
  // The Always-S gate held the pre-admitted publication at least once.
  bool preadmitted_gate_held = false;
  uint64_t preadmitted_gate_hold_ns = 0;
  uint64_t preadmitted_acquire_ns = 0;
  uint64_t preadmitted_queue_ns = 0;
  // When the pre-admission slot last became free (an admission).
  uint64_t preadmission_free_since_ns = 0;
  uint64_t publication_order_regression_seen = 0;
  uint64_t last_accepted_publication_id = 0;
  uint64_t last_accepted_publish_thread_tag = 0;
  uint64_t accepted_source_gap_total = 0;
  IncomingRealAuthority incoming_real;
  uint64_t incoming_real_arm_total = 0;
  uint64_t incoming_real_admit_total = 0;
  uint64_t incoming_real_release_total = 0;
  uint64_t incoming_real_gate_total = 0;
  uint64_t incoming_real_invariant_violation_total = 0;
  uint32_t incoming_real_high_water = 0;
  // Split sidecar (accepted, not materialized) and its falsifier sensors.
  uint64_t split_sidecar_hold_begin_ns = 0;
  uint64_t split_sidecar_hold_total = 0;
  uint64_t split_sidecar_hold_s_capacity_total = 0;
  uint64_t split_sidecar_hold_timeout_total = 0;
  bool split_sidecar_hold_alarm_raised = false;
  // State of the live Synthetics when a hold starts (sums over holds).
  uint64_t split_sidecar_hold_s_generation_total = 0;
  uint64_t split_sidecar_hold_s_post_total = 0;
  uint64_t split_sidecar_hold_s_submitted_total = 0;
  uint64_t split_sidecar_hold_s_ready_total = 0;
  uint64_t split_sidecar_hold_s_anchored_total = 0;
  uint64_t split_sidecar_hold_predecessor_committed_total = 0;
  // Split pre-wake quiet-window cycles.
  uint64_t split_quiet_window_cycle_total = 0;
  // Issue age of the Real whose promotion blocked a decay on the floor.
  PresenterSampleWindow<128> latency_decay_floor_issue_age_ns;
  PresenterSampleWindow<128> split_sidecar_hold_ns;
  // Phase 0 sensor 1: accepted-Real end-to-end residence (armed -> exit) and
  // how each occupancy ended. split_sidecar_hold_ns above stays as it is: it
  // measures only the materialization-capacity subinterval (D8-5).
  PresenterSampleWindow<128> split_sidecar_residence_ns;
  uint64_t split_sidecar_exit_total[size_t(SidecarExitReason::kCount)] = {};
  // Phase 0 sensor 7: hold provenance. Accrued time and entries per cause,
  // attributed at pump granularity.
  uint64_t split_sidecar_hold_cause_ns[size_t(SidecarHoldCause::kCount)] = {};
  uint64_t split_sidecar_hold_cause_total[size_t(SidecarHoldCause::kCount)] =
      {};
  SidecarHoldCause split_sidecar_hold_cause = SidecarHoldCause::kNone;
  uint64_t split_sidecar_hold_cause_since_ns = 0;
  // Phase 0 sensor 2: true Synthetic backing ownership. The adapter's
  // output_busy[i] is cleared only by synthetic_release_callback, so mirroring
  // that callback is the honest measure of physical occupancy.
  // CountGenerationJobs() is the false authority of blocker #22: it falls when
  // a Generation completes on a poll before it is chained, while the Post
  // still holds the image.
  uint32_t synthetic_backing_owned_mask = 0;
  uint32_t synthetic_backing_owned_high_water = 0;
  uint64_t synthetic_backing_false_free_total = 0;
  uint32_t synthetic_backing_false_free_high_water = 0;
  // Track B gate counter: submits the true authority refused that the job
  // count would have admitted, and the adapter would then have rejected.
  // mutable: incremented from the const capacity query it measures.
  mutable uint64_t synthetic_backing_refusal_avoided_total = 0;
  // P2 gate counter: Synthetic Posts deferred to keep a Real-progress lane.
  // The frontier deliberately stops bounding live logical S tickets: an S past
  // Post leaves it. More tickets are therefore EXPECTED, and this records how
  // many, so the change is read as its intent and not as a regression. The
  // identity pool stays safe because materialization still needs two free
  // logical slots and each promotion allocates exactly two.
  uint32_t logical_synthetic_ticket_high_water = 0;
  // P3 signal: frontier at its bound with the Synthetic pool completely idle.
  uint64_t frontier_saturated_backing_idle_total = 0;
  // Track C step 4 (#3): a confirmed transition refills the phase window to
  // the size the stability gate demands.
  uint64_t source_transition_faster_refilled_total = 0;
  // Track C step 5 (#11): phase debt that was ours, refused instead of
  // attributed to the Source.
  uint64_t source_phase_debt_ours_reset_total = 0;
  uint64_t latency_depth_decay_armed_reanchor_total = 0;
  // Blocker #2: a decay may compact the future lattice, never expand it.
  bool latency_depth_reanchor_armed_by_decay = false;
  uint64_t latency_depth_reanchor_previous_operating_ns = 0;
  uint64_t latency_depth_decay_projection_compactable_ns = 0;
  uint64_t latency_depth_decay_actual_compacted_total = 0;
  uint64_t latency_depth_decay_already_tighter_total = 0;
  uint64_t latency_depth_decay_blocked_immutable_total = 0;
  uint64_t latency_depth_decay_blocked_reserve_only_total = 0;
  uint64_t latency_depth_decay_forward_dilation_total = 0;
  // Blocker #2 decomposition, telemetry only.
  uint64_t reanchor_floor_term_observations_total = 0;
  uint64_t reanchor_floor_winner_total[size_t(ReanchorFloorTerm::kCount)] = {};
  uint64_t reanchor_floor_reserve_only_block_total = 0;
  uint64_t latency_depth_lattice_compacted_total = 0;
  uint64_t latency_depth_lattice_compacted_ns = 0;
  uint64_t source_issue_p_censored_transition_kept_total = 0;
  // P3 counter: quiet-window cycles the sidecar's progress claim overrode.
  uint64_t split_quiet_window_progress_override_total = 0;
  // P3 arbiter counters: quanta handed to Generation because an accepted Real
  // was blocked with the chain idle, and the falsifier for that being the
  // right place to act at all.
  uint64_t arbitration_generation_after_blocked_post_total = 0;
  uint64_t arbitration_progress_no_candidate_total = 0;
  // Phase 0 sensor 3: Post completion against the FinalReady deadline, beside
  // the submit stamp the judgement currently uses (blocker #21). Recording
  // only; the judgement itself changes in phase 1.
  PresenterSampleWindow<128> post_submit_to_completion_ns;
  uint64_t final_ready_false_success_total = 0;
  uint64_t final_ready_completion_judged_total = 0;
  uint64_t post_completion_proven_total = 0;
  uint64_t completion_legacy_unproven_total = 0;
  uint64_t post_completion_observation_error_total = 0;
  uint64_t final_ready_judgement_skipped_no_proof_total = 0;
  uint64_t final_ready_judgement_duplicate_prevented_total = 0;
  // Phase 0 sensor 6: which mechanism actually asked for a kGpuPoll wake.
  uint64_t gpu_poll_source_total[size_t(GpuPollSource::kCount)] = {};
  // Phase 0 sensor 8: the terms of C_pair, recorded separately rather than
  // synthesized. The net sample needs provenance that only phases 0-2 supply,
  // so this records what is already honest: A issue -> S Post completion.
  PresenterSampleWindow<128> pair_causal_raw_ns;
  PresenterSampleWindow<128> pair_causal_clean_ns;
  PresenterSampleWindow<128> pair_policy_wait_ns;
  uint64_t pair_causal_clean_subtracted_total = 0;
  PresenterSampleWindow<128> pair_chain_raw_ns;
  uint64_t pair_causal_sample_total = 0;
  // Phase 0 sensor 9: peak occupancy per pool, for P2 verification and the
  // later kFinalOutputPoolSize re-derivation. Synthetic ownership of the
  // egress capacity is the one that decides whether a Real lane survives.
  uint32_t pool_high_water_capture = 0;
  uint32_t pool_high_water_residency = 0;
  uint32_t pool_high_water_final_output = 0;
  uint32_t pool_high_water_final_output_synthetic = 0;
  uint32_t pool_high_water_logical_synthetic = 0;
  uint64_t final_output_synthetic_owns_all_but_one_total = 0;
  uint64_t final_output_synthetic_owns_all_total = 0;
  uint64_t split_sidecar_same_pump_promotion_total = 0;
  uint64_t split_sidecar_promote_with_preadmitted_total = 0;
  uint64_t split_sidecar_promote_reanchor_pending_total = 0;
  uint64_t split_sidecar_bp_interval_total = 0;
  uint64_t split_sidecar_drain_discard_total = 0;
  uint32_t split_live_unanchored_synthetic_high_water = 0;
  uint64_t source_reanchor_live_anchor_won_total = 0;
  uint64_t source_reanchor_live_anchor_won_held_total = 0;
  uint64_t accepted_order_failure_current_publication = 0;
  uint64_t accepted_order_failure_current_source = 0;
  uint32_t accepted_order_failure_current_mailbox = UINT32_MAX;
  uint32_t accepted_order_failure_current_context = UINT32_MAX;
  uint32_t accepted_order_failure_current_capture = UINT32_MAX;
  uint64_t accepted_order_failure_previous_publication = 0;
  uint64_t accepted_order_failure_previous_source = 0;
  std::atomic<uint64_t> last_accepted_source_id{0};
  std::atomic<uint64_t> last_applied_source_id{0};
  std::atomic<uint64_t> last_released_source_id{0};
  std::atomic<uint64_t> last_applied_logical_sequence{0};
  uint64_t next_logical_token = 1;
  uint64_t next_logical_sequence = 1;
  uint64_t next_apply_sequence = 1;
  uint64_t last_summary_transaction = 0;
  uint32_t history_slot = UINT32_MAX;
  uint32_t last_transaction_output_slot = UINT32_MAX;

  std::atomic<uint64_t> source_reserved_total{0};
  std::atomic<uint64_t> ingress_attempt_total{0};
  std::atomic<uint64_t> ingress_acquire_total{0};
  std::atomic<uint64_t> ingress_submit_total{0};
  std::atomic<uint64_t> ingress_accept_ready_total{0};
  std::atomic<uint64_t> ingress_accepted_total{0};
  std::atomic<uint32_t> ingress_context_high_water{0};
  std::atomic<uint64_t> source_submitted_total{0};
  std::atomic<uint64_t> source_published_total{0};
  std::atomic<uint64_t> source_handoff_total{0};
  std::atomic<uint64_t> source_reserve_miss_total{0};
  std::atomic<uint64_t> capture_transfer_submitted_total{0};
  std::atomic<uint64_t> capture_transfer_completed_total{0};
  std::atomic<uint64_t> capture_committed_waiting_residency_total{0};
  std::atomic<uint32_t> capture_committed_waiting_residency_high_water{0};
  std::atomic<uint64_t> capture_pool_high_water{0};
  std::atomic<uint32_t> normal_capture_high_water{0};
  std::atomic<uint64_t> capture_transfer_high_water{0};
  std::atomic<uint32_t> capture_free_min{kCapturePoolSize};
  std::atomic<uint64_t> source_timeline_cpu_poll_total{0};
  std::atomic<uint64_t> committed_capture_cpu_promotion_total{0};
  std::atomic<uint64_t> direct_capture_timeline_wait_submit_total{0};
  std::atomic<uint64_t> capture_sync_fd_poll_total{0};
  std::atomic<uint64_t> capture_sync_fd_ready_total{0};
  std::atomic<uint64_t> capture_sync_fd_export_total{0};
  std::atomic<uint64_t> capture_sync_fd_export_failure_total{0};
  std::atomic<uint64_t> capture_sync_fd_fallback_total{0};
  // D3 Generation sync_fd telemetry (presenter thread only).
  uint64_t generation_sync_fd_poll_total = 0;
  uint64_t generation_sync_fd_ready_total = 0;
  uint64_t generation_sync_fd_export_total = 0;
  uint64_t generation_sync_fd_export_failure_total = 0;
  uint64_t generation_sync_fd_fallback_total = 0;
  // Completion/ownership v2 falsifiers (presenter thread only), cumulative
  // since creation. The Post and Generation owners live in VulkanPresenter
  // and report their counters with every result.
  ZeroFGCompletionOwnerStats capture_owner_stats;
  ZeroFGCompletionOwnerStats ingress_owner_stats;
  ZeroFGCompletionOwnerStats last_post_owner_stats;
  ZeroFGCompletionOwnerStats last_generation_owner_stats;
  uint64_t post_timeline_query_total = 0;
  std::array<uint64_t, kInputReadinessCount> generation_input_class_total = {};
  std::array<uint64_t, kInputReadinessCount> generation_input_class_over_1ms =
      {};
  uint64_t generation_input_poll_total = 0;
  uint64_t generation_input_class_host_ns_total = 0;
  uint64_t capture_proactive_turnover_attempt_total = 0;
  uint64_t capture_proactive_turnover_submit_total = 0;
  uint64_t capture_emergency_progress_attempt_total = 0;
  uint64_t capture_emergency_progress_success_total = 0;
  std::atomic<uint64_t> handoffs_consumed_total{0};
  std::atomic<uint64_t> readiness_wait_total{0};
  std::atomic<uint64_t> transaction_applied_total{0};
  std::atomic<uint64_t> release_event_total{0};
  std::atomic<uint64_t> candidate_release_total{0};
  std::atomic<uint64_t> post_submit_total{0};
  std::atomic<uint64_t> ordering_error_total{0};
  std::atomic<uint64_t> fail_open_total{0};
  std::atomic<uint64_t> lifecycle_unresolved_total{0};
  std::atomic<uint32_t> pipeline_snapshot_logical_total{0};
  std::atomic<uint32_t> pipeline_snapshot_logical_real{0};
  std::atomic<uint32_t> pipeline_snapshot_logical_synthetic{0};
  std::atomic<uint32_t> pipeline_snapshot_logical_high_water{0};
  std::atomic<uint32_t> pipeline_snapshot_live_future_real_commitments{0};
  std::atomic<uint32_t> pipeline_snapshot_future_real_commitment_high_water{0};
  std::atomic<uint64_t> pipeline_snapshot_tail_real_target_minus_head_quanta{0};
  std::atomic<uint32_t> pipeline_snapshot_pending_uncommitted_real{0};
  std::atomic<uint32_t> pipeline_snapshot_committed_real{0};
  std::atomic<uint32_t> pipeline_snapshot_committed_real_high_water{0};
  std::atomic<uint32_t> pipeline_snapshot_generation_jobs{0};
  std::atomic<uint32_t> pipeline_snapshot_post_jobs{0};
  std::array<
      std::array<std::atomic<uint32_t>, size_t(LogicalOutputState::kCount)>,
      2>
      pipeline_snapshot_logical_states = {};
  std::atomic<bool> pipeline_snapshot_head_valid{false};
  std::atomic<uint64_t> pipeline_snapshot_head_sequence{0};
  std::atomic<uint32_t> pipeline_snapshot_head_kind{0};
  std::atomic<uint32_t> pipeline_snapshot_head_state{0};
  std::atomic<uint64_t> pipeline_snapshot_head_nominal_target_ns{0};
  std::atomic<uint64_t> pipeline_snapshot_head_assigned_target_ns{0};
  std::atomic<uint64_t> pipeline_snapshot_head_nominal_tick{0};
  std::atomic<uint64_t> pipeline_snapshot_head_assigned_tick{0};
  std::atomic<uint64_t> pipeline_snapshot_head_exec_debt{0};
  std::atomic<bool> pipeline_snapshot_tail_valid{false};
  std::atomic<uint32_t> pipeline_snapshot_tail_kind{0};
  std::atomic<uint64_t> pipeline_snapshot_tail_source_id{0};
  std::atomic<uint64_t> pipeline_snapshot_tail_nominal_tick{0};
  std::atomic<uint64_t> pipeline_snapshot_tail_assigned_tick{0};
  std::atomic<uint64_t> pipeline_snapshot_tail_exec_debt{0};
  std::atomic<int64_t> pipeline_snapshot_head_target_delta_ns{0};
  std::atomic<uint64_t> pipeline_snapshot_last_applied_semantic_target_ns{0};
  std::atomic<uint64_t> pipeline_snapshot_last_accounted_semantic_target_ns{0};
  std::atomic<int64_t> pipeline_snapshot_next_semantic_target_delta_ns{0};
  std::atomic<uint32_t> pipeline_snapshot_real_final_ready_behind_head{0};
  std::array<std::atomic<int64_t>, 3>
      pipeline_snapshot_real_final_ready_target_deltas_ns = {};
  std::atomic<uint32_t> pipeline_snapshot_unfunded_total{0};
  std::atomic<uint32_t> pipeline_snapshot_real_unfunded{0};
  std::atomic<uint32_t> pipeline_snapshot_synthetic_unfunded{0};
  std::atomic<uint32_t> pipeline_snapshot_max_earlier_unfunded{0};
  std::atomic<uint32_t> pipeline_snapshot_candidate_free{0};
  std::atomic<uint32_t> pipeline_snapshot_candidate_ingress{0};
  std::atomic<uint32_t> pipeline_snapshot_candidate_residency{0};
  std::atomic<uint32_t> pipeline_snapshot_resource_starvation_mask{0};
  std::array<std::atomic<uint32_t>, kPoolSize>
      pipeline_snapshot_candidate_owner_refs = {};
  std::array<std::atomic<uint64_t>, kPoolSize>
      pipeline_snapshot_candidate_source_ids = {};
  std::array<std::atomic<uint32_t>, kPoolSize>
      pipeline_snapshot_candidate_states = {};
  std::atomic<uint64_t> pipeline_snapshot_residency_pressure_drops{0};
  std::atomic<uint32_t> pipeline_snapshot_orphan_ready_final_outputs{0};
  std::array<std::atomic<uint32_t>, size_t(FinalOutputState::kCount)>
      pipeline_snapshot_final_states = {};
  std::atomic<uint64_t> pool_high_water{0};
  std::atomic<uint64_t> ingress_high_water{0};
  std::atomic<uint64_t> residency_high_water{0};
  std::atomic<uint64_t> residency_capacity_exceeded_total{0};
  std::atomic<uint64_t> residency_capacity_excess_high_water{0};
  std::atomic<uint64_t> final_output_high_water{0};
  std::atomic<uint64_t> source_queue_wait_last_samples{0};
  std::atomic<uint64_t> source_queue_wait_last_total_ns{0};
  std::atomic<int> presenter_priority_effective{0};
  std::atomic<bool> presenter_priority_set_succeeded{false};

  PresenterSampleWindow<128> readiness_wait_ns;
  PresenterSampleWindow<128> capture_transfer_queue_wait_ns;
  PresenterSampleWindow<128> capture_transfer_submit_host_ns;
  PresenterSampleWindow<128> capture_transfer_residence_ns;
  PresenterSampleWindow<128> capture_reserve_to_recycle_ns;
  PresenterSampleWindow<128> capture_sync_fd_export_host_ns;
  PresenterSampleWindow<128> capture_sync_fd_poll_host_ns;
  PresenterSampleWindow<128> generation_sync_fd_export_host_ns;
  PresenterSampleWindow<128> generation_sync_fd_poll_host_ns;
  PresenterSampleWindow<128> release_latency_ns;
  PresenterSampleWindow<128> transaction_prepare_ns;
  PresenterSampleWindow<128> surface_apply_host_ns;
  PresenterSampleWindow<128> dispatch_start_late_ns;
  PresenterSampleWindow<128> hard_ready_dispatch_late_ns;
  PresenterSampleWindow<128> h14_b_apply_lateness_after_ready_s_ns;
  PresenterSampleWindow<128> pump_logical_order_ns;
  PresenterSampleWindow<128> e1f_shadow_impulse_ns;
  PresenterSampleWindow<128> e1f_commit_boundary_impulse_ns;
  PresenterSampleWindow<128> synthetic_anchor_latency_ns;
  PresenterSampleWindow<128> pre_first_pump_ns;
  PresenterSampleWindow<128> cycle_callback_ns;
  PresenterSampleWindow<128> cycle_release_ns;
  PresenterSampleWindow<128> active_capture_poll_host_ns;
  PresenterSampleWindow<128> capture_issue_to_transfer_submit_ns;
  // IssueSwap to capture reservation: the Source-side dependency that enters
  // D. It excludes the wait for a free residency slot, which is presenter
  // queueing that grows with D itself.
  PresenterSampleWindow<128> capture_issue_to_reserve_ns;
  // D's dependency term: issue to reserve minus known ZeroFG-imposed waits.
  PresenterSampleWindow<128> capture_issue_to_reserve_dependency_ns;
  uint64_t capture_dependency_clamped_total = 0;
  PresenterSampleWindow<128> split_preadmission_gate_hold_ns;
  // P: raw Source issue intervals, the clean ones that taught P, and the
  // intervals censored by ZeroFG backpressure (bp) or right after it.
  PresenterSampleWindow<128> source_issue_raw_interval_ns;
  PresenterSampleWindow<128> source_issue_learned_interval_ns;
  PresenterSampleWindow<128> source_issue_bp_wait_ns;
  uint64_t source_issue_bp_interval_total = 0;
  uint64_t source_issue_bp_wait_ns_total = 0;
  // BP not smaller than its raw interval (attribution sanity check).
  uint64_t source_issue_bp_clamped_total = 0;
  uint64_t source_issue_p_censored_bp_total = 0;
  uint64_t source_issue_p_censored_post_bp_total = 0;
  bool source_issue_previous_bp = false;
  // Runs of consecutive intervals free of our backpressure, and the recent
  // backpressure history the rate probe arms on.
  uint32_t source_issue_clean_run_length = 0;
  uint32_t source_issue_clean_run_max = 0;
  uint64_t source_issue_clean_run_learnable_total = 0;
  PresenterSampleWindow<64> source_issue_clean_run_samples;
  uint32_t source_issue_recent_bp_mask = 0;
  uint32_t source_issue_recent_bp_count = 0;
  // Source rate probe (startup DEV switch): a short window in which the
  // handoff is drained eagerly so the Source runs free and its own cadence
  // becomes observable again.
  enum class SourceRateProbeState : uint8_t { kIdle, kDraining, kMeasuring };
  bool split_physical_operating_point_enabled = false;
  SourceRateProbeState source_rate_probe_state = SourceRateProbeState::kIdle;
  SourceRateProbeReason source_rate_probe_reason = SourceRateProbeReason::kNormal;
  uint64_t source_rate_probe_begin_ns = 0;
  uint64_t source_rate_probe_next_due_ns = 0;
  uint32_t source_rate_probe_interval_budget = 0;
  PresenterSampleWindow<16> source_rate_probe_samples_ns;
  uint64_t source_rate_probe_reference_period_ns = 0;
  uint64_t source_rate_probe_min_candidate_ns = 0;
  uint64_t source_rate_probe_max_candidate_ns = 0;
  uint64_t source_rate_probe_start_issue_sequence = 0;
  uint64_t source_rate_probe_deadline_issue_sequence = 0;
  uint64_t source_rate_probe_attempt_total = 0;
  uint64_t source_rate_probe_episode_started_total = 0;
  uint64_t source_rate_probe_candidate_in_range_total = 0;
  uint64_t source_rate_probe_candidate_out_of_range_total = 0;
  uint64_t source_rate_probe_episode_expired_total = 0;
  uint64_t source_rate_probe_confirmed_transition_total = 0;
  uint64_t source_rate_probe_confirm_total = 0;
  uint64_t source_rate_probe_no_change_total = 0;
  uint64_t source_rate_probe_abort_total = 0;
  uint64_t source_rate_probe_drop_total = 0;
  uint64_t source_rate_probe_last_candidate_ns = 0;
  uint64_t source_rate_probe_recovery_requested_total = 0;
  uint64_t source_rate_probe_recovery_requested_ns = 0;
  uint64_t source_rate_probe_recovery_requested_issue = 0;
  uint64_t source_rate_probe_recovery_attempt_total = 0;
  uint64_t source_rate_probe_recovery_confirm_total = 0;
  uint64_t source_rate_probe_recovery_no_change_total = 0;
  uint64_t source_rate_probe_recovery_abort_total = 0;
  uint64_t source_rate_probe_recovery_expired_total = 0;
  bool source_rate_probe_recovery_pending = false;
  PhysicalOperatingPointState physical_operating_point_state =
      PhysicalOperatingPointState::kIdle;
  std::array<PhysicalOperatingPointObservation,
             kPhysicalOperatingPointWindow> physical_op_observations = {};
  uint32_t physical_op_observation_count = 0;
  uint32_t physical_op_observation_next = 0;
  uint32_t physical_op_candidate_windows = 0;
  uint64_t physical_op_candidate_start_ns = 0;
  uint64_t physical_op_candidate_start_issue = 0;
  uint64_t physical_op_candidate_period_ns = 0;
  bool physical_op_refinement_active = false;
  uint64_t physical_op_refinement_start_ns = 0;
  uint64_t physical_op_refinement_start_issue = 0;
  uint64_t physical_op_refinement_candidate_period_ns = 0;
  uint32_t physical_op_refinement_windows = 0;
  uint64_t physical_op_refinement_enter_total = 0;
  uint64_t physical_op_refinement_confirm_total = 0;
  uint64_t physical_op_refinement_abort_total = 0;
  uint64_t physical_op_candidate_abort_total = 0;
  uint64_t physical_op_candidate_enter_total = 0;
  uint64_t physical_op_confirm_total = 0;
  uint64_t physical_op_exit_total = 0;
  uint64_t physical_op_reentry_total = 0;
  uint64_t physical_op_last_entry_ns = 0;
  uint64_t physical_op_last_exit_ns = 0;
  uint64_t physical_op_idle_since_ns = 0;
  uint64_t physical_op_last_raw_period_ns = 0;
  uint64_t physical_op_last_bp_fraction_per_mille = 0;
  uint64_t physical_op_last_post_bp_fraction_per_mille = 0;
  uint64_t physical_op_last_censored_fraction_per_mille = 0;
  uint64_t physical_op_last_clean_fraction_per_mille = 0;
  uint64_t physical_op_last_hold_fraction_per_mille = 0;
  uint64_t physical_op_last_pressure_fraction_per_mille = 0;
  uint32_t physical_op_slack_streak = 0;
  uint64_t physical_op_probe_cooldown_until_ns = 0;
  bool physical_op_confirmed = false;
  bool physical_op_hold_snapshot_initialized = false;
  uint64_t physical_op_previous_hold_total = 0;
  std::atomic<uint64_t> physical_op_real_finalready_before_target_total{0};
  uint64_t physical_op_real_finalready_observed_total = 0;
  bool physical_op_real_finalready_snapshot_initialized = false;
  bool physical_op_last_recovery_signal = false;
  uint64_t physical_op_last_raw_ratio_per_mille = 0;
  uint64_t physical_op_candidate_reference_ns = 0;
  uint32_t physical_op_candidate_below_entry_windows = 0;
  bool physical_op_candidate_below_entry_logged = false;
  uint64_t physical_op_candidate_below_entry_window_total = 0;
  uint32_t physical_op_refinement_below_entry_windows = 0;
  bool physical_op_refinement_below_entry_logged = false;
  uint64_t physical_op_refinement_below_entry_window_total = 0;
  std::array<uint64_t, size_t(PhysicalOperatingPointAbortReason::kCount)>
      physical_op_candidate_abort_reason_total = {};
  std::array<uint64_t, size_t(PhysicalOperatingPointAbortReason::kCount)>
      physical_op_refinement_abort_reason_total = {};
  uint64_t physical_op_recovery_partial_total = 0;
  uint64_t physical_op_recovery_full_total = 0;
  uint64_t physical_op_recovery_normal_exit_total = 0;
  uint64_t physical_op_recovery_last_candidate_ns = 0;
  const char* physical_op_recovery_last_outcome = "none";
  PresenterSampleWindow<32> source_period_learned_samples_ns;
  uint64_t source_period_learned_ns = 0;
  TransitionPlannedSpaceState transition_planned_space_state =
      TransitionPlannedSpaceState::kIdle;
  uint64_t h1c_fast_sequence_start_ns = 0;
  uint64_t h1c_fast_sequence_start_issue = 0;
  uint64_t h1c_fast_evidence_start_ns = 0;
  uint64_t h1c_fast_evidence_start_issue = 0;
  uint64_t h1c_fast_evidence_last_issue = 0;
  bool h1c_fast_evidence_valid = false;
  bool h1c_fast_evidence_attempted = false;
  bool h1c_fast_evidence_inhibited = false;
  uint64_t h1c_confirm_issue = 0;
  uint64_t h1c_drain_start_ns = 0;
  uint64_t h1c_drain_deadline_issue = 0;
  uint64_t h1c_drain_synthetic_source_watermark = 0;
  uint64_t h1c_episode_real_only_pairs = 0;
  uint64_t h1c_episode_quarantined_pairs = 0;
  uint64_t h1c_episode_s_at_start = 0;
  uint64_t h1c_episode_s_at_drain_start = 0;
  uint32_t h1c_episode_frontier_at_start = 0;
  uint32_t h1c_episode_chains_at_start = 0;
  uint32_t h1c_episode_frontier_at_drain_start = 0;
  uint32_t h1c_episode_chains_at_drain_start = 0;
  uint64_t h1c_fast_sequence_start_total = 0;
  uint64_t h1c_transition_real_only_pair_total = 0;
  uint64_t h1c_false_positive_pair_total = 0;
  uint64_t h1c_post_confirm_quarantine_total = 0;
  uint64_t h1c_drain_complete_total = 0;
  // The presenter loop is draining toward its exit (PresenterThreadMain).
  bool presenter_draining = false;
  uint64_t h1c_timeout_total = 0;
  uint64_t h1c_reset_total = 0;
  uint64_t h1c_recovery_probe_inhibit_total = 0;
  uint64_t h1c_planned_ticks_saved_total = 0;
  uint64_t h1c_short_boundary_commit_total = 0;
  uint64_t h1c_s_retired_during_drain_total = 0;
  uint64_t h1c_pair_compressed_normal_total = 0;
  uint64_t h1c_pair_planned_floor_blocked_total = 0;
  uint64_t h1c_commit_trace_total = 0;
  // H1c gate A window: owned by H1c, read only by its gate and telemetry.
  PresenterSampleWindow<kH1cShortRawWindow> h1c_short_raw_interval_ns;
  uint64_t h1c_short_raw_median_ns = 0;
  uint64_t h1c_short_raw_mean_ns = 0;
  bool h1c_short_raw_ready = false;
  bool h1c_short_raw_slow = false;
  uint32_t h1c_fast_evidence_clean_count = 0;
  uint64_t h1c_evidence_sequence_total = 0;
  uint64_t h1c_second_sample_total = 0;
  uint64_t h1c_single_sample_sequence_total = 0;
  std::array<uint64_t, size_t(H1cResetReason::kCount)>
      h1c_single_sample_end_reason_total = {};
  std::array<uint64_t, size_t(H1cResetReason::kCount)>
      h1c_episode_end_reason_total = {};
  uint64_t h1c_raw_slow_reset_total = 0;
  uint64_t h1c_raw_slow_blocked_sample_total = 0;
  uint64_t h1c_raw_slow_bimodal_suspect_total = 0;
  uint64_t h1c_physical_op_block_total = 0;
  uint64_t h1c_evidence_log_total = 0;
  PresenterSampleWindow<128> capture_submit_to_first_poll_begin_ns;
  PresenterSampleWindow<128> capture_submit_to_ready_poll_begin_ns;
  PresenterSampleWindow<128> active_generation_poll_host_ns;
  PresenterSampleWindow<128> active_post_poll_host_ns;
  std::array<PresenterSampleWindow<128>, size_t(PresenterWakeReason::kCount)>
      presenter_wait_overshoot_ns = {};
  PresenterSampleWindow<128> apply_call_phase_error_ns;
  PresenterSampleWindow<128> apply_call_phase_early_ns;
  PresenterSampleWindow<128> apply_call_phase_late_ns;
  PresenterSampleWindow<128> actual_apply_lead_ns;
  PresenterSampleWindow<128> ordered_ready_wait_ns;
  PresenterSampleWindow<128> hard_real_target_lateness_ns;
  PresenterSampleWindow<128> hard_real_waiting_post_not_before_lateness_ns;
  PresenterSampleWindow<128> hard_real_waiting_post_soft_deadline_lateness_ns;
  PresenterSampleWindow<128> hard_real_waiting_post_free_final_outputs;
  PresenterSampleWindow<128> hard_real_waiting_post_required_surplus;
  PresenterSampleWindow<128> hard_real_waiting_post_earlier_unfunded;
  PresenterSampleWindow<128> hard_real_final_ready_age_ns;
  PresenterSampleWindow<128> hard_real_final_ready_runway_ns;
  PresenterSampleWindow<128> hard_real_final_ready_late_at_ready_ns;
  PresenterSampleWindow<128> generation_submit_ahead_of_latest_ns;
  PresenterSampleWindow<128> generation_submit_late_ns;
  PresenterSampleWindow<128> generation_submit_after_not_before_ns;
  PresenterSampleWindow<128> post_submit_ahead_of_latest_ns;
  PresenterSampleWindow<128> post_submit_late_ns;
  PresenterSampleWindow<128> post_submit_after_not_before_ns;
  PresenterSampleWindow<128> publish_to_apply_ns;
  PresenterSampleWindow<128> ready_to_apply_ns;
  PresenterSampleWindow<128> post_queue_wait_ns;
  PresenterSampleWindow<128> post_submit_host_ns;
  PresenterSampleWindow<128> post_host_block_ns;
  PresenterSampleWindow<128> post_gpu_completion_ns;
  PresenterSampleWindow<128> post_service_gpu_ns;
  PresenterSampleWindow<128> post_submit_to_ready_ns;
  // Post submit host time split by whether the candidate dependency had
  // completed before the Post entered q0, and the sync_file export host time.
  PresenterSampleWindow<128> post_submit_host_input_complete_ns;
  PresenterSampleWindow<128> post_submit_host_input_pending_ns;
  PresenterSampleWindow<128> post_export_host_ns;
  PresenterSampleWindow<128> post_submit_host_real_ns;
  PresenterSampleWindow<128> post_submit_host_synthetic_ns;
  PresenterSampleWindow<128> generation_queue_wait_ns;
  PresenterSampleWindow<128> generation_submit_host_ns;
  PresenterSampleWindow<128> generation_host_block_ns;
  PresenterSampleWindow<128> generation_gpu_completion_ns;
  PresenterSampleWindow<128> generation_service_gpu_ns;
  uint64_t profile_readback_attempt_total = 0;
  uint64_t profile_readback_success_total = 0;
  uint64_t profile_readback_not_ready_total = 0;
  uint64_t profile_readback_error_total = 0;
  PresenterSampleWindow<128> generation_submit_to_ready_ns;
  PresenterSampleWindow<128> generation_setup_ns;
  PresenterSampleWindow<128> generation_total_ns;
  PresenterSampleWindow<16> generation_bootstrap_total_ns;
  PresenterSampleWindow<128> generation_steady_total_ns;
  PresenterSampleWindow<32> synthetic_chain_residence_ns;
  PresenterSampleWindow<128> synthetic_nominal_lateness_ns;
  PresenterSampleWindow<128> synthetic_final_ready_lead_ns;
  PresenterSampleWindow<128> issue_to_candidate_ready_ns;
  PresenterSampleWindow<128> source_issue_to_publish_ns;
  PresenterSampleWindow<128> ingress_submit_host_ns;
  PresenterTimestampWindow<kSourceEffectiveRateTimestampSamples>
      source_effective_rate_timestamps_ns;
  PresenterTimestampWindow<kRealOnlyValidationSamples + 1>
      source_real_only_c2_rate_timestamps_ns;
  PresenterSampleWindow<128> source_pair_interval_ns;
  PresenterSampleWindow<32> source_period_samples_ns;
  PresenterSampleWindow<8> source_transition_samples_ns;
  PlannedSpaceTrace planned_space_trace;
  PresenterSampleWindow<32> source_protection_treatment_ns;
  uint32_t post_effect_count = 0;
  uint64_t source_period_ns = 0;
  uint64_t period_epoch = 1;
  uint64_t period_sample_total = 0;
  uint64_t source_rejected_sample_total = 0;
  uint64_t source_transition_confirmed_total = 0;
  uint64_t source_transition_slower_confirmed_total = 0;
  uint64_t source_transition_sample_total = 0;
  uint64_t source_effective_rate_last_ns = 0;
  uint64_t source_effective_rate_observation_total = 0;
  uint64_t source_effective_rate_unstable_total = 0;
  uint64_t source_pair_shorter_than_rate_total = 0;
  uint64_t source_pair_longer_than_rate_total = 0;
  // Split Always-S presentation falsifiers. Semantic targets remain
  // unchanged; same-latch accounting is resolved by logical predecessor
  // links, never by callback arrival order.
  uint64_t split_apply_fence_wait_real_total = 0;
  uint64_t split_apply_fence_wait_synthetic_total = 0;
  uint64_t split_present_spacing_push_total = 0;
  uint64_t split_present_spacing_push_max_ns = 0;
  uint64_t split_preaccept_capacity_gate_total = 0;
  // The presentation advance (ObservePresentationAdvance), always on in
  // the release line.
  static constexpr bool kPresentationAdvanceEnabled = true;
  static constexpr uint64_t kPresentationAdvanceEvery = 4;
  static constexpr uint64_t kPresentationAdvanceMaxNs = 250000000ull;
  static constexpr uint32_t kPresentationAdvanceRiseStreak = 4;
  static constexpr uint32_t kPresentationAdvanceLateStreak = 2;
  uint64_t presentation_advance_ns = 0;
  uint64_t presentation_advance_outputs_since = 0;
  uint32_t presentation_advance_up_streak = 0;
  uint32_t presentation_advance_late_streak = 0;
  uint32_t presentation_advance_below_streak = 0;
  uint64_t presentation_advance_gate_seen = 0;
  uint64_t presentation_advance_capacity_down = 0;
  uint64_t presentation_advance_steps_up = 0;
  uint64_t presentation_advance_steps_down = 0;
  uint64_t presentation_advance_max_ns = 0;
  uint64_t presentation_advance_excess_max_ns = 0;
  uint64_t split_preadmission_total = 0;
  uint64_t split_preadmission_gate_deferred_total = 0;
  uint64_t split_preadmission_other_deferred_total = 0;
  uint64_t split_s_obligation_created_total = 0;
  uint64_t split_s_waiting_for_physical_chain_total = 0;
  uint64_t split_s_obligation_high_water = 0;
  uint64_t split_b_ready_waiting_for_s_total = 0;
  uint64_t split_forbidden_s_drop_total = 0;
  uint64_t split_live_real_retarget_total = 0;
  bool stable_latency_metronome_armed = false;
  uint64_t stable_output_quantum_ns = 0;
  uint64_t operating_latency_ns = 0;
  uint64_t latency_required_last_ns = 0;
  uint64_t next_semantic_target_ns = 0;
  uint64_t last_semantic_target_ns = 0;
  uint64_t semantic_epoch_origin_ns = 0;
  uint64_t next_semantic_tick_index = 0;
  uint64_t last_accounted_semantic_target_ns = 0;
  uint64_t last_applied_semantic_target_ns = 0;
  // Real first-commitment cause telemetry (observation only; cumulative per
  // Connect).
  OrderedBoundaryProvenance ordered_boundary_provenance;
  uint64_t commit_route_pair_boundary_total = 0;
  uint64_t commit_route_projection_total = 0;
  std::array<uint64_t, 8> commit_projection_mask_total = {};
  std::array<uint64_t, 8> commit_projection_mask_extra_tick_total = {};
  std::array<uint64_t, 5> commit_boundary_kind_total = {};
  uint64_t commit_ready_at_commit_total = 0;
  uint64_t real_commit_late_total = 0;
  uint64_t real_commit_observed_total = 0;
  PresenterSampleWindow<128> real_ready_age_at_commit_ns;
  PresenterSampleWindow<128> real_commit_runway_pair_ns;
  PresenterSampleWindow<128> real_commit_runway_projection_ns;
  PresenterSampleWindow<128> real_commit_lateness_ns;
  OutputCandidate last_applied_real_commitment;
  OutputCandidate last_applied_synthetic_commitment;
  uint64_t source_phase_epoch_pair_period_last_ns = 0;
  uint64_t source_phase_current_source_period_last_ns = 0;
  int64_t source_phase_period_error_last_ns = 0;
  uint64_t source_phase_source_anchor_last_ns = 0;
  uint64_t source_phase_next_semantic_target_last_ns = 0;
  int64_t source_phase_error_last_ns = 0;
  uint64_t source_phase_lead_high_ns = 0;
  uint64_t source_phase_lag_high_ns = 0;
  uint64_t source_phase_observation_total = 0;
  uint64_t source_phase_stable_source_sample_total = 0;
  bool source_phase_previous_valid = false;
  int64_t source_phase_previous_error_ns = 0;
  uint64_t source_phase_confirmation_start_issue_ns = 0;
  int64_t source_phase_confirmation_period_error_ns = 0;
  uint64_t source_phase_confirmation_frequency_accumulated_ns = 0;
  uint32_t source_phase_confirmation_samples = 0;
  uint32_t source_phase_confirmation_consistent_samples = 0;
  uint64_t source_phase_confirmation_forward_skip_current = 0;
  uint64_t source_phase_confirmation_forward_skip_total = 0;
  uint64_t source_phase_confirmation_direction_restart_total = 0;
  bool source_phase_reanchor_pending = false;
  SourceReanchorProvenance source_reanchor_provenance;
  uint64_t source_phase_reanchor_request_total = 0;
  uint64_t source_phase_reanchor_faster_source_request_total = 0;
  uint64_t source_phase_reanchor_slower_source_request_total = 0;
  uint64_t source_phase_reanchor_total = 0;
  uint64_t source_phase_reanchor_faster_source_total = 0;
  uint64_t source_phase_reanchor_slower_source_total = 0;
  uint64_t source_phase_reanchor_old_output_quantum_ns = 0;
  uint64_t source_phase_reanchor_new_output_quantum_ns = 0;
  int64_t source_phase_reanchor_phase_last_ns = 0;
  uint64_t source_phase_reanchor_frequency_evidence_last_ns = 0;
  uint64_t source_phase_reanchor_forward_skip_confirmation_last = 0;
  uint64_t source_phase_reanchor_live_real_invalidation_total = 0;
  int64_t source_phase_frequency_drift_total_ns = 0;
  int64_t source_phase_reanchor_impulse_total_ns = 0;
  int64_t source_phase_residual_total_ns = 0;
  uint64_t phase_reserve_target_ns = 0;
  uint64_t phase_reserve_actual_ns = 0;
  uint64_t phase_reserve_minimum_ns = 0;
  uint64_t phase_reserve_deficit_ns = 0;
  uint64_t pending_source_reanchor_period_ns = 0;
  uint64_t phase_debt_ns = 0;
  int64_t phase_debt_block_error_ns = 0;
  uint64_t phase_debt_block_span_ns = 0;
  uint64_t phase_debt_block_period_ns = 0;
  uint64_t phase_debt_period_epoch = 0;
  uint32_t phase_debt_block_intervals = 0;
  uint64_t phase_debt_blocks_total = 0;
  uint64_t phase_debt_fire_total = 0;
  uint64_t phase_debt_reset_total = 0;
  uint64_t phase_debt_partial_discard_total = 0;
  ResidualPhaseShadow residual_phase_shadow;
  uint64_t latency_epoch = 1;
  // H12 deferred latency feedback (S judged when its anchor arrives).
  uint64_t synthetic_deferred_feedback_success_total = 0;
  uint64_t synthetic_deferred_feedback_miss_total = 0;
  bool latency_depth_reanchor_pending = false;
  uint64_t latency_depth_reanchor_total = 0;
  uint64_t latency_depth_reanchor_live_real_invalidation_total = 0;
  uint64_t latency_depth_reanchor_future_anchor_ns = 0;
  int64_t latency_depth_reanchor_impulse_ns = 0;
  uint64_t latency_metronome_arm_total = 0;
  uint64_t latency_feasibility_miss_total = 0;
  uint64_t latency_feedback_excluded_total = 0;
  uint64_t latency_feedback_true_feasibility_total = 0;
  uint64_t latency_feedback_excluded_residency_pressure_total = 0;
  uint64_t latency_feedback_excluded_final_output_pressure_total = 0;
  uint64_t latency_feedback_excluded_funding_total = 0;
  uint64_t latency_feedback_excluded_ordering_total = 0;
  uint64_t latency_feedback_excluded_scheduler_total = 0;
  uint64_t latency_feedback_excluded_transition_total = 0;
  std::array<uint64_t, size_t(LatencyMissProvenance::kCount)>
      latency_miss_provenance_total = {};
  std::array<uint64_t, size_t(LatencyMissProvenance::kCount)>
      latency_miss_provenance_one_quantum_help_total = {};
  uint64_t latency_success_total = 0;
  uint64_t semantic_hold_total = 0;
  uint64_t dispatch_lead_ns = kDispatchLeadBootstrapNs;
  bool dispatch_lead_controller_armed = false;
  uint64_t dispatch_lead_attack_total = 0;
  uint64_t dispatch_lead_last_attack_step_ns = 0;
  uint64_t dispatch_lead_last_attack_time_ns = 0;
  uint64_t dispatch_lead_hard_ready_miss_sample_total = 0;
  uint64_t dispatch_lead_hard_ready_attack_total = 0;
  uint64_t residency_recovery_post_physical_release_total = 0;
  uint64_t presenter_arbiter_cycle_total = 0;
  // Presenter CPU accounting: arbiter cycles that found no progress, and the
  // time the presenter thread actually spent blocked in its wake wait. The
  // cpu_summary_last_* snapshots turn the totals into per-summary deltas.
  uint64_t presenter_zero_progress_cycle_total = 0;
  uint64_t presenter_wait_call_total = 0;
  uint64_t presenter_wait_blocked_ns_total = 0;
  uint64_t cpu_summary_last_wall_ns = 0;
  uint64_t cpu_summary_last_thread_ns = 0;
  uint64_t cpu_summary_last_process_ns = 0;
  uint64_t cpu_summary_last_cycles = 0;
  uint64_t cpu_summary_last_zero_progress = 0;
  uint64_t cpu_summary_last_wait_calls = 0;
  uint64_t cpu_summary_last_wait_blocked_ns = 0;
  uint64_t cpu_summary_last_applied = 0;
  uint32_t arbiter_blocking_submits_current = 0;
  uint32_t arbiter_blocking_submits_high_water = 0;
  uint32_t arbiter_host_driver_ops_current = 0;
  uint32_t arbiter_host_driver_ops_high_water = 0;
  bool arbiter_first_pump_completed = false;
  bool normal_driver_turn_prefers_poll = true;
  uint64_t active_capture_poll_total = 0;
  uint64_t active_generation_poll_total = 0;
  uint64_t active_post_poll_total = 0;
  uint64_t generation_poll_not_due_total = 0;
  uint64_t post_poll_not_due_total = 0;
  uint64_t generation_poll_retry_scheduled_total = 0;
  uint64_t post_poll_retry_scheduled_total = 0;
  uint64_t head_observation_not_due_total = 0;
  uint64_t observation_bounded_retry_wake_total = 0;
  uint64_t poll_allowed_head_critical_total = 0;
  uint64_t poll_allowed_source_critical_total = 0;
  BlockingOperation arbiter_last_blocking_operation =
      BlockingOperation::kNone;
  uint64_t arbiter_last_blocking_begin_ns = 0;
  uint64_t arbiter_last_blocking_end_ns = 0;
  BlockingOperation previous_cycle_blocking_operation =
      BlockingOperation::kNone;
  uint64_t previous_cycle_blocking_begin_ns = 0;
  uint64_t previous_cycle_blocking_end_ns = 0;
  uint64_t pump_logical_order_cpu_total_ns = 0;
  uint64_t pump_logical_order_call_total = 0;
  std::array<uint64_t, size_t(BlockingOperation::kCount)>
      hard_ready_miss_by_blocking_operation = {};
  std::array<uint64_t, size_t(BlockingOperation::kCount)>
      hard_ready_miss_crossed_by_blocking_operation = {};
  std::array<uint64_t, size_t(PresenterWakeReason::kCount)>
      hard_ready_miss_after_wake_reason = {};
  uint64_t hard_ready_miss_attribution_sample_total = 0;
  std::array<uint64_t, size_t(PresenterWakeReason::kCount)>
      presenter_wait_requested_by_reason = {};
  std::array<uint64_t, size_t(PresenterWakeReason::kCount)>
      presenter_wait_deadline_return_by_reason = {};
  uint64_t presenter_wait_event_wake_total = 0;
  uint64_t requested_wake_deadline_ns = 0;
  uint64_t actual_wait_return_ns = 0;
  PresenterWakeReason last_presenter_wake_reason =
      PresenterWakeReason::kProducerEvent;
  uint64_t accepted_real_drop_violation_total = 0;
  uint64_t apply_before_planned_violation_total = 0;
  uint64_t multi_blocking_submit_quantum_violation_total = 0;
  uint64_t multi_host_driver_op_quantum_violation_total = 0;
  uint64_t first_pump_before_driver_violation_total = 0;
  uint64_t semantic_real_tick_total = 0;
  uint64_t semantic_synthetic_tick_total = 0;
  uint64_t semantic_real_deferred_total = 0;
  uint64_t semantic_real_deferred_tick_total = 0;
  uint64_t future_cursor_advanced_by_real_defer_total = 0;
  uint64_t future_cursor_advanced_by_real_defer_tick_total = 0;
  uint64_t real_exec_debt_high_water = 0;
  uint64_t synthetic_dropped_for_real_backlog_total = 0;
  uint64_t real_exec_debt_discharge_tick_total = 0;
  uint64_t real_exec_debt_zero_transition_total = 0;
  uint64_t semantic_target_order_violation_total = 0;
  uint64_t shallow_real_commit_total = 0;
  uint64_t shallow_real_commit_nominal_total = 0;
  uint64_t shallow_real_commit_backlog_total = 0;
  uint64_t shallow_real_commit_backlog_tick_total = 0;
  uint64_t shallow_real_commitment_violation_total = 0;
  uint64_t pending_real_nominal_violation_total = 0;
  uint64_t physical_real_acceptance_cursor_advance_total = 0;
  uint32_t committed_real_high_water = 0;
  bool shallow_real_commitment_violation_latched = false;
  bool pending_real_nominal_violation_latched = false;
  uint64_t soft_dispatch_miss_salvaged_real_total = 0;
  uint64_t soft_dispatch_miss_salvaged_synthetic_total = 0;
  uint64_t hard_target_miss_real_total = 0;
  uint64_t hard_target_miss_synthetic_total = 0;
  uint64_t real_late_eligible_total = 0;
  uint64_t real_late_applied_total = 0;
  uint64_t real_late_blocked_total = 0;
  uint64_t real_late_blocked_order_total = 0;
  uint64_t real_late_blocked_not_head_total = 0;
  uint64_t real_late_blocked_structural_total = 0;
  uint64_t real_old_policy_would_defer_total = 0;
  uint64_t real_multi_late_apply_same_pump_total = 0;
  // Presenter-owned telemetry, reset only at Connect like the E1F windows.
  UnifiedRunwayShadow unified_runway_shadow;
  E2BClockCanary e2b;
  // H12 ALWAYS-S: production is work-conserving, presentation is paced. These
  // prove the new contract: admission versus anchor arrival, production before
  // versus after the anchor, and the two falsifiers that must stay at/near
  // zero.
  uint64_t synthetic_admitted_anchored_total = 0;
  uint64_t synthetic_admitted_unanchored_total = 0;
  uint64_t synthetic_generation_submitted_before_anchor_total = 0;
  uint64_t synthetic_generation_submitted_after_anchor_total = 0;
  uint64_t synthetic_post_submitted_before_anchor_total = 0;
  uint64_t synthetic_post_submitted_after_anchor_total = 0;
  uint64_t synthetic_anchored_before_generation_total = 0;
  uint64_t synthetic_anchored_after_generation_total = 0;
  uint64_t synthetic_anchored_after_post_total = 0;
  uint64_t synthetic_anchored_after_final_ready_total = 0;
  uint64_t synthetic_final_ready_before_anchor_total = 0;
  uint64_t synthetic_apply_blocked_unanchored_total = 0;
  uint64_t synthetic_structural_orphan_reclaimed_total = 0;
  // MUST remain zero: no unanchored S may ever receive a semantic hard cutoff.
  uint64_t synthetic_unanchored_semantic_hard_drop_total = 0;
  // H12 historical name. After H13 this is "presentation expired before the
  // Generation submit", not a Production failure: Generation is still
  // submitted afterwards.
  uint64_t anchored_s_reached_b_safe_without_generation_job_total = 0;
  // H13 — presentation expiry does not cancel admitted production.
  uint64_t synthetic_presentation_hold_total = 0;
  uint64_t presentation_expired_before_generation_submit_total = 0;
  uint64_t presentation_expired_generation_inflight_total = 0;
  uint64_t presentation_expired_waiting_post_total = 0;
  uint64_t presentation_expired_post_submitted_total = 0;
  uint64_t presentation_expired_final_ready_total = 0;
  // Production that continued after the presentation window closed. Non-zero
  // here is the direct proof that B-safe no longer vetoes Production.
  uint64_t generation_submitted_after_presentation_hold_total = 0;
  uint64_t generation_completed_after_presentation_hold_total = 0;
  uint64_t post_submitted_after_presentation_hold_total = 0;
  uint64_t production_retired_after_presentation_hold_total = 0;
  // Hard safety falsifiers. Both MUST remain zero.
  uint64_t presentation_retired_reached_apply_total = 0;
  uint64_t presentation_retired_blocked_real_total = 0;
  // MUST be ~0 except for explicit lifecycle/structural/failure retirement.
  uint64_t admitted_s_retired_without_generation_attempt_total = 0;
  // H14 — factual rolling-pair presentation decisions. The old B-safe value
  // remains observational and cannot increment the first two counters.
  uint64_t h14_b_due_ready_s_not_ready_hold_total = 0;
  uint64_t h14_b_due_ready_s_ready_total = 0;
  uint64_t h14_s_applied_after_old_bsafe_total = 0;
  uint64_t h14_b_applied_after_ready_s_total = 0;
  uint64_t h14_b_head_pair_commit_total = 0;
  uint64_t h14_b_head_current_epoch_commit_total = 0;
  uint64_t h14_b_head_reanchor_cutover_total = 0;
  uint64_t semantic_cursor_late_streak = 0;
  uint64_t semantic_cursor_late_streak_max = 0;
  // E1F-SHADOW telemetry-only totals (see ComputeE1FShadowFirstTargetNs).
  // request_total == would_move_total + noop_total.
  uint64_t e1f_shadow_request_total = 0;
  uint64_t e1f_shadow_would_move_total = 0;
  uint64_t e1f_shadow_noop_total = 0;
  uint64_t e1f_next_real_commit_same_pump_total = 0;
  uint64_t e1f_same_pump_commit_would_use_old_cursor_total = 0;
  std::array<std::atomic<uint64_t>, size_t(LogicalOutputState::kCount)>
      hard_real_miss_by_state = {};
  std::atomic<uint64_t>
      hard_real_waiting_post_not_before_lateness_last_ns{0};
  std::atomic<uint64_t>
      hard_real_waiting_post_soft_deadline_lateness_last_ns{0};
  std::atomic<uint32_t> hard_real_waiting_post_free_final_outputs_last{0};
  std::atomic<uint32_t> hard_real_waiting_post_required_surplus_last{0};
  std::atomic<uint32_t> hard_real_waiting_post_earlier_unfunded_last{0};
  std::atomic<uint64_t> hard_real_final_ready_age_last_ns{0};
  std::atomic<int64_t> hard_real_final_ready_target_slack_last_ns{0};
  std::atomic<uint64_t> hard_real_target_lateness_last_ns{0};
  std::atomic<uint64_t> hard_real_target_lateness_max_ns{0};
  uint64_t semantic_stale_synthetic_drop_total = 0;
  uint64_t semantic_apply_diagnostic_total = 0;
  uint64_t final_ready_success_total = 0;
  uint64_t final_ready_miss_total = 0;
  uint64_t pair_opportunity_total = 0;
  uint64_t stable_pair_opportunity_total = 0;
  uint64_t stable_synthetic_applied_total = 0;
  uint64_t stable_synthetic_dropped_total = 0;
  uint64_t generation_attempt_total = 0;
  uint64_t generation_to_post_chain_candidate_total = 0;
  uint64_t generation_to_post_chain_submit_total = 0;
  uint64_t generation_to_post_chain_retired_total = 0;
  uint64_t synthetic_chain_warmup_attempt_total = 0;
  uint64_t synthetic_chain_warmup_submitted_total = 0;
  uint64_t synthetic_chain_warmup_completed_total = 0;
  uint64_t synthetic_chain_predicted_bypassed_total = 0;
  uint64_t synthetic_chain_warmup_applied_total = 0;
  uint64_t synthetic_chain_warmup_hard_drop_total = 0;
  uint64_t synthetic_chain_warmup_real_hard_miss_total = 0;
  uint64_t synthetic_chain_warmup_source_protection_drop_total = 0;
  uint64_t synthetic_chain_warmup_residency_pressure_drop_total = 0;
  uint64_t synthetic_chain_warmup_capture_near_miss_baseline = 0;
  uint64_t synthetic_chain_warmup_capture_near_miss_final = 0;
  bool synthetic_chain_warmup_started = false;
  bool synthetic_chain_estimator_armed = false;
  uint64_t generation_bootstrap_attempt_total = 0;
  uint64_t generation_bootstrap_success_total = 0;
  uint64_t generation_bootstrap_estimate_ns = 0;
  bool generation_bootstrap_armed = false;
  bool generation_unsustainable = false;
  uint64_t generation_unsustainable_total = 0;
  uint64_t synthetic_residency_surplus_admission_total = 0;
  uint64_t synthetic_residency_eager_admission_total = 0;
  uint64_t synthetic_residency_active_pressure_drop_total = 0;
  uint64_t synthetic_generation_eager_residency_total = 0;
  uint64_t last_synthetic_slack_ns = 0;
  uint64_t last_real_slack_ns = 0;
  uint64_t synthetic_generated_total = 0;
  uint64_t real_applied_total = 0;
  uint64_t synthetic_applied_total = 0;
  uint32_t synthetic_pool_high_water = 0;
  uint32_t synthetic_pool_occupancy = 0;
  uint32_t logical_output_high_water = 0;
  uint32_t logical_synthetic_high_water = 0;
  uint32_t future_real_commitment_high_water = 0;
  uint32_t generation_in_flight_high_water = 0;
  uint32_t post_in_flight_high_water = 0;
  uint64_t post_acquire_fence_ready_total = 0;
  uint64_t post_sync_fd_poll_total = 0;
  uint64_t post_sync_fd_ready_total = 0;
  uint64_t post_output_release_deferred_total = 0;
  uint32_t transaction_in_flight_high_water = 0;
  uint64_t arbitration_generation_selected_total = 0;
  uint64_t arbitration_post_selected_total = 0;
  uint64_t arbitration_post_urgent_total = 0;
  uint64_t synthetic_post_submit_late_total = 0;
  uint64_t synthetic_post_ready_late_total = 0;
  uint64_t synthetic_nominal_late_total = 0;
  uint64_t late_salvage_started_total = 0;
  uint64_t late_salvage_applied_total = 0;
  uint64_t late_salvage_safe_window_expired_total = 0;
  uint32_t late_salvage_high_water = 0;
  uint64_t new_s_blocked_no_logical_with_salvage_total = 0;
  uint64_t new_s_blocked_no_s_slot_with_salvage_total = 0;
  uint64_t real_hard_miss_while_late_salvage_total = 0;
  uint64_t real_defer_while_late_salvage_total = 0;
  uint64_t generation_jit_deferral_total = 0;
  uint64_t post_jit_deferral_total = 0;
  uint64_t synthetic_generation_would_jit_defer_total = 0;
  uint64_t synthetic_post_would_jit_defer_total = 0;
  uint64_t real_post_would_jit_defer_total = 0;
  uint64_t synthetic_generation_eager_submit_total = 0;
  uint64_t synthetic_post_eager_submit_total = 0;
  uint64_t real_early_transfer_total = 0;
  std::atomic<uint64_t> early_real_transfer_attempt_total{0};
  std::atomic<uint64_t> early_real_transfer_success_total{0};
  std::atomic<uint64_t> early_real_transfer_blocked_funding_total{0};
  std::atomic<uint64_t> early_real_transfer_blocked_final_pool_total{0};
  std::atomic<uint64_t> early_real_transfer_blocked_ownership_total{0};
  std::atomic<uint64_t> early_real_transfer_actual_release_total{0};
  std::atomic<uint64_t> early_real_transfer_no_release_total{0};
  std::atomic<uint64_t> early_real_transfer_compound_pending_total{0};
  std::atomic<uint32_t> residency_recovery_debt_high_water{0};
  std::atomic<uint64_t> residency_recovery_with_prior_debt_total{0};
  std::atomic<uint64_t> residency_recovery_debt_discharged_total{0};
  std::atomic<uint64_t>
      residency_recovery_debt_recycled_before_apply_total{0};
  uint64_t residency_recovery_generation_attempt_total = 0;
  uint64_t residency_recovery_generation_submit_total = 0;
  uint64_t residency_recovery_generation_poll_total = 0;
  uint64_t residency_recovery_generation_drop_total = 0;
  uint64_t residency_recovery_generation_physical_release_total = 0;
  std::atomic<uint64_t> synthetic_real_funding_yield_total{0};
  std::atomic<uint64_t> final_output_funding_block_total{0};
  // Phase 5 B2: the same blocks measured in time rather than counted.
  uint64_t funding_block_begin_ns = 0;
  uint64_t funding_block_total_ns = 0;
  PresenterSampleWindow<128> funding_block_ns;
  // Phase 5 B2c. Per-logical funding wait, segmented in time by the but-for
  // answer on the gate that logical actually failed.
  PresenterSampleWindow<128> funding_policy_wait_ns;
  PresenterSampleWindow<128> funding_physical_wait_ns;
  PresenterSampleWindow<128> pair_funding_policy_wait_ns;
  uint64_t funding_class_total_ns[size_t(FundingWaitClass::kCount)] = {};
  uint64_t funding_episode_open_total[size_t(FundingWaitClass::kCount)] = {};
  uint64_t funding_segment_switch_total = 0;
  uint64_t funding_closed_by_condition_total = 0;
  uint64_t funding_closed_by_state_total = 0;
  uint64_t funding_closed_by_acquire_total = 0;
  uint64_t pair_causal_clean_funding_subtracted_total = 0;
  FundingCensus last_funding_census;
  // Phase 5 Better D.
  PresenterSampleWindow<kBetterDWindowCapacity> better_d_need_ns;
  PresenterSampleWindow<128> better_d_deficit_ns;
  uint64_t better_d_window_epoch = 0;
  uint64_t better_d_f_ns = 0;
  uint64_t better_d_causal_last_ns = 0;
  uint64_t better_d_armed_depth_ns = 0;
  uint64_t better_d_qualified_total = 0;
  uint64_t better_d_censored_total[size_t(BetterDCensor::kCount)] = {};
  uint64_t better_d_backlog_marked_total = 0;
  uint64_t funding_backlog_marked_total = 0;
  uint64_t funding_backlog_censored_total = 0;
  uint64_t behind_physical_total = 0;
  uint64_t better_d_rise_total = 0;
  uint64_t better_d_rise_ns_total = 0;
  uint64_t better_d_decay_step_total = 0;
  bool better_d_f_at_ceiling = false;
  uint64_t better_d_f_ceiling_enter_total = 0;
  uint64_t better_d_f_ceiling_exit_total = 0;
  uint64_t better_d_f_ceiling_excess_max_ns = 0;
  // B3a / B3b.
  uint64_t better_d_deferred_total = 0;
  uint64_t better_d_deferred_resolved_total = 0;
  uint64_t better_d_double_score_total = 0;
  // B4.
  uint64_t better_d_issues_since_qualified = 0;
  bool better_d_tail_stale_active = false;
  bool better_d_stale_renewal_pending = false;
  uint64_t better_d_qualified_since_stale = 0;
  uint64_t better_d_stale_episode_total = 0;
  uint64_t better_d_stale_renewal_recovered_total = 0;
  uint64_t better_d_rise_unready_total = 0;
  PresenterSampleWindow<128> better_d_anchor_lag_ns;
  PresenterSampleWindow<128> better_d_source_late_ab_ns;
  uint64_t better_d_epoch_reset_total = 0;
  uint64_t better_d_arm_rise_total = 0;
  uint64_t better_d_arm_decay_total = 0;
  uint64_t better_d_rise_already_deep_total = 0;
  std::atomic<uint32_t> final_output_earlier_unfunded_high_water{0};
  std::atomic<uint64_t> residency_recovery_generation_advance_total{0};
  std::atomic<uint64_t> capture_reserve_near_miss_total{0};
  std::atomic<uint64_t> orphan_ready_final_output_total{0};
  std::atomic<uint64_t> final_output_owner_mismatch_total{0};
  uint64_t source_protection_baseline_ns = 0;
  uint32_t source_protection_strikes = 0;
  uint64_t source_protection_would_strike_total = 0;
  uint64_t source_protection_would_probe_total = 0;
  uint64_t source_real_only_validation_pass_total = 0;
  uint64_t source_real_only_baseline_period_ns = 0;
  bool source_real_only_c2_baseline_valid = false;
  bool source_real_only_c2_native_reference_valid = false;
  uint64_t source_real_only_c2_baseline_period_ns = 0;
  uint64_t source_real_only_c2_baseline_epoch = 0;
  uint64_t source_real_only_c2_window_epoch = 0;
  uint64_t source_real_only_c2_window_bp_wait_ns = 0;
  uint64_t source_real_only_c2_last_bp_explained_ns = 0;
  uint32_t source_real_only_c2_window_remaining = 0;
  uint64_t source_real_only_c2_baseline_acquired_total = 0;
  uint64_t source_real_only_c2_baseline_invalidated_total = 0;
  uint64_t source_real_only_c2_comparison_same_regime_total = 0;
  uint64_t source_real_only_c2_comparison_skipped_regime_mismatch_total = 0;
  uint64_t source_real_only_c2_faster_regime_evidence_total = 0;
  uint64_t source_real_only_c2_slower_regime_evidence_total = 0;
  uint64_t source_real_only_c2_unstable_window_total = 0;
  uint64_t source_real_only_c2_window_overrun_total = 0;
  uint64_t source_real_only_c2_bp_attributed_slowdown_total = 0;
  uint64_t c2_bootstrap_completed_by_comparison_total = 0;
  uint64_t c2_bootstrap_completed_without_comparison_total = 0;
  std::atomic<uint64_t> source_validation_start_sequence{0};
  bool source_issue_cursor_initialized = false;
  uint64_t last_processed_source_issue_period_sequence = 0;
  uint64_t source_issue_interval_accumulator_ns = 1;
  // Raw issue timeline for REAL_ONLY sovereignty (never BP-corrected).
  uint64_t source_issue_raw_accumulator_ns = 1;
  uint64_t source_issue_period_consumed_total = 0;
  uint64_t source_issue_period_overrun_total = 0;
  uint64_t source_issue_snapshot_inconsistent_total = 0;
  uint64_t source_real_only_ratio_per_mille = 0;
  std::array<uint64_t, size_t(xe::SourceContractFault::kCount)>
      source_contract_baseline{};
  uint64_t source_protection_pair_floor = 0;
  uint64_t source_protection_after_issue_sequence = 0;
  uint64_t source_phase_previous_source_id = 0;
  uint64_t armed_real_without_eligibility = 0;
  uint64_t armed_real_apply_without_target = 0;
  uint64_t accepted_gap_epoch_reset = 0;
  // Since the 2026-09-10 accepted-gap pair-distance reopen this counts
  // accepted gap PAIRS (d = B.source_id - A.source_id > 1). A gap no longer
  // suppresses S; the histogram below splits them as d=2 / d=3 / d>=4.
  uint64_t accepted_gap_observation_total = 0;
  std::array<uint64_t, 3> accepted_gap_pair_distance_total = {};
  uint64_t accepted_gap_synthetic_admitted_total = 0;
  uint64_t accepted_gap_synthetic_applied_total = 0;
  uint64_t accepted_gap_synthetic_hold_total = 0;
  uint64_t accepted_gap_real_scaled_commit_total = 0;
  uint64_t accepted_gap_real_floor_lead_max_ns = 0;
  // MUST remain zero: a same-epoch pair head committed inside its lattice
  // distance.
  uint64_t pair_distance_compressed_real_total = 0;
  // 2026-09-10 phase-bound correction: gap pairs whose IssueSwap distance
  // Source phase refused (and the ticks not materialized), gaps whose
  // IssueSwaps were bunched, and pair-local B commitments beyond B's Source
  // phase target, split consecutive/gap.
  uint64_t accepted_gap_pair_phase_capped_total = 0;
  uint64_t accepted_gap_pair_phase_capped_tick_total = 0;
  uint64_t accepted_gap_issue_bunched_total = 0;
  uint64_t accepted_gap_issue_bunch_max_ns = 0;
  uint64_t accepted_gap_b_phase_excess_max_ns = 0;
  uint64_t consecutive_pair_b_phase_excess_max_ns = 0;
  uint64_t structural_timing_failure_total = 0;
  bool source_rate_transition_reanchor_pending = false;
  bool source_rate_transition_slower_pending = false;
  uint64_t source_real_only_validation_fail_total = 0;
  bool last_pair_synthetic_applied = false;
  GenerationFailureStage last_generation_failure_stage =
      GenerationFailureStage::kNone;
  uint32_t last_generation_status = 0;
  VkResult last_generation_vk_result = VK_SUCCESS;
  bool last_generation_vk_result_valid = false;
  bool last_generation_submission_accepted = false;
  std::array<uint64_t, size_t(SyntheticDropReason::kCount)>
      synthetic_drop_total = {};
  PresenterTimestampWindow<128> applied_timestamps;
};

#else

struct ZeroFGIndependentPresenter::Impl {};

#endif

ZeroFGIndependentPresenter::ZeroFGIndependentPresenter(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ZeroFGIndependentPresenter::~ZeroFGIndependentPresenter() = default;

std::unique_ptr<ZeroFGIndependentPresenter>
ZeroFGIndependentPresenter::Create(
    VulkanDevice* vulkan_device, bool elevated_presenter_priority,
    uint64_t synthetic_cost_seed_ns,
    IngressSourceAcquireCallback ingress_source_acquire_callback,
    PaintConfigProvider paint_config_provider,
    GenerationCallback generation_callback,
    GenerationPollCallback generation_poll_callback,
    SyntheticReleaseCallback synthetic_release_callback,
    GenerationShutdownCallback generation_shutdown_callback,
    PostProcessCallback post_process_callback,
    PostProcessReleaseCallback post_process_release_callback,
    PostProcessShutdownCallback post_process_shutdown_callback,
    PresenterDeviceDrainCallback presenter_device_drain_callback) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  auto impl = std::make_unique<Impl>(
      vulkan_device, elevated_presenter_priority, synthetic_cost_seed_ns,
      std::move(ingress_source_acquire_callback),
      std::move(paint_config_provider), std::move(generation_callback),
      std::move(generation_poll_callback),
      std::move(synthetic_release_callback),
      std::move(generation_shutdown_callback),
      std::move(post_process_callback),
      std::move(post_process_release_callback),
      std::move(post_process_shutdown_callback),
      std::move(presenter_device_drain_callback));
  if (!impl->Initialize()) {
    return nullptr;
  }
  return std::unique_ptr<ZeroFGIndependentPresenter>(
      new ZeroFGIndependentPresenter(std::move(impl)));
#else
  XELOGW("ZeroFGC0: independent presenter is Android-only");
  return nullptr;
#endif
}

bool ZeroFGIndependentPresenter::Connect(void* native_window,
                                           VkExtent2D extent,
                                           VkFormat final_output_format,
                                           VkSurfaceKHR main_surface) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  return impl_ && impl_->Connect(native_window, extent, final_output_format,
                                 main_surface);
#else
  return false;
#endif
}

bool ZeroFGIndependentPresenter::main_surface_handoff_pending() const {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  return impl_ && impl_->MainSurfaceHandoffPending();
#else
  return false;
#endif
}

void ZeroFGIndependentPresenter::MainSurfaceReleasedByA() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (impl_) {
    impl_->MainSurfaceReleasedByA();
  }
#endif
}

bool ZeroFGIndependentPresenter::MainSurfaceAllowsProducerA() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (!impl_ || impl_->main_surface_producer.state() !=
                    ZeroFGMainSurfaceProducer::State::kB) {
    return true;
  }
  impl_->main_surface_producer.CountViolation(
      "A_swapchain_while_B_produces");
  return false;
#else
  return true;
#endif
}

void ZeroFGIndependentPresenter::NoteMainSurfaceProducedByA() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (impl_) {
    impl_->NoteMainSurfaceProducedByA();
  }
#endif
}

bool ZeroFGIndependentPresenter::main_surface_reconnect_requested() const {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  return impl_ && impl_->main_surface_reconnect_requested.load(
                      std::memory_order_acquire);
#else
  return false;
#endif
}

void ZeroFGIndependentPresenter::SetMainSurfaceUIRequest(
    std::function<void()> request) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (impl_) {
    std::lock_guard<std::mutex> lock(impl_->main_surface_ui_request_mutex);
    impl_->main_surface_ui_request = std::move(request);
  }
#endif
}

void ZeroFGIndependentPresenter::BeginSurfaceDisconnect() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (impl_) {
    impl_->BeginSurfaceDisconnect();
  }
#endif
}

void ZeroFGIndependentPresenter::DestroySurfaceResourcesAfterSourceIdle() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (impl_) {
    impl_->DestroySurfaceResourcesAfterSourceIdle();
  }
#endif
}

bool ZeroFGIndependentPresenter::accepting() const {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  return impl_ && impl_->accepting.load(std::memory_order_acquire);
#else
  return false;
#endif
}

bool ZeroFGIndependentPresenter::TryActivateFromNativeSourceCadence() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  return impl_ && impl_->TryActivateFromNativeSourceCadence();
#else
  return false;
#endif
}

bool ZeroFGIndependentPresenter::owns_final_output() const {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  return impl_ &&
         impl_->final_output_authority.load(std::memory_order_acquire);
#else
  return false;
#endif
}

void ZeroFGIndependentPresenter::PublicationCommitted() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (impl_) {
    impl_->PublicationCommitted();
  }
#endif
}

void ZeroFGIndependentPresenter::ClaimPublicationAuthority() {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (impl_) {
    impl_->ClaimPublicationAuthority();
  }
#endif
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe
