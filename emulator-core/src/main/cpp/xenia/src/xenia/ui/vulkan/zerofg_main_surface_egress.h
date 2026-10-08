/**
 ******************************************************************************
 * XenDroid ZeroFG V2 Main Surface Authority egress                           *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_ZEROFG_MAIN_SURFACE_EGRESS_H_
#define XENIA_UI_VULKAN_ZEROFG_MAIN_SURFACE_EGRESS_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "xenia/ui/vulkan/vulkan_device.h"

namespace xe {
namespace ui {
namespace vulkan {

// Main Surface Authority (MSA): who produces into the SurfaceView's
// VkSurfaceKHR. Exactly one of A (the normal presenter) or B (ZeroFG egress)
// at any instant, with a gap state on each side of a transfer. A && B is a
// fatal experiment violation: a transition whose source state does not match
// is refused, logged loudly and counted, never performed.
class ZeroFGMainSurfaceProducer {
 public:
  enum class State : uint32_t { kA, kHandoffGap, kB, kHandbackGap };

  // Every transition publishes whether B produces (the host's vote signal);
  // a producer destroyed while B still produces withdraws it.
  ~ZeroFGMainSurfaceProducer();

  static const char* Name(State state);

  State state() const { return state_.load(std::memory_order_acquire); }
  bool Transition(State from, State to, const char* actor);
  // A falsifier observed outside a transition, such as A about to create its
  // swapchain while B still produces.
  void CountViolation(const char* what);

  uint64_t handoff_begin_total() const {
    return handoff_begin_total_.load(std::memory_order_relaxed);
  }
  uint64_t handoff_done_total() const {
    return handoff_done_total_.load(std::memory_order_relaxed);
  }
  uint64_t handback_begin_total() const {
    return handback_begin_total_.load(std::memory_order_relaxed);
  }
  uint64_t handback_done_total() const {
    return handback_done_total_.load(std::memory_order_relaxed);
  }
  uint64_t violation_total() const {
    return violation_total_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<State> state_{State::kA};
  std::atomic<uint64_t> handoff_begin_total_{0};
  std::atomic<uint64_t> handoff_done_total_{0};
  std::atomic<uint64_t> handback_begin_total_{0};
  std::atomic<uint64_t> handback_done_total_{0};
  std::atomic<uint64_t> violation_total_{0};
};

// Device B's FIFO egress onto the main Surface. One dedicated thread owns the
// swapchain: it creates it once A has retired its own, then copies each
// Presentation-approved FinalOutput into an acquired image and presents it.
// A HOLD never reaches this class, so a HOLD tick presents nothing and the
// FIFO keeps the last image on screen.
//
// The owner (the independent presenter thread) never blocks on it: requests
// go through a ring bounded by the FinalOutput pool, and a FinalOutput comes
// back as a completion once the copy that read it has finished.
class ZeroFGMainSurfaceEgress {
 public:
  static constexpr uint32_t kMaxSlots = 8;

  enum class Failure : uint32_t {
    kNone,
    kNoSurfaceSupport,
    kNoTransferDst,
    kNoFormat,
    kExtentMismatch,
    kSwapchainCreate,
    kSwapchainImages,
    kResourceCreate,
    kProducerOrder,
    kAcquire,
    kRecord,
    kSubmit,
    kCopyFenceExport,
    kPresent,
    kSlotReuse,
    kCompletionOverflow,
    kDeviceLost,
  };
  static const char* FailureName(Failure failure);

  struct Request {
    uint32_t slot = UINT32_MAX;
    uint64_t sequence_id = 0;
    bool synthetic = false;
    // A "not before" time, never a ratchet: FIFO shows at most one image per
    // vertical blank, and the actual time only ever feeds telemetry.
    uint64_t desired_present_ns = 0;
    // The pacing's output quantum for this output (0 before it is armed). The
    // output shaper engages only while it is at least one 60 Hz period.
    uint64_t output_quantum_ns = 0;
    uint64_t apply_time_ns = 0;
    VkImage image = VK_NULL_HANDLE;
    // The Post's completion timeline point: the copy waits on it on the GPU.
    VkSemaphore post_semaphore = VK_NULL_HANDLE;
    uint64_t post_value = 0;
  };

  struct Completion {
    uint32_t slot = UINT32_MAX;
    uint64_t sequence_id = 0;
    uint64_t copy_done_ns = 0;
    // False when the request was abandoned (stop, teardown, outdated surface)
    // and never reached the swapchain.
    bool presented = false;
  };

  struct Stats {
    bool swapchain_ready = false;
    bool timing_available = false;
    // GPU guard is on and display timing lets it reach the
    // compositor.
    bool apocalypse_guard = false;
    uint32_t queue_index = 0;
    uint32_t image_count = 0;
    uint64_t refresh_cycle_ns = 0;
    uint64_t presents = 0;
    uint64_t presents_synthetic = 0;
    // The guard's latch: whether it latched on, the 1 s arrival windows
    // evaluated, those showing the temporal pathology and those also confirmed
    // by GPU saturation, and the last GPU busy reading in permille (-1:
    // unavailable). Then its shaper: presents whose desired time it raised,
    // and by how much past max(original desired, present call).
    bool apocalypse_latched = false;
    uint64_t latch_windows_evaluated = 0;
    uint64_t latch_windows_pathology = 0;
    uint64_t latch_windows_confirmed = 0;
    int32_t gpu_busy_permille = -1;
    uint64_t shaper_raised = 0;
    uint64_t shaper_push_p50_ns = 0;
    uint64_t shaper_push_p90_ns = 0;
    uint64_t shaper_push_max_ns = 0;
    uint64_t acquire_timeouts = 0;
    uint64_t suboptimal = 0;
    uint64_t outdated = 0;
    uint64_t abandoned = 0;
    uint64_t copies_completed = 0;
    uint64_t copy_fence_errors = 0;
    uint64_t timings_returned = 0;
    uint64_t timings_missing = 0;
    uint64_t timings_unmatched = 0;
    uint64_t timing_query_errors = 0;
    uint64_t actual_early = 0;
    uint64_t actual_interval_min_ns = 0;
    uint64_t actual_interval_p10_ns = 0;
    uint64_t actual_interval_p50_ns = 0;
    uint64_t actual_interval_p90_ns = 0;
    uint64_t actual_interval_p99_ns = 0;
    uint64_t actual_interval_max_ns = 0;
    uint64_t desired_to_actual_p50_ns = 0;
    uint64_t desired_to_actual_p90_ns = 0;
    uint64_t desired_to_actual_p99_ns = 0;
    uint64_t desired_to_actual_max_ns = 0;
    uint64_t acquire_block_p50_ns = 0;
    uint64_t acquire_block_p90_ns = 0;
    uint64_t acquire_block_max_ns = 0;
    uint64_t copy_residence_p50_ns = 0;
    uint64_t copy_residence_p90_ns = 0;
    uint64_t copy_residence_max_ns = 0;
    // Latency decomposition, all against the desired (not-before) time:
    // desired_to_actual = ready_vs_desired + ready_to_actual. ready is the
    // copy's observed completion (1 ms poll; slower only while acquire
    // blocks). Signed values are negative when early.
    uint64_t refresh_cycle_now_ns = 0;
    // Display mode changes (refresh moved by more than 1/8), not the small
    // jitter of the reported duration.
    uint64_t refresh_mode_changes = 0;
    // Older presents still unresolved when a later one reported a timing:
    // Android never reports them afterwards. Most likely dropped by the
    // compositor, or aged out of Android's short timing history.
    uint64_t overtaken = 0;
    // Timings whose earliestPresentTime is at least half a refresh before
    // actualPresentTime.
    uint64_t missed_earliest = 0;
    // vkQueuePresentKHR: call begin against desired, and the call itself.
    int64_t present_call_vs_desired_p50_ns = 0;
    int64_t present_call_vs_desired_p90_ns = 0;
    int64_t present_call_vs_desired_max_ns = 0;
    uint64_t present_call_p50_ns = 0;
    uint64_t present_call_p90_ns = 0;
    uint64_t present_call_max_ns = 0;
    int64_t ready_vs_desired_p50_ns = 0;
    int64_t ready_vs_desired_p90_ns = 0;
    int64_t ready_vs_desired_max_ns = 0;
    uint64_t ready_to_actual_p50_ns = 0;
    uint64_t ready_to_actual_p90_ns = 0;
    uint64_t ready_to_actual_max_ns = 0;
    // Android computes it as latch minus render complete and hands it over
    // as uint64: negative (the compositor latched before our GPU work
    // finished) arrives wrapped, so it is read back as signed.
    int64_t present_margin_p50_ns = 0;
    int64_t present_margin_p90_ns = 0;
    uint64_t earliest_gap_p50_ns = 0;
    uint64_t earliest_gap_p90_ns = 0;
    uint64_t earliest_gap_max_ns = 0;
    // Free output: whether it is on; the outputs it did not present
    // because the compositor still held every image; the outputs superseded by
    // a newer one while the egress was blocked in a present (all, and the
    // Synthetic ones among them).
    bool free_output = false;
    uint64_t free_dropped = 0;
    uint64_t free_dropped_synthetic = 0;
    uint64_t free_superseded = 0;
    uint64_t free_superseded_synthetic = 0;
    // The present limiter's grid period it learned (0: none), and how often
    // it engaged and cleared.
    uint64_t present_limiter_ns = 0;
    uint64_t present_limiter_engaged = 0;
    uint64_t present_limiter_cleared = 0;
    // Streaks of held presents that returned at no grid (GPU back-pressure).
    uint64_t present_limiter_rejected = 0;
    // Vsync quantizer: whether it is on; presents it gave a vsync of
    // their own; future-only phase advances (a slot the copy could not reach);
    // slots refused because the previous present already had them; offsets
    // shed into a free earlier vsync; and how long before its assigned vsync
    // each copy was observed ready (signed, p10/p50/p90).
    bool vsync_quantizer = false;
    uint64_t vsync_quantized = 0;
    uint64_t vsync_phase_reanchor = 0;
    uint64_t vsync_duplicate_refused = 0;
    uint64_t vsync_gap_absorbed = 0;
    int64_t ready_before_vsync_p10_ns = 0;
    int64_t ready_before_vsync_p50_ns = 0;
    int64_t ready_before_vsync_p90_ns = 0;
    // Timings that came back at their assigned vsync and later ones; the
    // latest a hit was queued/ready before its vsync, and the earliest a late
    // one was (signed): together they bracket the compositor's deadline.
    uint64_t vsync_hit = 0;
    uint64_t vsync_late = 0;
    int64_t hit_queue_before_vsync_min_ns = 0;
    int64_t hit_ready_before_vsync_min_ns = 0;
    int64_t late_queue_before_vsync_max_ns = 0;
    int64_t late_ready_before_vsync_max_ns = 0;
  };

  ZeroFGMainSurfaceEgress(VulkanDevice* device,
                          ZeroFGMainSurfaceProducer* producer,
                          VkSurfaceKHR surface, VkExtent2D extent,
                          VkFormat format, uint32_t slot_count,
                          bool elevated_priority, bool apocalypse_guard,
                          bool free_output, bool vsync_quantizer,
                          std::function<void()> wake_owner);
  ZeroFGMainSurfaceEgress(const ZeroFGMainSurfaceEgress&) = delete;
  ZeroFGMainSurfaceEgress& operator=(const ZeroFGMainSurfaceEgress&) = delete;
  ~ZeroFGMainSurfaceEgress();

  // Owner thread. Creates the per-slot copy contexts and starts the egress
  // thread, which stays idle until the producer state reaches the handoff gap.
  bool Start();
  // Owner thread; idempotent. Joins the egress thread (acquire is bounded),
  // idles the egress queue (vkQueueWaitIdle: not formally bounded), destroys
  // the swapchain, and only then moves the producer state off B.
  void Stop();
  // A has retired its swapchain: wake the egress thread.
  void NotifyProducerChanged();

  enum class EnqueueResult { kQueued, kNotProducing, kFull };
  bool HasCapacity() const;
  // Refuses while B does not produce (no swapchain yet, or handing back):
  // presentation is closed then, which is not ring pressure.
  EnqueueResult TryEnqueue(const Request& request);
  bool PopCompletion(Completion& completion_out);

  Failure failure() const { return failure_.load(std::memory_order_acquire); }
  bool surface_outdated() const {
    return surface_outdated_.load(std::memory_order_acquire);
  }
  bool timing_available() const { return timing_available_; }
  uint32_t queue_index() const { return queue_index_; }
  Stats SnapshotStats() const;

 private:
  template <size_t Capacity>
  class SampleWindow {
   public:
    void Add(uint64_t value);
    uint64_t Quantile(uint32_t numerator, uint32_t denominator) const;
    size_t count() const { return count_; }

   private:
    std::array<uint64_t, Capacity> values_ = {};
    size_t count_ = 0;
    size_t next_ = 0;
  };

  struct CopyContext {
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkSemaphore acquire_semaphore = VK_NULL_HANDLE;
    // Binary, exportable as a sync_file: the copy's completion is observed
    // with a non-waiting poll, never a Vulkan fence or counter query.
    VkSemaphore copy_done_semaphore = VK_NULL_HANDLE;
    // -1: already signaled (or none); -2: the copy's end is unobservable and
    // only a teardown queue idle retires it.
    int copy_done_fd = -1;
    bool in_flight = false;
    bool presented = false;
    bool acquire_dirty = false;
    uint64_t sequence_id = 0;
    uint64_t submit_ns = 0;
    // Vsync quantizer: the vsync this copy's present was given (0: none).
    uint64_t assigned_vsync_ns = 0;
  };

  struct TimingRecord {
    bool used = false;
    bool resolved = false;
    uint32_t present_id = 0;
    uint64_t sequence_id = 0;
    bool synthetic = false;
    uint64_t desired_ns = 0;
    uint64_t present_ns = 0;
    uint64_t ready_ns = 0;
    // Vsync quantizer: the vsync its present was given (0: none).
    uint64_t assigned_vsync_ns = 0;
  };

  static constexpr size_t kTimingRecordCount = 64;
  // Signed samples live in the unsigned windows offset by one second.
  static constexpr int64_t kSignedSampleBiasNs = 1000000000ll;
  static uint64_t BiasSigned(int64_t value_ns);
  static int64_t UnbiasSigned(uint64_t value_ns);
  static constexpr uint64_t kAcquireTimeoutNs = 20000000ull;
  // GPU guard: one 60 Hz period, the minimum spacing of two
  // outputs on the panel once its shaper acts.
  static constexpr uint64_t kShaperSpacingNs = 16666667ull;
  // Its degradation latch: consecutive 1 s windows of request arrivals that
  // must all show the pathology, the timings a window needs to be judged, the
  // arrival floor in milli-fps, the GPU busy share that confirms saturation,
  // and the intervals one window can hold.
  static constexpr uint64_t kLatchWindowNs = 1000000000ull;
  static constexpr uint32_t kLatchWindows = 3;
  static constexpr uint32_t kLatchMinIntervals = 20;
  static constexpr uint64_t kLatchRateFloorMfps = 30000;
  static constexpr uint32_t kLatchGpuBusyPermille = 970;
  static constexpr size_t kLatchIntervalCapacity = 256;

  void ThreadMain();
  bool CreateSwapchain();
  void DestroySwapchainAfterIdle();
  void Serve();
  bool ServeRequest(const Request& request, bool probe);
  uint64_t ShapeDesiredPresent(const Request& request, uint64_t now_ns);
  void ResetLatchWindow();
  void ObserveLatchArrival(uint64_t arrival_ns, uint64_t refresh_ns);
  void Abandon(const Request& request);
  // Free output: an image the compositor has already let go, or none.
  // At most one acquired image waits for its release fence between outputs.
  enum class FreeImage { kFree, kBusy, kStop };
  FreeImage TryTakeFreeImage(uint32_t& image_index);
  // Free output: an output that is not presented completes at once and
  // its FinalOutput returns (superseded: a newer output was waiting).
  void DropUnpresented(const Request& request, bool superseded);
  // Vsync quantizer: the physical vsync for an output whose logical
  // target is target_ns, on the display's observed vsync lattice; returns the
  // desired present time to hand to the compositor and the slot in
  // assigned_ns (0 and the logical target when the lattice is not known yet).
  uint64_t QuantizeToVsync(uint64_t target_ns, uint64_t now_ns,
                           uint64_t& assigned_ns);
  // The compositor latches a buffer for vsync V about one refresh before V
  // (shown frames were displayed 8.5-14 ms after ready, 2026-10-07): a slot is
  // reachable when the copy is predicted ready one refresh plus this margin
  // before it. Experimental, calibrated by ready_before_vsync and the vsync
  // hit/late bracket. Device facts (Adreno 840 phone, 2026-10-07): at 120 Hz
  // SurfaceFlinger starts composing 10.33 ms before V and auto-latches an
  // unsignaled buffer while a single layer updates.
  static constexpr uint64_t kVsyncLatchMarginNs = 1500000ull;
  // Free output: waits, outside every lock, until the next slot of the
  // learned limiter grid, serving only the newest output that arrives
  // meanwhile; sets probe when this present skips the grid to look for the
  // limiter's end. False when the egress is stopping.
  bool PaceFreePresent(Request& request, bool& probe);
  // Free output: learns the limiter grid from how long presents slept.
  void ObservePresentCall(uint64_t call_ns, uint64_t return_ns, bool probe);
  // The HyperOS limiter releases a present on a fixed grid (one slot every
  // 1/90 or 1/60 s), so a present that sleeps returns on a grid point.
  // A present that sleeps at least kLimiterBlockedNs was held; four in a row
  // engage the pacing with the shortest interval between their returns as the
  // period and the last return as the anchor, but only when every interval is
  // a whole number of periods within kLimiterGridTolerance: a present that
  // blocks in the driver while a saturated GPU drains its queue returns at no
  // grid (Arkham, no DynamicFPS, 2026-10-07: false 10.5 and 24.3 ms limiters,
  // ~41 of 60 outputs shown). Paced, a present is called kLimiterMarginNs after the next grid
  // point not yet used; a present that still sleeps kLimiterEarlyNs or more
  // re-anchors the grid on its return. The period is never refined from paced
  // presents (an off-grid present sleeps to the next point, which lengthens
  // the apparent interval: the 12:15 run let that feedback stretch 11.1 ms to
  // 29 ms); kLimiterResleepStreak sleeps in a row mean the grid itself changed
  // (90 -> 60 fps) and the pacing is learned again. After
  // kLimiterProbeNs without a sleep one present skips the grid: if it does not
  // sleep the limiter is gone and the pacing clears (a period under 1.1
  // refresh clears it too).
  static constexpr uint64_t kLimiterBlockedNs = 2000000ull;
  static constexpr uint32_t kLimiterEngageStreak = 4;
  // An interval may miss a whole multiple of the period by period / this.
  static constexpr uint64_t kLimiterGridTolerance = 8;
  static constexpr uint64_t kLimiterEarlyNs = 500000ull;
  static constexpr uint64_t kLimiterMarginNs = 300000ull;
  static constexpr uint64_t kLimiterProbeNs = 2000000000ull;
  static constexpr uint32_t kLimiterResleepStreak = 4;
  void PollCopyCompletions();
  void MaybePollTimings(bool force);
  void PushCompletion(const Completion& completion);
  void Fail(Failure failure, int32_t vk_result);
  void MarkSurfaceOutdated(int32_t vk_result);
  bool StopRequested();
  bool AnyCopyInFlight() const;
  void ApplyThreadPriority();

  VulkanDevice* const device_;
  ZeroFGMainSurfaceProducer* const producer_;
  const VkSurfaceKHR surface_;
  const VkExtent2D extent_;
  const VkFormat format_;
  const uint32_t slot_count_;
  const bool elevated_priority_;
  const std::function<void()> wake_owner_;
  const uint32_t queue_family_;
  const uint32_t queue_index_;
  const bool timing_available_;
  // GPU guard, which needs display timing to reach the
  // compositor.
  const bool apocalypse_guard_;
  // Free output: the pipeline never waits for the panel. The HyperOS
  // DynamicFPS cap (a hot phone held at 90, then 60 fps) blocks our own
  // vkQueuePresentKHR for one capped frame (present call p50 10.9 / 16.4 ms
  // against 0.2 ms); with one present at a time the requests
  // queued behind it held their FinalOutputs and the hold reached the Source
  // (Rayman 60 -> 44 -> 30). Worse, the present runs under the lock of queue
  // B:0, which Generation and the Post share on a one-queue device: a
  // blocked present blocks ZeroFG's GPU work. Free: the egress learns the
  // limiter from the blocked presents and calls present only once it lets
  // the next one through (no sleep under the queue lock), serving only the
  // newest output that waited (the older ones complete unpresented at once);
  // and an output finds an image the compositor has already released, or it
  // is not presented either. The panel shows the newest frames at whatever
  // rate Android allows, and the Source keeps its rate. Owner decision
  // 2026-10-07: accepted-Real sovereignty ends when ZeroFG emits the frame.
  const bool free_output_;
  // Vsync quantizer: the logical metronome (one output every O, from
  // the Source) and the panel's vsync are two nearly equal clocks (8.335 vs
  // 8.333 ms at 60 -> 120); their phase drifts slowly through the
  // compositor's latch deadline, and for minutes at a time each output missed
  // its vsync and met the next one in the same latch, which shows only the
  // newest: ~62 of 120 presents a second shown with the panel at 120 and no
  // cap (2026-10-07, also in the logs before any egress change). Presentation
  // quantizes each output to the observed vsync lattice: one vsync per
  // output, in order, the first one its copy can reach; a phase that can no
  // longer be reached advances future-only, and a free earlier vsync sheds it
  // again. Production, ordering and the Source are untouched.
  const bool vsync_quantizer_;

  // Egress-thread state, touched by the owner only after the join.
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  // Free output: the release fence of the held image, and that image
  // (UINT32_MAX: none held).
  VkFence image_fence_ = VK_NULL_HANDLE;
  uint32_t held_image_ = UINT32_MAX;
  // Free output's present limiter: when the last present returned,
  // the grid period (0: none) and anchor, the last grid point used, when a
  // present last slept and when the last probe ran, and the blocked presents
  // toward engaging.
  uint64_t last_present_return_ns_ = 0;
  // The vsync quantizer's lattice: the last observed actual present (a
  // vsync instant), the last vsync given to a present, and the phase offset in
  // whole refreshes that only moves forward until a free vsync sheds it.
  uint64_t vsync_anchor_ns_ = 0;
  uint64_t vsync_last_slot_ns_ = 0;
  uint64_t vsync_offset_periods_ = 0;
  uint64_t limiter_period_ns_ = 0;
  uint64_t limiter_anchor_ns_ = 0;
  uint64_t limiter_last_slot_ns_ = 0;
  uint64_t limiter_last_sleep_ns_ = 0;
  uint64_t limiter_last_probe_ns_ = 0;
  uint32_t limiter_block_streak_ = 0;
  uint32_t limiter_sleep_streak_ = 0;
  std::array<uint64_t, kLimiterEngageStreak> limiter_samples_ns_ = {};
  std::vector<VkImage> images_;
  std::vector<VkSemaphore> render_done_;
  std::array<CopyContext, kMaxSlots> contexts_ = {};
  std::array<TimingRecord, kTimingRecordCount> timing_records_ = {};
  size_t timing_record_next_ = 0;
  uint64_t last_actual_ns_ = 0;
  // Upper bound of the last present's display time, for the shaper (0: none).
  uint64_t shaper_latest_display_ns_ = 0;
  // The shaper's degradation latch, under stats_mutex_: the open 1 s window of
  // request arrivals (0: none), the actual intervals seen during it, and the
  // consecutive degraded windows so far.
  uint64_t latch_window_begin_ns_ = 0;
  uint32_t latch_window_arrivals_ = 0;
  uint32_t latch_degraded_streak_ = 0;
  uint32_t latch_interval_count_ = 0;
  std::array<uint64_t, kLatchIntervalCapacity> latch_intervals_ns_ = {};
  bool gpu_busy_unavailable_logged_ = false;
  uint64_t next_timing_poll_ns_ = 0;
  uint64_t next_refresh_poll_ns_ = 0;
  bool started_ = false;
  bool stopped_ = false;

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_requested_ = false;
  std::array<Request, kMaxSlots> ring_ = {};
  uint32_t ring_head_ = 0;
  uint32_t ring_count_ = 0;
  std::array<Completion, kMaxSlots * 2> completions_ = {};
  uint32_t completion_head_ = 0;
  uint32_t completion_count_ = 0;

  std::atomic<Failure> failure_{Failure::kNone};
  std::atomic<bool> surface_outdated_{false};

  mutable std::mutex stats_mutex_;
  Stats counters_;
  SampleWindow<128> actual_interval_ns_;
  SampleWindow<128> desired_to_actual_ns_;
  SampleWindow<128> acquire_block_ns_;
  SampleWindow<128> copy_residence_ns_;
  SampleWindow<128> present_call_vs_desired_ns_;
  SampleWindow<128> present_call_ns_;
  SampleWindow<128> ready_vs_desired_ns_;
  SampleWindow<128> ready_to_actual_ns_;
  SampleWindow<128> present_margin_ns_;
  SampleWindow<128> earliest_gap_ns_;
  SampleWindow<128> ready_before_vsync_ns_;
  SampleWindow<128> hit_queue_before_vsync_ns_;
  SampleWindow<128> hit_ready_before_vsync_ns_;
  SampleWindow<128> late_queue_before_vsync_ns_;
  SampleWindow<128> late_ready_before_vsync_ns_;
  SampleWindow<128> shaper_push_ns_;

  std::thread thread_;
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_ZEROFG_MAIN_SURFACE_EGRESS_H_
