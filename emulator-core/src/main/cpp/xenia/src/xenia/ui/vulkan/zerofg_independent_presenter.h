/**
 ******************************************************************************
 * XenDroid ZeroFG V2 independent Android presenter                           *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_ZEROFG_INDEPENDENT_PRESENTER_H_
#define XENIA_UI_VULKAN_ZEROFG_INDEPENDENT_PRESENTER_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>

#include "xenia/ui/presenter.h"
#include "xenia/ui/vulkan/vulkan_device.h"
#include "xenia/ui/vulkan/zerofg_completion_owner.h"

namespace xe {
namespace ui {
namespace vulkan {

// The independent presenter owns immutable Real candidates, optional Synthetic
// candidates and a bounded final-output pool. Real and Synthetic converge
// before XenDroid's normal spatial presentation pipeline. This presenter never
// owns or calls the normal Vulkan swapchain / WSI path.
class ZeroFGIndependentPresenter final {
 public:
  // Source ingress and temporal Real residency are logically separate pools of
  // identical device-local images. Source writes only to the small Capture
  // pool. At the committed handoff, the presenter exchanges that image with a
  // physically-free Residency image, so Capture recycles without a second
  // full-frame GPU copy. The Source timeline still gates Residency readiness.
  // D/history/A-B ownership therefore cannot consume the storage required by
  // the next Source publication.
  static constexpr uint32_t kRealResidencyCapacity = 6;
  static constexpr uint32_t kCapturePoolSize = 4;
  static constexpr uint32_t kPoolSize = kRealResidencyCapacity;
  static constexpr uint32_t kSyntheticPoolSize = 3;
  static constexpr uint32_t kFinalOutputPoolSize = 5;
  static constexpr uint32_t kMailboxCount = 3;

  using PaintConfigProvider =
      std::function<Presenter::GuestOutputPaintConfig()>;

  struct GenerationRequest {
    VkImage previous_image = VK_NULL_HANDLE;
    VkImageView previous_view = VK_NULL_HANDLE;
    VkImageLayout previous_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImage current_image = VK_NULL_HANDLE;
    VkImageView current_view = VK_NULL_HANDLE;
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Physical allocation extent shared by the A/B Residency images. This is
    // intentionally distinct from extent, which is the active content and
    // Synthetic working extent.
    VkExtent2D source_physical_extent = {};
    VkExtent2D extent = {};
    uint64_t previous_source_id = 0;
    uint64_t current_source_id = 0;
    // GPU->GPU readiness of the A/B Residency images: one wait when both live
    // on the same timeline (the shared Capture-to-Residency timeline), two
    // when each Residency slot owns its readiness timeline (V2-2). Each
    // semaphore appears once.
    std::array<VkSemaphore, 2> input_wait_semaphores = {};
    std::array<uint64_t, 2> input_wait_values = {};
    uint32_t input_wait_count = 0;
    // An admitted S owns its q0 submit opportunity until acquired or until
    // Source/lifecycle state revokes that authority.
    std::function<bool()> queue_commit_allowed;
  };

  // Opaque integration view of one coherently acquired GuestOutput mailbox
  // publication. mailbox_acquisition keeps consumer ownership only through a
  // future ingress submit attempt; image_lifetime may outlive it until GPU
// retirement without exposing VulkanPresenter::GuestOutputImage to the core.
  struct HandoffReadyObservation {
    HandoffReadyObservation() = default;
    HandoffReadyObservation(const HandoffReadyObservation&) = delete;
    HandoffReadyObservation& operator=(const HandoffReadyObservation&) = delete;
    int fd = -1;
    uint64_t source_id = 0;
    uint64_t publish_ns = 0;
    uint64_t first_seen_ready_ns = 0;
    bool error = false;
    ~HandoffReadyObservation();
    uint64_t Observe();  // B-thread-only sync_fd poll, never a Vulkan query.
  };
  struct IngressSourcePublication {
    uint32_t mailbox_index = UINT32_MAX;
    uint32_t content_width = 0;
    uint32_t content_height = 0;
    uint32_t display_aspect_ratio_x = 0;
    uint32_t display_aspect_ratio_y = 0;
    bool is_8bpc = false;
    xe::SourcePublicationTelemetry publication;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent2D extent = {};
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::shared_ptr<void> mailbox_acquisition;
    std::shared_ptr<void> image_lifetime;
    // Host-only cross-device input. Both semaphores belong to this presenter's
    // device. The lease cancels an unsubmitted acquisition, never an accepted
    // GPU read. Completion exports the release after relinquishing the queue.
    VkSemaphore external_wait = VK_NULL_HANDLE;
    VkSemaphore external_release = VK_NULL_HANDLE;
    std::function<bool()> external_submitted;
    bool external_failure = false;
    // Set with a false return while the newest publication's Source copy is
    // still in flight on the Source device; the presenter then retries at the
    // handoff poll interval instead of the ownership one.
    bool external_pending = false;
    std::shared_ptr<HandoffReadyObservation> handoff_ready;
    // Split handoff: Source wait ZeroFG backpressure imposed on this
    // publication (slot + depth-1 Publish), and when it became Published.
    uint64_t source_bp_wait_ns = 0;
    uint64_t handoff_published_ns = 0;

    explicit operator bool() const {
      return mailbox_index != UINT32_MAX && image != VK_NULL_HANDLE &&
             content_width && content_height && mailbox_acquisition &&
             image_lifetime;
    }
    void ReleaseMailboxAcquisition() { mailbox_acquisition.reset(); }
  };

  using IngressSourceAcquireCallback =
      std::function<bool(IngressSourcePublication&)>;

  enum class GenerationFailureStage : uint8_t {
    kNone,
    kRequestValidation,
    kAdapter,
    kResize,
    kSyntheticPool,
    kCommandReset,
    kCommandBegin,
    kRecord,
    kCommandEnd,
    kPreparationYield,
    kQueueBusy,
    kQueueSubmit,
    kTimelineQuery,
  };

  // Profiling for a generation is drained when its frame-context slot is
  // reused.  Keeping this separate from GenerationResult prevents a delayed
  // sample from being mistaken for the generation currently being submitted.
  struct GenerationProfilingSample {
    bool timestamp_valid = false;
    uint64_t service_gpu_ns = 0;
    uint64_t submit_call_ns = 0;
  };

  struct GenerationResult {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent2D extent = {};
    uint32_t synthetic_index = UINT32_MAX;
    uint64_t queue_wait_ns = 0;
    uint64_t submit_host_ns = 0;
    uint64_t gpu_completion_ns = 0;
    uint64_t setup_ns = 0;
    uint64_t total_ns = 0;
    uint64_t signal_value = 0;
    VkSemaphore completion_semaphore = VK_NULL_HANDLE;
    uint64_t submit_time_ns = 0;
    uint64_t service_gpu_ns = 0;
    uint64_t submit_to_ready_ns = 0;
    uint32_t pool_occupancy = 0;
    uint32_t pool_high_water = 0;
    // Telemetry only, D3: the sync_fd
    // export after the Generation submit, and whether a completion poll
    // observed the Generation through that fd (non-waiting) instead of the
    // completion timeline.
    bool completion_fd_export_attempted = false;
    bool completion_fd_exported = false;
    bool completion_fd_fallback = false;
    bool completion_fd_polled = false;
    uint64_t completion_fd_export_host_ns = 0;
    uint64_t completion_fd_poll_host_ns = 0;
    // Whether completion_semaphore is this Generation context's own timeline
    // (V2-1); the shared one only if the context's own failed to create.
    bool completion_per_context = false;
    // Completion/ownership falsifiers of the Generation path, cumulative
    // since the Generation context was created.
    ZeroFGCompletionOwnerStats completion_owner_stats = {};
    GenerationFailureStage failure_stage = GenerationFailureStage::kNone;
    uint32_t zerofg_status = 0;
    VkResult vk_result = VK_SUCCESS;
    bool vk_result_valid = false;
    bool submission_accepted = false;
    bool timestamp_valid = false;
    bool drained_profile_sample_valid = false;
    GenerationProfilingSample drained_profile_sample = {};
    uint32_t profile_readback_attempts = 0;
    uint32_t profile_readback_successes = 0;
    uint32_t profile_readback_not_ready = 0;
    uint32_t profile_readback_errors = 0;
    VkResult profile_readback_last_error = VK_SUCCESS;
  };

  using GenerationCallback =
      std::function<bool(const GenerationRequest&, GenerationResult&)>;
  using GenerationPollCallback =
      std::function<bool(uint32_t, GenerationResult&, bool&)>;
  using SyntheticReleaseCallback = std::function<void(uint32_t)>;
  using GenerationShutdownCallback = std::function<void()>;

// Invoked only by the dedicated ZeroFG output thread. The callback records
  // and submits XenDroid's normal spatial presentation pipeline, then returns
  // immediately after queue acceptance. Completion is observed separately;
// the immutable candidate remains owned by the presenter throughout the GPU read.
  struct PostProcessRequest {
    bool candidate_is_synthetic = false;
    uint64_t source_id = 0;
    uint64_t pair_a_source_id = 0;
    uint64_t logical_sequence = 0;
    VkImage candidate_image = VK_NULL_HANDLE;
    VkImageLayout candidate_layout = VK_IMAGE_LAYOUT_GENERAL;
    VkExtent2D candidate_storage_extent = {};
    uint32_t frontbuffer_width = 0;
    uint32_t frontbuffer_height = 0;
    uint32_t display_aspect_ratio_x = 0;
    uint32_t display_aspect_ratio_y = 0;
    bool is_8bpc = false;
    Presenter::GuestOutputPaintConfig config = {};
    VkImage final_output_image = VK_NULL_HANDLE;
    VkImageView final_output_view = VK_NULL_HANDLE;
    VkExtent2D final_output_extent = {};
    VkFormat final_output_format = VK_FORMAT_UNDEFINED;
    uint32_t final_output_index = UINT32_MAX;
    // Optional candidate dependency. Synthetic waits on Generation; Real
    // waits on Capture-to-Residency readiness in the GPU submission graph.
    VkSemaphore candidate_wait_semaphore = VK_NULL_HANDLE;
    uint64_t candidate_wait_value = 0;
  };

  enum class PostFailureStage : uint8_t {
    kNone,
    kRequestValidation,
    kContextCommandPool,
    kContextCommandBuffer,
    kContextTimeline,
    kDescriptorPool,
    kDescriptorSets,
    kFinalRenderPass,
    kFinalFormat,
    kLogicalInput,
    kPaintFlow,
    kIntermediateResource,
    kIntermediateFramebuffer,
    kIntermediatePipeline,
    kFinalPipeline,
    kFinalFramebuffer,
    kCommandReset,
    kCommandBegin,
    kCommandEnd,
    kQueueSubmit,
    kAcquireFenceExport,
  };

  struct PostProcessResult {
    uint32_t effect_count = 0;
    uint64_t queue_wait_ns = 0;
    uint64_t submit_host_ns = 0;
    uint64_t gpu_completion_ns = 0;
    uint64_t signal_value = 0;
    // Retained only as a teardown fallback if sync_file export fails after
    // Vulkan has accepted the Post submission. The shared Post timeline, or
    // this FinalOutput slot's own timeline when completion_per_slot is set.
    VkSemaphore completion_semaphore = VK_NULL_HANDLE;
    uint64_t submit_time_ns = 0;
    uint64_t service_gpu_ns = 0;
    uint64_t submit_to_ready_ns = 0;
    // Telemetry only: host time of the sync_file export after queue
    // acceptance.
    uint64_t export_host_ns = 0;
    // Telemetry only: monotonic time just before the Post requested q0.
    uint64_t submit_request_ns = 0;
    // Android sync_file exported from the Post completion signal. Ownership
// transfers to the presenter on success. -1 is a valid already-signaled fence.
    int acquire_fence_fd = -1;
    PostFailureStage failure_stage = PostFailureStage::kNone;
    VkResult vk_result = VK_SUCCESS;
    bool vk_result_valid = false;
    bool submission_accepted = false;
    bool timestamp_valid = false;
    bool acquire_fence_exported = false;
    // Telemetry only: whether the candidate dependency had already completed
    // (timeline counter at or past the wait value) just before the Post
    // entered q0. Unknown for a Post that waits on an input: the counter
    // query that could tell waits on pending GPU work on Turnip/KGSL.
    bool input_readiness_known = false;
    bool input_complete_at_submit = false;
    // Telemetry only: whether the Post carried a candidate wait semaphore.
    bool input_has_wait = false;
    // Whether completion_semaphore is this FinalOutput slot's own timeline
    // (every Post job creates one).
    bool completion_per_slot = false;
    // Completion/ownership falsifiers of the Post path, cumulative since the
    // Post context was created.
    ZeroFGCompletionOwnerStats completion_owner_stats = {};
  };

  using PostProcessCallback =
      std::function<bool(const PostProcessRequest&, PostProcessResult&)>;
  using PostProcessReleaseCallback = std::function<void(uint32_t)>;
  using PostProcessShutdownCallback = std::function<void()>;
  // Teardown-only hook owned by the host presenter. It drains device-B before
  // the Generation/Post shutdown callbacks destroy device-B objects.
  using PresenterDeviceDrainCallback = std::function<bool()>;

  static std::unique_ptr<ZeroFGIndependentPresenter> Create(
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
      PresenterDeviceDrainCallback presenter_device_drain_callback);

  ZeroFGIndependentPresenter(const ZeroFGIndependentPresenter&) = delete;
  ZeroFGIndependentPresenter& operator=(
      const ZeroFGIndependentPresenter&) = delete;
  ~ZeroFGIndependentPresenter();

  // Surface lifecycle methods are UI-thread-only. Teardown closes ingress and
  // proves completion of presenter-owned GPU work before destroying storage.
  // main_surface is the normal presenter's VkSurfaceKHR, borrowed (never
  // destroyed here) by Main Surface Authority; ignored otherwise.
  bool Connect(void* native_window, VkExtent2D extent,
               VkFormat final_output_format,
               VkSurfaceKHR main_surface = VK_NULL_HANDLE);
  void BeginSurfaceDisconnect();
  void DestroySurfaceResourcesAfterSourceIdle();

  // Main Surface Authority (MSA), ZeroFG's only output. The normal presenter
  // (A) and ZeroFG's device-B egress (B) never produce into the Surface at the
  // same time.
  // ZeroFG owns final output while A still holds its swapchain: A must retire
  // it, UI thread, under painting ownership.
  bool main_surface_handoff_pending() const;
  // A has destroyed its swapchain and kept the VkSurfaceKHR.
  void MainSurfaceReleasedByA();
  // A && B falsifier, asked before A creates a swapchain: false (counted as a
  // violation) while B still produces into the Surface.
  bool MainSurfaceAllowsProducerA();
  // A owns the Surface lifecycle again (its swapchain was recreated, or the
  // Surface was disconnected).
  void NoteMainSurfaceProducedByA();
  // The egress saw the Surface go out of date: the UI thread must reconnect.
  bool main_surface_reconnect_requested() const;
  // Asks the UI thread to paint; set once by the normal presenter.
  void SetMainSurfaceUIRequest(std::function<void()> request);

  bool accepting() const;
// Called from the Source publication boundary while ZeroFG is still
  // dormant. It observes only the lock-free global IssueSwap window. On a
// stable native cadence it arms ZeroFG for the following Source frame;
  // the frame performing qualification remains on the normal presenter.
  bool TryActivateFromNativeSourceCadence();
  // True while the independent child layer, including an accepted-Real drain,
  // is the final visible output authority. The normal Vulkan presenter uses
  // this only to avoid double-counting output during fail-open transition.
  bool owns_final_output() const;
  void ClaimPublicationAuthority();
  // Source publication notification. The Source thread only wakes the owner
  // thread; mailbox acquisition and acceptance happen asynchronously there.
  void PublicationCommitted();

 private:
  struct Impl;

  explicit ZeroFGIndependentPresenter(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_ZEROFG_INDEPENDENT_PRESENTER_H_
