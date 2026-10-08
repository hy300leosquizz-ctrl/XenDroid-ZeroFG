/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_VULKAN_PRESENTER_H_
#define XENIA_UI_VULKAN_VULKAN_PRESENTER_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "xenia/base/assert.h"
#include "xenia/ui/presenter.h"
#include "xenia/ui/surface.h"
#include "xenia/ui/vulkan/ui_samplers.h"
#include "xenia/ui/vulkan/vulkan_device.h"
#include "xenia/ui/vulkan/vulkan_gpu_completion_timeline.h"
#include "xenia/ui/vulkan/vulkan_instance.h"
#include "xenia/ui/vulkan/zerofg_independent_presenter.h"

namespace xe {
namespace ui {
namespace vulkan {

class ZeroFGDeviceHandoff;

class VulkanUIDrawContext final : public UIDrawContext {
 public:
  VulkanUIDrawContext(Presenter& presenter, uint32_t render_target_width,
                      uint32_t render_target_height,
                      VkCommandBuffer draw_command_buffer,
                      uint64_t submission_index_current,
                      uint64_t submission_index_completed,
                      VkRenderPass render_pass, VkFormat render_pass_format)
      : UIDrawContext(presenter, render_target_width, render_target_height),
        draw_command_buffer_(draw_command_buffer),
        submission_index_current_(submission_index_current),
        submission_index_completed_(submission_index_completed),
        render_pass_(render_pass),
        render_pass_format_(render_pass_format) {}

  VkCommandBuffer draw_command_buffer() const { return draw_command_buffer_; }
  uint64_t submission_index_current() const {
    return submission_index_current_;
  }
  uint64_t submission_index_completed() const {
    return submission_index_completed_;
  }
  VkRenderPass render_pass() const { return render_pass_; }
  VkFormat render_pass_format() const { return render_pass_format_; }

 private:
  VkCommandBuffer draw_command_buffer_;
  uint64_t submission_index_current_;
  uint64_t submission_index_completed_;
  VkRenderPass render_pass_;
  VkFormat render_pass_format_;
};

class VulkanPresenter final : public Presenter {
 public:
  static constexpr size_t kMaxActiveGuestOutputImageVersions =
      kGuestOutputMailboxSize;

  static constexpr VkFormat kGuestOutputFormat =
      VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  static constexpr VkPipelineStageFlagBits kGuestOutputInternalStageMask =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  static constexpr VkAccessFlags kGuestOutputInternalAccessMask =
      VK_ACCESS_SHADER_READ_BIT;
  static constexpr VkImageLayout kGuestOutputInternalLayout =
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  class VulkanGuestOutputRefreshContext final
      : public GuestOutputRefreshContext {
   public:
    VulkanGuestOutputRefreshContext(bool& is_8bpc_out_ref, VkImage image,
                                    VkImageView image_view,
                                    uint64_t image_version,
                                    bool image_ever_written_previously)
        : GuestOutputRefreshContext(is_8bpc_out_ref),
          image_(image),
          image_view_(image_view),
          image_version_(image_version),
          image_ever_written_previously_(image_ever_written_previously) {}

    VkImage image() const { return image_; }
    VkImageView image_view() const { return image_view_; }
    uint64_t image_version() const { return image_version_; }
    bool image_ever_written_previously() const {
      return image_ever_written_previously_;
    }
   private:
    VkImage image_;
    VkImageView image_view_;
    uint64_t image_version_;
    bool image_ever_written_previously_;
  };

  static std::unique_ptr<VulkanPresenter> Create(
      HostGpuLossCallback host_gpu_loss_callback, VulkanDevice* vulkan_device,
      const UISamplers* ui_samplers) {
    auto presenter = std::unique_ptr<VulkanPresenter>(new VulkanPresenter(
        host_gpu_loss_callback, vulkan_device, ui_samplers));
    if (!presenter->InitializeSurfaceIndependent()) {
      return nullptr;
    }
    return presenter;
  }

  ~VulkanPresenter();

  VulkanDevice* vulkan_device() const { return vulkan_device_; }

  static Surface::TypeFlags GetSurfaceTypesSupportedByInstance(
      const VulkanInstance::Extensions& instance_extensions);
  Surface::TypeFlags GetSupportedSurfaceTypes() const override;

  bool CaptureGuestOutput(RawImage& image_out) override;

  void AwaitUISubmissionCompletionFromUIThread(uint64_t submission_index) {
    ui_completion_timeline_.AwaitSubmissionAndUpdateCompleted(submission_index);
  }
  VkCommandBuffer AcquireUISetupCommandBufferFromUIThread();

 protected:
  SurfacePaintConnectResult ConnectOrReconnectPaintingToSurfaceFromUIThread(
      Surface& new_surface, uint32_t new_surface_width,
      uint32_t new_surface_height, bool was_paintable,
      bool& is_vsync_implicit_out) override;
  void DisconnectPaintingFromSurfaceFromUIThreadImpl() override;

  bool RefreshGuestOutputImpl(
      uint32_t mailbox_index, uint32_t frontbuffer_width,
      uint32_t frontbuffer_height,
      std::function<bool(GuestOutputRefreshContext& context)> refresher,
      bool& is_8bpc_out_ref) override;
  bool OnGuestOutputPublished(
      uint32_t mailbox_index, bool guest_output_active, uint64_t source_id,
      uint64_t issue_time_ns, uint64_t publish_time_ns,
      const GuestOutputProperties& properties) override;
  bool IsGuestOutputPresentationBackendActive() const override {
    if (!zerofg_independent_presenter_) {
      return false;
    }
    if (zerofg_independent_presenter_->accepting()) {
      return true;
    }
    // Qualification occurs after this frame's refresher, so even if it arms
    // ZeroFG now this publication remains on the normal XenDroid presenter.
    // The next frame receives the first immutable ZeroFG Source publication.
    zerofg_independent_presenter_->TryActivateFromNativeSourceCadence();
    return false;
  }
  void OnGuestOutputPublicationCommitted() override;

  PaintResult PaintAndPresentImpl(bool execute_ui_drawers) override;

 private:
  struct NativePresentHostOperationStats {
    uint64_t calls = 0;
    uint64_t total_ns = 0;
    uint64_t max_ns = 0;
    uint64_t over_1ms = 0;
    uint64_t over_4ms = 0;
  };
  struct NativePresentHostTelemetry {
    uint64_t interval_start_ns = 0;
    NativePresentHostOperationStats completion;
    NativePresentHostOperationStats acquire;
    NativePresentHostOperationStats submit;
    NativePresentHostOperationStats present_queue_lock;
    NativePresentHostOperationStats present;
  };

  static void RecordNativePresentHostOperation(
      NativePresentHostOperationStats& stats, uint64_t duration_ns);
  void MaybeLogNativePresentHostTelemetry();

  bool AcquireZeroFGIngressSource(
      ZeroFGIndependentPresenter::IngressSourcePublication& publication_out);

  struct ZeroFGGenerationContext;
  struct ZeroFGPostContext;

  class GuestOutputImage {
   public:
    static std::unique_ptr<GuestOutputImage> Create(
        const VulkanDevice* const vulkan_device, const uint32_t width,
        const uint32_t height,
        VkImageUsageFlags additional_usage = 0) {
      assert_not_zero(width);
      assert_not_zero(height);
      auto image = std::unique_ptr<GuestOutputImage>(
          new GuestOutputImage(vulkan_device, width, height,
                               additional_usage));
      if (!image->Initialize()) {
        return nullptr;
      }
      return std::move(image);
    }

    GuestOutputImage(const GuestOutputImage& image) = delete;
    GuestOutputImage& operator=(const GuestOutputImage& image) = delete;
    ~GuestOutputImage();

    const VkExtent2D& extent() const { return extent_; }

    VkImage image() const { return image_; }
    VkDeviceMemory memory() const { return memory_; }
    VkImageView view() const { return view_; }

   private:
    GuestOutputImage(const VulkanDevice* const vulkan_device,
                     const uint32_t width, const uint32_t height,
                     VkImageUsageFlags additional_usage)
        : vulkan_device_(vulkan_device),
          additional_usage_(additional_usage) {
      extent_.width = width;
      extent_.height = height;
    }

    bool Initialize();

    const VulkanDevice* vulkan_device_;

    VkExtent2D extent_;
    VkImageUsageFlags additional_usage_ = 0;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
  };

  struct GuestOutputImageInstance {
    std::shared_ptr<GuestOutputImage> image;
    uint64_t version = UINT64_MAX;
    uint64_t last_refresher_submission = 0;
    bool ever_successfully_refreshed = false;

    void SetToNewImage(const std::shared_ptr<GuestOutputImage>& new_image,
                       uint64_t new_version) {
      image = new_image;
      version = new_version;
      last_refresher_submission = 0;
      ever_successfully_refreshed = false;
    }
  };

  struct GuestOutputPaintRectangleConstants {
    union {
      struct {
        float x;
        float y;
      };
      float offset[2];
    };
    union {
      struct {
        float width;
        float height;
      };
      float size[2];
    };
  };

  enum GuestOutputPaintPipelineLayoutIndex : size_t {
    kGuestOutputPaintPipelineLayoutIndexBilinear,
    kGuestOutputPaintPipelineLayoutIndexCasSharpen,
    kGuestOutputPaintPipelineLayoutIndexCasResample,
    kGuestOutputPaintPipelineLayoutIndexFsrEasu,
    kGuestOutputPaintPipelineLayoutIndexFsrRcas,
    kGuestOutputPaintPipelineLayoutIndexSgsr,

    kGuestOutputPaintPipelineLayoutCount,
  };

  static constexpr GuestOutputPaintPipelineLayoutIndex
  GetGuestOutputPaintPipelineLayoutIndex(GuestOutputPaintEffect effect) {
    switch (effect) {
      case GuestOutputPaintEffect::kBilinear:
      case GuestOutputPaintEffect::kBilinearDither:
        return kGuestOutputPaintPipelineLayoutIndexBilinear;
      case GuestOutputPaintEffect::kCasSharpen:
      case GuestOutputPaintEffect::kCasSharpenDither:
        return kGuestOutputPaintPipelineLayoutIndexCasSharpen;
      case GuestOutputPaintEffect::kCasResample:
      case GuestOutputPaintEffect::kCasResampleDither:
        return kGuestOutputPaintPipelineLayoutIndexCasResample;
      case GuestOutputPaintEffect::kFsrEasu:
        return kGuestOutputPaintPipelineLayoutIndexFsrEasu;
      case GuestOutputPaintEffect::kFsrRcas:
      case GuestOutputPaintEffect::kFsrRcasDither:
        return kGuestOutputPaintPipelineLayoutIndexFsrRcas;
      case GuestOutputPaintEffect::kSgsr:
      case GuestOutputPaintEffect::kSgsrEdgeDirection:
        return kGuestOutputPaintPipelineLayoutIndexSgsr;
      default:
        assert_unhandled_case(effect);
        return kGuestOutputPaintPipelineLayoutCount;
    }
  }

  struct PaintContext {
    class Submission {
     public:
      static std::unique_ptr<Submission> Create(
          const VulkanDevice* const vulkan_device) {
        auto submission =
            std::unique_ptr<Submission>(new Submission(vulkan_device));
        if (!submission->Initialize()) {
          return nullptr;
        }
        return submission;
      }

      Submission(const Submission& submission) = delete;
      Submission& operator=(const Submission& submission) = delete;
      ~Submission();

      VkSemaphore acquire_semaphore() const { return acquire_semaphore_; }
      VkCommandPool draw_command_pool() const { return draw_command_pool_; }
      VkCommandBuffer draw_command_buffer() const {
        return draw_command_buffer_;
      }

     private:
      explicit Submission(const VulkanDevice* const vulkan_device)
          : vulkan_device_(vulkan_device) {}
      bool Initialize();

      const VulkanDevice* vulkan_device_;
      VkSemaphore acquire_semaphore_ = VK_NULL_HANDLE;
      VkCommandPool draw_command_pool_ = VK_NULL_HANDLE;
      VkCommandBuffer draw_command_buffer_ = VK_NULL_HANDLE;
    };

    static constexpr uint32_t kSubmissionCount = 3;
    static constexpr uint32_t kGuestOutputPaintRefCount = 15;

    struct GuestOutputPaintPipeline {
      VkPipeline intermediate_pipeline = VK_NULL_HANDLE;
      VkPipeline swapchain_pipeline = VK_NULL_HANDLE;
      VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
    };

    enum GuestOutputDescriptorSet : uint32_t {
      kGuestOutputDescriptorSetGuestOutput0Sampled,

      kGuestOutputDescriptorSetIntermediate0Sampled =
          kGuestOutputDescriptorSetGuestOutput0Sampled +
          kGuestOutputPaintRefCount,

      kGuestOutputDescriptorSetCount =
          kGuestOutputDescriptorSetIntermediate0Sampled +
          kMaxGuestOutputPaintEffects - 1,
    };

    struct UISetupCommandBuffer {
      UISetupCommandBuffer(VkCommandPool command_pool,
                           VkCommandBuffer command_buffer,
                           uint64_t last_usage_submission_index = 0)
          : command_pool(command_pool),
            command_buffer(command_buffer),
            last_usage_submission_index(last_usage_submission_index) {}

      VkCommandPool command_pool;
      VkCommandBuffer command_buffer;
      uint64_t last_usage_submission_index;
    };

    struct SwapchainFramebuffer {
      SwapchainFramebuffer(VkImageView image_view, VkFramebuffer framebuffer)
          : image_view(image_view), framebuffer(framebuffer) {}

      VkImageView image_view;
      VkFramebuffer framebuffer;
    };

    explicit PaintContext(VulkanDevice* const vulkan_device)
        : vulkan_device(vulkan_device),
          completion_timeline(vulkan_device, "paint") {}
    PaintContext(const PaintContext& paint_context) = delete;
    PaintContext& operator=(const PaintContext& paint_context) = delete;

    static VkSwapchainKHR CreateSwapchainForVulkanSurface(
        const VulkanDevice* vulkan_device, VkSurfaceKHR surface, uint32_t width,
        uint32_t height, VkSwapchainKHR old_swapchain,
        uint32_t& present_queue_family_out, VkFormat& image_format_out,
        VkExtent2D& image_extent_out, bool& is_fifo_out,
        uint32_t& requested_image_count_out,
        VkPresentModeKHR& present_mode_out, bool& ui_surface_unusable_out);

    VkSwapchainKHR PrepareForSwapchainRetirement();
    void DestroySwapchainAndVulkanSurface();

    VulkanDevice* vulkan_device;

    std::array<std::unique_ptr<PaintContext::Submission>, kSubmissionCount>
        submissions;
    VulkanGPUCompletionTimeline completion_timeline;

    std::array<GuestOutputPaintPipeline, size_t(GuestOutputPaintEffect::kCount)>
        guest_output_paint_pipelines;

    VkDescriptorPool guest_output_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet
        guest_output_descriptor_sets[kGuestOutputDescriptorSetCount];

    std::array<std::pair<uint64_t, std::shared_ptr<GuestOutputImage>>,
               kGuestOutputPaintRefCount>
        guest_output_image_paint_refs;
    uint64_t guest_output_image_paint_last_submission = 0;

    std::array<std::unique_ptr<GuestOutputImage>,
               kMaxGuestOutputPaintEffects - 1>
        guest_output_intermediate_images;
    std::array<VkFramebuffer, kMaxGuestOutputPaintEffects - 1>
        guest_output_intermediate_framebuffers = {};
    uint64_t guest_output_intermediate_image_last_submission = 0;

    std::vector<UISetupCommandBuffer> ui_setup_command_buffers;
    size_t ui_setup_command_buffer_current_index = SIZE_MAX;

    VkRenderPass swapchain_render_pass = VK_NULL_HANDLE;
    VkFormat swapchain_render_pass_format = VK_FORMAT_UNDEFINED;
    bool swapchain_render_pass_clear_load_op = false;

    VkSurfaceKHR vulkan_surface = VK_NULL_HANDLE;
    uint32_t present_queue_family = UINT32_MAX;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D swapchain_extent = {};
    bool swapchain_is_fifo = false;
    uint32_t swapchain_requested_image_count = 0;
    VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;
    std::vector<VkImage> swapchain_images;
    std::vector<SwapchainFramebuffer> swapchain_framebuffers;
    std::vector<VkSemaphore> swapchain_image_present_semaphores;
  };

  explicit VulkanPresenter(HostGpuLossCallback host_gpu_loss_callback,
                           VulkanDevice* vulkan_device,
                           const UISamplers* ui_samplers);

  bool InitializeSurfaceIndependent();

  [[nodiscard]] VkPipeline CreateGuestOutputPaintPipeline(
      GuestOutputPaintEffect effect, VkRenderPass render_pass,
      VkResult* result_out = nullptr, bool zerofg = false);

  bool InitializeZeroFGDeviceContext();
  VulkanDevice* ZeroFGDevice() const;

  void BeginZeroFGSurfaceDisconnect();
  void DestroyZeroFGSurfaceResourcesAfterSourceIdle();
  // Main Surface Authority: A gives up its swapchain, keeping the surface.
  void RetireSwapchainForMainSurfaceAuthority();
  bool ProcessZeroFGPost(
      const ZeroFGIndependentPresenter::PostProcessRequest& request,
      ZeroFGIndependentPresenter::PostProcessResult& result);
  void ReleaseZeroFGPost(uint32_t final_output_index);
  bool PrepareZeroFGGenerationContext(
      ZeroFGIndependentPresenter::GenerationResult& result,
      uint64_t& setup_ns, bool& created_out);
  bool ProcessZeroFGGeneration(
      const ZeroFGIndependentPresenter::GenerationRequest& request,
      ZeroFGIndependentPresenter::GenerationResult& result);
  bool ReadZeroFGPreviousProfilingSample(
      uint32_t synthetic_index,
      ZeroFGIndependentPresenter::GenerationResult& result);
  bool PollZeroFGGeneration(
      uint32_t synthetic_index,
      ZeroFGIndependentPresenter::GenerationResult& result,
      bool& ready_out);
  void ReleaseZeroFGSynthetic(uint32_t synthetic_index);
  void DestroyZeroFGGenerationContext();
  void DestroyZeroFGPostContext();
  VulkanDevice* vulkan_device_;
  const UISamplers* ui_samplers_;

  // Host-only execution resources. A is borrowed in the default path; B and
  // its Post paint subset are owned for the entire VulkanPresenter lifetime.
  // No swapchain or native paint resources are created on B.
  struct ZeroFGPresenterVulkanContext {
    explicit ZeroFGPresenterVulkanContext(VulkanDevice* source);
    ~ZeroFGPresenterVulkanContext();
    std::unique_ptr<VulkanDevice> owned_device;
    VulkanDevice* device;
    std::unique_ptr<UISamplers> samplers;
    std::unique_ptr<ZeroFGDeviceHandoff> handoff;
    std::unique_ptr<ZeroFGGenerationContext> generation;
    // Fail-open for the Modern/QCOM backend candidates: set when a candidate
    // made the generation context or its Resize fail; every later context of
    // this session then forces the candidate switches to Compat.
    bool backend_fallback_active = false;
    std::unique_ptr<ZeroFGPostContext> post;
    VkDescriptorSetLayout image_layout = VK_NULL_HANDLE;
    std::array<VkPipelineLayout, kGuestOutputPaintPipelineLayoutCount> layouts =
        {};
    VkShaderModule vertex = VK_NULL_HANDLE;
    std::array<VkShaderModule, size_t(GuestOutputPaintEffect::kCount)>
        fragments = {};
    VkRenderPass render_pass = VK_NULL_HANDLE;
    std::array<VkPipeline, size_t(GuestOutputPaintEffect::kCount)> pipelines =
        {};
  };
  std::unique_ptr<ZeroFGPresenterVulkanContext> zerofg_vulkan_context_;

  // ZeroFG owns immutable Real/Synthetic candidates, the common post path
  // and the Main Surface egress on device B. It never owns the normal
  // swapchain or its VkQueue presentation boundary.
  std::unique_ptr<ZeroFGIndependentPresenter> zerofg_independent_presenter_;

  VkDescriptorSetLayout guest_output_paint_image_descriptor_set_layout_ =
      VK_NULL_HANDLE;
  std::array<VkPipelineLayout, kGuestOutputPaintPipelineLayoutCount>
      guest_output_paint_pipeline_layouts_ = {};
  VkShaderModule guest_output_paint_vs_ = VK_NULL_HANDLE;
  std::array<VkShaderModule, size_t(GuestOutputPaintEffect::kCount)>
      guest_output_paint_fs_ = {};
  VkRenderPass guest_output_intermediate_render_pass_ = VK_NULL_HANDLE;

  uint64_t guest_output_image_next_version_ = 0;
  std::array<GuestOutputImageInstance, kGuestOutputMailboxSize>
      guest_output_images_;
  VulkanGPUCompletionTimeline guest_output_image_refresher_completion_timeline_;

  VulkanGPUCompletionTimeline ui_completion_timeline_;

  PaintContext paint_context_;
  NativePresentHostTelemetry native_present_host_telemetry_;
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_D3D12_D3D12_PRESENTER_H_
