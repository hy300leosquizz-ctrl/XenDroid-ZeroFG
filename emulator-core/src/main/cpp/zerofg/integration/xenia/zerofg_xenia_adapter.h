#pragma once

#include <cstdint>
#include <memory>

#include "zerofg/zerofg.h"

namespace zerofg::xenia {

// Thin host adapter. Queue submission, synchronization, swapchain ownership,
// presentation and frame pacing remain responsibilities of VulkanPresenter.
class Adapter {
 public:
  static std::unique_ptr<Adapter> Create(
      const VulkanContext& vulkan, uint32_t frame_context_count,
      Mode mode, Backend backend, const Capabilities& capabilities,
      Status* status = nullptr);

  ~Adapter();

  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;

  Status Resize(uint32_t width, uint32_t height, VkFormat input_format,
                VkFormat output_format);

  Status Interpolate(VkCommandBuffer command_buffer,
                     uint32_t frame_context_index, const Image& previous,
                     const Image& current, float phase, const Image& output);

 private:
  explicit Adapter(std::unique_ptr<Interpolator> interpolator);

  std::unique_ptr<Interpolator> interpolator_;
};

}  // namespace zerofg::xenia
