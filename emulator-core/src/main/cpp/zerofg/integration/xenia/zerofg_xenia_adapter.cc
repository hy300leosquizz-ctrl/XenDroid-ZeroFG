#include "zerofg/integration/xenia/zerofg_xenia_adapter.h"

#include <utility>

namespace zerofg::xenia {

std::unique_ptr<Adapter> Adapter::Create(
    const VulkanContext& vulkan, uint32_t frame_context_count, Mode mode,
    Backend backend, const Capabilities& capabilities, Status* status) {
  CreateInfo create_info;
  create_info.vulkan = vulkan;
  create_info.frame_context_count = frame_context_count;
  create_info.mode = mode;
  create_info.backend = backend;
  create_info.capabilities = capabilities;

  std::unique_ptr<Interpolator> interpolator =
      Interpolator::Create(create_info, status);
  if (!interpolator) {
    return nullptr;
  }

  return std::unique_ptr<Adapter>(new Adapter(std::move(interpolator)));
}

Adapter::Adapter(std::unique_ptr<Interpolator> interpolator)
    : interpolator_(std::move(interpolator)) {}

Adapter::~Adapter() = default;

Status Adapter::Resize(uint32_t width, uint32_t height, VkFormat input_format,
                       VkFormat output_format) {
  return interpolator_->Resize(width, height, input_format, output_format);
}

Status Adapter::Interpolate(VkCommandBuffer command_buffer,
                            uint32_t frame_context_index,
                            const Image& previous, const Image& current,
                            float phase, const Image& output) {
  return interpolator_->Interpolate(command_buffer, frame_context_index,
                                    previous, current, phase, output);
}

}  // namespace zerofg::xenia
