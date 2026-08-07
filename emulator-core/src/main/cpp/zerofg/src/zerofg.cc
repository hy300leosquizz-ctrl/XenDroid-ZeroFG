#include "zerofg/zerofg.h"
#include "vulkan_dispatch.h"

#include <cmath>
#include <utility>

namespace zerofg {

class Interpolator::Impl {
 public:
  explicit Impl(const CreateInfo& create_info)
      : vulkan_(create_info.vulkan) {}

  bool Initialize() {
    return dispatch_.Load(
        vulkan_.instance,
        vulkan_.device,
        vulkan_.get_instance_proc_addr);
  }

  Status Resize(uint32_t width,
                uint32_t height,
                VkFormat input_format,
                VkFormat output_format) {
    if (width == 0 || height == 0 ||
        input_format == VK_FORMAT_UNDEFINED ||
        output_format == VK_FORMAT_UNDEFINED) {
      return Status::kInvalidArgument;
    }

    width_ = width;
    height_ = height;
    input_format_ = input_format;
    output_format_ = output_format;
    initialized_ = true;

    return Status::kSuccess;
  }

  Status Interpolate(VkCommandBuffer command_buffer,
                     const Image& previous,
                     const Image& current,
                     float phase,
                     const Image& output) {
    if (!initialized_ ||
        command_buffer == VK_NULL_HANDLE ||
        previous.image == VK_NULL_HANDLE ||
        previous.view == VK_NULL_HANDLE ||
        current.image == VK_NULL_HANDLE ||
        current.view == VK_NULL_HANDLE ||
        output.image == VK_NULL_HANDLE ||
        output.view == VK_NULL_HANDLE) {
      return Status::kInvalidArgument;
    }

    if (!std::isfinite(phase) || phase < 0.0f || phase > 1.0f) {
      return Status::kInvalidArgument;
    }

    if (previous.width != width_ || previous.height != height_ ||
        current.width != width_ || current.height != height_ ||
        output.width != width_ || output.height != height_) {
      return Status::kInvalidArgument;
    }

    if (previous.format != input_format_ ||
        current.format != input_format_ ||
        output.format != output_format_) {
      return Status::kInvalidArgument;
    }

    // ZeroFG MVP initially targets exactly one synthetic midpoint frame.
    if (std::abs(phase - 0.5f) > 0.0001f) {
      return Status::kUnsupported;
    }

    // Vulkan compute implementation comes next.
    return Status::kUnsupported;
  }

 private:
  VulkanContext vulkan_;
  VulkanDispatch dispatch_;

  uint32_t width_ = 0;
  uint32_t height_ = 0;

  VkFormat input_format_ = VK_FORMAT_UNDEFINED;
  VkFormat output_format_ = VK_FORMAT_UNDEFINED;

  bool initialized_ = false;
};

std::unique_ptr<Interpolator> Interpolator::Create(
    const CreateInfo& create_info,
    Status* status) {
  if (create_info.vulkan.instance == VK_NULL_HANDLE ||
      create_info.vulkan.physical_device == VK_NULL_HANDLE ||
      create_info.vulkan.device == VK_NULL_HANDLE ||
      create_info.vulkan.get_instance_proc_addr == nullptr) {
    if (status) {
      *status = Status::kInvalidArgument;
    }
    return nullptr;
  }

  auto impl = std::make_unique<Impl>(create_info);

  if (!impl->Initialize()) {
    if (status) {
      *status = Status::kUnsupported;
    }
    return nullptr;
  }

  auto interpolator =
      std::unique_ptr<Interpolator>(new Interpolator(std::move(impl)));

  if (status) {
    *status = Status::kSuccess;
  }

  return interpolator;
}

Interpolator::Interpolator(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Interpolator::~Interpolator() = default;

Status Interpolator::Resize(uint32_t width,
                            uint32_t height,
                            VkFormat input_format,
                            VkFormat output_format) {
  return impl_->Resize(width, height, input_format, output_format);
}

Status Interpolator::Interpolate(VkCommandBuffer command_buffer,
                                 const Image& previous,
                                 const Image& current,
                                 float phase,
                                 const Image& output) {
  return impl_->Interpolate(
      command_buffer, previous, current, phase, output);
}

}  // namespace zerofg
