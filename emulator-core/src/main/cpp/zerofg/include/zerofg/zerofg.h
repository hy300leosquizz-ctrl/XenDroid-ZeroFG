#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

namespace zerofg {

// ZeroFG deliberately does not own a Vulkan queue, swapchain, presentation,
// frame pacing, or synchronization with the host renderer.
//
// The host is responsible for:
//   - beginning/ending and submitting the command buffer;
//   - synchronization and image barriers;
//   - keeping input/output images alive;
//   - presenting the generated frame.
//
// ZeroFG only records temporal interpolation work into the supplied
// VkCommandBuffer.

enum class Status {
  kSuccess = 0,
  kInvalidArgument,
  kUnsupported,
  kOutOfMemory,
  kVulkanError,
};

struct VulkanContext {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  // ZeroFG resolves Vulkan entry points through the host loader so custom
  // ICDs and drivers use exactly the same Vulkan dispatch path as the host.
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;

  const VkAllocationCallbacks* allocator = nullptr;
};

struct Image {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0;
  uint32_t height = 0;
  VkImageUsageFlags usage = 0;
};

struct CreateInfo {
  VulkanContext vulkan;

  // Number of independent GPU resource contexts used for
  // frames in flight.
  uint32_t frame_context_count = 3;
};

class Interpolator {
 public:
  static std::unique_ptr<Interpolator> Create(
      const CreateInfo& create_info,
      Status* status = nullptr);

  ~Interpolator();

  Interpolator(const Interpolator&) = delete;
  Interpolator& operator=(const Interpolator&) = delete;

  // Allocates/reallocates the internal working resources.
  // This should be called outside the hot presentation path whenever possible.
  Status Resize(uint32_t width,
                uint32_t height,
                VkFormat input_format,
                VkFormat output_format);

  // Records commands that generate a temporal frame between previous and
  // current.
  //
  // phase:
  //   0.0 = previous
  //   0.5 = midpoint (initial ZeroFG 2x target)
  //   1.0 = current
  //
  // The initial implementation will target phase == 0.5.
//
// previous/current must have VK_IMAGE_USAGE_SAMPLED_BIT.
// output must be a distinct image with
// VK_IMAGE_USAGE_TRANSFER_DST_BIT.
//
// The command buffer must support VK_QUEUE_GRAPHICS_BIT because the
// MVP output path uses vkCmdBlitImage for format conversion.
  // frame_context_index selects an independent GPU resource context.
// The caller must not reuse the same index until the GPU submission
// containing the previous Interpolate call for that index has completed.
Status Interpolate(VkCommandBuffer command_buffer,
                     uint32_t frame_context_index,
                     const Image& previous,
                     const Image& current,
                     float phase,
                     const Image& output);

 private:
  class Impl;

  explicit Interpolator(std::vector<std::unique_ptr<Impl>> impls);

  std::vector<std::unique_ptr<Impl>> impls_;
};

}  // namespace zerofg
