/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/vulkan/vulkan_device.h"

#include "xenia/base/assert.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/platform.h"
#include "xenia/ui/vulkan/zerofg_config.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
#include "third_party/libadrenotools/include/adrenotools/driver.h"
#endif

DEFINE_int32(
    vulkan_clamp_storage_buffer_range, 0,
    "If > 0, clamp the reported maxStorageBufferRange to this many bytes. "
    "Drivers reporting very large limits get a single 512 MB storage buffer "
    "binding for shared memory emulation; clamping to 134217728 (128 MB) "
    "forces the split 4-binding path (what Turnip uses) for debugging "
    "driver-specific large-binding misbehavior.",
    "Vulkan");

namespace xe {
namespace ui {
namespace vulkan {

template <typename Structure, VkStructureType StructureType>
struct VulkanFeatures {
  Structure supported = {StructureType};
  Structure enabled = {StructureType};

  void Link(VkPhysicalDeviceFeatures2& supported_features_2,
            VkDeviceCreateInfo& device_create_info) {
    supported.pNext = supported_features_2.pNext;
    supported_features_2.pNext = &supported;
    enabled.pNext = const_cast<void*>(device_create_info.pNext);
    device_create_info.pNext = &enabled;
  }
};

zerofg::Capabilities::Format VulkanDevice::QueryZeroFGBackendFormat(
    const VkFormat format, const VkImageUsageFlags usage,
    const VkImageTiling tiling, const VkImageCreateFlags create_flags,
    const VkImageViewType view_type) const {
  zerofg::Capabilities::Format result;
  result.format = format;
  result.usage = usage;
  result.tiling = tiling;
  result.create_flags = create_flags;
  result.view_type = view_type;
  if (!is_zerofg_presenter_device()) return result;
  const auto& ifn = vulkan_instance_->functions();
  const auto get_format_properties2 = PFN_vkGetPhysicalDeviceFormatProperties2(
      ifn.vkGetInstanceProcAddr(vulkan_instance_->instance(),
                               "vkGetPhysicalDeviceFormatProperties2"));
  const auto get_image_properties2 =
      PFN_vkGetPhysicalDeviceImageFormatProperties2(
          ifn.vkGetInstanceProcAddr(vulkan_instance_->instance(),
                                   "vkGetPhysicalDeviceImageFormatProperties2"));
  if (get_format_properties2) {
    VkFormatProperties3 properties3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 properties2 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
    properties2.pNext = &properties3;
    get_format_properties2(physical_device_, format, &properties2);
    result.optimal_tiling_features = properties3.optimalTilingFeatures;
  }
  if (!get_image_properties2) {
    result.image_query_result = VK_ERROR_EXTENSION_NOT_PRESENT;
    return result;
  }
  VkPhysicalDeviceImageFormatInfo2 image_info = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
  image_info.format = format;
  image_info.type = VK_IMAGE_TYPE_2D;
  image_info.tiling = tiling;
  image_info.usage = usage;
  image_info.flags = create_flags;
  VkPhysicalDeviceImageViewImageFormatInfoEXT view_info = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_IMAGE_FORMAT_INFO_EXT};
  view_info.imageViewType = view_type;
  VkFilterCubicImageViewImageFormatPropertiesEXT cubic_info = {
      VK_STRUCTURE_TYPE_FILTER_CUBIC_IMAGE_VIEW_IMAGE_FORMAT_PROPERTIES_EXT};
  VkImageFormatProperties2 image_properties = {
      VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
  if (zerofg_backend_capabilities_.filter_cubic_supported) {
    image_info.pNext = &view_info;
    image_properties.pNext = &cubic_info;
  }
  result.image_query_result =
      get_image_properties2(physical_device_, &image_info, &image_properties);
  if (result.image_query_result == VK_SUCCESS) {
    result.cubic = cubic_info.filterCubic != VK_FALSE;
    result.cubic_minmax = cubic_info.filterCubicMinmax != VK_FALSE;
  }
  return result;
}

bool VulkanDevice::DrainZeroFGForTeardown() {
  if (!is_zerofg_presenter_device()) return false;
  if (zerofg_teardown_idle()) return true;
  // Every enabled queue of B: the main-surface profile adds B:1 for egress.
  VkResult result = VK_SUCCESS;
  const size_t queue_count =
      queue_families_[queue_family_graphics_compute()].queues.size();
  for (size_t queue_index = 0; queue_index < queue_count; ++queue_index) {
    auto queue =
        AcquireQueue(queue_family_graphics_compute(), uint32_t(queue_index));
    const VkResult queue_result = functions().vkQueueWaitIdle(queue.queue());
    if (queue_result != VK_SUCCESS && result == VK_SUCCESS) {
      result = queue_result;
    }
  }
  bool retired = result == VK_SUCCESS;
  if (result == VK_ERROR_DEVICE_LOST) {
    SetLost();
    // A lost device is terminal for B, so child-resource retirement is no
    // longer required before teardown destroys the device-owned objects.
    retired = true;
  } else if (!retired) {
    // Teardown may block, and a queue-idle error alone is not retirement
    // proof. Make one explicit device-wide attempt before reporting failure;
    // the live path never calls either idle operation.
    const VkResult device_idle_result = vkDeviceWaitIdle()(device());
    XELOGW("ZeroFGDeviceB teardown_queue_idle fallback_device_idle result={}",
           int32_t(device_idle_result));
    retired = device_idle_result == VK_SUCCESS;
    if (device_idle_result == VK_ERROR_DEVICE_LOST) {
      SetLost();
      retired = true;
    }
  }
  zerofg_teardown_idle_.store(retired, std::memory_order_release);
  XELOGI(
      "ZeroFGDeviceB teardown_queue_idle result={} retired={} A_untouched=true",
      int32_t(result), retired);
  return retired;
}

bool VulkanDevice::RejectZeroFGSubmitAfterTeardownIdle() {
  if (!is_zerofg_presenter_device() || !zerofg_teardown_idle()) return false;
  if (!teardown_submit_rejected_.exchange(true, std::memory_order_acq_rel)) {
    XELOGE("ZeroFGDeviceB submit_after_teardown_idle rejected");
  }
  return true;
}

std::unique_ptr<VulkanDevice> VulkanDevice::CreateIfSupported(
    const VulkanInstance* const vulkan_instance,
    const VkPhysicalDevice physical_device, const bool with_gpu_emulation,
    const bool with_swapchain, const CreationProfile profile,
    const ZeroFGBackendFeatureRequests backend_requests) {
  assert_not_null(vulkan_instance);
  assert_not_null(physical_device);
  // Device B: ZeroFG's own device and the main Surface's WSI producer.
  const bool presenter_device =
      profile == CreationProfile::kZeroFGMainSurfacePresenter;
  // Driver pipeline statistics were a development diagnostic.
  constexpr bool pipeline_statistics_requested = false;
  if (presenter_device && (with_gpu_emulation || !with_swapchain)) {
    XELOGE(
        "ZeroFGDeviceB invalid creation profile: emulation prohibited, WSI "
        "required");
    return nullptr;
  }

  const VulkanInstance::Functions& ifn = vulkan_instance->functions();

  // Get supported Vulkan 1.0 properties and features.

  VkPhysicalDeviceProperties properties = {};
  ifn.vkGetPhysicalDeviceProperties(physical_device, &properties);

  // From the VkApplicationInfo specification:
  //
  // "The Khronos validation layers will treat apiVersion as the highest API
  // version the application targets, and will validate API usage against the
  // minimum of that version and the implementation version (instance or device,
  // depending on context). If an application tries to use functionality from a
  // greater version than this, a validation error will be triggered."
  //
  // "Vulkan 1.0 implementations were required to return
  // VK_ERROR_INCOMPATIBLE_DRIVER if apiVersion was larger than 1.0."
  //
  // Make sure that all usages of the API version in Xenia receive the highest
  // minor version that Xenia has been tested on.
  // Libraries such as the Vulkan Memory Allocator also may expect a minor
  // version that is known to them.
  const uint32_t unclamped_api_version = properties.apiVersion;
  const uint32_t clamped_api_minor_version = std::min(
      VK_MAKE_API_VERSION(VK_API_VERSION_VARIANT(unclamped_api_version),
                          VK_API_VERSION_MAJOR(unclamped_api_version),
                          VK_API_VERSION_MINOR(unclamped_api_version), 0),
      vulkan_instance->api_version() >= VK_MAKE_API_VERSION(0, 1, 1, 0)
          ? kHighestUsedApiMinorVersion
          : VK_MAKE_API_VERSION(0, 1, 0, 0));
  properties.apiVersion =
      VK_MAKE_API_VERSION(VK_API_VERSION_VARIANT(clamped_api_minor_version),
                          VK_API_VERSION_MAJOR(clamped_api_minor_version),
                          VK_API_VERSION_MINOR(clamped_api_minor_version),
                          VK_API_VERSION_PATCH(unclamped_api_version));

  VkPhysicalDeviceFeatures supported_features = {};
  ifn.vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
  if (presenter_device &&
      (vulkan_instance->api_version() < VK_API_VERSION_1_3 ||
       properties.apiVersion < VK_API_VERSION_1_3)) {
    XELOGE("ZeroFGDeviceB requires effective Vulkan 1.3");
    return nullptr;
  }

  if (with_gpu_emulation) {
    if (!supported_features.independentBlend) {
      // Not trivial to work around:
      // - Affects not only the blend equation, but also the color write mask.
      // - Can't reuse the blend state of the first attachment for all because
      //   some attachments may have a format that doesn't support blending.
      // - Not possible to split the draw into per-attachment draws because of
      //   depth / stencil.
      // Not supported only on the proprietary driver for the Qualcomm
      // Adreno 4xx, where the driver is largely experimental and doesn't expose
      // a lot of the functionality available in the hardware.
      XELOGW(
          "Vulkan device '{}' doesn't support the independentBlend feature "
          "required for GPU emulation",
          properties.deviceName);
      return nullptr;
    }
  }

  // Enable needed extensions.

  std::unique_ptr<VulkanDevice> device(
      new VulkanDevice(vulkan_instance, physical_device));
  device->creation_profile_ = profile;
  auto& backend = device->zerofg_backend_capabilities_;
  backend.effective_api_version = properties.apiVersion;

  const bool get_physical_device_properties2_supported =
      vulkan_instance->extensions().ext_1_1_KHR_get_physical_device_properties2;

  // Name pointers from `requested_extensions` will be used in the enabled
  // extensions vector.
  std::unordered_map<std::string, bool*> requested_extensions;

  const auto request_promoted_extension =
      [&](const char* const name, uint32_t const major, uint32_t const minor,
          bool* const supported_ptr) {
        assert_not_null(supported_ptr);
        if (properties.apiVersion >= VK_MAKE_API_VERSION(0, major, minor, 0)) {
          *supported_ptr = true;
        } else {
          requested_extensions.emplace(name, supported_ptr);
        }
      };

#define XE_UI_VULKAN_STRUCT_EXTENSION(name) \
  requested_extensions.emplace("VK_" #name, &device->extensions_.ext_##name);
#define XE_UI_VULKAN_LOCAL_EXTENSION(name) \
  requested_extensions.emplace("VK_" #name, &ext_##name);
#define XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(name, major, minor) \
  request_promoted_extension(                                      \
      "VK_" #name, major, minor,                                   \
      &device->extensions_.ext_##major##_##minor##_##name);
#define XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(name, major, minor) \
  request_promoted_extension("VK_" #name, major, minor,           \
                             &ext_##major##_##minor##_##name);

  bool ext_KHR_portability_subset = false;
  bool ext_1_2_KHR_driver_properties = false;
  bool ext_1_2_KHR_timeline_semaphore = false;
  if (get_physical_device_properties2_supported) {
    // #164. Must be enabled according to the specification if the physical
    // device is a portability subset one.
    XE_UI_VULKAN_LOCAL_EXTENSION(KHR_portability_subset)
    // #197
    XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_driver_properties, 1, 2)
  }
  if (pipeline_statistics_requested &&
      get_physical_device_properties2_supported) {
    XE_UI_VULKAN_STRUCT_EXTENSION(KHR_pipeline_executable_properties)
  }

  // Used by the Vulkan Memory Allocator and potentially by Xenia.
  // #128.
  XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_dedicated_allocation, 1, 1)
  // #147. Also must be enabled for VK_KHR_dedicated_allocation and
  // VK_KHR_sampler_ycbcr_conversion.
  XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_get_memory_requirements2, 1, 1)
  // #158. Also must be enabled for VK_KHR_sampler_ycbcr_conversion.
  XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_bind_memory2, 1, 1)
  if (get_physical_device_properties2_supported) {
    // #238.
    XE_UI_VULKAN_STRUCT_EXTENSION(EXT_memory_budget)
    // #208. GPU readiness for Source-first ZeroFG ingest submissions.
    XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_timeline_semaphore, 1, 2)
  }
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    // #414.
    XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_maintenance4, 1, 3)
  }
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    // #55. Dynamic rendering removes render pass/framebuffer overhead.
    XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_dynamic_rendering, 1, 3)
  }

  bool ext_GOOGLE_display_timing = false;
  if (with_swapchain) {
    // #2.
    XE_UI_VULKAN_STRUCT_EXTENSION(KHR_swapchain)
    requested_extensions.emplace(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME,
                                 &ext_GOOGLE_display_timing);
#if XE_PLATFORM_WIN32
    // #256. Windows-only extension to control fullscreen exclusive behavior.
    // Used to prevent HDR state corruption during fullscreen transitions.
    // Requires the VK_KHR_get_surface_capabilities2 instance extension.
    if (vulkan_instance->extensions().ext_KHR_get_surface_capabilities2) {
      XE_UI_VULKAN_STRUCT_EXTENSION(EXT_full_screen_exclusive)
    }
#endif
  }

  bool ext_1_2_KHR_sampler_mirror_clamp_to_edge = false;
  bool ext_1_2_EXT_host_query_reset = false;
  bool ext_1_2_KHR_shader_float16_int8 = false;
  bool ext_1_1_KHR_maintenance1 = false;
  bool ext_1_2_KHR_shader_float_controls = false;
  bool ext_EXT_fragment_shader_interlock = false;
  bool ext_1_3_EXT_shader_demote_to_helper_invocation = false;
  bool ext_1_3_KHR_dynamic_rendering = false;
  bool ext_EXT_non_seamless_cube_map = false;
  bool ext_EXT_custom_border_color = false;
  bool ext_1_3_EXT_subgroup_size_control = false;
  bool ext_KHR_fragment_shader_barycentric = false;
  bool ext_NV_fragment_shader_barycentric = false;
  bool ext_QCOM_image_processing = false;
  if (presenter_device) {
    // Optional backend requests belong exclusively to B. Querying the core
    // subgroup properties does not itself enable either optional feature.
    ext_1_3_EXT_subgroup_size_control =
        properties.apiVersion >= VK_API_VERSION_1_3;
    if (backend_requests.filter_cubic) {
      requested_extensions.emplace(VK_EXT_FILTER_CUBIC_EXTENSION_NAME,
                                   &backend.filter_cubic_enabled);
    }
    if (backend_requests.qcom_image_processing ||
        backend_requests.qcom_block_match) {
      requested_extensions.emplace(VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME,
                                   &ext_QCOM_image_processing);
    }
  }
  if (with_gpu_emulation) {
    // #15.
    XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_sampler_mirror_clamp_to_edge, 1,
                                          2)
    // #70. Must be enabled for VK_KHR_sampler_ycbcr_conversion.
    XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_maintenance1, 1, 1)
    // #141.
    XE_UI_VULKAN_STRUCT_EXTENSION(EXT_shader_stencil_export)
    // #148.
    XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_image_format_list, 1, 2)
    if (get_physical_device_properties2_supported) {
      // #157.
      XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_sampler_ycbcr_conversion, 1, 1)
      // #198. Also must be enabled for VK_KHR_spirv_1_4.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_shader_float_controls, 1, 2)
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(EXT_host_query_reset, 1, 2)
      // #83. Float16 and Int16 capabilities are declared by system shaders.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_shader_float16_int8, 1, 2)
      // #252.
      XE_UI_VULKAN_LOCAL_EXTENSION(EXT_fragment_shader_interlock)
      // #55.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(KHR_dynamic_rendering, 1, 3)
      // #233. In-pass reads of current attachments for tiler-native resolves.
      XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_dynamic_rendering_local_read,
                                             1, 4)
      // #277.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(
          EXT_shader_demote_to_helper_invocation, 1, 3)
      // #423.
      XE_UI_VULKAN_LOCAL_EXTENSION(EXT_non_seamless_cube_map)
      // #288. Custom sampler border colors (for YCbCr border colors).
      XE_UI_VULKAN_LOCAL_EXTENSION(EXT_custom_border_color)
      // #226.
      XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION(EXT_subgroup_size_control, 1, 3)
      // #322 (KHR) / #203 (NV). Barycentric coordinates for manual
      // interpolation.
      XE_UI_VULKAN_LOCAL_EXTENSION(KHR_fragment_shader_barycentric)
      XE_UI_VULKAN_LOCAL_EXTENSION(NV_fragment_shader_barycentric)
      // #456. Per-sub-feature dynamic state used to collapse pipeline key
      // permutations. EDS1/EDS2 are core in 1.3; only EDS3 needs an extension.
      XE_UI_VULKAN_STRUCT_EXTENSION(EXT_extended_dynamic_state3)
    }
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
      // #237.
      XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION(KHR_spirv_1_4, 1, 2)
    }
    // #342. Driver-side fault description after VK_ERROR_DEVICE_LOST.
    if (get_physical_device_properties2_supported) {
      XE_UI_VULKAN_STRUCT_EXTENSION(EXT_device_fault)
    }
    // #179. Import guest RAM as device memory for a zero-copy shared-memory
    // buffer that aliases guest RAM.
    if (get_physical_device_properties2_supported) {
      XE_UI_VULKAN_STRUCT_EXTENSION(EXT_external_memory_host)
    }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
    // ZeroFG's A->B handoff shares Source images with device B as
    // AHardwareBuffers and synchronizes through sync_files. Request these only
    // for an opted-in ZeroFG session so the normal XenDroid device extension
    // set is unchanged when ZeroFG is off.
    if (IsZeroFGRequested() &&
        properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
      XE_UI_VULKAN_STRUCT_EXTENSION(
          ANDROID_external_memory_android_hardware_buffer)
      XE_UI_VULKAN_STRUCT_EXTENSION(KHR_external_semaphore_fd)
    }
#endif
  }

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (presenter_device) {
    XE_UI_VULKAN_STRUCT_EXTENSION(
        ANDROID_external_memory_android_hardware_buffer)
    XE_UI_VULKAN_STRUCT_EXTENSION(KHR_external_semaphore_fd)
  }
#endif

#undef XE_UI_VULKAN_STRUCT_EXTENSION
#undef XE_UI_VULKAN_LOCAL_EXTENSION
#undef XE_UI_VULKAN_STRUCT_PROMOTED_EXTENSION
#undef XE_UI_VULKAN_LOCAL_PROMOTED_EXTENSION

  std::vector<const char*> enabled_extensions;
  bool zerofg_display_timing_available = false;
  // Device B asks for a high queue priority through the driver as well
  // (VK_KHR/EXT_global_priority), for drivers that honour it.
  bool global_priority_khr = false;
  bool global_priority_ext = false;
  {
    uint32_t supported_extension_count = 0;
    const VkResult get_supported_extension_count_result =
        ifn.vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &supported_extension_count, nullptr);
    if (get_supported_extension_count_result != VK_SUCCESS &&
        get_supported_extension_count_result != VK_INCOMPLETE) {
      XELOGW("Failed to get the Vulkan device '{}' extension count",
             properties.deviceName);
      return nullptr;
    }
    if (supported_extension_count) {
      std::vector<VkExtensionProperties> supported_extensions(
          supported_extension_count);
      if (ifn.vkEnumerateDeviceExtensionProperties(
              physical_device, nullptr, &supported_extension_count,
              supported_extensions.data()) != VK_SUCCESS) {
        XELOGW("Failed to get the Vulkan device '{}' extensions",
               properties.deviceName);
        return nullptr;
      }
      assert_true(supported_extension_count == supported_extensions.size());
      for (const VkExtensionProperties& supported_extension :
           supported_extensions) {
        if (presenter_device) {
          const char* name = supported_extension.extensionName;
          if (!std::strcmp(name, VK_EXT_FILTER_CUBIC_EXTENSION_NAME)) {
            backend.filter_cubic_supported = true;
          } else if (!std::strcmp(name,
                                 VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME)) {
            backend.qcom_image_processing_supported = true;
          } else if (!std::strcmp(name,
                                 VK_QCOM_IMAGE_PROCESSING_2_EXTENSION_NAME)) {
            backend.qcom_image_processing2_advertised = true;
          } else if (!std::strcmp(name, "VK_QCOM_image_processing3")) {
            backend.qcom_image_processing3_advertised = true;
          } else if (!std::strcmp(name,
                                 VK_QCOM_FILTER_CUBIC_WEIGHTS_EXTENSION_NAME)) {
            backend.qcom_cubic_weights_advertised = true;
          } else if (!std::strcmp(name,
                                 VK_QCOM_FILTER_CUBIC_CLAMP_EXTENSION_NAME)) {
            backend.qcom_cubic_clamp_advertised = true;
          }
        }
        if (presenter_device &&
            !std::strcmp(supported_extension.extensionName,
                         VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
          device->properties_.pipelineExecutablePropertiesAdvertised = true;
        }
        if (!std::strcmp(supported_extension.extensionName,
                         VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME)) {
          zerofg_display_timing_available = true;
        }
        if (presenter_device) {
          const char* name = supported_extension.extensionName;
          if (!std::strcmp(name, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME)) {
            global_priority_khr = true;
          } else if (!std::strcmp(name,
                                  VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME)) {
            global_priority_ext = true;
          }
        }
        const auto requested_extension_it =
            requested_extensions.find(supported_extension.extensionName);
        if (requested_extension_it == requested_extensions.cend()) {
          continue;
        }
        assert_not_null(requested_extension_it->second);
        if (!*requested_extension_it->second) {
          enabled_extensions.emplace_back(
              requested_extension_it->first.c_str());
          *requested_extension_it->second = true;
        }
      }
    }
  }

  if (with_swapchain && !device->extensions_.ext_KHR_swapchain) {
    XELOGW("Vulkan device '{}' doesn't support swapchains",
           properties.deviceName);
    return nullptr;
  }
  if (global_priority_khr) {
    enabled_extensions.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
  } else if (global_priority_ext) {
    enabled_extensions.push_back(VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME);
  }

  VkDeviceCreateInfo device_create_info = {
      VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};

  device_create_info.enabledExtensionCount =
      uint32_t(enabled_extensions.size());
  device_create_info.ppEnabledExtensionNames = enabled_extensions.data();

  // Get supported Vulkan 1.1+ and extension properties and features.
  //
  // The property and feature structures are initialized to zero or to the
  // minimum / maximum requirements for the simplicity of handling unavailable
  // VK_KHR_get_physical_device_properties2.

  VkPhysicalDeviceProperties2 properties_2 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

  VkPhysicalDeviceFeatures2 supported_features_2 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

  VulkanFeatures<VkPhysicalDeviceVulkan11Features,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES>
      features_1_1;
  VulkanFeatures<VkPhysicalDeviceVulkan12Features,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES>
      features_1_2;
  VulkanFeatures<VkPhysicalDeviceHostQueryResetFeatures,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES>
      features_EXT_host_query_reset;
  VulkanFeatures<VkPhysicalDeviceTimelineSemaphoreFeatures,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES>
      features_KHR_timeline_semaphore;
  VulkanFeatures<VkPhysicalDeviceShaderFloat16Int8Features,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES>
      features_KHR_shader_float16_int8;
  VulkanFeatures<VkPhysicalDeviceVulkan13Features,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES>
      features_1_3;
  VulkanFeatures<
      VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR>
      features_KHR_pipeline_executable_properties;
  VulkanFeatures<
      VkPhysicalDevicePortabilitySubsetFeaturesKHR,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR>
      features_KHR_portability_subset;
  VkPhysicalDeviceDriverPropertiesKHR properties_1_2_KHR_driver_properties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceFloatControlsProperties
      properties_1_2_KHR_shader_float_controls = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
  VulkanFeatures<
      VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT>
      features_EXT_fragment_shader_interlock;
  VulkanFeatures<
      VkPhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES_EXT>
      features_1_3_EXT_shader_demote_to_helper_invocation;
  VulkanFeatures<
      VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT>
      features_EXT_non_seamless_cube_map;
  VulkanFeatures<
      VkPhysicalDeviceCustomBorderColorFeaturesEXT,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT>
      features_EXT_custom_border_color;
  VulkanFeatures<VkPhysicalDeviceDynamicRenderingFeatures,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES>
      features_1_3_KHR_dynamic_rendering;
  VulkanFeatures<
      VkPhysicalDeviceDynamicRenderingLocalReadFeatures,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES>
      features_1_4_KHR_dynamic_rendering_local_read;
  // Vulkan 1.1 core subgroup properties.
  VkPhysicalDeviceSubgroupProperties properties_subgroup = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  // VK_EXT_subgroup_size_control (#226, promoted to 1.3).
  VkPhysicalDeviceSubgroupSizeControlProperties
      properties_1_3_EXT_subgroup_size_control = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
  VulkanFeatures<
      VkPhysicalDeviceSubgroupSizeControlFeatures,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES>
      features_1_3_EXT_subgroup_size_control;
  // VK_KHR_fragment_shader_barycentric (#322) /
  // VK_NV_fragment_shader_barycentric (#203). KHR and NV share the same feature
  // structure type.
  VulkanFeatures<
      VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR>
      features_KHR_fragment_shader_barycentric;
  VulkanFeatures<VkPhysicalDeviceFaultFeaturesEXT,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT>
      features_EXT_device_fault;
  // VK_EXT_extended_dynamic_state3 (#456). Per-sub-feature dynamic state.
  VulkanFeatures<
      VkPhysicalDeviceExtendedDynamicState3FeaturesEXT,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT>
      features_EXT_extended_dynamic_state3;
  // dynamicPrimitiveTopologyUnrestricted is a property, not a feature.
  VkPhysicalDeviceExtendedDynamicState3PropertiesEXT
      properties_EXT_extended_dynamic_state3 = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT};
  // VK_EXT_external_memory_host (#179) properties (host pointer import
  // alignment).
  VkPhysicalDeviceExternalMemoryHostPropertiesEXT
      properties_EXT_external_memory_host = {
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};

  // All QCOM probe structures are linked for supported queries only. v2,
  // cubic weights/clamp are never linked to device creation in this batch.
  VkPhysicalDeviceImageProcessingFeaturesQCOM features_qcom = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM};
  VkPhysicalDeviceImageProcessingFeaturesQCOM enabled_qcom = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM};
  VkPhysicalDeviceImageProcessingPropertiesQCOM properties_qcom = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES_QCOM};
  VkPhysicalDeviceImageProcessing2FeaturesQCOM features_qcom2 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_2_FEATURES_QCOM};
  VkPhysicalDeviceImageProcessing2PropertiesQCOM properties_qcom2 = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_2_PROPERTIES_QCOM};
  VkPhysicalDeviceCubicWeightsFeaturesQCOM features_cubic_weights = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUBIC_WEIGHTS_FEATURES_QCOM};
  VkPhysicalDeviceCubicClampFeaturesQCOM features_cubic_clamp = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUBIC_CLAMP_FEATURES_QCOM};
  VkPhysicalDeviceShaderIntegerDotProductProperties properties_dot_product = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES};
  VkPhysicalDeviceSamplerFilterMinmaxProperties properties_minmax = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_FILTER_MINMAX_PROPERTIES};

  if (get_physical_device_properties2_supported) {
    if (presenter_device) {
      const auto link_feature_query = [&](auto& feature) {
        feature.pNext = supported_features_2.pNext;
        supported_features_2.pNext = &feature;
      };
      const auto link_property_query = [&](auto& property) {
        property.pNext = properties_2.pNext;
        properties_2.pNext = &property;
      };
      link_property_query(properties_dot_product);
      link_property_query(properties_minmax);
      if (backend.qcom_image_processing_supported) {
        link_feature_query(features_qcom);
        link_property_query(properties_qcom);
      }
      if (backend.qcom_image_processing2_advertised) {
        link_feature_query(features_qcom2);
        link_property_query(properties_qcom2);
      }
      if (backend.qcom_cubic_weights_advertised) {
        link_feature_query(features_cubic_weights);
      }
      if (backend.qcom_cubic_clamp_advertised) {
        link_feature_query(features_cubic_clamp);
      }
    }
    if (presenter_device &&
        device->properties_.pipelineExecutablePropertiesAdvertised) {
      // Query support without enabling the feature. The enabled pNext is
      // linked only after this query proves support and the Test opt-in.
      features_KHR_pipeline_executable_properties.supported.pNext =
          supported_features_2.pNext;
      supported_features_2.pNext =
          &features_KHR_pipeline_executable_properties.supported;
    }
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
      features_1_1.Link(supported_features_2, device_create_info);
    }
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
      features_1_2.Link(supported_features_2, device_create_info);
    } else {
      if (ext_1_2_EXT_host_query_reset) {
        features_EXT_host_query_reset.Link(supported_features_2,
                                           device_create_info);
      }
      if (ext_1_2_KHR_shader_float16_int8) {
        features_KHR_shader_float16_int8.Link(supported_features_2,
                                              device_create_info);
      }
      if (ext_1_2_KHR_timeline_semaphore) {
        features_KHR_timeline_semaphore.Link(supported_features_2,
                                             device_create_info);
      }
    }
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
      features_1_3.Link(supported_features_2, device_create_info);
    } else {
      if (ext_1_3_EXT_shader_demote_to_helper_invocation) {
        features_1_3_EXT_shader_demote_to_helper_invocation.Link(
            supported_features_2, device_create_info);
      }
      if (ext_1_3_KHR_dynamic_rendering) {
        features_1_3_KHR_dynamic_rendering.Link(supported_features_2,
                                                device_create_info);
      }
    }
    if (device->extensions_.ext_1_4_KHR_dynamic_rendering_local_read) {
      features_1_4_KHR_dynamic_rendering_local_read.Link(supported_features_2,
                                                         device_create_info);
    }
    if (ext_KHR_portability_subset) {
      features_KHR_portability_subset.Link(supported_features_2,
                                           device_create_info);
    }
    if (ext_1_2_KHR_driver_properties) {
      properties_1_2_KHR_driver_properties.pNext = properties_2.pNext;
      properties_2.pNext = &properties_1_2_KHR_driver_properties;
    }
    if (ext_1_2_KHR_shader_float_controls) {
      properties_1_2_KHR_shader_float_controls.pNext = properties_2.pNext;
      properties_2.pNext = &properties_1_2_KHR_shader_float_controls;
    }
    if (ext_EXT_fragment_shader_interlock) {
      features_EXT_fragment_shader_interlock.Link(supported_features_2,
                                                  device_create_info);
    }
    if (ext_EXT_non_seamless_cube_map) {
      features_EXT_non_seamless_cube_map.Link(supported_features_2,
                                              device_create_info);
    }
    if (ext_EXT_custom_border_color) {
      features_EXT_custom_border_color.Link(supported_features_2,
                                            device_create_info);
    }
    // Subgroup properties are Vulkan 1.1 core.
    if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
      properties_subgroup.pNext = properties_2.pNext;
      properties_2.pNext = &properties_subgroup;
    }
    // VK_EXT_subgroup_size_control properties and features.
    if (ext_1_3_EXT_subgroup_size_control) {
      properties_1_3_EXT_subgroup_size_control.pNext = properties_2.pNext;
      properties_2.pNext = &properties_1_3_EXT_subgroup_size_control;
      // On Vulkan 1.3 these features come from VkPhysicalDeviceVulkan13Features
      // (linked above). The standalone structure must not also be in the chain.
      if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
        features_1_3_EXT_subgroup_size_control.Link(supported_features_2,
                                                    device_create_info);
      }
    }
    // VK_KHR_fragment_shader_barycentric / VK_NV_fragment_shader_barycentric.
    if (ext_KHR_fragment_shader_barycentric ||
        ext_NV_fragment_shader_barycentric) {
      features_KHR_fragment_shader_barycentric.Link(supported_features_2,
                                                    device_create_info);
    }
    if (device->extensions_.ext_EXT_device_fault) {
      features_EXT_device_fault.Link(supported_features_2, device_create_info);
    }
    if (device->extensions_.ext_EXT_extended_dynamic_state3) {
      features_EXT_extended_dynamic_state3.Link(supported_features_2,
                                                device_create_info);
      properties_EXT_extended_dynamic_state3.pNext = properties_2.pNext;
      properties_2.pNext = &properties_EXT_extended_dynamic_state3;
    }
    if (device->extensions_.ext_EXT_external_memory_host) {
      properties_EXT_external_memory_host.pNext = properties_2.pNext;
      properties_2.pNext = &properties_EXT_external_memory_host;
    }
    ifn.vkGetPhysicalDeviceProperties2(physical_device, &properties_2);
    ifn.vkGetPhysicalDeviceFeatures2(physical_device, &supported_features_2);
    device->properties_.pipelineExecutableInfoSupported =
        features_KHR_pipeline_executable_properties.supported
            .pipelineExecutableInfo != VK_FALSE;
    if (pipeline_statistics_requested &&
        device->extensions_.ext_KHR_pipeline_executable_properties &&
        device->properties_.pipelineExecutableInfoSupported) {
      features_KHR_pipeline_executable_properties.enabled
          .pipelineExecutableInfo = VK_TRUE;
      features_KHR_pipeline_executable_properties.enabled.pNext =
          const_cast<void*>(device_create_info.pNext);
      device_create_info.pNext =
          &features_KHR_pipeline_executable_properties.enabled;
      device->properties_.pipelineExecutableInfo = true;
    } else if (device->extensions_.ext_KHR_pipeline_executable_properties) {
      // Extension advertisement alone must not change the logical-device
      // extension set when its diagnostic feature cannot be enabled.
      enabled_extensions.erase(
          std::remove_if(
              enabled_extensions.begin(), enabled_extensions.end(),
              [](const char* name) {
                return !std::strcmp(
                    name,
                    VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
              }),
          enabled_extensions.end());
      device_create_info.enabledExtensionCount =
          uint32_t(enabled_extensions.size());
      device_create_info.ppEnabledExtensionNames = enabled_extensions.data();
      device->extensions_.ext_KHR_pipeline_executable_properties = false;
    }
    // Mirror supported deviceFault into the enabled struct so the driver
    // collects fault info during normal execution. Disable the extension flag
    // if the feature wasn't actually supported - vkGetDeviceFaultInfoEXT is
    // only valid to call when deviceFault was enabled at device creation.
    if (device->extensions_.ext_EXT_device_fault &&
        !features_EXT_device_fault.supported.deviceFault) {
      device->extensions_.ext_EXT_device_fault = false;
    }
    features_EXT_device_fault.enabled.deviceFault =
        features_EXT_device_fault.supported.deviceFault;
  }

  uint32_t queue_family_count = 0;
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                               &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(
      physical_device, &queue_family_count, queue_families.data());

  device->queue_families_.resize(queue_family_count);

  uint32_t first_queue_family_graphics_compute_sparse_binding = UINT32_MAX;
  uint32_t first_queue_family_graphics_compute = UINT32_MAX;
  uint32_t first_queue_family_sparse_binding = UINT32_MAX;
  uint32_t first_queue_family_transfer = UINT32_MAX;
  bool has_presentation_queue_family = false;

  for (uint32_t queue_family_index = 0; queue_family_index < queue_family_count;
       ++queue_family_index) {
    QueueFamily& queue_family = device->queue_families_[queue_family_index];
    const VkQueueFamilyProperties& queue_family_properties =
        queue_families[queue_family_index];
    queue_family.physical_queue_count = queue_family_properties.queueCount;
    queue_family.timestamp_valid_bits =
        queue_family_properties.timestampValidBits;

    const VkQueueFlags queue_unsupported_flags =
        ~queue_family_properties.queueFlags;

    if (!(queue_unsupported_flags &
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
      first_queue_family_graphics_compute =
          std::min(queue_family_index, first_queue_family_graphics_compute);
    }

    if (with_gpu_emulation && supported_features.sparseBinding &&
        !(queue_unsupported_flags & VK_QUEUE_SPARSE_BINDING_BIT)) {
      first_queue_family_sparse_binding =
          std::min(queue_family_index, first_queue_family_sparse_binding);
      if (!(queue_unsupported_flags &
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
        first_queue_family_graphics_compute_sparse_binding =
            std::min(queue_family_index,
                     first_queue_family_graphics_compute_sparse_binding);
      }
    }

    // A transfer-only family (transfer supported, graphics and compute not) is
    // the dedicated DMA / copy engine, used for resolve readback copies.
    if (with_gpu_emulation &&
        !(queue_unsupported_flags & VK_QUEUE_TRANSFER_BIT) &&
        (queue_unsupported_flags & VK_QUEUE_GRAPHICS_BIT) &&
        (queue_unsupported_flags & VK_QUEUE_COMPUTE_BIT)) {
      first_queue_family_transfer =
          std::min(queue_family_index, first_queue_family_transfer);
    }

    if (with_swapchain) {
#if XE_PLATFORM_WIN32
      queue_family.may_support_presentation =
          vulkan_instance->extensions().ext_KHR_win32_surface &&
          ifn.vkGetPhysicalDeviceWin32PresentationSupportKHR(
              physical_device, queue_family_index);
#else
      queue_family.may_support_presentation = true;
#endif
      if (queue_family.may_support_presentation) {
        // Device B presents from its graphics/compute family only; an idle
        // queue on another family would be one more KGSL context for nothing.
        if (!presenter_device) {
          queue_family.queues.resize(
              std::max(size_t(1), queue_family.queues.size()));
        }
        has_presentation_queue_family = true;
      }
    }
  }

  if (first_queue_family_graphics_compute == UINT32_MAX) {
    // Not valid according to the Vulkan specification, but for safety.
    XELOGW(
        "Vulkan device '{}' doesn't provide a graphics and compute queue "
        "family",
        properties.deviceName);
    return nullptr;
  }

  if (with_swapchain && !has_presentation_queue_family) {
    XELOGW(
        "Vulkan device '{}' doesn't provide a queue family that supports "
        "presentation",
        properties.deviceName);
    return nullptr;
  }

  // Get the queues to create.

  if (first_queue_family_sparse_binding == UINT32_MAX) {
    // Not valid not to provide a sparse binding queue if the sparseBinding
    // feature is supported according to the Vulkan specification, but for
    // safety and simplicity.
    supported_features.sparseBinding = VK_FALSE;
  }
  if (!supported_features.sparseBinding) {
    supported_features.sparseResidencyBuffer = VK_FALSE;
    supported_features.sparseResidencyImage2D = VK_FALSE;
    supported_features.sparseResidencyImage3D = VK_FALSE;
    supported_features.sparseResidency2Samples = VK_FALSE;
    supported_features.sparseResidency4Samples = VK_FALSE;
    supported_features.sparseResidency8Samples = VK_FALSE;
    supported_features.sparseResidency16Samples = VK_FALSE;
    supported_features.sparseResidencyAliased = VK_FALSE;
  }

  // Prefer using one queue for the normal XenDroid paths. The independent
  // presenter is the one exception: when opted in and physically available,
  // enable queue 1 from the
  // same graphics/compute family so its submissions don't share Vulkan's
  // per-queue host-synchronization mutex with the sovereign Source queue 0.
  // This does not assume that the implementation executes the queues in
  // parallel; it only provides distinct VkQueue host submission domains.

  if (first_queue_family_graphics_compute_sparse_binding != UINT32_MAX) {
    device->queue_family_graphics_compute_ =
        first_queue_family_graphics_compute_sparse_binding;
    device->queue_family_sparse_binding_ =
        first_queue_family_graphics_compute_sparse_binding;
  } else {
    device->queue_family_graphics_compute_ =
        first_queue_family_graphics_compute;
    device->queue_family_sparse_binding_ = first_queue_family_sparse_binding;
  }

  QueueFamily& graphics_compute_queue_family =
      device->queue_families_[device->queue_family_graphics_compute_];
  size_t graphics_compute_queue_count = 1;
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  // Device B: queue 1 for Main Surface egress, so copy/present is its own
  // KGSL context and the Generation/Post backlog on B:0 cannot drag it.
  // Device A keeps XenDroid's normal queue request.
  if (presenter_device) {
    graphics_compute_queue_count = std::min(
        size_t(2), size_t(graphics_compute_queue_family.physical_queue_count));
  }
#endif
  graphics_compute_queue_family.queues.resize(
      std::max(graphics_compute_queue_count,
               graphics_compute_queue_family.queues.size()));
  XELOGI(
      "VulkanDevice: graphics/compute family {} physical queues={} enabled "
      "queues={} ZeroFG presenter queue={} main surface egress queue={}",
      device->queue_family_graphics_compute_,
      graphics_compute_queue_family.physical_queue_count,
      graphics_compute_queue_family.queues.size(),
      device->queue_index_zerofg_presenter(),
      device->queue_index_zerofg_main_surface_present());
  if (device->queue_family_sparse_binding_ != UINT32_MAX) {
    device->queue_families_[device->queue_family_sparse_binding_].queues.resize(
        std::max(size_t(1),
                 device->queue_families_[device->queue_family_sparse_binding_]
                     .queues.size()));
  }
  device->queue_family_transfer_ = first_queue_family_transfer;
  if (device->queue_family_transfer_ != UINT32_MAX) {
    device->queue_families_[device->queue_family_transfer_].queues.resize(
        std::max(size_t(1),
                 device->queue_families_[device->queue_family_transfer_]
                     .queues.size()));
    XELOGI(
        "VulkanDevice: using dedicated transfer queue family {} for readback",
        device->queue_family_transfer_);
  } else {
    XELOGI(
        "VulkanDevice: no dedicated transfer queue family; readback copies "
        "stay "
        "on the graphics queue");
  }

  size_t max_enabled_queues_per_family = 0;
  for (const QueueFamily& queue_family : device->queue_families_) {
    max_enabled_queues_per_family =
        std::max(queue_family.queues.size(), max_enabled_queues_per_family);
  }
  const std::vector<float> queue_priorities(max_enabled_queues_per_family,
                                            1.0f);
  // Device B's work must not wait behind the game's on a saturated GPU: its
  // queues ask the driver for a high global priority (the game stays at the
  // default medium). Turnip's KGSL backend ignores it; the adrenotools hook
  // below covers that driver.
  const bool request_global_priority =
      presenter_device && (global_priority_khr || global_priority_ext);
  VkDeviceQueueGlobalPriorityCreateInfoKHR global_priority_info = {
      VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR};
  global_priority_info.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  for (size_t queue_family_index = 0;
       queue_family_index < device->queue_families_.size();
       ++queue_family_index) {
    const QueueFamily& queue_family =
        device->queue_families_[queue_family_index];
    if (queue_family.queues.empty()) {
      continue;
    }
    VkDeviceQueueCreateInfo& queue_create_info =
        queue_create_infos.emplace_back();
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.pNext =
        request_global_priority ? &global_priority_info : nullptr;
    queue_create_info.flags = 0;
    queue_create_info.queueFamilyIndex = uint32_t(queue_family_index);
    queue_create_info.queueCount = uint32_t(queue_family.queues.size());
    queue_create_info.pQueuePriorities = queue_priorities.data();
  }
  device_create_info.queueCreateInfoCount = uint32_t(queue_create_infos.size());
  device_create_info.pQueueCreateInfos = queue_create_infos.data();

  // Enable needed features and copy the properties.
  //
  // Enabling only actually used features because drivers may take more optimal
  // paths when certain features are disabled. Also, in VK_EXT_shader_object,
  // the state that the application must set for the draw depends on which
  // features are enabled.

  device->properties_.apiVersion = properties.apiVersion;
  device->properties_.driverVersion = properties.driverVersion;
  device->properties_.vendorID = properties.vendorID;
  device->properties_.deviceID = properties.deviceID;
  std::strcpy(device->properties_.deviceName, properties.deviceName);

  XELOGI(
      "Vulkan device '{}': API {}.{}.{} ({}.{} used), vendor 0x{:04X}, device "
      "0x{:04X}, driver version 0x{:X}",
      properties.deviceName, VK_VERSION_MAJOR(unclamped_api_version),
      VK_VERSION_MINOR(unclamped_api_version),
      VK_VERSION_PATCH(properties.apiVersion),
      VK_VERSION_MAJOR(properties.apiVersion),
      VK_VERSION_MINOR(properties.apiVersion), properties.vendorID,
      properties.deviceID, properties.driverVersion);

  XELOGI("Enabled Vulkan device extensions:");
  for (uint32_t enabled_extension_index = 0;
       enabled_extension_index < device_create_info.enabledExtensionCount;
       ++enabled_extension_index) {
    XELOGI("* {}",
           device_create_info.ppEnabledExtensionNames[enabled_extension_index]);
  }

  XELOGI("Vulkan device properties and enabled features:");

  VkPhysicalDeviceFeatures enabled_features = {};
  device_create_info.pEnabledFeatures = &enabled_features;

#define XE_UI_VULKAN_LIMIT(name)                     \
  device->properties_.name = properties.limits.name; \
  XELOGI("* " #name ": {}", properties.limits.name);
#define XE_UI_VULKAN_ENUM_LIMIT(name, type)          \
  device->properties_.name = properties.limits.name; \
  XELOGI("* " #name ": {}", vk::to_string(vk::type(properties.limits.name)));
#define XE_UI_VULKAN_FEATURE(name)                    \
  enabled_features.name = supported_features.name;    \
  device->properties_.name = supported_features.name; \
  if (supported_features.name) {                      \
    XELOGI("* " #name);                               \
  }
#define XE_UI_VULKAN_PROPERTY_2(structure, name) \
  device->properties_.name = structure.name;     \
  XELOGI("* " #name ": {}", structure.name);
#define XE_UI_VULKAN_ENUM_PROPERTY_2(structure, name, type) \
  device->properties_.name = structure.name;                \
  XELOGI("* " #name ": {}", vk::to_string(vk::type(structure.name)));
#define XE_UI_VULKAN_FEATURE_2(structure, name)        \
  structure.enabled.name = structure.supported.name;   \
  device->properties_.name = structure.supported.name; \
  if (structure.supported.name) {                      \
    XELOGI("* " #name);                                \
  }
#define XE_UI_VULKAN_FEATURE_IMPLIED(name) \
  device->properties_.name = true;         \
  XELOGI("* " #name);

  if (ext_1_2_KHR_driver_properties) {
    XE_UI_VULKAN_ENUM_PROPERTY_2(properties_1_2_KHR_driver_properties, driverID,
                                 DriverId);
    XELOGI("* driverName: {}", properties_1_2_KHR_driver_properties.driverName);
    if (properties_1_2_KHR_driver_properties.driverInfo[0]) {
      XELOGI("* driverInfo: {}",
             properties_1_2_KHR_driver_properties.driverInfo);
    }
    XELOGI("* conformanceVersion: {}.{}.{}.{}",
           properties_1_2_KHR_driver_properties.conformanceVersion.major,
           properties_1_2_KHR_driver_properties.conformanceVersion.minor,
           properties_1_2_KHR_driver_properties.conformanceVersion.subminor,
           properties_1_2_KHR_driver_properties.conformanceVersion.patch);
  }

  XE_UI_VULKAN_LIMIT(maxImageDimension2D)
  XE_UI_VULKAN_LIMIT(maxImageDimension3D)
  XE_UI_VULKAN_LIMIT(maxImageDimensionCube)
  XE_UI_VULKAN_LIMIT(maxImageArrayLayers)
  XE_UI_VULKAN_LIMIT(maxStorageBufferRange)
  if (cvars::vulkan_clamp_storage_buffer_range > 0) {
    device->properties_.maxStorageBufferRange =
        std::min(device->properties_.maxStorageBufferRange,
                 uint32_t(cvars::vulkan_clamp_storage_buffer_range));
    XELOGI("* maxStorageBufferRange clamped to {} by configuration",
           device->properties_.maxStorageBufferRange);
  }
  XE_UI_VULKAN_LIMIT(maxSamplerAllocationCount)
  XE_UI_VULKAN_LIMIT(maxPerStageDescriptorSamplers)
  XE_UI_VULKAN_LIMIT(maxPerStageDescriptorStorageBuffers)
  XE_UI_VULKAN_LIMIT(maxPerStageDescriptorSampledImages)
  XE_UI_VULKAN_LIMIT(maxPerStageResources)
  XE_UI_VULKAN_LIMIT(maxVertexOutputComponents)
  XE_UI_VULKAN_LIMIT(maxTessellationEvaluationOutputComponents)
  XE_UI_VULKAN_LIMIT(maxGeometryInputComponents)
  XE_UI_VULKAN_LIMIT(maxGeometryOutputComponents)
  XE_UI_VULKAN_LIMIT(maxFragmentInputComponents)
  XE_UI_VULKAN_LIMIT(maxFragmentCombinedOutputResources)
  XE_UI_VULKAN_LIMIT(maxSamplerAnisotropy)
  XE_UI_VULKAN_LIMIT(timestampPeriod)
  XE_UI_VULKAN_LIMIT(maxViewportDimensions[0])
  XE_UI_VULKAN_LIMIT(maxViewportDimensions[1])
  XE_UI_VULKAN_LIMIT(minUniformBufferOffsetAlignment)
  XE_UI_VULKAN_LIMIT(maxUniformBufferRange)
  XE_UI_VULKAN_LIMIT(maxDescriptorSetUniformBuffersDynamic)
  XE_UI_VULKAN_LIMIT(minStorageBufferOffsetAlignment)
  XE_UI_VULKAN_LIMIT(maxFramebufferWidth)
  XE_UI_VULKAN_LIMIT(maxFramebufferHeight)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferColorSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferDepthSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferStencilSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(framebufferNoAttachmentsSampleCounts,
                          SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageColorSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageIntegerSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageDepthSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_ENUM_LIMIT(sampledImageStencilSampleCounts, SampleCountFlags)
  XE_UI_VULKAN_LIMIT(standardSampleLocations)
  XE_UI_VULKAN_LIMIT(optimalBufferCopyOffsetAlignment)
  XE_UI_VULKAN_LIMIT(optimalBufferCopyRowPitchAlignment)
  XE_UI_VULKAN_LIMIT(nonCoherentAtomSize)

  if (with_gpu_emulation) {
    XE_UI_VULKAN_FEATURE(robustBufferAccess)
    XE_UI_VULKAN_FEATURE(fullDrawIndexUint32)
    XE_UI_VULKAN_FEATURE(independentBlend)
    XE_UI_VULKAN_FEATURE(geometryShader)
    XE_UI_VULKAN_FEATURE(tessellationShader)
    XE_UI_VULKAN_FEATURE(sampleRateShading)
    XE_UI_VULKAN_FEATURE(depthClamp)
    XE_UI_VULKAN_FEATURE(fillModeNonSolid)
    XE_UI_VULKAN_FEATURE(samplerAnisotropy)
    XE_UI_VULKAN_FEATURE(occlusionQueryPrecise)
    XE_UI_VULKAN_FEATURE(vertexPipelineStoresAndAtomics)
    XE_UI_VULKAN_FEATURE(fragmentStoresAndAtomics)
    XE_UI_VULKAN_FEATURE(shaderClipDistance)
    XE_UI_VULKAN_FEATURE(shaderCullDistance)
    XE_UI_VULKAN_FEATURE(shaderInt16)
    XE_UI_VULKAN_FEATURE(sparseBinding)
    XE_UI_VULKAN_FEATURE(sparseResidencyBuffer)
  }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if ((with_gpu_emulation && IsZeroFGRequested()) || presenter_device) {
    XE_UI_VULKAN_FEATURE(shaderStorageImageExtendedFormats)
  }
#endif
  if (presenter_device) {
    // Existing host Post system shaders declare Int16/Float16. This is a
    // host-pipeline requirement, not a new RC1 Compat algorithm capability.
    XE_UI_VULKAN_FEATURE(shaderInt16)
  }

  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
    XE_UI_VULKAN_FEATURE_2(features_1_2, timelineSemaphore);
    if (presenter_device) {
      XE_UI_VULKAN_FEATURE_2(features_1_2, shaderFloat16);
    }
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_1_2, samplerMirrorClampToEdge);
      XE_UI_VULKAN_FEATURE_2(features_1_2, uniformBufferStandardLayout);
      XE_UI_VULKAN_FEATURE_2(features_1_2, scalarBlockLayout);
      XE_UI_VULKAN_FEATURE_2(features_1_2, hostQueryReset);
      XE_UI_VULKAN_FEATURE_2(features_1_2, shaderFloat16);
    }
  } else {
    if (ext_1_2_KHR_timeline_semaphore) {
      XE_UI_VULKAN_FEATURE_2(features_KHR_timeline_semaphore,
                             timelineSemaphore);
    }
    if (ext_1_2_KHR_sampler_mirror_clamp_to_edge) {
      XE_UI_VULKAN_FEATURE_IMPLIED(samplerMirrorClampToEdge)
    }
    if (ext_1_2_EXT_host_query_reset && with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_host_query_reset, hostQueryReset);
    }
    if (ext_1_2_KHR_shader_float16_int8 && with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_KHR_shader_float16_int8, shaderFloat16);
    }
  }
  device->extensions_.ext_1_2_EXT_host_query_reset =
      ext_1_2_EXT_host_query_reset;
  device->extensions_.ext_1_2_KHR_timeline_semaphore =
      ext_1_2_KHR_timeline_semaphore;

  // shaderDrawParameters (Vulkan 1.1). Needed by shaders reading SV_VertexID
  // with Direct3D semantics, which are compiled to VertexIndex minus BaseVertex
  // (declaring the DrawParameters SPIR-V capability). The guest output triangle
  // strip vertex shader used by the presenter is one such shader.
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    features_1_1.enabled.shaderDrawParameters =
        features_1_1.supported.shaderDrawParameters;
    if (features_1_1.supported.shaderDrawParameters) {
      XELOGI("* shaderDrawParameters");
    }
  }

  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_1_3, shaderDemoteToHelperInvocation);
      XE_UI_VULKAN_FEATURE_2(features_1_3, dynamicRendering);
    }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
    if ((with_gpu_emulation && IsZeroFGRequested()) || presenter_device) {
      XE_UI_VULKAN_FEATURE_2(features_1_3, synchronization2);
    }
#endif
  } else {
    if (ext_1_3_EXT_shader_demote_to_helper_invocation) {
      if (with_gpu_emulation) {
        XE_UI_VULKAN_FEATURE_2(
            features_1_3_EXT_shader_demote_to_helper_invocation,
            shaderDemoteToHelperInvocation);
      }
    }
    if (ext_1_3_KHR_dynamic_rendering) {
      if (with_gpu_emulation) {
        XE_UI_VULKAN_FEATURE_2(features_1_3_KHR_dynamic_rendering,
                               dynamicRendering);
      }
    }
  }

  if (device->extensions_.ext_1_4_KHR_dynamic_rendering_local_read &&
      with_gpu_emulation) {
    XE_UI_VULKAN_FEATURE_2(features_1_4_KHR_dynamic_rendering_local_read,
                           dynamicRenderingLocalRead);
  }

  if (ext_KHR_portability_subset) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset,
                             constantAlphaColorBlendFactors)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset,
                             imageViewFormatReinterpretation)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset,
                             imageViewFormatSwizzle)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset, pointPolygons)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset,
                             separateStencilMaskRef)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset,
                             shaderSampleRateInterpolationFunctions)
      XE_UI_VULKAN_FEATURE_2(features_KHR_portability_subset, triangleFans)
    }
  } else {
    // Not a portability subset device.
    XE_UI_VULKAN_FEATURE_IMPLIED(constantAlphaColorBlendFactors)
    XE_UI_VULKAN_FEATURE_IMPLIED(imageViewFormatReinterpretation)
    XE_UI_VULKAN_FEATURE_IMPLIED(imageViewFormatSwizzle)
    XE_UI_VULKAN_FEATURE_IMPLIED(pointPolygons)
    XE_UI_VULKAN_FEATURE_IMPLIED(separateStencilMaskRef)
    XE_UI_VULKAN_FEATURE_IMPLIED(shaderSampleRateInterpolationFunctions)
    XE_UI_VULKAN_FEATURE_IMPLIED(triangleFans)
  }

  if (ext_1_2_KHR_shader_float_controls) {
    XE_UI_VULKAN_PROPERTY_2(properties_1_2_KHR_shader_float_controls,
                            shaderSignedZeroInfNanPreserveFloat32);
    XE_UI_VULKAN_PROPERTY_2(properties_1_2_KHR_shader_float_controls,
                            shaderDenormFlushToZeroFloat32);
    XE_UI_VULKAN_PROPERTY_2(properties_1_2_KHR_shader_float_controls,
                            shaderRoundingModeRTEFloat32);
  }

  if (ext_EXT_fragment_shader_interlock) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_fragment_shader_interlock,
                             fragmentShaderSampleInterlock)
      XE_UI_VULKAN_FEATURE_2(features_EXT_fragment_shader_interlock,
                             fragmentShaderPixelInterlock)
    }
  }

  if (ext_EXT_non_seamless_cube_map) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_non_seamless_cube_map,
                             nonSeamlessCubeMap)
    }
  }

  if (ext_EXT_custom_border_color) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_custom_border_color,
                             customBorderColors)
      XE_UI_VULKAN_FEATURE_2(features_EXT_custom_border_color,
                             customBorderColorWithoutFormat)
    }
  }

  // Vulkan 1.1 core subgroup properties.
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    XE_UI_VULKAN_PROPERTY_2(properties_subgroup, subgroupSize);
    device->properties_.subgroupSupportedStages =
        properties_subgroup.supportedStages;
    XELOGI("* subgroupSupportedStages: {}",
           vk::to_string(
               vk::ShaderStageFlags(properties_subgroup.supportedStages)));
    device->properties_.subgroupSupportedOperations =
        properties_subgroup.supportedOperations;
    XELOGI("* subgroupSupportedOperations: {}",
           vk::to_string(vk::SubgroupFeatureFlags(
               properties_subgroup.supportedOperations)));
  }

  // VK_EXT_subgroup_size_control (#226, promoted to 1.3).
  if (ext_1_3_EXT_subgroup_size_control) {
    XE_UI_VULKAN_PROPERTY_2(properties_1_3_EXT_subgroup_size_control,
                            minSubgroupSize);
    XE_UI_VULKAN_PROPERTY_2(properties_1_3_EXT_subgroup_size_control,
                            maxSubgroupSize);
    XE_UI_VULKAN_PROPERTY_2(properties_1_3_EXT_subgroup_size_control,
                            maxComputeWorkgroupSubgroups);
    if (with_gpu_emulation) {
      // On Vulkan 1.3 these are enabled through
      // VkPhysicalDeviceVulkan13Features.
      if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
        XE_UI_VULKAN_FEATURE_2(features_1_3, subgroupSizeControl);
        XE_UI_VULKAN_FEATURE_2(features_1_3, computeFullSubgroups);
      } else {
        XE_UI_VULKAN_FEATURE_2(features_1_3_EXT_subgroup_size_control,
                               subgroupSizeControl);
        XE_UI_VULKAN_FEATURE_2(features_1_3_EXT_subgroup_size_control,
                               computeFullSubgroups);
      }
    }
  }
  device->extensions_.ext_1_3_EXT_subgroup_size_control =
      ext_1_3_EXT_subgroup_size_control;

  // VK_EXT_external_memory_host (#179).
  if (device->extensions_.ext_EXT_external_memory_host) {
    XE_UI_VULKAN_PROPERTY_2(properties_EXT_external_memory_host,
                            minImportedHostPointerAlignment);
  }

  // VK_KHR_fragment_shader_barycentric (#322) /
  // VK_NV_fragment_shader_barycentric (#203).
  // MoltenVK advertises this, but SPIRV-Cross can't translate Xenia's
  // PerVertexKHR usage to MSL, so leave the shader path on its fallback.
  const bool driver_is_moltenvk =
      device->properties_.driverID == VK_DRIVER_ID_MOLTENVK;
  if ((ext_KHR_fragment_shader_barycentric ||
       ext_NV_fragment_shader_barycentric) &&
      !driver_is_moltenvk) {
    if (with_gpu_emulation) {
      XE_UI_VULKAN_FEATURE_2(features_KHR_fragment_shader_barycentric,
                             fragmentShaderBarycentric);
    }
  }

  device->extensions_.ext_KHR_fragment_shader_barycentric =
      (ext_KHR_fragment_shader_barycentric ||
       ext_NV_fragment_shader_barycentric) &&
      !driver_is_moltenvk;

  // Extended dynamic state. EDS1 (#268) and EDS2 (#378) are promoted to Vulkan
  // 1.3 core - always available on a 1.3 device. EDS3 (#456) is a standalone
  // extension with per-sub-feature bools; mirror only the sub-features Xenia
  // actually makes dynamic into the enabled struct (leaving the rest disabled
  // avoids requesting unsupported sub-features at device creation).
  if (with_gpu_emulation) {
    device->properties_.extendedDynamicState =
        properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0);
    if (device->properties_.extendedDynamicState) {
      XELOGI("* extendedDynamicState (core 1.3)");
    }
    if (device->extensions_.ext_EXT_extended_dynamic_state3 &&
        device->properties_.extendedDynamicState) {
      XE_UI_VULKAN_FEATURE_2(features_EXT_extended_dynamic_state3,
                             extendedDynamicState3DepthClampEnable);
      XE_UI_VULKAN_FEATURE_2(features_EXT_extended_dynamic_state3,
                             extendedDynamicState3PolygonMode);
      XE_UI_VULKAN_FEATURE_2(features_EXT_extended_dynamic_state3,
                             extendedDynamicState3ColorBlendEnable);
      XE_UI_VULKAN_FEATURE_2(features_EXT_extended_dynamic_state3,
                             extendedDynamicState3ColorBlendEquation);
      XE_UI_VULKAN_FEATURE_2(features_EXT_extended_dynamic_state3,
                             extendedDynamicState3ColorWriteMask);
    }
  }
  // dynamicPrimitiveTopologyUnrestricted is a property of the extension, not a
  // feature to enable - read it from the properties structure (valid only when
  // the extension is present and queried above). When false, dynamic topology
  // must stay within its class, so the class is kept in the pipeline key.
  if (device->extensions_.ext_EXT_extended_dynamic_state3) {
    device->properties_.extendedDynamicState3PrimitiveTopologyUnrestricted =
        properties_EXT_extended_dynamic_state3
            .dynamicPrimitiveTopologyUnrestricted;
  }

#undef XE_UI_VULKAN_LIMIT
#undef XE_UI_VULKAN_ENUM_LIMIT
#undef XE_UI_VULKAN_FEATURE
#undef XE_UI_VULKAN_PROPERTY_2
#undef XE_UI_VULKAN_ENUM_PROPERTY_2
#undef XE_UI_VULKAN_FEATURE_2

  if (presenter_device) {
    backend.synchronization2_enabled = device->properties_.synchronization2;
    backend.shader_storage_image_extended_formats_enabled =
        device->properties_.shaderStorageImageExtendedFormats;
    backend.pipeline_executable_extension_advertised =
        device->properties_.pipelineExecutablePropertiesAdvertised;
    backend.pipeline_executable_feature_supported =
        device->properties_.pipelineExecutableInfoSupported;
    backend.pipeline_executable_feature_enabled =
        device->properties_.pipelineExecutableInfo;
    backend.shader_float16_supported = features_1_2.supported.shaderFloat16;
    backend.shader_float16_enabled = features_1_2.enabled.shaderFloat16;
    backend.shader_int8_supported = features_1_2.supported.shaderInt8;
    backend.shader_int8_enabled = features_1_2.enabled.shaderInt8;
    backend.shader_int16_supported = supported_features.shaderInt16;
    backend.shader_int16_enabled = enabled_features.shaderInt16;
    backend.storage_buffer_16bit_access_supported =
        features_1_1.supported.storageBuffer16BitAccess;
    // A backend request must not implicitly enable unrelated storage features.
    backend.storage_buffer_16bit_access_enabled =
        features_1_1.enabled.storageBuffer16BitAccess;
    backend.subgroup_size = properties_subgroup.subgroupSize;
    backend.subgroup_min_size =
        properties_1_3_EXT_subgroup_size_control.minSubgroupSize;
    backend.subgroup_max_size =
        properties_1_3_EXT_subgroup_size_control.maxSubgroupSize;
    backend.max_compute_workgroup_subgroups =
        properties_1_3_EXT_subgroup_size_control.maxComputeWorkgroupSubgroups;
    backend.subgroup_supported_stages = properties_subgroup.supportedStages;
    backend.subgroup_supported_operations =
        properties_subgroup.supportedOperations;
    backend.subgroup_size_control_supported =
        features_1_3.supported.subgroupSizeControl;
    backend.compute_full_subgroups_supported =
        features_1_3.supported.computeFullSubgroups;
    backend.required_subgroup_size_stages =
        properties_1_3_EXT_subgroup_size_control.requiredSubgroupSizeStages;
    backend.maintenance4_supported = features_1_3.supported.maintenance4;
    backend.maintenance4_enabled = features_1_3.enabled.maintenance4;
    backend.subgroup_size_control_enabled =
        features_1_3.enabled.subgroupSizeControl;
    backend.compute_full_subgroups_enabled =
        features_1_3.enabled.computeFullSubgroups;
    device->properties_.subgroupSizeControl =
        backend.subgroup_size_control_enabled;
    device->properties_.computeFullSubgroups =
        backend.compute_full_subgroups_enabled;
    backend.shader_integer_dot_product_supported =
        features_1_3.supported.shaderIntegerDotProduct;
    backend.shader_integer_dot_product_enabled =
        features_1_3.enabled.shaderIntegerDotProduct;
    backend.integer_dot_product_4x8_unsigned_accelerated =
        properties_dot_product.integerDotProduct4x8BitPackedUnsignedAccelerated;
    backend.integer_dot_product_4x8_signed_accelerated =
        properties_dot_product.integerDotProduct4x8BitPackedSignedAccelerated;
    backend.integer_dot_product_4x8_mixed_signedness_accelerated =
        properties_dot_product
            .integerDotProduct4x8BitPackedMixedSignednessAccelerated;
    backend.sampler_filter_minmax_supported =
        features_1_2.supported.samplerFilterMinmax;
    backend.sampler_filter_minmax_enabled =
        features_1_2.enabled.samplerFilterMinmax;
    backend.filter_minmax_single_component_formats =
        properties_minmax.filterMinmaxSingleComponentFormats;
    backend.filter_minmax_image_component_mapping =
        properties_minmax.filterMinmaxImageComponentMapping;
    backend.qcom_texture_sample_weighted_supported =
        features_qcom.textureSampleWeighted;
    backend.qcom_texture_box_filter_supported = features_qcom.textureBoxFilter;
    backend.qcom_texture_block_match_supported = features_qcom.textureBlockMatch;
    enabled_qcom.textureBlockMatch = backend_requests.qcom_block_match &&
        features_qcom.textureBlockMatch;
    // Weighted sampling and the box filter are queried only; block match is
    // the one QCOM image-processing feature a backend requests.
    if (ext_QCOM_image_processing && enabled_qcom.textureBlockMatch) {
      enabled_qcom.pNext = const_cast<void*>(device_create_info.pNext);
      device_create_info.pNext = &enabled_qcom;
      backend.qcom_image_processing_enabled = true;
    } else if (ext_QCOM_image_processing) {
      // No usable feature: leave advertisement visible but do not enable it.
      enabled_extensions.erase(
          std::remove_if(enabled_extensions.begin(), enabled_extensions.end(),
                         [](const char* name) {
                           return !std::strcmp(
                               name, VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME);
                         }),
          enabled_extensions.end());
    }
    backend.qcom_texture_sample_weighted_enabled =
        enabled_qcom.textureSampleWeighted;
    backend.qcom_texture_box_filter_enabled = enabled_qcom.textureBoxFilter;
    backend.qcom_texture_block_match_enabled = enabled_qcom.textureBlockMatch;
    backend.qcom_max_weight_filter_phases = properties_qcom.maxWeightFilterPhases;
    backend.qcom_max_weight_filter_dimension =
        properties_qcom.maxWeightFilterDimension;
    backend.qcom_max_block_match_region = properties_qcom.maxBlockMatchRegion;
    backend.qcom_max_box_filter_block_size = properties_qcom.maxBoxFilterBlockSize;
    backend.qcom_texture_block_match2_supported = features_qcom2.textureBlockMatch2;
    backend.qcom_max_block_match_window = properties_qcom2.maxBlockMatchWindow;
    backend.qcom_selectable_cubic_weights_supported =
        features_cubic_weights.selectableCubicWeights;
    backend.qcom_cubic_range_clamp_supported = features_cubic_clamp.cubicRangeClamp;
    device_create_info.enabledExtensionCount = uint32_t(enabled_extensions.size());
    device_create_info.ppEnabledExtensionNames = enabled_extensions.data();
  }

  // Create the device.

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  // ZeroFG's device asks KGSL for a higher priority than the game's (KGSL's
  // default 8): on a saturated GPU its small work then runs first instead of
  // waiting behind the game's frames and completing in clumps (a copy of 8 ms
  // took 40-170 ms, Arkham GT off, 2026-10-07). Turnip creates its KGSL draw
  // context without one, so the adrenotools hook sets it during this call.
  // 4 is the second of the four levels: above the game, below the top level.
  constexpr uint32_t kZeroFGKgslContextPriority = 4;
  const uint32_t contexts_raised_before =
      presenter_device ? adrenotools_context_priority_raised() : 0;
  if (presenter_device) {
    adrenotools_set_context_priority(kZeroFGKgslContextPriority);
  }
#endif
  VkResult device_create_result = ifn.vkCreateDevice(
      physical_device, &device_create_info, nullptr, &device->device_);
  const char* global_priority_state =
      request_global_priority ? "high" : "unsupported";
  if (request_global_priority &&
      (device_create_result == VK_ERROR_NOT_PERMITTED_KHR ||
       device_create_result == VK_ERROR_INITIALIZATION_FAILED)) {
    // A driver may refuse a high priority to an application: B at the
    // default priority is still B.
    global_priority_state = "refused";
    for (VkDeviceQueueCreateInfo& queue_create_info : queue_create_infos) {
      queue_create_info.pNext = nullptr;
    }
    device_create_result = ifn.vkCreateDevice(
        physical_device, &device_create_info, nullptr, &device->device_);
  }
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (presenter_device) {
    adrenotools_set_context_priority(0);
    XELOGI(
        "ZeroFGDeviceB vk_global_priority={} kgsl_priority requested={} "
        "contexts_raised={}",
        global_priority_state, kZeroFGKgslContextPriority,
        adrenotools_context_priority_raised() - contexts_raised_before);
  }
#else
  if (presenter_device) {
    XELOGI("ZeroFGDeviceB vk_global_priority={}", global_priority_state);
  }
#endif
  if (device_create_result != VK_SUCCESS) {
    XELOGE(
        "Failed to create a Vulkan logical device from the physical device "
        "'{}': {}",
        properties.deviceName, vk::to_string(vk::Result(device_create_result)));
    return nullptr;
  }

  // Load device functions.

  bool functions_loaded = true;

  Functions& dfn = device->functions_;

#define XE_UI_VULKAN_FUNCTION(name)                                   \
  functions_loaded &= (dfn.name = PFN_##name(ifn.vkGetDeviceProcAddr( \
                           device->device_, #name))) != nullptr;

  // Vulkan 1.0.
#include "xenia/ui/vulkan/functions/device_1_0.inc"

  // Extensions promoted to a Vulkan version supported by the device.
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name) \
  functions_loaded &=                                             \
      (dfn.core_name = PFN_##core_name(                           \
           ifn.vkGetDeviceProcAddr(device->device_, #core_name))) != nullptr;
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)) {
#include "xenia/ui/vulkan/functions/device_1_1_khr_bind_memory2.inc"
#include "xenia/ui/vulkan/functions/device_1_1_khr_get_memory_requirements2.inc"
  }
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)) {
#include "xenia/ui/vulkan/functions/device_1_2_ext_host_query_reset.inc"
  }
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 3, 0)) {
#include "xenia/ui/vulkan/functions/device_1_3_khr_dynamic_rendering.inc"
#include "xenia/ui/vulkan/functions/device_1_3_khr_maintenance4.inc"
    // VK_EXT_extended_dynamic_state (#268) + VK_EXT_extended_dynamic_state2
    // (#378) setters and the core topology/restart setters, all promoted to
    // 1.3 core and loaded by their core name.
    if (with_gpu_emulation) {
#include "xenia/ui/vulkan/functions/device_ext_extended_dynamic_state.inc"
#include "xenia/ui/vulkan/functions/device_1_3_core_dynamic_topology.inc"
    }
  }
  if (properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 4, 0)) {
#include "xenia/ui/vulkan/functions/device_1_4_khr_dynamic_rendering_local_read.inc"
  }
#undef XE_UI_VULKAN_FUNCTION_PROMOTED

  // Non-promoted extensions, and extensions promoted to a Vulkan version not
  // supported by the device.
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name) \
  functions_loaded &=                                             \
      (dfn.core_name = PFN_##core_name(ifn.vkGetDeviceProcAddr(   \
           device->device_, #extension_name))) != nullptr;
  if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 1, 0)) {
    if (device->extensions_.ext_1_1_KHR_get_memory_requirements2) {
#include "xenia/ui/vulkan/functions/device_1_1_khr_get_memory_requirements2.inc"
    }
    if (device->extensions_.ext_1_1_KHR_bind_memory2) {
#include "xenia/ui/vulkan/functions/device_1_1_khr_bind_memory2.inc"
    }
  }
  if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 2, 0)) {
    if (device->extensions_.ext_1_2_EXT_host_query_reset) {
#include "xenia/ui/vulkan/functions/device_1_2_ext_host_query_reset.inc"
    }
  }
  if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) {
    if (device->extensions_.ext_1_3_KHR_maintenance4) {
#include "xenia/ui/vulkan/functions/device_1_3_khr_maintenance4.inc"
    }
    if (device->extensions_.ext_1_3_KHR_dynamic_rendering) {
#include "xenia/ui/vulkan/functions/device_1_3_khr_dynamic_rendering.inc"
    }
  }
  if (properties.apiVersion < VK_MAKE_API_VERSION(0, 1, 4, 0)) {
    if (device->extensions_.ext_1_4_KHR_dynamic_rendering_local_read) {
#include "xenia/ui/vulkan/functions/device_1_4_khr_dynamic_rendering_local_read.inc"
    }
  }
  if (device->extensions_.ext_KHR_swapchain) {
#include "xenia/ui/vulkan/functions/device_khr_swapchain.inc"
  }
#undef XE_UI_VULKAN_FUNCTION_PROMOTED

#undef XE_UI_VULKAN_FUNCTION

  // VK_EXT_extended_dynamic_state3 (#456) is optional and never promoted - load
  // its EXT-named setters non-fatally (like vkGetDeviceFaultInfoEXT). If any are
  // missing, disable the extension so the GPU backend bakes those fields into
  // the pipeline key instead of trying to call null function pointers.
  if (device->extensions_.ext_EXT_extended_dynamic_state3) {
    bool eds3_functions_loaded = true;
#define XE_UI_VULKAN_FUNCTION(name)                                   \
  eds3_functions_loaded &= (dfn.name = PFN_##name(ifn.vkGetDeviceProcAddr( \
                               device->device_, #name))) != nullptr;
#include "xenia/ui/vulkan/functions/device_ext_extended_dynamic_state3.inc"
#undef XE_UI_VULKAN_FUNCTION
    if (!eds3_functions_loaded) {
      device->extensions_.ext_EXT_extended_dynamic_state3 = false;
      device->properties_.extendedDynamicState3DepthClampEnable = false;
      device->properties_.extendedDynamicState3PolygonMode = false;
      device->properties_.extendedDynamicState3ColorBlendEnable = false;
      device->properties_.extendedDynamicState3ColorBlendEquation = false;
      device->properties_.extendedDynamicState3ColorWriteMask = false;
      device->properties_
          .extendedDynamicState3PrimitiveTopologyUnrestricted = false;
    }
  }

  // Optional fault-info function pointer - failing to load is not fatal.
  if (device->extensions_.ext_EXT_device_fault) {
    device->vkGetDeviceFaultInfoEXT_ = PFN_vkGetDeviceFaultInfoEXT(
        ifn.vkGetDeviceProcAddr(device->device_, "vkGetDeviceFaultInfoEXT"));
    if (!device->vkGetDeviceFaultInfoEXT_) {
      device->extensions_.ext_EXT_device_fault = false;
    }
  }

  // Optional host-pointer import function - failing to load disables zero-copy.
  if (device->extensions_.ext_EXT_external_memory_host) {
    device->vkGetMemoryHostPointerPropertiesEXT_ =
        PFN_vkGetMemoryHostPointerPropertiesEXT(ifn.vkGetDeviceProcAddr(
            device->device_, "vkGetMemoryHostPointerPropertiesEXT"));
    if (!device->vkGetMemoryHostPointerPropertiesEXT_) {
      device->extensions_.ext_EXT_external_memory_host = false;
    }
  }

  // Observational display-timing entry points. Missing pointers only disable
  // the probe; they must not make logical-device creation fail.
  if (ext_GOOGLE_display_timing) {
    device->vkGetRefreshCycleDurationGOOGLE_ =
        PFN_vkGetRefreshCycleDurationGOOGLE(ifn.vkGetDeviceProcAddr(
            device->device_, "vkGetRefreshCycleDurationGOOGLE"));
    device->vkGetPastPresentationTimingGOOGLE_ =
        PFN_vkGetPastPresentationTimingGOOGLE(ifn.vkGetDeviceProcAddr(
            device->device_, "vkGetPastPresentationTimingGOOGLE"));
    if (!device->vkGetRefreshCycleDurationGOOGLE_ ||
        !device->vkGetPastPresentationTimingGOOGLE_) {
      device->vkGetRefreshCycleDurationGOOGLE_ = nullptr;
      device->vkGetPastPresentationTimingGOOGLE_ = nullptr;
    }
  }

  // Optional diagnostic entry points. Failure here disables statistics only:
  // it must not join the generic device function-loading failure authority.
  if (device->properties_.pipelineExecutableInfo) {
    device->vkGetPipelineExecutablePropertiesKHR_ =
        PFN_vkGetPipelineExecutablePropertiesKHR(ifn.vkGetDeviceProcAddr(
            device->device_, "vkGetPipelineExecutablePropertiesKHR"));
    device->vkGetPipelineExecutableStatisticsKHR_ =
        PFN_vkGetPipelineExecutableStatisticsKHR(ifn.vkGetDeviceProcAddr(
            device->device_, "vkGetPipelineExecutableStatisticsKHR"));
    if (!device->vkGetPipelineExecutablePropertiesKHR_ ||
        !device->vkGetPipelineExecutableStatisticsKHR_) {
      device->vkGetPipelineExecutablePropertiesKHR_ = nullptr;
      device->vkGetPipelineExecutableStatisticsKHR_ = nullptr;
      XELOGW("ZeroFGPipelineStats unavailable reason=query_entry_points");
    }
  }

  device->vkDeviceWaitIdle_ = PFN_vkDeviceWaitIdle(
      ifn.vkGetDeviceProcAddr(device->device_, "vkDeviceWaitIdle"));
  if (device->properties_.synchronization2) {
    device->vkCmdPipelineBarrier2_ = PFN_vkCmdPipelineBarrier2(
        ifn.vkGetDeviceProcAddr(device->device_, "vkCmdPipelineBarrier2"));
  }
  if (device->properties_.timelineSemaphore) {
    const char* counter_function_name =
        properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)
            ? "vkGetSemaphoreCounterValue"
            : "vkGetSemaphoreCounterValueKHR";
    device->vkGetSemaphoreCounterValue_ = PFN_vkGetSemaphoreCounterValue(
        ifn.vkGetDeviceProcAddr(device->device_, counter_function_name));
    const char* wait_function_name =
        properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 2, 0)
            ? "vkWaitSemaphores"
            : "vkWaitSemaphoresKHR";
    device->vkWaitSemaphores_ = PFN_vkWaitSemaphores(
        ifn.vkGetDeviceProcAddr(device->device_, wait_function_name));
    if (!device->vkGetSemaphoreCounterValue_ ||
        !device->vkWaitSemaphores_) {
      device->properties_.timelineSemaphore = false;
    }
  }

#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  if (device->extensions_.ext_KHR_external_semaphore_fd) {
    device->vkGetSemaphoreFdKHR_ = PFN_vkGetSemaphoreFdKHR(
        ifn.vkGetDeviceProcAddr(device->device_, "vkGetSemaphoreFdKHR"));
    device->vkImportSemaphoreFdKHR_ = PFN_vkImportSemaphoreFdKHR(
        ifn.vkGetDeviceProcAddr(device->device_, "vkImportSemaphoreFdKHR"));
    if (!device->vkGetSemaphoreFdKHR_) {
      device->extensions_.ext_KHR_external_semaphore_fd = false;
    }
  }
  if (device->extensions_.ext_ANDROID_external_memory_android_hardware_buffer) {
    device->vkGetAndroidHardwareBufferPropertiesANDROID_ =
        PFN_vkGetAndroidHardwareBufferPropertiesANDROID(
            ifn.vkGetDeviceProcAddr(
                device->device_,
                "vkGetAndroidHardwareBufferPropertiesANDROID"));
    const char* image_format_properties_function_name =
        properties.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0)
            ? "vkGetPhysicalDeviceImageFormatProperties2"
            : "vkGetPhysicalDeviceImageFormatProperties2KHR";
    device->vkGetPhysicalDeviceImageFormatProperties2_ =
        PFN_vkGetPhysicalDeviceImageFormatProperties2(
            ifn.vkGetInstanceProcAddr(vulkan_instance->instance(),
                                      image_format_properties_function_name));
    if (!device->vkGetAndroidHardwareBufferPropertiesANDROID_ ||
        !device->vkGetPhysicalDeviceImageFormatProperties2_) {
      device->extensions_.ext_ANDROID_external_memory_android_hardware_buffer =
          false;
    }
  }
#endif

  if (!functions_loaded) {
    XELOGE("Failed to get all Vulkan device function pointers for '{}'",
           properties.deviceName);
    return nullptr;
  }

  if (presenter_device) {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
    const auto& p = device->properties_;
    if (!p.synchronization2 || !device->vkCmdPipelineBarrier2_ ||
        !device->vkDeviceWaitIdle_ || !p.timelineSemaphore || !p.shaderInt16 ||
        !p.shaderFloat16 || !features_1_1.enabled.shaderDrawParameters ||
        !device->vkImportSemaphoreFdKHR_ || !device->vkGetSemaphoreFdKHR_ ||
        !device->extensions_
             .ext_ANDROID_external_memory_android_hardware_buffer) {
      XELOGE("ZeroFGDeviceB required host feature/extension unavailable");
      return nullptr;
    }
    if (!device->extensions_.ext_KHR_swapchain) {
      XELOGE("ZeroFGDeviceB without VK_KHR_swapchain");
      return nullptr;
    }
    XELOGI(
        "ZeroFGDeviceB enabled sync2={} timeline={} int16={} float16={} "
        "storage_extended={} draw_parameters={} queues={} wsi={} "
        "display_timing={}",
        p.synchronization2, p.timelineSemaphore, p.shaderInt16, p.shaderFloat16,
        p.shaderStorageImageExtendedFormats,
        bool(features_1_1.enabled.shaderDrawParameters),
        device->queue_families_[device->queue_family_graphics_compute_]
            .queues.size(),
        device->extensions_.ext_KHR_swapchain,
        device->vkGetPastPresentationTimingGOOGLE_ != nullptr);
#else
    return nullptr;
#endif
  }

  // Get the queues.

  for (size_t queue_family_index = 0;
       queue_family_index < device->queue_families_.size();
       ++queue_family_index) {
    QueueFamily& queue_family = device->queue_families_[queue_family_index];
    for (size_t queue_index = 0; queue_index < queue_family.queues.size();
         ++queue_index) {
      VkQueue queue;
      dfn.vkGetDeviceQueue(device->device_, uint32_t(queue_family_index),
                           uint32_t(queue_index), &queue);
      queue_family.queues[queue_index] = std::make_unique<Queue>(queue);
    }
  }

  // Get the memory types.

  VkPhysicalDeviceMemoryProperties memory_properties;
  ifn.vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
  for (uint32_t memory_type_index = 0;
       memory_type_index < memory_properties.memoryTypeCount;
       ++memory_type_index) {
    const uint32_t memory_type_bit = uint32_t(1) << memory_type_index;
    const VkMemoryPropertyFlags memory_type_flags =
        memory_properties.memoryTypes[memory_type_index].propertyFlags;
    if (memory_type_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
      device->memory_types_.device_local |= memory_type_bit;
    }
    if (memory_type_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      device->memory_types_.host_visible |= memory_type_bit;
    }
    if (memory_type_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
      device->memory_types_.host_coherent |= memory_type_bit;
    }
    if (memory_type_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
      device->memory_types_.host_cached |= memory_type_bit;
    }
    XELOGI(
        "Vulkan memory type {}: {}{}{}{}{}heap {}", memory_type_index,
        (memory_type_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            ? "DEVICE_LOCAL "
            : "",
        (memory_type_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            ? "HOST_VISIBLE "
            : "",
        (memory_type_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            ? "HOST_COHERENT "
            : "",
        (memory_type_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
            ? "HOST_CACHED "
            : "",
        (memory_type_flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
            ? "LAZILY_ALLOCATED "
            : "",
        memory_properties.memoryTypes[memory_type_index].heapIndex);

    // Detect ReBAR/SAM memory (both device-local and host-visible)
    if ((memory_type_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (memory_type_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      uint32_t heap_index =
          memory_properties.memoryTypes[memory_type_index].heapIndex;
      VkDeviceSize heap_size = memory_properties.memoryHeaps[heap_index].size;
      // Require at least 256MB to consider this usable ReBAR memory.
      // Smaller heaps aren't worth using for staging buffers.
      constexpr VkDeviceSize kMinRebarHeapSize = 256 * 1024 * 1024;
      if (heap_size >= kMinRebarHeapSize) {
        device->memory_types_.device_local_host_visible |= memory_type_bit;
        XELOGI(
            "Vulkan memory type {}: HOST_VISIBLE | DEVICE_LOCAL (ReBAR/SAM), "
            "heap {} ({} MB)",
            memory_type_index, heap_index, heap_size >> 20);
      } else {
        XELOGI(
            "Vulkan memory type {}: HOST_VISIBLE | DEVICE_LOCAL but heap {} "
            "too small ({} MB < 256 MB), not using as ReBAR",
            memory_type_index, heap_index, heap_size >> 20);
      }
    }
  }

  XELOGI("ZeroFGProbe: VK_GOOGLE_display_timing available={}",
         zerofg_display_timing_available ? "YES" : "NO");
  XELOGI("ZeroFGProbe: VK_GOOGLE_display_timing enabled={}",
         ext_GOOGLE_display_timing ? "YES" : "NO");
  if (presenter_device) {
    const VkFormat formats[] = {
        VK_FORMAT_R8_UNORM, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R16_UNORM};
    std::string format_summary;
    for (size_t i = 0; i < std::size(formats); ++i) {
      backend.backend_formats[i] = device->QueryZeroFGBackendFormat(
          formats[i], VK_IMAGE_USAGE_SAMPLED_BIT);
      const auto& f = backend.backend_formats[i];
      const auto bits = f.optimal_tiling_features;
      format_summary += fmt::format(
          "{}[format={},usage={},flags={},tiling={},view={},status={},"
          "cubic={},cubic_minmax={},features={},block_match={},box_filter={},"
          "weight_sampled={},weight_image={}]",
          i ? ";" : "", int32_t(f.format), f.usage, f.create_flags,
          int32_t(f.tiling), int32_t(f.view_type), int32_t(f.image_query_result),
          f.cubic, f.cubic_minmax, uint64_t(bits),
          bool(bits & VK_FORMAT_FEATURE_2_BLOCK_MATCHING_BIT_QCOM),
          bool(bits & VK_FORMAT_FEATURE_2_BOX_FILTER_SAMPLED_BIT_QCOM),
          bool(bits & VK_FORMAT_FEATURE_2_WEIGHT_SAMPLED_IMAGE_BIT_QCOM),
          bool(bits & VK_FORMAT_FEATURE_2_WEIGHT_IMAGE_BIT_QCOM));
    }
    XELOGI(
        "ZeroFGBackendCapabilities api={} api_physical={} driverID={} driverName={} "
        "driverInfo={} shaderFloat16={}/{} shaderInt8={}/{} shaderInt16={}/{} "
        "storageBuffer16BitAccess={}/{} subgroup_size={} subgroup_min={} "
        "subgroup_max={} subgroup_workgroup_groups={} subgroup_stages={} "
        "subgroup_operations={} "
        "subgroupSizeControl={}/{} computeFullSubgroups={}/{} "
        "maintenance4={}/{} "
        "requiredSubgroupSizeStages={} shaderIntegerDotProduct={}/{} "
        "dot4x8_unsigned_accelerated={} dot4x8_signed_accelerated={} "
        "dot4x8_mixed_accelerated={} filter_cubic={}/{} "
        "sampler_filter_minmax={}/{} minmax_single_component={} "
        "minmax_component_mapping={} qcom_image_processing={}/{} "
        "qcom_weighted={}/{} qcom_box={}/{} qcom_block_match={}/{} "
        "maxWeightFilterPhases={} maxWeightFilterDimension={}x{} "
        "maxBlockMatchRegion={}x{} maxBoxFilterBlockSize={}x{} "
        "qcom_v2_advertised={} textureBlockMatch2={} maxBlockMatchWindow={}x{} "
        "qcom_v3_advertised={} qcom_v3_features=header_unavailable "
        "qcom_cubic_weights_advertised={} selectableCubicWeights={} "
        "qcom_cubic_clamp_advertised={} cubicRangeClamp={} "
        "probe_only_v2_v3_cubic=true supported_enabled_order=true formats={}",
        backend.effective_api_version, unclamped_api_version,
        uint32_t(properties_1_2_KHR_driver_properties.driverID),
        properties_1_2_KHR_driver_properties.driverName,
        properties_1_2_KHR_driver_properties.driverInfo,
        backend.shader_float16_supported, backend.shader_float16_enabled,
        backend.shader_int8_supported, backend.shader_int8_enabled,
        backend.shader_int16_supported, backend.shader_int16_enabled,
        backend.storage_buffer_16bit_access_supported,
        backend.storage_buffer_16bit_access_enabled, backend.subgroup_size,
        backend.subgroup_min_size, backend.subgroup_max_size,
        backend.max_compute_workgroup_subgroups,
        backend.subgroup_supported_stages, backend.subgroup_supported_operations,
        backend.subgroup_size_control_supported,
        backend.subgroup_size_control_enabled,
        backend.compute_full_subgroups_supported,
        backend.compute_full_subgroups_enabled,
        backend.maintenance4_supported, backend.maintenance4_enabled,
        backend.required_subgroup_size_stages,
        backend.shader_integer_dot_product_supported,
        backend.shader_integer_dot_product_enabled,
        backend.integer_dot_product_4x8_unsigned_accelerated,
        backend.integer_dot_product_4x8_signed_accelerated,
        backend.integer_dot_product_4x8_mixed_signedness_accelerated,
        backend.filter_cubic_supported, backend.filter_cubic_enabled,
        backend.sampler_filter_minmax_supported,
        backend.sampler_filter_minmax_enabled,
        backend.filter_minmax_single_component_formats,
        backend.filter_minmax_image_component_mapping,
        backend.qcom_image_processing_supported,
        backend.qcom_image_processing_enabled,
        backend.qcom_texture_sample_weighted_supported,
        backend.qcom_texture_sample_weighted_enabled,
        backend.qcom_texture_box_filter_supported,
        backend.qcom_texture_box_filter_enabled,
        backend.qcom_texture_block_match_supported,
        backend.qcom_texture_block_match_enabled,
        backend.qcom_max_weight_filter_phases,
        backend.qcom_max_weight_filter_dimension.width,
        backend.qcom_max_weight_filter_dimension.height,
        backend.qcom_max_block_match_region.width,
        backend.qcom_max_block_match_region.height,
        backend.qcom_max_box_filter_block_size.width,
        backend.qcom_max_box_filter_block_size.height,
        backend.qcom_image_processing2_advertised,
        backend.qcom_texture_block_match2_supported,
        backend.qcom_max_block_match_window.width,
        backend.qcom_max_block_match_window.height,
        backend.qcom_image_processing3_advertised,
        backend.qcom_cubic_weights_advertised,
        backend.qcom_selectable_cubic_weights_supported,
        backend.qcom_cubic_clamp_advertised,
        backend.qcom_cubic_range_clamp_supported, format_summary);
  }
  return device;
}

VulkanDevice::~VulkanDevice() {
  if (device_) {
    vulkan_instance_->functions().vkDestroyDevice(device_, nullptr);
  }
}

VulkanDevice::VulkanDevice(const VulkanInstance* const vulkan_instance,
                           const VkPhysicalDevice physical_device)
    : vulkan_instance_(vulkan_instance), physical_device_(physical_device) {
  assert_not_null(vulkan_instance);
  assert_not_null(physical_device);
}

void VulkanDevice::LogFaultInfo() {
  if (!extensions_.ext_EXT_device_fault || !vkGetDeviceFaultInfoEXT_) {
    return;
  }
  // Log only once, even if many observers call us.
  if (fault_info_logged_.test_and_set(std::memory_order_acq_rel)) {
    return;
  }

  // First pass: counts only.
  VkDeviceFaultCountsEXT counts = {VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
  if (vkGetDeviceFaultInfoEXT_(device_, &counts, nullptr) != VK_SUCCESS) {
    XELOGE("VK_EXT_device_fault: failed to query fault info counts");
    return;
  }

  std::vector<VkDeviceFaultAddressInfoEXT> address_infos(
      counts.addressInfoCount);
  std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(counts.vendorInfoCount);
  VkDeviceFaultInfoEXT info = {VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
  info.pAddressInfos = address_infos.empty() ? nullptr : address_infos.data();
  info.pVendorInfos = vendor_infos.empty() ? nullptr : vendor_infos.data();
  // Skip vendor binary - we don't have a place to dump it anyway.
  counts.vendorBinarySize = 0;
  if (vkGetDeviceFaultInfoEXT_(device_, &counts, &info) != VK_SUCCESS) {
    XELOGE("VK_EXT_device_fault: failed to query fault info");
    return;
  }

  XELOGE("VK_EXT_device_fault: \"{}\" - {} address(es), {} vendor code(s)",
         info.description, counts.addressInfoCount, counts.vendorInfoCount);
  for (uint32_t i = 0; i < counts.addressInfoCount; ++i) {
    const VkDeviceFaultAddressInfoEXT& a = address_infos[i];
    const char* type_name = "unknown";
    switch (a.addressType) {
      case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT:
        type_name = "none";
        break;
      case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT:
        type_name = "read-invalid";
        break;
      case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT:
        type_name = "write-invalid";
        break;
      case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT:
        type_name = "execute-invalid";
        break;
      case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT:
        type_name = "instruction-pointer-unknown";
        break;
      case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT:
        type_name = "instruction-pointer-invalid";
        break;
      case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:
        type_name = "instruction-pointer-fault";
        break;
      default:
        break;
    }
    XELOGE("  fault addr: 0x{:016X} (precision ±0x{:X}) type={}",
           a.reportedAddress, a.addressPrecision, type_name);
  }
  for (uint32_t i = 0; i < counts.vendorInfoCount; ++i) {
    const VkDeviceFaultVendorInfoEXT& v = vendor_infos[i];
    XELOGE("  vendor code: {} fault=0x{:016X} \"{}\"", v.vendorFaultCode,
           v.vendorFaultData, v.description);
  }
}

std::optional<uint32_t> VulkanDevice::QueryGpuBusyPermille() const {
#if XE_PLATFORM_ANDROID || XE_PLATFORM_xendroid
  // Adreno only: KGSL's busy and total time over its last update period.
  if (properties_.vendorID != 0x5143) {
    return std::nullopt;
  }
  std::FILE* file = std::fopen("/sys/class/kgsl/kgsl-3d0/gpubusy", "r");
  if (!file) {
    return std::nullopt;
  }
  unsigned long long busy = 0;
  unsigned long long total = 0;
  const int fields = std::fscanf(file, "%llu %llu", &busy, &total);
  std::fclose(file);
  if (fields != 2) {
    return std::nullopt;
  }
  // 0 0 while the GPU is powered down: idle, not unavailable.
  return total ? uint32_t(std::min(busy, total) * 1000 / total) : 0u;
#else
  return std::nullopt;
#endif
}

}  // namespace vulkan
}  // namespace ui
}  // namespace xe
