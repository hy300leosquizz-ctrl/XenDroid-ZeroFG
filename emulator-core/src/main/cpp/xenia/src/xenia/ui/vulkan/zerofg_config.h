#ifndef XENIA_UI_VULKAN_ZEROFG_CONFIG_H_
#define XENIA_UI_VULKAN_ZEROFG_CONFIG_H_

#include <atomic>
#include <cstdint>

#include "xenia/base/cvar.h"

DECLARE_bool(zerofg_frame_generation);
DECLARE_string(zerofg_mode);
DECLARE_uint32(zerofg_fps_limit);

namespace xe::ui::vulkan {

enum class ZeroFGSelection { kOff, kZero, kReallyZero };

inline ZeroFGSelection GetZeroFGSelection() {
  const std::string& mode = cvars::zerofg_mode;
  if (mode == "off") {
    return ZeroFGSelection::kOff;
  }
  if (mode == "zero" || mode == "rc3") {
    return ZeroFGSelection::kZero;
  }
  if (mode == "reallyzero" || mode == "rc3_lite") {
    return ZeroFGSelection::kReallyZero;
  }
  // Earlier releases wrote other variant names (or only the boolean); a
  // configuration that had frame generation on keeps it on, as Zero.
  if (mode == "subzero" || mode == "rc1" || mode == "rc1_golden" ||
      mode == "rc1_test" || mode == "rc2" ||
      (mode.empty() && cvars::zerofg_frame_generation)) {
    return ZeroFGSelection::kZero;
  }
  return ZeroFGSelection::kOff;
}

inline bool IsZeroFGRequested() {
  return GetZeroFGSelection() != ZeroFGSelection::kOff;
}

// True while a ZeroFG presenter is connected and has not failed open. The
// guest-side pacing (the vblank pull and the game frame rate cap) acts only
// then: a session that fell back to the native path, or whose device B never
// qualified, runs the guest natively.
inline std::atomic<bool>& ZeroFGGuestPacingLive() {
  static std::atomic<bool> live{false};
  return live;
}

inline bool IsReallyZeroRequested() {
  return GetZeroFGSelection() == ZeroFGSelection::kReallyZero;
}

inline const char* ZeroFGSelectionName() {
  switch (GetZeroFGSelection()) {
    case ZeroFGSelection::kOff:
      return "OFF";
    case ZeroFGSelection::kZero:
      return "ZERO";
    case ZeroFGSelection::kReallyZero:
      return "REALLYZERO";
  }
  return "UNSUPPORTED";
}

}  // namespace xe::ui::vulkan

#endif  // XENIA_UI_VULKAN_ZEROFG_CONFIG_H_
