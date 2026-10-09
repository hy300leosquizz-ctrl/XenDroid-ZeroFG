/**
 ******************************************************************************
 * XenDroid ZeroFG pipeline build guard                                       *
 ******************************************************************************
 */

#ifndef XENIA_UI_VULKAN_ZEROFG_BUILD_GUARD_H_
#define XENIA_UI_VULKAN_ZEROFG_BUILD_GUARD_H_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "xenia/base/logging.h"

namespace xe {
namespace ui {
namespace vulkan {

// A driver can die while it compiles ZeroFG's pipelines: a SIGSEGV inside
// vkCreateComputePipelines returns no VkResult, so ZeroFGBackendFallback never
// sees it and every launch dies at the same frame (first seen 2026-10-08: an
// Adreno 830 on a Qualcomm proprietary driver package, building RC3 with the
// fast paths that driver offers and Turnip does not, subgroup D4 and FP16).
// The guard turns that crash loop into at most two crashes:
//  - before a build, a marker naming the driver and the backend is written to
//    the storage root, and it is removed when the build returns;
//  - a marker found at the next start means that build killed the process:
//    an Auto build moves this driver to Compat, a Compat build turns ZeroFG off
//    for it;
//  - the verdict belongs to one driver and one app build, so another driver or
//    a new release tries again.
// A process killed while a build runs (rare: builds take a fraction of a
// second) reads as a crash too; its cost is a Compat run, then a new try with
// the next release.
class ZeroFGBuildGuard {
 public:
  enum class Verdict { kNone, kCompat, kOff };

  static const char* VerdictName(Verdict verdict) {
    switch (verdict) {
      case Verdict::kNone:
        return "none";
      case Verdict::kCompat:
        return "compat";
      case Verdict::kOff:
        return "off";
    }
    return "none";
  }

  static ZeroFGBuildGuard& Get() {
    static ZeroFGBuildGuard guard;
    return guard;
  }

  // Once per process, before the first build: reads the verdicts and turns a
  // marker left by a build that never returned into a verdict. Later calls
  // return the verdict already taken. An empty directory leaves the guard off.
  Verdict Initialize(const std::filesystem::path& directory,
                     const std::string& identity) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
      return verdict_;
    }
    initialized_ = true;
    if (directory.empty()) {
      XELOGW("ZeroFGBuildGuard inactive: no storage root");
      return verdict_;
    }
    marker_path_ = directory / "zerofg_build.marker";
    verdicts_path_ = directory / "zerofg_driver_verdicts.txt";
    identity_ = identity;

    std::vector<std::pair<Verdict, std::string>> verdicts = ReadVerdicts();
    std::string marker_identity, marker_backend, marker_stage;
    {
      std::ifstream marker(marker_path_);
      if (marker) {
        std::getline(marker, marker_identity);
        std::getline(marker, marker_backend);
        std::getline(marker, marker_stage);
      }
    }
    if (!marker_identity.empty()) {
      // The last build on that driver never returned: Compat after a build
      // on the fast paths, off after a Compat build or a second crash (an
      // engine without a Compat form crashes the same way twice).
      Verdict promoted =
          marker_backend == "compat" ? Verdict::kOff : Verdict::kCompat;
      bool found = false;
      for (auto& entry : verdicts) {
        if (entry.second == marker_identity) {
          found = true;
          if (entry.first != Verdict::kNone) {
            promoted = Verdict::kOff;
          }
          entry.first = promoted;
        }
      }
      if (!found) {
        verdicts.emplace_back(promoted, marker_identity);
      }
      WriteVerdicts(verdicts);
      XELOGE(
          "ZeroFGBuildGuard: the last session ended inside the driver while it "
          "built ZeroFG's pipelines (stage={} backend={}); verdict={} for {}",
          marker_stage, marker_backend, VerdictName(promoted),
          marker_identity);
      std::error_code error;
      std::filesystem::remove(marker_path_, error);
    }
    for (const auto& entry : verdicts) {
      if (entry.second == identity_) {
        verdict_ = entry.first;
      }
    }
    active_ = true;
    XELOGI("ZeroFGBuildGuard verdict={} identity={}", VerdictName(verdict_),
           identity_);
    return verdict_;
  }

  Verdict verdict() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return verdict_;
  }

  // Marks one build in flight for as long as it lives.
  class Scope {
   public:
    Scope(ZeroFGBuildGuard& guard, const char* stage, bool compat)
        : guard_(&guard) {
      guard_->Begin(stage, compat);
    }
    ~Scope() {
      if (guard_) {
        guard_->End();
      }
    }
    Scope(Scope&& other) noexcept : guard_(other.guard_) {
      other.guard_ = nullptr;
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope& operator=(Scope&&) = delete;

   private:
    ZeroFGBuildGuard* guard_;
  };

  Scope Build(const char* stage, bool compat) {
    return Scope(*this, stage, compat);
  }

 private:
  ZeroFGBuildGuard() = default;

  void Begin(const char* stage, bool compat) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || builds_in_flight_++) {
      return;
    }
    // A plain write is enough: the kernel keeps it when the process dies.
    std::ofstream marker(marker_path_, std::ios::trunc);
    marker << identity_ << '\n'
           << (compat ? "compat" : "auto") << '\n'
           << stage << '\n';
  }

  void End() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_ || !builds_in_flight_ || --builds_in_flight_) {
      return;
    }
    std::error_code error;
    std::filesystem::remove(marker_path_, error);
  }

  std::vector<std::pair<Verdict, std::string>> ReadVerdicts() const {
    std::vector<std::pair<Verdict, std::string>> verdicts;
    std::ifstream file(verdicts_path_);
    std::string line;
    while (std::getline(file, line)) {
      const size_t tab = line.find('\t');
      if (tab == std::string::npos) {
        continue;
      }
      const std::string name = line.substr(0, tab);
      const Verdict verdict = name == "off"      ? Verdict::kOff
                              : name == "compat" ? Verdict::kCompat
                                                 : Verdict::kNone;
      if (verdict != Verdict::kNone) {
        verdicts.emplace_back(verdict, line.substr(tab + 1));
      }
    }
    return verdicts;
  }

  void WriteVerdicts(
      const std::vector<std::pair<Verdict, std::string>>& verdicts) const {
    std::ofstream file(verdicts_path_, std::ios::trunc);
    for (const auto& entry : verdicts) {
      if (entry.first != Verdict::kNone) {
        file << VerdictName(entry.first) << '\t' << entry.second << '\n';
      }
    }
  }

  mutable std::mutex mutex_;
  bool initialized_ = false;
  bool active_ = false;
  uint32_t builds_in_flight_ = 0;
  Verdict verdict_ = Verdict::kNone;
  std::filesystem::path marker_path_;
  std::filesystem::path verdicts_path_;
  std::string identity_;
};

}  // namespace vulkan
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_VULKAN_ZEROFG_BUILD_GUARD_H_
