# How ZeroFG fits into XenDroid

This page explains how a generated frame travels from the game to the screen,
and which part of the code owns each step. For using ZeroFG, see the
[README](../README.md).

## The path of a frame

```text
 Device A (the emulator)                 Device B (ZeroFG)
 ───────────────────────                 ─────────────────────────────────────────────
 guest frame N rendered  ──capture──▶    Residency (a copy of each real frame)
                                              │
                                              ├─▶ Generation: frame N-1 + frame N
                                              │         → the midpoint frame S
                                              │
                                              ├─▶ Post: the same output pipeline as
                                              │   normal XenDroid (scaling, CAS, FSR,
                                              │   SGSR, sharpening) for R and S alike
                                              │
                                              └─▶ Main Surface egress: one present per
                                                  display refresh, in order: R, S, R, S…
```

- **The game keeps its own Vulkan device (A).** Nothing in the emulator's
  rendering changes. When a frame is finished, the presenter copies it into a
  ZeroFG-owned image on device B.
- **ZeroFG runs on a second Vulkan device (B)**, created on the same GPU. Its
  queue, its submissions and its waits belong to ZeroFG alone, so its work
  never sits in the emulator's queue. Device B also requests a higher GPU
  scheduling priority than the game's, so its short frame-generation work is
  not stranded behind the game's long GPU workload; under GPU saturation this
  may trade some game throughput for much lower frame-generation latency.
  Without it, the two devices are not isolated in time on a saturated GPU and
  device B cannot do its job.
  It asks through Vulkan's global priority (high) on every driver that offers
  it. Turnip's KGSL backend ignores that request, so with a custom Turnip
  driver XenDroid's adrenotools also sets the kernel's priority while device B
  is created (KGSL's default is 8, B gets 4). The log line `ZeroFGDeviceB
  vk_global_priority=... kgsl_priority requested=4 contexts_raised=1` shows
  both.
- **The engine** (`emulator-core/src/main/cpp/zerofg`) only records GPU
  commands: given two real frames and an output image, it fills the output
  with the frame halfway between them. It owns no queue, no swapchain and no
  timing, which is why it can also live in its own repository.
- **The independent presenter** (`zerofg_independent_presenter.cc`) owns
  everything around the engine: the capture and residency slots, when each
  frame is generated, post-processed and shown, and the order of what reaches
  the screen.
- **The Main Surface egress** (`zerofg_main_surface_egress.cc`) presents on the
  game's own Android surface from device B, at the display's full refresh rate.

## Pacing

The game's frame rate sets the clock. With a 60 fps game, ZeroFG plans one
output every 8.3 ms, alternating a real frame (R) and a generated one (S):

- **Real frames are sovereign** until ZeroFG hands them to the system: a real
  frame is never dropped or reordered to make room for a generated one. A
  generated frame that is late is simply not shown.
- **The game is never slowed by the screen.** ZeroFG copies a frame only into a
  buffer the system has already released, and when the system limits the
  frame rate (a thermal cap, for example), it learns the limit and presents
  only the newest frame at each of its slots. The game keeps its speed; fewer
  frames reach the panel.
- **One frame per refresh.** Each frame is assigned its own display refresh,
  the first it can make, in order. Without this, the game's clock and the
  panel's refresh drift slowly against each other, and for minutes at a time
  two frames meet in one refresh and one of them is lost.

## Modes and backends

| Mode | Engine |
| --- | --- |
| Zero | full refinement (four fine passes) |
| ReallyZero | the same engine with two fine passes |

The engine runs the generic fast paths the device really has (half-precision
colour, subgroup operations, hardware cubic filtering) and falls back once to
its portable form if those fail. Both produce the same logical result.

## Requirements and fallback

- An effective Vulkan 1.3 device with synchronization2 enabled.
- If device B cannot be created, if the engine is unsupported, or if anything
  fails while running, ZeroFG hands the surface back to the normal XenDroid
  presenter and the game continues without frame generation.

## Logs

With ZeroFG on, the log (`xe.log`, exported from the app) carries a short health
summary every few seconds:

| Line | What it tells |
| --- | --- |
| `ZeroFGC0Summary` | frames applied per second, ordering errors, fallbacks |
| `ZeroFGPhaseCInvariants` | the safety invariants (all zero when healthy) |
| `ZeroFGSourceRate` | the game's frame period as ZeroFG sees it |
| `ZeroFGMainSurface event=summary` | what reached the screen: presents, timings, frame caps, refresh assignment |
| `ZeroFGProfile` | the GPU time of each generated frame |
| `ZeroFGDeviceB` | the health of ZeroFG's own Vulkan device |

A fallback writes a `ZeroFGC0Terminal…` block with the reason. That block is the
most useful thing to attach to a bug report.
