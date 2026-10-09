# XenDroid-ZeroFG

> Frame generation for Xbox 360 emulation on Android. **ZeroFG 1.0.**

ZeroFG is a frame generation engine for mobile GPUs: it takes two frames a game
rendered and creates the one halfway between them, so a 60 fps game reaches 120
frames per second on a 120 Hz screen, and a 30 fps game reaches 60.

XenDroid is where ZeroFG was born. It is the first place ZeroFG runs, not the
reason it exists. ZeroFG was built with XenDroid, but not for XenDroid: the
engine does not know it is inside an emulator, and the standalone engine lives
in its own [ZeroFG repository](https://github.com/hy300leosquizz-ctrl/ZeroFG).

Compared with XenDroid, this app also brings Qualcomm's Snapdragon Game Super
Resolution (SGSR) upscaler, which the original XenDroid does not have, and a
set of fixes to its presentation (see
[What changed in XenDroid itself](#what-changed-in-xendroid-itself)). It
installs as its own app (`xendroid.zerofg`), so it lives next to XenDroid.

## Why an emulator

Xbox 360 emulation on a phone is about as heavy as mobile gets. XenDroid
translates PowerPC code on the fly, rebuilds a GPU that never existed on
Android, and keeps CPU and GPU near their limits for whole sessions. Then you
ask that same phone to generate a new frame between every two.

That was the blessing and the curse. Every timing problem, every driver quirk,
every thermal limit shows up in XenDroid first and hits hardest. Nothing could
hide. So we had to solve things properly: the pacing, the presentation, the
fight for the GPU. What survives XenDroid survives almost anywhere.

## By the numbers

In two months (August to October 2026):

- **736 commits** and more than **137,000 lines written**, with 64,000 rewritten
  away along the way;
- about **64,000 lines of original code** in the project today: the engines and
  their shaders (from RC1 to Zero), the presenter that paces and shows every
  frame, and the measurement laboratory and its tools that test the engine
  against ground truth;
- **34,000 lines of engineering notes**: every hypothesis, every run, every
  wrong turn, written down.

All of it original. Blood, sweat and many, many logs.

## Two modes

| Mode | What it is | GPU time per generated frame (1280 x 720) |
| --- | --- | --- |
| **Zero** | The full engine: the best image | Adreno 840: 2.2 ms · Adreno 650: 7.2 ms |
| **ReallyZero** | The same engine with half the refinement passes, for weaker GPUs | Adreno 840: 1.7 ms · Adreno 650: 5.4 ms |

ReallyZero gives almost the same picture as Zero: within about 0.2 dB on our
real-content measurement streams (fast synthetic motion loses more), and hard to
tell apart in play. The times come from the
laboratory, on a fixed workload. In a game, the phone lowers the GPU clock
whenever there is slack, so the saving shows up as headroom, not as
milliseconds.

Pick one in **Settings → ZeroFG → Frame generation**. It applies when a game
starts.

## How it works

ZeroFG estimates motion on a 20-pixel grid and checks every vector at full
resolution. It carries what it learned from the previous pair into the next
one, it tells exposure changes apart from motion, and it refuses vectors that
would tear the scene apart. HUDs, text and static overlays stay where they
belong instead of smearing with the world behind them.

Around the engine sits the presenter, and that is where most of the work went:

- **ZeroFG has its own Vulkan device.** The emulator keeps its device; ZeroFG
  captures, generates, post-processes and presents on a second one, so its work
  never sits in the emulator's queue.
- **ZeroFG's work is not stranded behind the game's.** ZeroFG requests higher
  GPU scheduling priority, so its short frame-generation work is not stranded
  behind the game's long GPU workload, and its frames keep coming evenly when
  the game saturates the GPU. Under GPU saturation, this may trade some game
  throughput for much lower frame-generation latency. ZeroFG asks every driver through Vulkan's standard global priority,
  and sets the priority itself under the custom Turnip drivers XenDroid loads
  (the usual setup), which ignore that request. Measured with Turnip; with
  other drivers it depends on whether they honour the request.
- **The game is never slowed by the screen.** ZeroFG never waits for the
  display. When Android caps the frame rate (HyperOS does it on a hot phone:
  90, then 60, then 45), fewer generated frames reach the screen, and the game
  keeps its speed.
- **One frame per refresh.** Every frame gets a display refresh of its own, in
  order, so two frames never fight over the same one.
- **Real frames come first.** A frame the game rendered is never dropped or
  reordered for a generated one. A generated frame that comes late is simply
  not shown.
- **It gets out of the way.** If the device cannot run ZeroFG (it needs Vulkan
  1.3 with synchronization2), or anything fails while it runs, ZeroFG hands the
  screen back and the game carries on without it.

[docs/ZEROFG_OVERVIEW.md](docs/ZEROFG_OVERVIEW.md) follows a frame from the
game to the screen.

## Measured

On an Adreno 840 phone at 120 Hz, with Rayman Origins at 60 fps:

- **about 120 frames on screen per second**, sustained for minutes, with 99.5%
  of them on the refresh they were given in the validation run;
- under the HyperOS thermal caps the game stayed at 60 while the screen showed
  about 87, 58 and 44 frames per second at the 90, 60 and 45 caps;
- no ordering, invariant or fallback failures across the test sessions.

In Batman: Arkham City on the same phone, hot, with the GPU 95–99% busy, the
game ran at about 23 fps and ZeroFG put about 46 frames a second on screen,
every frame it generated and none dropped. Its copy to the screen took about
1 ms; before ZeroFG had GPU priority, the same copy waited 40 to 170 ms behind
the game and the frames reached the screen in bursts.

## Limitations

ZeroFG 1.0 is ready to use, and it has real limits:

- **Latency.** A generated frame needs the next real frame, so ZeroFG adds about
  one game frame of latency (about 17 ms at 60 fps), plus a short presentation
  buffer. Fast competitive games feel it more.
- **Artifacts.** Very fast motion, things appearing from behind others,
  transparent effects, scene cuts and UI elements that animate on their own can
  show brief artifacts.
- **Heavy games.** ZeroFG competes with the game for the GPU and requests a
  higher scheduling priority. In games that already saturate the GPU, this may
  cost the game some throughput while the output stays smooth; at a low game
  frame rate the motion is smoother, but the latency is that of the slower
  game.
- **Hot phones.** When the phone heats up, Android may cap the frame rate:
  the game keeps its speed, but fewer frames reach the screen. When throttling
  chokes the GPU instead, ZeroFG's priority keeps its frames coming evenly,
  possibly at some cost to the game's frame rate; with a driver that ignores
  the priority request, the image can stutter in bursts.
- **Devices.** ZeroFG needs a Vulkan 1.3 driver with synchronization2. One
  device is runtime-validated in games (Adreno 840, HyperOS); the Adreno 650 is
  additionally qualified in the laboratory, where the numbers above come from.
  Other GPUs and drivers have not been tested by us yet.
- **Displays.** It shines with 60 fps games on 120 Hz screens.

## What's new in 1.0.2

- **The output follows the game's real rhythm.** Many 360 games wait for the
  console's 60 Hz vblank, so a frame that misses one waits for the next: the
  game alternates 33 and 50 ms frames and never settles. ZeroFG's clock only
  moved in big confirmed steps and could stay faster or slower than the game:
  in Batman: Arkham City the delay grew to about half a second for seconds at
  a time. ZeroFG now follows the average of the game's last frames, measured
  without the time the game spent waiting on ZeroFG, and runs a little faster
  than the game while frames sit deeper than it needs, until the delay is
  gone. Nothing is skipped.
- **Late game frames go out sooner.** With ZeroFG on, a game frame that is
  already late for its vblank gets it at once instead of waiting up to 16 ms
  for the next one. The game still sees 60 vblanks a second, so it never
  speeds up.
- **Fewer stutters when the frame rate changes.** When the game changed pace,
  ZeroFG re-laid its output timing and could leave a slot nothing filled: the
  frame before it stayed on screen too long. In Modern Warfare 3 these holds
  fell from about 3 % of the generated frames to under 1 % in its heavy
  scenes, and the longest gaps got shorter.
- **A game frame rate cap** (Settings → ZeroFG, or per game in the game's own
  settings). For games that keep the GPU at 100 %, a cap gives frame
  generation room to work: Modern Warfare 3 at 40 keeps about 80 frames a
  second on screen with less lag (70 → 55 ms) and almost no missed refreshes;
  Halo 3 at 20 halves its lag. A game whose logic runs per frame (Rayman
  Origins) plays slower under a cap below its own frame rate.
- **The cap and the vblank help turn off when ZeroFG does.** If ZeroFG has to
  stop in the middle of a game, the game goes back to running natively.

## What's new in 1.0.1

- **Lower latency.** ZeroFG's delay could get stuck several frames deeper than
  it needs and only came back when the game itself stuttered: in Halo 3, 85-175
  ms from the game's frame to the screen on 1.0. 1.0.1 keeps two game frames in
  flight instead of three, and drops an old pacing rule (the "physical operating
  point") that, after a few hiccups, could lock the output at 40 fps with half
  a second of delay. Halo 3 at 30 fps now measures about 65-80 ms.
- **A steadier cadence below 30 fps.** ZeroFG took the game waiting for its own
  GPU for ZeroFG's backpressure, so it ignored the game's real frame rate and
  kept a 30 fps rhythm, and frames came out in pairs. It now follows the game's
  real rate.
- **Halo 3 gets frame generation.** ZeroFG refused pictures its motion grid
  does not divide evenly (Halo 3 renders at 1152 x 640); it now pads them
  invisibly and runs.
- **No frozen screen when ZeroFG stops.** When ZeroFG had to stop in the middle
  of a game, as it did in Halo 3, the screen could freeze on its last frame
  while the game kept running. It now hands the screen back.
- **Protection against a Qualcomm driver crash** (below).

## Tips and known issues

- **Use a Turnip driver.** Settings → Vulkan → Custom Vulkan driver. ZeroFG
  was validated with Turnip, and its GPU priority works best there.
- **Qualcomm's own driver.** On some phones (seen on an Adreno 830), the
  phone's own Qualcomm driver crashes while compiling ZeroFG's shaders, about
  two seconds into the game. Since 1.0.1 the app remembers it: after such a
  crash the next launch uses ZeroFG's simplest shaders, and after a second one
  ZeroFG turns itself off for that driver, so the game keeps running. A new
  driver or a new app version tries again. Turnip avoids it altogether.
- **Let the phone run at full speed.** Turn on the phone's game or performance
  mode and add XenDroid-ZeroFG to it. Many phones otherwise hold the CPU at
  half speed, and the emulator, which leans on the CPU, falls below 30 fps.
- **Pick the mode.** Zero gives the best image. If the game's own frame rate
  drops with Zero, try ReallyZero, which is lighter on the GPU.
- **Cap a game that keeps the GPU busy.** If a game wobbles between frame rates
  with the GPU at 100 %, set its own game frame rate cap a little below its
  usual rate (Modern Warfare 3: 40, Halo 3: 20): the same frames reach the
  screen with less lag and a steadier rhythm. Check that the game still plays
  at normal speed; some games tie their logic to the frame rate.
- **Below 30 fps.** ZeroFG still makes the motion smoother, but the controls
  feel heavier: it waits for the next game frame, and slow game frames are
  long. In fast shooters below 30 fps you may prefer ZeroFG off.
- **Reporting a problem.** Settings → Logging → Export session logs to
  Downloads, then open an issue with the `xendroid-logs-....zip` attached and
  your phone, GPU, driver, game and ZeroFG mode. The issue forms ask for each.

## What changed in XenDroid itself

Building ZeroFG meant pulling XenDroid's presentation apart. We found and fixed
real bugs on the way, most of them by accident while chasing something else,
and added a few things:

- **The native presenter painted one frame at a time.** It never recorded the
  format its final paint pipeline was built for, so every frame it rebuilt the
  pipeline and first waited for the previous frame to finish painting. It now
  keeps up to three paints in flight, as it was designed to.
- **Some GPU checks could block.** On the Turnip driver's KGSL backend, a
  completion check that should return at once could wait until all queued GPU
  work finished. These checks are now bounded.
- **A driver's GPU contexts can be given a priority.** Turnip creates every
  GPU context at the kernel's default priority. XenDroid's adrenotools can now
  set the priority of the contexts a custom driver creates; ZeroFG uses it for
  its own device.
- **Snapdragon Game Super Resolution.** The first thing this fork added, before
  ZeroFG: Qualcomm's SGSR 1 upscaler, plain and with edge direction, in
  XenDroid's scaling and sharpening options next to CAS and FSR. The original
  XenDroid still does not have it.
- **Guest frames in flight follow ZeroFG.** With ZeroFG off, XenDroid now keeps
  two guest frames in flight instead of three, so less work queues between the
  game and the screen. With ZeroFG on it keeps three, the room ZeroFG needs.

## History

- **August 2026, where it started: SGSR.** Before any frame generation, this
  fork added Qualcomm's Snapdragon Game Super Resolution to XenDroid. ZeroFG
  came right after.
- **August 2026, V1, the proof of concept.** It worked, and it was ugly: strong
  ghosting and smearing, and a high cost. It is kept as the
  [V1 POC release](https://github.com/hy300leosquizz-ctrl/XenDroid-ZeroFG/releases/tag/v1-poc).
- **August–September, V2, the platform.** An independent presenter that owns
  pacing and presentation, then a Vulkan device of ZeroFG's own, and finally
  ZeroFG presenting straight onto the game's screen at the display's full rate.
- **September, the algorithm rounds.** RC1, a frozen portable motion estimator,
  and RC2, better warping and occlusion handling, each measured in play.
- **October, Zero.** A laboratory that runs the real engine against known
  ground truth, and in it RC3, the engine that became Zero, with its economy
  tier ReallyZero. The same week brought the fixes that keep the game's speed
  under system frame caps and give every frame its own refresh and, on the eve
  of the release, the GPU priority that ended the stutter in games that
  saturate the GPU.

## Building

See [BUILD.md](BUILD.md). To launch games from emulation frontends, see
[docs/frontend-integration.md](docs/frontend-integration.md); for per-game
settings, [GAME_COMPAT.md](GAME_COMPAT.md).

The standalone engine, the presenter and integration notes for other hosts are
in the [ZeroFG repository](https://github.com/hy300leosquizz-ctrl/ZeroFG).

## Contact

Questions, bug reports and collaboration: **fgzerofg@gmail.com**. A bug report
helps most with the log the app exports and the phone, GPU and driver you used.

## Development and credits

ZeroFG is developed through human–AI collaboration.

- **hy300leosquizz** ([`hy300leosquizz-ctrl`](https://github.com/hy300leosquizz-ctrl)) — creator and maintainer. Lawyer by trade, Formula 1 podcast presenter by passion, and software architect by sheer determination and curiosity, developing latent talents after asking himself whether he could modify a few features in emulators and falling down the rabbit hole for months: he drew the architecture, set the direction, had an instinct trust rate of 98%, became an expert in Vulkan and in frame generation features and capabilities, made every final call, and ran every test on his own phones, usually a hot one.
- **Zeromeia** — the project name for an AI development collaborator powered by ChatGPT by OpenAI: architecture, runtime and log analysis, experiment design, code review, documentation and release preparation.
- **Zé Raio** — the project name for an AI development collaborator powered by Claude by Anthropic: implementation, the measurement laboratory, log analysis, documentation and release preparation.

AI-generated analysis, designs, code and documentation are engineering inputs. Final decisions, device testing, validation and publication stay with the human maintainer. The names Zeromeia and Zé Raio describe the project's use of ChatGPT and Claude and do not imply sponsorship or endorsement by OpenAI or Anthropic.

## Licensing

- **ZeroFG**: the engine (`emulator-core/src/main/cpp/zerofg/`) and the ZeroFG
  host code (`emulator-core/src/main/cpp/xenia/src/xenia/ui/vulkan/zerofg_*`
  and `vulkan_presenter_zerofg_source_adapter.inc`) are licensed under the
  **Apache License, Version 2.0**. See [LICENSE-ZeroFG](LICENSE-ZeroFG).
  `vulkan_presenter_zerofg_device_context.inc` is adapted from Xenia's
  presenter and stays under Xenia's BSD 3-Clause license, as its header says.
- **SGSR**: Qualcomm's Snapdragon Game Super Resolution shaders
  (`guest_output_sgsr*.ps.glsl`) keep their BSD 3-Clause license, as their
  headers say.
- **libadrenotools**: XenDroid's copy, including the GPU priority hook added
  for ZeroFG, stays under its BSD 2-Clause license.
- **XenDroid and Xenia**: the rest of this repository carries XenDroid- and
  Xenia-derived code with its existing licensing obligations. Existing
  copyright notices, the Xenia BSD 3-Clause terms and third-party licenses
  still apply to their code. No relicensing of XenDroid or Xenia is implied.
