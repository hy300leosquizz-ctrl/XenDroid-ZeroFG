# XenDroid-ZeroFG

> Reference XenDroid integration and functional proof of concept for ZeroFG V1.

**Status: Experimental — Functional V1 Proof of Concept**

XenDroid-ZeroFG integrates the independent [ZeroFG](https://github.com/hy300leosquizz-ctrl/ZeroFG) Vulkan frame-generation engine into a real Android Vulkan host. This repository demonstrates the V1 interpolation path working inside XenDroid, including the generation and presentation of real synthetic frames observed during device testing.

This is an experimental proof of concept. It is preserved because the integration works technically, not because its visual quality is suitable for normal use.

## Reference integration

ZeroFG receives host-owned Vulkan resources and records interpolation work into a command buffer supplied by XenDroid. XenDroid retains responsibility for queue submission, swapchain ownership, presentation, synchronization, and frame pacing.

The integration:

- connects the ZeroFG V1 engine to XenDroid's Vulkan presenter;
- preserves the normal XenDroid rendering path when frame generation is disabled;
- generates a synthetic midpoint (`phase = 0.5`) for the experimental 2× path;
- demonstrates that the independent engine can run in a real Vulkan host.

The standalone V1 engine source is published separately in the [ZeroFG repository](https://github.com/hy300leosquizz-ctrl/ZeroFG).

## Known limitations

V1 works technically and demonstrates real frame generation, but the result is visually poor. In practical terms: it works, but it is ugly.

- severe ghosting and smearing, especially during camera motion;
- prominent motion, edge, and blending artifacts;
- poor disocclusion handling;
- limited motion estimation;
- high GPU overhead;
- possible performance regressions rather than gains;
- image quality unsuitable for normal gameplay.

Do not treat this implementation as a stable feature or a normal gameplay enhancement. Its value is technical demonstration and historical preservation of the first functional ZeroFG path.

## Experimental V1 APK

The [ZeroFG V1 Functional POC — Experimental](https://github.com/hy300leosquizz-ctrl/XenDroid-ZeroFG/releases/tag/v1-poc) release provides a debug APK built from this runtime-validated V1 implementation. It is a demonstration artifact, not a daily-use build, and its limitations should be read before installation.

## Building

See [BUILD.md](BUILD.md) for the canonical Android build instructions. The V1 POC is built through the XenDroid Gradle/NDK pipeline; the standalone ZeroFG repository does not claim an independently validated build system.

## Device requirements

- Snapdragon Gen 2 or newer is recommended.
- Adreno 740 or newer is recommended; lower Adreno 7xx devices have not been validated by this project.

Custom Vulkan drivers may be loaded through **Settings → Vulkan → Custom Vulkan Driver**. Driver compatibility is device-specific.

## Development and credits

ZeroFG is developed through human–AI collaboration.

- **hy300leosquizz** ([`hy300leosquizz-ctrl`](https://github.com/hy300leosquizz-ctrl)) — creator and project maintainer; responsible for engineering direction, integration, device and runtime testing, and final technical decisions.
- **Zeromeia** — the project name for an AI development collaborator powered by ChatGPT by OpenAI. Zeromeia is used extensively across the development pipeline, including architecture, technical and runtime analysis, algorithm and experiment design, code-generation guidance, Codex task design, code review, debugging, log analysis, documentation, repository organization, and release preparation.

AI-generated analysis, designs, code suggestions, and documentation are treated as engineering inputs. Final project decisions, device testing, validation, and publication remain under the control of the human maintainer. The name Zeromeia describes the project's use of ChatGPT and does not imply sponsorship or endorsement by OpenAI, legal personhood, copyright ownership, or independent publication authority.

## Project lineage and licensing

XenDroid derives from Android Xbox 360 emulator work based on Xenia. Existing copyright notices, licenses, and third-party terms remain applicable in their respective files and directories. See the repository's license files and source headers for details. No new license or provenance claim is introduced by the ZeroFG presentation in this README.

## Disclaimer

XenDroid-ZeroFG is free experimental software. Do not pay for unofficial copies, and do not assume third-party APKs are authentic or safe. This project provides no guarantee of game compatibility, performance, or visual quality.
