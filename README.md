# frame2frame

**Make the experience inside AR glasses visible from the outside.**

[![CI](https://github.com/jasperzhao05/frame2frame-virtual-screen/actions/workflows/ci.yml/badge.svg)](https://github.com/jasperzhao05/frame2frame-virtual-screen/actions/workflows/ci.yml)
[![Python 3.9–3.13](https://img.shields.io/badge/python-3.9--3.13-3776AB)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/license-MIT-2ea44f)](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/LICENSE)

## Contents

- [Overview](#overview)
- [Research](#research)
  - [Screen Age](#screen-age)
  - [Filter trade-off](#filter-trade-off)
- [Quick start](#quick-start)
- [Engineering](#engineering)
  - [Reliability](#reliability)
  - [Backends and filters](#backends-and-filters)
- [Reference](#reference)

## Overview

AR glasses have a demonstration problem. The wearer sees a spatial display;
the audience sees only a person wearing glasses. Product footage can show the
device, but not the experience.

`frame2frame` combines ordinary third-person footage with content supplied as
an image, a video, or a stream of live application frames. It estimates head
pose, maintains a coherent temporal state, and projects that content as a
head-relative virtual screen—turning an otherwise private experience into
shareable AR-style footage.

The system was first developed during a Rokid internship; this repository is an
independent open-source rebuild. It is a visualization pipeline, not a headset
integration. It never reads headset output: the image, video, or live
application frames shown on the virtual screen must be supplied separately. It
does not use headset telemetry, eye gaze, SLAM, or world anchors. This
open-source repository is not affiliated with or endorsed by Rokid.

### Virtual screen in motion

<p align="center">
  <img
    src="https://raw.githubusercontent.com/jasperzhao05/frame2frame-virtual-screen/main/docs/demo-rokid-outdoor.webp"
    width="560"
    alt="Two real AR-glasses scenes shown beside FIR distance-4 output playing Black Myth Wukong content"
  >
</p>

*Two outdoor clips, six seconds total at 20 fps. Original footage appears on
the left; MediaPipe + FIR output at screen distance 4 appears on the right. A
separately supplied* Black Myth: Wukong *video advances on the projected
screen. Neither clip uses per-frame reframing, manual screen keyframing, slow
motion, interpolation, or reversed footage.*

*Wearer footage: BooredAtWork,
[“Rokid Glasses 2025 – Next Level Augmented Reality Experience!”](https://www.youtube.com/watch?v=abE88Vve0o4),
and Naomi Wu,
[“Vuzix Blade Review—Wearable Display for a Cyberpunk Future!”](https://www.youtube.com/watch?v=sS90qEPgc50),
both [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Screen content:
Game Science,
[“Black Myth: Wukong — 13 Minutes Official Gameplay Trailer”](https://www.youtube.com/watch?v=oRLhCxC886o).
The gameplay is supplied content rendered by `frame2frame`, not headset output.
Source footage, likenesses, game content, and marks are not covered by this
repository's MIT license. No affiliation or endorsement is implied.*

```text
Video / webcam → pose → temporal state → screen geometry ────────┐
Image / video / latest live frame → content preparation ─────────┴→ warp + composite
```

Orientation, face center, and apparent scale advance on the same frame
timeline. Short detection gaps are held and faded; sustained gaps reset the
state before reacquisition.

Static images are prepared once. Screen videos follow the scene timeline. Live
producers publish into a capacity-one latest-frame source, so stale application
frames are replaced rather than queued. All three sources share the same
projection and compositing path.

The [architecture notes](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/ARCHITECTURE.md)
define coordinate conventions, temporal invariants, resource ownership, and
extension boundaries.

## Research

Making content visible is only the first step. Once a display is attached to a
moving head, pose noise becomes visible jitter—and suppressing that jitter can
make the screen represent an older moment. `frame2frame` isolates this
stability–responsiveness trade-off: Screen Age measures the moment represented
by projected geometry, while a deterministic signal benchmark measures
residual noise and motion distortion.

### Screen Age

For every retained output frame `t`, the protocol compares the filter output
with screens projected from earlier ground-truth attitudes while keeping the
current center and apparent scale fixed. Screen Age is the single age that
minimizes pooled squared corner error across all development frames; it is
frozen before held-out evaluation and reported in frames and milliseconds.

<p align="center">
  <a href="https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/screen-age.svg">
    <img
      src="https://raw.githubusercontent.com/jasperzhao05/frame2frame-virtual-screen/main/docs/screen-age.png"
      width="640"
      alt="A real-person projected-screen worldline beside a compact calibration plot of analytic FIR delay against recovered Screen Age"
    >
  </a>
</p>

*Figure 1 — Projected Screen Chronometry. Panel A shows one protocol-selected
held-out BIWI event at screen distance 4. The project-default FIR output at `t`
is closest to `GT(t−11)`, giving a Screen Age of 11 frames, or 367 ms at 30 fps;
ten thin solid outlines trace the intervening counterfactual screens from
`t−1` through `t−10`. Panel B shows that all six registered FIR designs recover
their known 7-, 11-, or 19-frame delays across 45 continuous segments from 20
subjects.*

*Source:
[BIWI Kinect Head Pose Database](https://vision.ee.ethz.ch/datsets.html),
available for non-commercial research and education. BIWI assets retain their
upstream terms and are not covered by this repository's MIT License. This
figure calibrates Screen Age against known FIR response; it does not measure
pose-model accuracy, perceptual preference, or end-to-end latency.*

The [Screen Age protocol and aggregate receipt](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/SCREEN_AGE.md)
specify the metric, frozen split, bootstrap audit, and evidence boundary.

### Filter trade-off

A separate seeded, model-free workload compares the same clean synthetic motion
with and without injected observation noise. It reports residual noise,
response delay, and motion distortion without invoking a pose model.

```bash
python -m scripts.benchmark_smoothing --check
```

Default results at 30 fps (seed `20260730`):

| Filter | Residual jitter RMS ↓ | Jitter reduction ↑ | 50% step delay ↓ | Aligned motion RMSE ↓ |
|---|---:|---:|---:|---:|
| `none` | 1.329° | 0.0% | 0 frames | 0.000° |
| `fir` | 0.383° | 71.2% | 11 frames / 366.7 ms | 0.072° |
| `oneeuro` | 0.867° | 34.7% | 0 frames in this test | 0.387° |

`none` is a pass-through reference: it introduces no delay or motion
distortion, but it removes no observation noise.

Among the two smoothing filters in this model-free workload, FIR removes more
injected noise and introduces a fixed 11-frame causal age. One Euro reacts
immediately to the scripted step threshold but retains more noise and motion
distortion. This receipt explains the offline trade-off; it does not evaluate
Kalman or select a production live policy. The registered Kalman A100
configuration is the project's current causal live candidate for further
evaluation; One Euro is retained as a comparison and compatibility baseline.
That policy is not evidence of universal superiority.

The 50% step metric is the time required to cross half of a scripted 20° yaw
change; it is not capture-to-display latency. One Euro crossing that threshold
on the first updated sample does not imply zero lag for every motion.

Together, the two experiments separate complementary claims: Screen Age
calibrates the time represented by projected geometry, while the synthetic
benchmark isolates noise suppression and motion preservation. Neither
experiment evaluates pose-model accuracy or establishes a universally best
filter.

Read the [benchmark protocol](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/BENCHMARKS.md)
before comparing changes.

## Quick start

Clone and install the project:

```bash
git clone https://github.com/jasperzhao05/frame2frame-virtual-screen.git
cd frame2frame-virtual-screen
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e .
```

Render a video:

```bash
frame2frame --input clip.mp4 --output output/clip.mp4
```

Play a second video on the virtual screen:

```bash
frame2frame \
  --input wearer.mp4 \
  --screen-video interface.mp4 \
  --screen-video-end loop \
  --screen-fit contain \
  --screen-distance 4 \
  --output output/demonstration.mp4
```

The person video is the master clock: screen-video frames are selected from its
media time and remain aligned through offline FIR buffering and detection gaps.

Exercise the real-time content interface with a webcam and generated live UI:

```bash
python -m scripts.live_content_demo --webcam 0 --output ""
```

An application can publish OpenCV frames to `LatestFrameSource` from its own
producer thread. Every publication replaces the previous snapshot, keeping the
interface bounded when content arrives faster than pose estimation or
rendering. The [usage guide](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/USAGE.md#dynamic-and-real-time-content)
shows the minimal adapter and a looping-video producer.

The first MediaPipe run downloads a roughly 3.6 MB Face Landmarker asset to
`~/.cache/frame2frame`. Inputs are processed locally.

<details>
<summary>Webcam, custom content, and environment checks</summary>

Run a webcam preview with the current causal live candidate:

```bash
frame2frame --webcam 0 --filter kalman --display --output output/webcam.mp4
```

The registered `kalman` configuration is the A100 constant-angular-velocity
design (`acceleration_std=100`, `measurement_std=1`). It consumes measured
capture intervals and filters attitude only; MediaPipe remains the pose backend.
This project policy is not a claim of superior perceptual quality or end-to-end
latency. `oneeuro` remains available for comparison and backward compatibility;
it is not the current live default.

Render a static screen image:

```bash
frame2frame --input clip.mp4 --texture screen.png --output output/custom.mp4
```

Run the deterministic, model- and codec-free component receipt:

```bash
python -m scripts.benchmark_dynamic_content --check --format markdown
```

It checks latest-frame replacement, zero-conversion BGR preparation, exact
frame mapping, and repeatable compositing. Its component timings are not an
end-to-end camera-to-display latency claim.

Check the local environment without downloading a model or opening a camera:

```bash
frame2frame --doctor
```

`--doctor` checks the supported runtime, binary dependencies, cache, model
asset, and optional `ffmpeg`. It prints a terminal report and creates no file.

**Example `--doctor` output:**

```text
frame2frame doctor 0.3.0b2
[ok] python: 3.13.5 is supported
[ok] numpy: 2.5.1 imports successfully
[ok] scipy: 1.17.1 imports successfully
[ok] opencv: 4.13.0 imports successfully
[ok] mediapipe: 0.10.35 Tasks API imports successfully
[ok] matplotlib: 3.10.9 is installed
[ok] cache: writable (target: ~/.cache/frame2frame)
[warning] mediapipe-model: not cached; first run will download 3,758,596 verified bytes
[warning] ffmpeg: not found; required only for --preserve-audio and showcase generation
summary: 7 ok, 2 warning, 0 error
```

A warning describes an optional tool or normal first-run condition. An error
identifies a required dependency or filesystem problem and makes `--doctor`
exit unsuccessfully. The first MediaPipe import may take several seconds.

</details>

The [usage guide](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/USAGE.md)
covers camera calibration, diagnostics, cache redirection, audio handling, and
advanced configuration. See the
[privacy notes](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/PRIVACY.md)
for the complete data flow.

## Engineering

### Reliability

Reliability checks target pipeline behavior, not pose accuracy or perceptual
quality.

Run the deterministic file-pipeline check for frame conservation, dropout
reset behavior, and repeatable decoded output:

```bash
python -m scripts.benchmark_pipeline --check
```

Create a checksum-pinned real-video receipt for the primary MediaPipe + FIR
path:

```bash
python -m scripts.fetch_examples --download-only --limit 1
python -m scripts.validate_real_video \
  --input examples/inputs/head-pose-face-detection-female.mp4 \
  --output output/real-video-validation.mp4 \
  --receipt output/real-video-validation.json
```

These checks record exact frame handling, configuration, media hashes,
observation coverage, and run timings. They do not establish pose accuracy,
camera-to-display latency, perceptual quality, or a production SLO. See the
[reliability protocol](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/RELIABILITY.md) and
[real-video receipt protocol](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/VALIDATION.md).

<details>
<summary>Per-clip dropout recovery example</summary>

![Fixed crop from an Intel driver-action scene beside MediaPipe and FIR output in a changing car interior](https://raw.githubusercontent.com/jasperzhao05/frame2frame-virtual-screen/main/docs/demo-mediapipe.gif)

*One continuous 15-second excerpt, shown at 2x playback. Source: Intel IoT
DevKit
[`driver-action-recognition.mp4`](https://github.com/intel-iot-devkit/sample-videos/raw/master/driver-action-recognition.mp4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A single fixed crop
is applied before inference; there is no per-frame reframing.*

- **Natural variation:** changing background and illumination, continuous
  yaw/pitch motion, and small body translations.
- **Graceful recovery:** one 0.83-second observation gap is left intact; stale
  geometry clears and the next fresh segment reacquires the plane.
- **Measured scope:** 422 of 450 source frames produced fresh MediaPipe
  observations (93.8%); this is per-clip behavior, not general robustness.

</details>

<details>
<summary>Model-free pipeline integration demo</summary>

```bash
python -m scripts.make_demo --out output/demo.mp4
```

`ScriptedEstimator` replaces pose inference, so this exercises decoding,
temporal processing, projection, compositing, diagnostics, and encoding without
a camera, model download, or real face.

</details>

### Backends and filters

| Backend | Install | Pose source | Intended use |
|---|---|---|---|
| `mediapipe` | included by default | Face Landmarker + canonical-face PnP | Primary maintained/default adapter; validate on target footage and hardware |
| `hopenet` | `pip install -e ".[hopenet]"` | ResNet-50 binned Euler regression | Research checkpoint-compatibility adapter; paper metrics are not reproduced here |
| `6drepnet` | manual, isolated setup only | Continuous 6D rotation representation | Experimental third-party compatibility adapter; isolated setup required |

<details>
<summary>Optional backend installation notes</summary>

6DRepNet's published package conflicts with MediaPipe's OpenCV provider, so the
project deliberately does not advertise a one-command extra for it. Hopenet's
first run downloads a pinned 95,924,799-byte checkpoint.

</details>

MediaPipe is the default because it is the primary dependency-complete and
maintained path, not because this repository establishes superior accuracy.
Deterministic CI does not run real pose-model inference; every backend requires
validation on the intended footage and hardware.

- **`fir`** — default fixed-delay filter for offline processing; file rendering
  can compensate its known group delay.
- **`kalman`** — the project's current causal live candidate: the registered
  A100 constant-angular-velocity attitude design, with no fixed frame delay.
- **`oneeuro`** — retained comparison and compatibility baseline; its effective
  lag depends on motion.
- **`none`** — diagnostic baseline for observing detector noise.

## Reference

### Limitations

This is a local research renderer, not a hardened desktop application, hosted
service, biometric system, or safety system. Public demos and validation
receipts are scoped examples rather than general guarantees.

- One face is tracked; identity association and multi-face rendering are not
  implemented.
- Camera intrinsics default to `focal = max(width, height)` unless supplied;
  face-size depth remains an approximation.
- Rotation is filtered as Euler-angle streams; extreme poses near gimbal lock
  remain a limitation.
- OpenCV writes constant-frame-rate video with a portability-oriented `mp4v`
  codec. Audio is omitted unless `--preserve-audio` is used.
- The system estimates head orientation, not eye gaze, identity, attention,
  intent, or medical state.
- Optional deep backends need more cross-device and real-footage validation
  than the default MediaPipe path.

### Repository

```text
frame2frame/
  content.py    video and latest-frame content sources
  pose/         pose contracts and backend adapters
  filters.py    FIR, Kalman, and One Euro filters
  _tracking.py  temporal alignment and dropout state
  geometry.py   camera math and screen projection
  render.py     perspective warp and compositing
  pipeline.py   end-to-end orchestration
scripts/        reproducible demos, benchmarks, and receipts
tests/          unit, invariant, integration, packaging, and CLI tests
docs/           architecture, protocols, usage, privacy, and validation
```

New contributors should start with the
[architecture reading order](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/ARCHITECTURE.md#reading-order), then use the
change-to-test map in
[CONTRIBUTING.md](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/CONTRIBUTING.md).

### Contributing, citation, and license

Please use [GitHub issues](https://github.com/jasperzhao05/frame2frame-virtual-screen/issues)
for bugs and [GitHub Security Advisories](https://github.com/jasperzhao05/frame2frame-virtual-screen/security/advisories/new)
for vulnerabilities. Do not attach private face footage to a public report.

If this project supports published work, cite [`CITATION.cff`](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/CITATION.cff)
and the underlying pose/filter papers listed in the
[third-party notices](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/THIRD_PARTY_NOTICES.md).

Project code and owned documentation are released under the
[MIT License](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/LICENSE). Downloaded models, research data, and example footage
retain their upstream terms.
