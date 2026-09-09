# frame2frame

**Make the experience inside AR glasses visible from the outside.**

[![CI](https://github.com/jasperzhao05/frame2frame-virtual-screen/actions/workflows/ci.yml/badge.svg)](https://github.com/jasperzhao05/frame2frame-virtual-screen/actions/workflows/ci.yml)
[![Python 3.9–3.13](https://img.shields.io/badge/python-3.9--3.13-3776AB)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/license-MIT-2ea44f)](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/LICENSE)

<p align="center">
  <img
    src="https://raw.githubusercontent.com/jasperzhao05/frame2frame-virtual-screen/main/docs/demo-vuzix-outdoor.gif"
    width="500"
    alt="A real AR-glasses scene shown beside FIR distance-4 output playing Black Myth Wukong content"
  >
</p>

*Original footage on the left; a virtual screen playing supplied*
Black Myth: Wukong *gameplay on the right. MediaPipe estimates head pose and
FIR stabilizes the screen. One continuous shot, with no screen keyframing.*

*Sources: Naomi Wu,
[“Vuzix Blade Review”](https://www.youtube.com/watch?v=sS90qEPgc50)
([CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)); Game Science,
[Black Myth: Wukong gameplay](https://www.youtube.com/watch?v=oRLhCxC886o).
The overlay is rendered by `frame2frame`, not captured from the headset.*

## Overview

AR glasses have a demonstration problem. The wearer sees a spatial display;
the audience sees only a person wearing glasses.

`frame2frame` brings that experience into the picture. Combine footage of the
wearer with an image, a second video, or live application frames. The system
estimates head pose and projects the supplied content onto a virtual screen
that follows the wearer, creating AR-style footage from an ordinary camera view.

Originally developed during a Rokid internship, this repository is an
independent open-source rebuild. It uses separately supplied content and reads
no headset output or telemetry.

## Contents

- [Quick start](#quick-start)
- [How to choose a good filter (research-oriented)](#how-to-choose-a-good-filter-research-oriented)
- [Building a reliable pipeline](#building-a-reliable-pipeline)
- [Reference](#reference)

## Quick start

Install from a source checkout with Python 3.9–3.13:

```bash
git clone https://github.com/jasperzhao05/frame2frame-virtual-screen.git
cd frame2frame-virtual-screen
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e .
```

The first MediaPipe run downloads a verified 3.6 MB model. Video processing
runs locally.

### Render a second video

Use `wearer.mp4` as the camera view and `interface.mp4` as the screen content:

```bash
# --screen-video-end loop: restart the content when it ends.
# --screen-fit contain: preserve aspect ratio without cropping.
# --screen-distance 4: place the screen farther along the head-forward ray.
frame2frame \
  --input wearer.mp4 \
  --screen-video interface.mp4 \
  --screen-video-end loop \
  --screen-fit contain \
  --screen-distance 4 \
  --output output/demonstration.mp4
```

The wearer video sets the timeline; screen content follows its playback time.

<a id="render-live-content"></a>

### Render live content (engineering-oriented)

<p align="center">
  <img
    src="https://raw.githubusercontent.com/jasperzhao05/frame2frame-virtual-screen/main/docs/demo-live-desmos.gif"
    width="640"
    alt="Actual Desmos application frames beside the same content projected onto a virtual screen following a real person's head"
  >
</p>

*Actual Desmos application capture on the left; the same frames composited onto
prerecorded person footage on the right. This recorded replay uses Kalman and
plays at source speed. The person footage substitutes for a webcam.*

*Sources: [Desmos Graphing Calculator](https://www.desmos.com/calculator) and
[Intel IoT DevKit footage](https://github.com/intel-iot-devkit/sample-videos)
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
[Capture and rendering details](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/THIRD_PARTY_NOTICES.md#recorded-application-content-illustration).*

```bash
# Webcam 0 supplies wearer footage; the default producer publishes test cards at 60 fps.
python -m scripts.live_content_demo --webcam 0
```

This example opens a preview and records `output/live-content.mp4`. Press Escape
to stop. To display your own application, capture its visual output and publish the frames to
`LatestFrameSource`; the [usage guide](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/USAGE.md#dynamic-and-real-time-content)
shows the adapter. Application-window capture is supplied by the caller.

<a id="how-to-choose-a-good-filter"></a>

## How to choose a good filter (research-oriented)

Even when a wearer appears still, their head makes small natural movements,
and pose estimates add frame-to-frame fluctuations. The virtual screen sits
at a distance along the head's forward direction. Small changes in that
direction shift the screen's position, and placing it farther from the face
can amplify the movement in the camera image.

We smooth the pose over time to reduce this visible jitter while preserving
deliberate head movement.

<p align="center">
  <img
    src="https://raw.githubusercontent.com/jasperzhao05/frame2frame-virtual-screen/main/docs/demo-comparison.gif"
    width="640"
    alt="Clean synthetic head motion compared with raw observations, EMA, FIR, One Euro, and Kalman virtual-screen responses"
  >
</p>

*One motion, five responses: Raw, EMA, FIR, One Euro, and Kalman, alongside a
clean reference. All five policies receive the same noisy input and share the
same geometry. This periodic
loop illustrates behavior; the benchmark below uses non-periodic noise.
FIR is shown without delay compensation.*

Smoothing often comes at a cost: using past poses to damp rapid fluctuations
can also slow the response to a real turn. A steady screen may therefore lag
behind the wearer, especially when the motion changes direction.
Choosing a filter means weighing **how much jitter it removes**
against **how much delay it introduces**.

### Screen Age

To put a number on that delay, we define **Screen Age**: how far back in the
reference motion we must look to best match the filtered screen. We compare
projected screen corners while holding position and scale fixed, isolating the
time shift introduced by orientation filtering.

<p align="center">
  <a href="https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/screen-age.svg">
    <img
      src="https://raw.githubusercontent.com/jasperzhao05/frame2frame-virtual-screen/main/docs/screen-age.png"
      width="640"
      alt="A real-person projected-screen worldline beside a compact calibration plot of analytic FIR delay against recovered Screen Age"
    >
  </a>
</p>

*The default FIR screen at frame `t` best matches `GT(t−11)` in this example:
**11 frames, or 367 ms at 30 fps**. Across 45 BIWI segments from 20 subjects,
the metric recovers the known 7-, 11-, and 19-frame delays of six FIR designs.*

BIWI supplies measured head-pose ground truth. We select one age per filter
configuration on development subjects and freeze it for held-out evaluation.
This calibrates the metric against known filter delay; it does not measure
camera-to-display latency. The [protocol and receipt](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/SCREEN_AGE.md)
explain the calculation and reproduction scope.

*Source: [BIWI Kinect Head Pose Database](https://vision.ee.ethz.ch/datsets.html),
for non-commercial research and education.*

### Balancing jitter and delay

With Screen Age, we can compare jitter reduction alongside its timing cost.
We also check how well each filter preserves intended motion. To isolate
observation noise, we compare each filter's response to clean and noisy versions
of the same motion. Two motions—smooth turns and stop-and-reverse movement—are
crossed with four errors: independent noise, correlated noise, bursts, and
sample-and-hold error.

The table averages noise reduction across all eight conditions and 20 fixed
seeds. Screen Age and motion error are measured on the two clean motions;
motion error is reported after removing their best time offset. All policies
use one fixed parameter set at 30 fps, with the same screen geometry.

| Policy | Noise reduction ↑ | Screen Age, frames ↓ | Motion error after alignment ↓ |
|---|---:|---:|---:|
| Raw | 0.00 ± 0.00% | 0 | 0.000 px |
| EMA 2.5 Hz | 31.64 ± 1.15% | 2 | 1.704 px |
| FIR angular default | 41.94 ± 1.84% | 11 | 1.854 px |
| One Euro angular default | 14.49 ± 0.75% | 0.5 [0–1] | 2.024 px |
| Kalman A100 | 18.07 ± 1.45% | 0.5 [0–1] | 3.646 px |

*Noise reduction: equal-condition mean ± sample standard deviation across
seeds. Fractional ages average two integer motion-level estimates. EMA is a
benchmark-only baseline; translation smoothing is excluded.*

FIR removes the most noise in all 20 trials, with an 11-frame delay. EMA
preserves the clean motion slightly better after alignment and responds sooner.
These results describe the tested settings; selecting among them depends on
how the output will be used.

For offline footage, frame buffering lets us align FIR output with its source
video. A live preview keeps the causal response, so the same delay is visible.
The [full comparison](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/BENCHMARKS.md)
includes worst cases, error against the current screen position, filter
parameters, and commands to reproduce the results. These geometric measures
do not establish human preference or a universal filter ranking.

## Building a reliable pipeline

The screen's motion is only one part of the finished shot. Its content must
stay synchronized with the wearer, and a missed detection must be handled
without breaking the video.

```text
Video / webcam → pose → temporal state → screen geometry ────────┐
Image / video / latest live frame → content preparation ─────────┴→ warp + composite
```

Orientation, face position, scale, and video content stay on the same source
frame timeline through offline filtering. During a brief detection gap, the
screen holds and fades. A sustained gap clears the tracking state so a new
detection starts fresh. Every source frame stays in sequence, including frames
without a face.

Live applications publish into a single latest-frame slot: a new frame
replaces the old one. That keeps application content from accumulating in a
playback queue while the renderer works.

[Deterministic checks](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/RELIABILITY.md)
verify frame conservation and repeatable output; focused tests cover recovery
and content synchronization. A separate [real-video check](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/VALIDATION.md)
records input identity, detection coverage, and run timings for the MediaPipe
path. The [architecture notes](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/ARCHITECTURE.md)
explain the coordinate conventions and implementation boundaries.

## Reference

### Backends and filters

MediaPipe is included by default. Hopenet and 6DRepNet are optional research
adapters with additional setup and validation requirements.

| Backend | Setup | Role |
|---|---|---|
| `mediapipe` | Included | Default maintained path: face landmarks + canonical-face PnP |
| `hopenet` | `pip install -e ".[hopenet]"` | Research checkpoint adapter |
| `6drepnet` | Isolated manual setup | Experimental third-party adapter; conflicting OpenCV dependency |

The runtime provides `fir` for fixed-delay offline smoothing, `kalman` as the
current causal live candidate, `oneeuro` as an adaptive comparison option, and
`none` for diagnostic pass-through. See the [configuration guide](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/USAGE.md)
for parameters and integration examples.

### Scope

- One person's head pose drives the screen. Eye gaze and headset telemetry are
  outside the model; multi-face identity tracking is not implemented.
- Camera intrinsics are approximated unless supplied; depth is estimated from
  apparent face size. Euler-angle filtering remains limited near extreme poses.
- Output uses constant-frame-rate video. Audio is omitted unless
  `--preserve-audio` is enabled; normalize variable-frame-rate inputs when
  precise synchronization matters.
- Demos and checks establish scoped behavior. Pose accuracy, human preference,
  and end-to-end live latency require separate evaluation on target footage
  and hardware.

### Contributing and citation

Start with the [architecture reading order](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/ARCHITECTURE.md#reading-order)
and [contributing guide](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/CONTRIBUTING.md).
Report bugs through [issues](https://github.com/jasperzhao05/frame2frame-virtual-screen/issues)
and vulnerabilities through [Security Advisories](https://github.com/jasperzhao05/frame2frame-virtual-screen/security/advisories/new).
For published work, use [CITATION.cff](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/CITATION.cff)
and cite the underlying methods listed in the [third-party notices](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/THIRD_PARTY_NOTICES.md).

Code and owned documentation are [MIT licensed](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/LICENSE).
Third-party footage, models, and data retain their own terms. The project is
not affiliated with or endorsed by Rokid or Vuzix.
