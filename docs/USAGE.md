# Usage and configuration

This guide covers the supported CLI and an intentionally narrow Python
integration seam. Renderer, filter, and backend internals remain non-public.
Commands under `scripts/` require a source checkout and must run from the
repository root; the `frame2frame` command and Python package are available
after install.

## Common CLI recipes

Check the local runtime without downloading a model or opening a source:

```bash
frame2frame --doctor
```

Missing required imports or an unusable cache produce an error exit. A missing
cached model or optional `ffmpeg` produces a warning and still exits zero.
The first check in a fresh environment can take several seconds while
MediaPipe's import path initializes Matplotlib's local font cache.

Process a file and write the default angle diagnostics:

```bash
frame2frame --input clip.mp4 --output output/clip.mp4
```

Process a file without writing a plot:

```bash
frame2frame --input clip.mp4 --output output/clip.mp4 --plot ""
```

Preview a webcam without recording it:

```bash
frame2frame --webcam 0 --filter kalman --display --output "" --plot ""
```

Press Escape to stop a preview. A webcam command writes to the configured output
unless `--output ""` is supplied.

Play a video on the virtual screen:

```bash
frame2frame \
  --input wearer.mp4 \
  --screen-video interface.mp4 \
  --screen-video-end loop \
  --screen-fit contain \
  --screen-distance 4 \
  --output output/demonstration.mp4
```

The person video owns the master timeline. The content video is sampled by
presentation time, its audio is ignored, and `hold`, `loop`, or `hide` defines
what happens after its final frame. Both file timelines are derived from frame
index and reported FPS rather than container presentation timestamps. Normalize
variable-frame-rate scene and content files to constant frame rate when exact
cross-video synchronization matters.

Configured screen video is decoded synchronously on the processing thread. It
is the deterministic file-to-file path, not the lowest-latency webcam path. For
live UI, capture, or network decoders, publish externally through
`LatestFrameSource` so codec stalls do not create a content-frame queue.

Copy source audio into a processed file:

```bash
frame2frame --input clip.mp4 --output output/clip.mp4 --preserve-audio
```

This last command requires a local `ffmpeg` executable on `PATH`. Audio
preservation is not available for webcams or runs without a video output.

Use a calibrated focal length, in pixels, when camera intrinsics are known:

```bash
frame2frame --input clip.mp4 --output output/clip.mp4 --focal-length-px 900
```

Without that flag, projection uses the larger decoded frame dimension as the
focal-length approximation.

## Python configuration entry point

The project entry point accepts a `PipelineConfig` and returns a `RunSummary`:

```python
from frame2frame import PipelineConfig, run

config = PipelineConfig(
    input="clip.mp4",
    output="output/clip.mp4",
    plot_path=None,
)
summary = run(config)

print(summary.frames)
print(summary.faces)
print(summary.mean_inference_ms)
print(summary.mean_content_ms)
print(summary.mean_render_ms)
print(summary.render_samples)
```

Exactly one of `input` or `webcam` must be set.

## `RunSummary`

| Field | Meaning |
|---|---|
| `frames` | Number of decoded frames processed, including frames without a face |
| `faces` | Number of frames with a fresh backend detection |
| `fps` | Validated source or declared webcam frame rate used by the pipeline |
| `mean_inference_ms` | Online mean estimator time per processed frame |
| `output` | Installed output path, or `None` when no video was written |
| `audio_remuxed` | Whether the optional audio-remux path completed |
| `mean_content_ms` | Online mean source-sampling and content-preparation time per emitted packet while screen rendering is enabled |
| `mean_render_ms` | Online mean virtual-screen render-attempt time for packets containing an observation and content |
| `content_samples` | Number of emitted packets for which the configured or injected content source was sampled |
| `render_samples` | Number of timed virtual-screen render attempts |

`faces` does not count a held observation during a short dropout as a new
detection. Stopping a preview early returns a summary for the frames processed
before Escape.

## `PipelineConfig`

### Source and backend

| Field | Default | Contract |
|---|---:|---|
| `input` | `None` | File path; mutually exclusive with `webcam` |
| `webcam` | `None` | Non-negative device index; mutually exclusive with `input` |
| `backend` | `"mediapipe"` | Registered pose backend name |
| `backend_kwargs` | `{}` | Keyword arguments forwarded to the backend constructor |
| `filter` | `FilterConfig()` | Temporal filter configuration described below |
| `screen` | `ScreenConfig()` | Screen geometry and appearance described below |

The pipeline supplies the validated source FPS to a configured backend. An
explicit `backend_kwargs["fps"]` takes precedence. Examples:

```python
from frame2frame import PipelineConfig

mediapipe_config = PipelineConfig(
    input="clip.mp4",
    backend_kwargs={
        "model_path": "/models/face_landmarker.task",
        "min_detection_confidence": 0.6,
        "min_tracking_confidence": 0.6,
    },
)

hopenet_config = PipelineConfig(
    input="clip.mp4",
    backend="hopenet",
    backend_kwargs={
        "weights": "/models/hopenet.pkl",
        "device": "cpu",
        "face_model_path": "/models/face_landmarker.task",
    },
)
```

Caller-supplied model paths must exist. Project-managed MediaPipe and Hopenet
assets are downloaded to `~/.cache/frame2frame` or `FRAME2FRAME_CACHE`, verified
by size and SHA-256, and installed atomically.

### Output and diagnostics

| Field | Default | Contract |
|---|---:|---|
| `output` | `"output/processed.mp4"` | Video destination; use `None` to skip writing |
| `display` | `False` | Show an OpenCV preview; Escape stops processing |
| `plot_path` | `"output/angle_processed.png"` | Angle diagnostics destination; use `None` to skip |
| `max_plot_samples` | `10_000` | Positive bound on retained diagnostic samples |
| `preserve_audio` | `False` | Remux source audio into file output with local `ffmpeg` |

Input, texture, screen-video, output, and plot paths must be distinct, including
existing symlink and hard-link aliases. Output parents are created when needed.

Each video or plot is published atomically from a temporary sibling after that
artifact succeeds. They are not one cross-file transaction: the plot is
published before the final video, so a later remux failure preserves the old
video but does not roll back the newly written plot.

### Rendering

| Field | Default | Meaning |
|---|---:|---|
| `draw_screen` | `True` | Composite the virtual screen |
| `draw_axis` | `False` | Draw a fresh-detection pose axis |
| `draw_bbox` | `False` | Draw a fresh-detection face box |

`ScreenConfig` controls screen placement and appearance:

| Field | Default | Meaning |
|---|---:|---|
| `distance_mul` | `2.0` | Plane distance as a multiple of `depth_scale` |
| `width_mul` | `4.0` | Screen width in face-size units |
| `height_mul` | `2.0` | Screen height in face-size units |
| `depth_scale` | `6.0` | Face-depth scale used by the approximate camera model |
| `min_size_px` | `40.0` | Lower bound for each projected screen dimension in pixels |
| `alpha` | `0.8` | Texture blend strength from 0 to 1 |
| `focal_length` | `None` | Pixel focal length; defaults to the larger frame dimension |
| `texture_path` | `None` | Optional image; otherwise a generated default texture is used |
| `border_color` | `(0, 200, 255)` | BGR diagnostic border color |
| `border_thickness` | `0` | Non-negative border thickness in pixels |
| `video_path` | `None` | Optional screen-content video; mutually exclusive with `texture_path` |
| `video_end` | `"hold"` | End policy: `hold`, `loop`, or `hide` |
| `content_fit` | `"stretch"` | Mapping policy: `stretch`, `contain`, or `cover` |

```python
from frame2frame import PipelineConfig, ScreenConfig, run

summary = run(
    PipelineConfig(
        input="clip.mp4",
        output="output/custom.mp4",
        screen=ScreenConfig(
            texture_path="screen.png",
            width_mul=3.0,
            height_mul=1.5,
            focal_length=900.0,
        ),
        draw_axis=True,
    )
)
```

`stretch` maps the full image to the full plane and preserves the historical
behavior. `contain` keeps the complete content and adds centered bars when its
aspect differs from the screen. `cover` fills the plane by cropping centered
source coordinates. These mappings change homography coordinates directly;
they do not allocate a resized content canvas for each frame.

### Dynamic and real-time content

`run(..., content_source=...)` accepts either an object implementing
`frame_at(ContentRequest)` or a callable taking one `ContentRequest`. The
request contains the zero-based person-frame index, that packet's presentation
time, and whether the person source is live. Returning `None` hides the screen
for that packet.

For a synchronously generated dashboard, the smallest adapter is a function:

```python
import cv2
import numpy as np

from frame2frame import FilterConfig, PipelineConfig, run


def dashboard(request):
    image = np.zeros((360, 640, 3), np.uint8)
    cv2.putText(
        image,
        f"t = {request.media_time_seconds:.2f} s",
        (40, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


run(
    PipelineConfig(
        webcam=0,
        display=True,
        output=None,
        plot_path=None,
        filter=FilterConfig(kind="kalman", smooth_translation=False),
    ),
    content_source=dashboard,
)
```

A synchronous callback runs on the rendering thread and must return promptly.
For an external camera, UI, decoder, or network producer, publish from its own
thread into `LatestFrameSource`:

Call one of these publication forms from your existing producer loop.

```python
from frame2frame import FilterConfig, LatestFrameSource, PipelineConfig, run

content = LatestFrameSource()


def publish_from_producer(next_bgr_frame):
    # Safe copy by default.
    content.publish(next_bgr_frame)


def transfer_from_producer(next_bgr_frame):
    # Zero-copy publication for a fresh, C-contiguous OpenCV frame that will
    # never be mutated again.
    content.publish(next_bgr_frame, copy=False)


# Consumer thread:
summary = run(
    PipelineConfig(
        webcam=0,
        display=True,
        output=None,
        plot_path=None,
        filter=FilterConfig(kind="kalman", smooth_translation=False),
    ),
    content_source=content,
)
```

The content contract is deliberately small:

- return `None` when no content is available;
- prefer a C-contiguous `uint8` BGR array for the fastest path;
- grayscale and BGRA arrays are also accepted; and
- keep synchronous callbacks short, or use `LatestFrameSource` for an external
  producer.

`LatestFrameSource` has one slot, not a queue. A newer publication replaces the
older one; the current snapshot remains available until another frame replaces
it or `clear()` is called. Sampling never calls or joins the producer; a short
lock protects only the reference swap. This bounds memory and prevents the
content interface itself from becoming an ever-older playback backlog.
Capacity-one storage does not establish camera-to-display latency or rule out
buffering elsewhere in capture, inference, rendering, encoding, or display.

With `copy=False`, publication retains the supplied array by reference and the
producer must never mutate it again. The renderer's no-conversion fast path
additionally requires a C-contiguous `uint8` BGR frame.

Injected content sources remain caller-owned. Configuration-created screen
videos are closed by the pipeline on success and failure. Timestamp-aware
sources such as `VideoContentSource` select from the original packet's
zero-based media time even after offline FIR buffering. `LatestFrameSource`
instead returns the freshest frame available at sampling time and is intended
for live pipelines; the built-in pipeline rejects it for file input. Use a
timestamp-aware callback or `VideoContentSource` for deterministic file
processing.

From a source checkout, run both levels of the real-time test:

```bash
# Visual webcam integration test
python -m scripts.live_content_demo --webcam 0 --producer-fps 60

# The same capacity-one path with a looping video producer
python -m scripts.live_content_demo --webcam 0 --content-video interface.mp4

# Deterministic component receipt
python -m scripts.benchmark_dynamic_content --check --format markdown
```

The webcam demo's independent producer renders a sequence number, timestamp,
scan line, and four distinct corners. When rendering is faster than production,
a sequence may repeat; when production is faster, sequences may jump. They must
never move backward or replay from an accumulated queue.

The deterministic receipt checks latest-frame replacement, zero-conversion BGR
preparation, exact frame mapping, and repeatable output digests across runs in
the same environment. It reports sampling, preparation, and compositing p50/p95
timings without timing gates. It excludes camera capture, pose inference,
codecs, and display, so it is a component receipt rather than an end-to-end
latency measurement.

### Temporal behavior

The canonical `FilterConfig.kind` values are `"fir"`, `"kalman"`, `"oneeuro"`,
and `"none"`. Configuration also accepts `"off"` and `"passthrough"` as
compatibility aliases for `"none"`; the CLI advertises only the canonical names.

- FIR is a per-channel Kaiser-window low-pass with a fixed group delay. It is the
  default for file processing.
- Kalman is the project's current live candidate for further evaluation. Its
  registered A100 constant-angular-velocity model filters attitude with
  pipeline-measured intervals between successive webcam frames and has no fixed
  frame delay.
- One Euro adapts its cutoff to movement speed and uses elapsed capture time. It
  remains available for comparison and backward compatibility, not as the
  current live default.
- None is the unfiltered diagnostic baseline.

Every `FilterConfig` field is listed here:

| Field | Default | Meaning |
|---|---:|---|
| `kind` | `"fir"` | FIR, Kalman, One Euro, or pass-through policy |
| `smooth_translation` | `True` | Smooth face center and size where the selected filter supports it |
| `cutoff_hz` | `2.5` | Base FIR cutoff in Hz |
| `transition_hz` | `5.0` | FIR transition width used by Kaiser design |
| `ripple_db` | `60.0` | FIR stop-band attenuation target; minimum 8 dB |
| `pitch_cutoff_scale` | `0.5` | Pitch cutoff relative to `cutoff_hz` |
| `roll_cutoff_scale` | `0.1` | Roll cutoff relative to `cutoff_hz` |
| `min_cutoff` | `1.0` | One Euro minimum cutoff in Hz |
| `beta` | `0.3` | Non-negative One Euro speed coefficient |
| `d_cutoff` | `1.0` | One Euro derivative cutoff in Hz |
| `acceleration_std` | `100.0` | Kalman angular-acceleration standard deviation in degrees/s² |
| `measurement_std` | `1.0` | Kalman attitude-measurement standard deviation in degrees |

`PipelineConfig` adds these temporal controls:

| Field | Default | Meaning |
|---|---:|---|
| `compensate_delay` | `True` | Align offline FIR output with its source frames |
| `dropout_hold_seconds` | `0.2` | Duration for full-opacity held tracking |
| `dropout_reset_seconds` | `0.5` | Positive duration before segment flush and reset |

FIR and One Euro can advance face center and size with the three pose angles.
Kalman deliberately passes center and size through unchanged because position
smoothing is outside the registered A100 configuration. The CLI applies this
contract automatically for `--filter kalman`; Python callers must set
`FilterConfig(kind="kalman", smooth_translation=False)` explicitly. This avoids
silently presenting position smoothing as part of the A100 result.

For offline FIR processing, `compensate_delay=True` holds decoded frames by the
known filter delay so each smoothed pose is composited onto its corresponding
source frame. Additional delay compensation is disabled for webcam and display
paths because the user would feel that latency.

`dropout_hold_seconds=0.2` keeps the last screen fully visible across a brief
miss. It then fades until `dropout_reset_seconds=0.5`, when the current tracking
segment is flushed and temporal state is reset. Held frames do not draw stale
axes or boxes and do not increment `RunSummary.faces`.

```python
from frame2frame import FilterConfig, PipelineConfig, run

run(
    PipelineConfig(
        input="clip.mp4",
        filter=FilterConfig(
            kind="fir",
            cutoff_hz=2.5,
            smooth_translation=True,
        ),
        dropout_hold_seconds=0.15,
        dropout_reset_seconds=0.45,
    )
)
```

## Injecting a custom estimator

Every estimator returns one `FaceObservation` or `None` for each BGR frame. Pose
angles are degrees in the renderer convention: positive yaw turns toward
image-left and positive pitch tilts toward image-up.

```python
from frame2frame import (
    FaceObservation,
    HeadPose,
    PipelineConfig,
    run,
)


class MyEstimator:
    def estimate(self, frame_bgr):
        # Replace this with model inference.
        return FaceObservation.from_bbox(
            HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
            bbox=(100, 80, 220, 240),
        )


estimator = MyEstimator()
summary = run(
    PipelineConfig(input="clip.mp4", output=None, plot_path=None),
    estimator=estimator,
)
```

An injected estimator remains caller-owned: `run()` does not close it. If a
custom estimator owns resources, its caller must close them. When no estimator
is injected, `run()` creates the configured backend and closes it on success or
failure. A custom estimator only needs the methods the pipeline calls; it does
not need to inherit a package base class.

Custom estimators should reject or avoid malformed observations. The pipeline
also validates finite angles and face geometry, positive size, and non-inverted
bounding boxes before temporal processing.

## Timestamp-aware estimators

Implement `estimate_at(frame_bgr, timestamp_ms)` when an estimator needs video
tracking timestamps. Otherwise the pipeline calls `estimate`, so frame-only
estimators need no timestamp code.

- file timestamps are deterministic and increase by `1000 / fps` milliseconds;
  the first decoded frame receives `1000 / fps` rather than zero;
- webcam timestamps use monotonic elapsed capture time;
- built-in video-tracking adapters convert timestamps to strictly increasing
  integer milliseconds even when adjacent values would round to the same number;
- Kalman and One Euro use elapsed webcam time, while FIR remains uniformly
  sampled from source FPS.

OpenCV still writes webcam output at a declared constant FPS. Timestamp-aware
inference does not turn that output into a variable-frame-rate recording.

## Audio, display, and output combinations

| Configuration | Supported behavior |
|---|---|
| File + output | Encode staged silent video, then install it |
| File + output + `preserve_audio` | Encode, close writer, stream-copy optional source audio, then install |
| File + `output=None` | Process and optionally display/plot without video writing |
| Webcam + output | Preview optionally and record at declared FPS |
| Webcam + `output=None` | Preview or process without recording |
| Webcam + `preserve_audio` | Invalid configuration |

Audio preservation copies rather than transcodes the source audio codec. The
run fails explicitly if that codec is incompatible with the destination
container. OpenCV re-encodes video at constant frame rate, so convert
variable-frame-rate sources to CFR before remuxing when exact synchronization
matters.

## Failure and privacy expectations

- Configuration errors are raised before opening the source whenever possible.
- The CLI prints concise runtime errors; add `--debug` for a traceback.
- Model downloads are the only normal first-use network operation.
- File and webcam frames, observations, plots, and rendered videos remain local.
- Do not attach private faces, audio, path names, or metadata to a public issue.

See [privacy and data flow](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/PRIVACY.md),
[architecture](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/docs/ARCHITECTURE.md),
and [third-party notices](https://github.com/jasperzhao05/frame2frame-virtual-screen/blob/main/THIRD_PARTY_NOTICES.md)
for the corresponding system boundaries and asset terms.
