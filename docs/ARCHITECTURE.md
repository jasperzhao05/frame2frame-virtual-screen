# Architecture

This note describes the invariants that make `frame2frame` more than a sequence
of unrelated video operations.

## Glossary

- **source** — one file or webcam selected by `PipelineConfig`;
- **backend** — a named pose implementation such as `mediapipe` or `hopenet`;
- **estimator** — the runtime object created for a backend, or injected by a
  caller;
- **observation** — the backend-independent `FaceObservation` passed from
  perception to temporal processing and rendering;
- **ready packet** — one source frame paired with the filtered observation and
  opacity that belong on that frame;
- **tracking segment** — consecutive temporal state between initial acquisition
  and a sustained-dropout reset;
- **nominal time** — deterministic file time derived from frame index and FPS;
- **capture time** — monotonic elapsed time measured for a live webcam;
- **publish** — the final atomic installation of a complete plot or video after
  processing and optional audio remuxing succeed.

The words model, detector, backend, and estimator are not interchangeable. A
model is an asset or neural architecture; a detector finds a face; a backend
converts perception results into the shared observation contract; an estimator
is one live backend instance.

## System invariants

These six rules are the shortest useful description of the architecture.

1. **Frame conservation and order.** During a successful run, unless a user
   stops preview, every decoded frame leaves the pipeline exactly once and in
   source order. A missed face never shortens the output.
2. **One temporal step per source step.** Rotation, center, and size advance on
   the same timeline. A detector miss may hold the previous observation, but it
   does not compress time or skip the filter update.
3. **Tracking-segment isolation.** A sustained dropout closes and flushes the
   current delayed segment, resets temporal state, and prevents reacquisition
   from blending with stale history.
4. **One renderer coordinate convention.** Backend-specific basis and sign
   conversions stay inside pose adapters. Geometry, rendering, and custom
   estimators share the convention documented below.
5. **Explicit resource ownership.** The pipeline closes readers, writers,
   preview windows, and estimators it creates. An estimator injected by a caller
   remains caller-owned.
6. **Per-artifact atomic publication.** A complete plot or video replaces its
   own destination atomically. The plot is published first; a later remux
   failure preserves the old video but does not roll back the new plot.

The strongest executable examples are in `tests/test_pipeline_timing.py`,
`tests/test_pipeline_resources.py`, `tests/test_pipeline_publish.py`,
`tests/test_geometry.py`, `tests/test_video_io.py`, and `tests/test_audio.py`.
Tests may inspect private temporal objects when necessary to protect an
invariant; those private names are not part of the public API.

## Reading order

For a first code review, read the package in this order:

1. `config.py` — supported configuration and validation boundaries;
2. `pose/base.py` — `HeadPose`, `FaceObservation`, and estimator ownership hooks;
3. `video.py` and `_media.py` — decode/encode geometry, writer validation, staged
   publication, and optional audio remuxing;
4. `pose/<backend>.py` — conversion from a backend into the shared observation;
5. `filters.py` — the three temporal policies and their delay contracts;
6. `_tracking.py` — dropout segments, filter progression, and source alignment;
7. `geometry.py` — the pure camera and gaze-plane math;
8. `_textures.py` and `render.py` — normalized texture input, projection guards,
   and compositing;
9. `_diagnostics.py` — bounded, delay-aligned plot data and atomic plot output;
10. `pipeline.py` — resource wiring, frame processing, emission, and publish;
11. `cli.py` — translation from command-line flags into `PipelineConfig`.

This order introduces contracts before orchestration. The CLI and `run()` are
short entry points, but starting with them before learning `FaceObservation` and
the temporal invariants makes the internal packet flow harder to interpret.

## Data flow

```text
source ─▶ VideoReader ─▶ decoded frame ─▶ PoseEstimator ─▶ FaceObservation
                              │                                  │
                              └────────▶ _TemporalTracker ◀──────┘
                                         smooth / hold / align
                                                  │
                                                  ▼
                                      ready frame/observation pair
                                                  │
                                                  ▼
                                        gaze-plane projection
                                                  │
                                                  ▼
                                      perspective composite
                                                  │
                                                  ▼
                                         display / VideoWriter
```

`FaceObservation` is the boundary between perception and rendering. It contains:

- yaw, pitch, and roll in degrees;
- face center in image pixels;
- half the face height as a scale proxy;
- a bounding box and optional landmarks.

The renderer does not know which model produced the observation. Tests and the
synthetic demo use this boundary to replace the model with `ScriptedEstimator`.

## Pipeline ownership

`pipeline.run` wires the reader, estimator, filter, renderer, writer, and final
publish step. It does not implement their state machines. The smoother owns its
filter coefficients and sample history. `_TemporalTracker` exclusively controls
when that smoother advances or resets, and owns dropout segmentation and offline
delay compensation.

The temporal tracker keeps one source-ordered queue of frames whose filtered
observations have not matured yet. A frame enters that queue once and leaves it
once. This makes output order an invariant of the data structure instead of a
condition coordinated between parallel queues and readiness flags.

Terminal rendering, writing, and preview are similarly isolated in
`_FrameEmitter`. Keeping these responsibilities separate makes resource
cleanup and publish ordering visible in the top-level function.

A ready packet is not necessarily delayed. Offline FIR compensation retains
packets for the smoother's fixed group delay; webcam, display, One Euro, and
passthrough paths use zero additional queue delay.

## Coordinate conventions

The camera frame is OpenCV-like:

- `+X` points image-right;
- `+Y` points image-down;
- `+Z` points away from the camera;
- positive yaw turns the forward ray toward image-left;
- positive pitch tilts it toward image-up;
- at neutral yaw/pitch, positive roll turns the local right axis clockwise in
  the image.

MediaPipe's transformation matrix uses a different basis. The default adapter
changes basis with `diag(1, -1, -1)` before decomposing the rotation. Keeping
that conversion inside the adapter lets geometry and custom backends share one
documented convention.

The renderer reconstructs that convention with one rigid rotation:

```text
R_head = Rx(-pitch) @ Ry(yaw) @ Rz(roll)
forward = R_head @ (0, 0, -1)
```

At neutral pose the person faces the camera, so the forward ray is `-Z`, not
the camera's `+Z` viewing direction. The screen right/down basis comes from the
first two columns of the same matrix, while the debug forward axis uses the ray
above. This keeps screen placement and perspective foreshortening consistent.
Backend-specific sign changes belong in adapters, never in screen geometry.

## Temporal stabilization

The temporal state has six channels:

```text
yaw, pitch, roll, center-x, center-y, face-size
```

Filtering only angles leaves the screen anchor and scale noisy; filtering those
six values on different timelines makes the screen lag behind its face. By
default, all channels advance together.

### FIR

The default filter is a per-channel Kaiser-window low-pass designed with
`scipy.signal.kaiserord` and `firwin`. Yaw, pitch, and roll have independently
scaled cutoffs because their detector noise differs; position and size use the
base cutoff. Filter state is primed from the first value to avoid a zero-valued
startup ramp.

An odd-length linear-phase FIR has constant group delay:

```text
delay = (number_of_taps - 1) / 2
```

At the default 30 fps design, that is 11 frames. For an offline file, the
pipeline holds decoded frames in a queue and pairs each with the filtered
observation that belongs to it. This corrects temporal alignment in the
rendered file; it does not make a causal filter instantaneous. Webcam/display
paths avoid the extra frame queue because a viewer would feel that latency.

### One Euro

The One Euro filter raises its cutoff as estimated motion speed rises. A steady
head receives stronger smoothing; a fast movement receives a faster response.
It is nonlinear and its effective lag is signal-dependent, so it does not have
one fixed group-delay number. File inputs use their deterministic frame cadence;
webcam inputs supply measured monotonic time deltas to the filter. FIR remains
a uniformly sampled filter and is therefore best suited to file processing.

### Missing detections

A missing observation must not shorten the video. Its frame passes through and
remains in sequence. During a short gap, the last observation is held and fed
to the filter once per decoded frame. The FIR therefore keeps the cutoff and
group delay implied by the video FPS, and its delayed-frame queue remains
bounded even when detections are sparse. The virtual screen stays fully visible
for `dropout_hold_seconds`, then fades until `dropout_reset_seconds`. A held
screen is not a new detection: stale boxes and pose axes are not drawn, and the
face count is unchanged. At reset, the current segment is padded and closed and
all temporal state is cleared so reacquisition cannot blend with stale history.

## Gaze-plane projection

“Gaze plane” is historical project terminology for a plane oriented from
estimated head pose; it does not imply eye-gaze estimation.

The renderer approximates camera intrinsics as:

```text
focal = max(frame_width, frame_height)
principal_point = image center
```

unless a focal length is supplied. Face depth is estimated from focal length
and detected face size. The face center is deprojected into camera space, and a
plane center is placed forward from it along the head ray.

The desired on-image screen width and height are converted to camera-space
half-extents at the plane depth. Head rotation orients the plane, its four
corners are projected, and the texture is mapped with a perspective transform.
Corners stay floating-point through that transform; only diagnostic borders
are rounded to integer pixels.

Corner order is texture-local top-left, top-right, bottom-right, bottom-left;
the renderer does not implicitly mirror content. If any corner is non-finite,
on the camera plane, or behind it, that frame's screen overlay is skipped rather
than sending a singular or inverted quadrilateral into OpenCV.

This is a visual screen-placement model, not metric 3D reconstruction. A
calibrated focal length improves perspective, but face-size depth remains an
approximation.

## I/O and failure boundaries

The top-level lifecycle is intentionally ordered:

```text
validate source and paths
    → preflight optional ffmpeg
    → open reader and validate source FPS
    → create or borrow estimator
    → prepare filter, texture, writer, and preview
    → process frames and flush the final tracking segment
    → release the encoder
    → save diagnostics
    → remux optional audio or install the staged video
    → close pipeline-owned resources
```

In particular, `ffmpeg` never reads from an open `VideoWriter`, and a plot
failure occurs before the staged video is published. Those orderings protect an
existing destination from a partially successful run.

- Readers determine output dimensions from a decoded, post-rotation frame.
- Output parents are created explicitly.
- Reader, writer, preview, and internally created estimator resources are
  closed on success and failure. An injected estimator remains caller-owned.
- Model downloads are written to a temporary sibling, SHA-256 checked, and
  atomically moved into the cache.
- OpenCV video encoding does not preserve audio. The opt-in audio path renders
  video first, asks local `ffmpeg` to stream-copy source audio, and only then
  installs the completed destination.
- OpenCV emits constant-frame-rate video and this pipeline does not preserve
  source frame timestamps. Exact audio synchronization therefore assumes CFR
  input; normalize VFR footage before remuxing.
- MediaPipe receives deterministic frame timestamps for files and monotonic
  capture timestamps for webcams. OpenCV webcam recording is still written at
  a declared constant FPS; this does not preserve irregular capture timing.

## Custom estimators and internal change seams

### Custom estimator

Implement `PoseEstimator.estimate(frame_bgr)` and return one
`FaceObservation | None`. Timestamp-aware estimators can additionally override
`estimate_at(frame_bgr, timestamp_ms)`; the default method delegates to
`estimate`. `run()` currently accepts a caller-owned estimator through this
narrow injection seam; this is not a general backend plugin API. Keep imports
for model-specific packages inside the adapter so the base package remains
lightweight.

### Internal screen-model seam

`gaze_plane_corners` is pure NumPy and independently testable. A calibrated
camera, different plane-placement rule, or quaternion rotation path can be
introduced there without coupling it to decoding or a model.

### Internal temporal-model seam

Filter implementations are internal. The tracker expects `update`,
`update_position`, and `reset`; a fixed-delay implementation also exposes
`group_delay` so offline frames can be aligned. These hooks describe the current
module boundary, not a supported third-party plugin API.
