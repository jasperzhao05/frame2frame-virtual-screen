# Pipeline reliability

`scripts/benchmark_pipeline.py` is a deterministic, model-free receipt for the
complete **file** pipeline. It joins a few high-signal contracts in one short
workload without duplicating the focused tracking, rendering, and lifecycle
tests.

## Run it

The CI-sized profile processes the same 180-frame, 320×180, 30 fps source three
times:

```bash
python -m scripts.benchmark_pipeline --check
```

The extended local profile processes a 1,800-frame source five times:

```bash
python -m scripts.benchmark_pipeline --extended --check
```

`--frames` and `--runs` override those profiles. JSON is the default receipt;
Markdown is available for a human-readable attachment:

```bash
python -m scripts.benchmark_pipeline \
  --check \
  --format json \
  --output output/pipeline-reliability.json
```

Every receipt records the Git commit and whether the worktree was clean. Use a
`clean_commit` receipt for a public comparison. A `dirty_worktree` receipt is
still honest, but the commit alone does not identify the measured source.

## Measured path

```text
synthetic MP4
  → OpenCV decode
  → deterministic ScriptedEstimator observations
  → FIR filtering and file-delay alignment
  → screen, axis, and bounding-box rendering
  → OpenCV encode
  → decoded-pixel verification
```

## Contracts and metric

### Frame conservation

Every run requires exact equality among:

```text
decoded source frames == RunSummary.frames == decoded output frames
```

The observed face count must also match the scripted schedule. This catches
dropped frames, early termination, and an unflushed FIR delay queue.

### Repeated decoded output

Every output is decoded and hashed with BLAKE2b. All run digests must match.
Decoded pixels are used so irrelevant container metadata is not mistaken for
pipeline state drift. This is a within-environment check; codec builds on two
platforms need not produce identical pixels.

### Scripted dropout schedule

The observation source contains a gap slightly longer than the configured
0.5-second tracking-reset threshold and schedules observations afterward. The
receipt records the gap duration and the number of later observations.

That proves only that the end-to-end workload includes both conditions. Direct
reset and reacquisition behavior is proved by the focused tests
`test_long_dropout_flushes_pending_segment_and_resets_filter` and
`test_reacquisition_after_wall_clock_reset_starts_a_fresh_filter_segment` in
`tests/test_pipeline_timing.py`.

### Pipeline throughput

`pipeline_frames_s` includes file decode, scripted estimation, filtering,
rendering, and file encode. It has no pass/fail floor because CPU, codec build,
power state, and background load affect it.

Do not call it neural-backend FPS, sustained webcam FPS, or camera-to-display
latency. Compare commits only on the same machine and environment.

## Focused evidence kept outside the benchmark

Declared normal operating poses—yaw ±45 degrees and pitch/roll ±30 degrees—are
tested directly against the runtime projection guard in `tests/test_render.py`.
That parameterized test requires every representative boundary combination to
return a finite valid quad. Extreme gimbal-lock and behind-camera cases are not
part of the normal envelope.

Reader, writer, failure-cleanup, and atomic-publication ownership remain covered
by `tests/test_pipeline_resources.py`, `tests/test_pipeline_publish.py`, and
`tests/test_video_io.py`. Keeping those direct oracles out of the benchmark
avoids a second private instrumentation framework.

## Evidence boundary

A passing receipt establishes frame conservation, repeated decoded-pixel
stability, inclusion of the scripted dropout schedule, and the recorded
machine-dependent throughput for this synthetic file workload. It does not
establish tracker recovery by itself, pose-model accuracy, camera latency,
hardware coverage, or a production SLO.
