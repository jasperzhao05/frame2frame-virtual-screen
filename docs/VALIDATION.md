# Real-video validation receipt

`scripts/validate_real_video.py` runs one file through the actual pipeline and
writes a machine-readable JSON receipt. Its purpose is narrower than a model
benchmark: it makes a real-footage usage example traceable and checks that the
pipeline decoded, processed, and encoded the expected media structure.

## Reproduce the primary path

From the repository root, first download a pinned, attributed Intel IoT DevKit
sample. `--limit 1` selects the first clip in the fixed registry:

```bash
python -m scripts.fetch_examples --download-only --limit 1
```

Then run the default MediaPipe + FIR path:

```bash
python -m scripts.validate_real_video \
  --input examples/inputs/head-pose-face-detection-female.mp4 \
  --output examples/outputs/head-pose-face-detection-female.validated.mp4 \
  --receipt output/real-video-validation.json
```

The first MediaPipe run may download the repository-pinned Face Landmarker
asset. The source video and model asset are independently checked against their
pinned byte counts and SHA-256 digests by the existing download paths.

The command exits zero only when every operational check in the receipt passes.
The processed video and receipt are local evidence and remain Git-ignored.

Repository maintainers can run the same primary path from GitHub Actions with
the manually dispatched `real-video-validation` workflow. It downloads exactly
one pinned clip, runs the default MediaPipe + FIR configuration, and retains the
processed video and JSON receipt as a seven-day workflow artifact. The workflow
is intentionally manual rather than a network-dependent pull-request gate.

For a publishable receipt, run from a clean commit and confirm
`project.source_revision_status` is `clean_commit`. An operational pass from a
`dirty_worktree` still describes the measured run, but the commit alone cannot
reconstruct the modified source and therefore is not a reproducible revision.

## What the receipt records

- exact input and output SHA-256 digests and byte counts;
- input source, license, attribution, and pinned-registry match;
- project version, Git commit/dirty state, command, runtime versions, and full
  `PipelineConfig` used for the run;
- frame counts, effective FPS, dimensions, and derived durations from separate
  full-decode passes before and after the pipeline;
- pipeline processed frames, fresh-detection frames, fresh-detection frame
  rate, mean estimator-call time, and whole-pipeline wall time;
- explicit frame-count, dimensions, FPS, and one-frame duration-conservation
  checks.

`fresh_detection_frame_rate_pct` means
`RunSummary.faces / RunSummary.frames * 100`. A held observation during a short
dropout is not a fresh detection. The value is neither a count of distinct
people nor an accuracy score.

Timing values describe this single machine and run. `mean_estimator_inference_ms`
measures estimator calls inside the pipeline. `pipeline_wall_seconds` includes
decode, estimation, smoothing, rendering, and encode, but excludes the two
separate full-decode probes performed before and after the run. Neither value is a
capture-to-display latency measurement.

## Evidence boundary

A passing receipt demonstrates only that this exact content, configuration,
revision, and environment completed the recorded operational checks. It does
not establish:

- ground-truth head-pose or face-detection accuracy;
- perceptual screen stability or visual quality;
- a latency service-level objective or cross-machine performance;
- long-running reliability, deployment safety, or production readiness.

Those claims require separate labeled data, perceptual evaluation, device
matrices, repeated latency measurements, and soak tests. Do not aggregate these
receipts into an accuracy claim unless a separately specified ground-truth
protocol is added.

Re-running a clean revision should reproduce the receipt schema and operational
checks, not necessarily byte-identical encoded output or timing. Codec builds,
hardware, and runtime versions can change both; the output digest identifies
the bytes from this particular run.

## Other footage

The five clips registered by `scripts/fetch_examples.py` are recognized by
content digest, even if a local file was renamed. Any other input must carry a
complete source note:

```bash
python -m scripts.validate_real_video \
  --input /path/to/owned-or-authorized-clip.mp4 \
  --output output/authorized-clip.mp4 \
  --receipt output/authorized-clip.receipt.json \
  --source-uri "internal-capture:session-2026-08-17" \
  --source-license "owned capture; publication permission recorded separately" \
  --source-attribution "ZHAO SIBO"
```

The tool records caller-supplied provenance but cannot independently verify its
legal accuracy. Do not use footage that you do not own or have permission to
process and publish.
