# Contributing

Bug reports, focused fixes, benchmark improvements, and new pose backends are
welcome. Please open an issue before a large behavioral change so its CLI or
configuration impact, latency, and validation plan can be agreed first.

## Development setup

These commands assume a source checkout. Run repository scripts and tests from
the repository root:

```bash
git clone https://github.com/jasperzhao05/frame2frame-virtual-screen.git
cd frame2frame-virtual-screen
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

The commands below use POSIX shell spelling, but each quality command is a
single cross-platform Python or tool invocation and can be run separately from
PowerShell. The test suite does not download models. Pipeline tests inject a
scripted estimator so local and CI runs stay fast and deterministic.

## Validation levels

### Every change

Run the core quality gate before opening a pull request:

```bash
ruff check .
ruff format --check .
mypy
python -m pytest -q -m "not integration" --cov=frame2frame --cov-report=term-missing
python -m scripts.benchmark_smoothing --check
python -m scripts.benchmark_pipeline --check
```

Ruff enforces a McCabe complexity ceiling of 10. Treat that limit as a design
signal: move a coherent responsibility behind a named interface instead of
silencing the check or mechanically fragmenting one control flow.

### Filtering, timing, rendering, or pipeline behavior

Run both deterministic benchmarks and the short synthetic end-to-end demo:

```bash
python -m scripts.benchmark_smoothing --check
python -m scripts.benchmark_pipeline --check
python -m scripts.make_demo --frames 36 --input output/contribution-input.mp4 --out output/contribution-demo.mp4 --plot ""
```

Generated videos, plots, weights, and downloaded examples must remain
untracked. Report benchmark changes with the exact command, seed, Python
version, machine, and before/after values. Do not present throughput from
different hardware as a regression.

### Real-video or pose-backend behavior

Run `frame2frame --doctor`, process an attributed or owned clip, and retain an
operational receipt:

```bash
python -m scripts.validate_real_video \
  --input examples/inputs/head-pose-face-detection-female.mp4 \
  --output output/real-video-validation.mp4 \
  --receipt output/real-video-validation.json
```

Use a clean commit for publishable evidence. Treat fresh-detection coverage and
timings as run facts, not accuracy, perceptual-quality, latency-SLO, or
production-readiness evidence. See `docs/VALIDATION.md` for provenance rules.

### Packaging or dependency changes

In addition to the core gate, build both artifacts and validate their package
metadata:

```bash
python -m build
python -m twine check dist/*
```

For a release-sensitive change, install the wheel and source distribution in
separate clean virtual environments. Verify `python -m frame2frame --help`,
`pip check`, the imported version, and that imports resolve from the installed
environment rather than the source checkout.

### Audio changes

Install `ffmpeg`, exercise `--preserve-audio` with an audio-bearing CFR source,
and test all of these paths:

- source with audio;
- source without audio;
- missing `ffmpeg`;
- incompatible or failed remux;
- shorter and longer source audio;
- an existing destination that must survive failure.

## Change-to-test map

Start with the smallest relevant command, then run the full gate before the
pull request. Some files appear in more than one row because the pipeline
contracts cross module boundaries.

| Area changed | Focused validation | Additional contract check |
|---|---|---|
| Configuration or CLI | `python -m pytest -q tests/test_cli.py tests/test_config.py tests/test_doctor.py` | `python -m frame2frame --help` and `--doctor` |
| Filters | `python -m pytest -q tests/test_filters.py tests/test_pipeline_timing.py` | smoothing benchmark |
| Geometry | `python -m pytest -q tests/test_geometry.py` | render tests, synthetic demo, and pipeline benchmark |
| Texture or compositing | `python -m pytest -q tests/test_textures.py tests/test_render.py` | geometry tests and synthetic demo |
| Pipeline timing or dropout | `python -m pytest -q tests/test_pipeline_timing.py tests/test_pipeline_diagnostics.py` | smoke tests and pipeline benchmark |
| Resource cleanup or publication | `python -m pytest -q tests/test_pipeline_resources.py tests/test_pipeline_publish.py` | failure-path review |
| Video reader/writer or audio | `python -m pytest -q tests/test_video_io.py tests/test_audio.py` | `python -m pytest -q -m integration` when applicable |
| Pose adapters or registry | `python -m pytest -q tests/test_pose_helpers.py` | scripted smoke test and attributed real-video receipt |
| Dependencies or metadata | `python -m pytest -q tests/test_packaging.py` | build, Twine, and clean installs |
| Benchmark protocol | `python -m pytest -q tests/test_benchmark.py tests/test_pipeline_benchmark.py` | both benchmark `--check` commands and schema review |
| README or distributed docs | `python -m pytest -q tests/test_packaging.py` | build and Twine rendering check |

## Pull-request checklist

- Explain the user-visible behavior and why the chosen design is appropriate.
- Add or update tests for success and failure paths.
- List the exact focused and full validation commands that were run.
- Preserve video/frame synchronization across missed detections and delayed
  filtering.
- Keep optional dependencies lazy; the base package must import without an
  optional deep backend installed.
- Document any download, model, dataset, sample, or dependency in
  `THIRD_PARTY_NOTICES.md`, including source and license.
- Do not include real-person footage unless distribution and subject-consent
  rights are explicit. Prefer `scripts/make_demo.py`.
- Do not include secrets, private footage, cache files, generated outputs, or
  model weights.
- Mark timing, download, audio, and third-party-asset checklist items as not
  applicable when the change cannot affect them; do not silently skip them.

## Adding a pose backend

1. Implement `PoseEstimator.estimate` in a module under `frame2frame/pose/`.
2. Return `FaceObservation` in the convention documented in `pose/base.py` and
   `docs/ARCHITECTURE.md`.
3. Override `estimate_at` only when the backend needs timestamp-aware tracking.
4. Import the backend lazily and register it in `frame2frame/pose/__init__.py`.
5. Put heavyweight packages in a named optional dependency unless a documented
   package conflict requires an isolated manual setup.
6. Make project-managed downloads checksum-verified, size-pinned, atomic,
   cacheable, and overrideable by a caller-supplied local path.
7. Add a model-free test using a fixture or fake object; do not make CI fetch
   weights.
8. Record upstream code, model, paper, license, and citation metadata.

An estimator created by the pipeline is pipeline-owned. An estimator injected
into `run()` remains caller-owned. New backends must preserve that distinction
and make `close()` safe after partial initialization.

## Benchmark discipline

`scripts/benchmark_smoothing.py` isolates detector noise by comparing each
filter's noisy-input output with a clean-input control. Keep the default seed,
trace, metric definitions, and self-check thresholds stable within a minor
release. If the protocol must change:

1. increment its `schema_version`;
2. explain the change in `docs/BENCHMARKS.md`;
3. regenerate the reference table;
4. do not compare new values directly with an older schema.

The benchmark's throughput loop updates all six filtered channels: three
angles, face-center x/y, and face size. Quality and latency values are
deterministic; wall-clock speed is not.

`scripts/benchmark_pipeline.py` protects a separate set of whole-file
contracts. Keep its synthetic pose envelope, dropout schedule, decoded-pixel
digest rule, and frame-conservation checks stable within a minor release. Its
reported throughput has no pass/fail floor and must not be labeled as backend
FPS or camera latency. See `docs/RELIABILITY.md` before changing that protocol.

## Test readability

- Prefer assertions on public behavior and externally visible events.
- White-box tests are appropriate for temporal alignment, numerical filters,
  atomic publication, and other invariants that cannot be diagnosed reliably
  from a final video alone.
- Keep white-box tests close to the invariant they protect; a private symbol
  used by a test does not become public API.
- Give parameterized cases descriptive IDs so a CI failure names the invalid
  condition rather than `config0`, `frame0`, or `observation0`.
- Use explicit helper imports. Avoid autouse fixtures or a test framework whose
  setup is harder to read than the behavior under test.
- Split combined assertions when they protect independent contracts, such as
  reader release and estimator closure.

## Style and scope

- Target the Python versions declared in `pyproject.toml`.
- Keep geometry and filtering functions independently testable.
- Give each mutable state machine one clear owner and one source of truth.
- Prefer explicit failures over silent output degradation.
- Keep backend-specific coordinate conversion inside its adapter.
- Preserve the distinction between nominal file time and monotonic webcam time.
- Keep pull requests focused; separate refactors from behavior changes when
  possible.
- Use American English in new prose to match public identifiers such as
  `center` and `border_color`.

Contributions are accepted under the repository's [MIT License](LICENSE).
