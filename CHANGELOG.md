# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[semantic versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Corrected the neutral face-forward direction from camera `+Z` to `-Z`, so
  screen yaw and pitch now produce the same near/far perspective as the person.
- Rebuilt the synthetic comparison with unambiguous left/right motion, replaced
  the real-video excerpt with a clear yaw turn, and recalibrated the default
  screen distance to keep normal poses in frame.

## [0.3.0b2] - 2026-08-17

### Added

- A no-download, no-device `frame2frame --doctor` readiness report for the
  supported Python runtime, core binary imports, primary MediaPipe Tasks API,
  plotting dependency, cache, cached model, and optional `ffmpeg`.
- A deterministic whole-file reliability benchmark covering frame
  conservation, repeated decoded-pixel stability, a scripted reset-length
  dropout schedule followed by later observations, and scoped throughput
  reporting.
- Focused projection checks for representative boundary combinations across
  the declared normal operating envelope of yaw ±45° and pitch/roll ±30°.
- An evidence-bounded real-video validation command and manually dispatched
  GitHub workflow that record content identity, declared provenance metadata,
  configuration, revision, decoded media facts, operational checks, and run
  timings.
- A CLI `--focal-length-px` option for known camera intrinsics.

### Changed

- The default MediaPipe dependency is capped below 1.0 after a clean install of
  1.0.1 aborted the primary macOS graph during initialization; the validated
  0.10.35 release remains the maintained path.
- CI now exercises the runtime doctor and short pipeline reliability gate in
  addition to the existing tests, type checks, packaging checks, and synthetic
  smoothing benchmark.
- GitHub Actions are pinned to immutable commit SHAs, checkout credentials are
  not persisted, and Dependabot tracks reviewed action updates.
- Reliability and validation documentation now separates directly measured
  operational evidence from model accuracy, perceptual quality, latency SLO,
  cross-machine performance, and production-readiness claims.

### Security

- Pull requests receive dependency review for newly introduced high- or
  critical-severity runtime vulnerabilities.
- Repository operations now use the dependency graph, Dependabot alerts and
  security updates, private vulnerability reporting, secret scanning with push
  protection, and CodeQL default analysis for Python and GitHub Actions.

## [0.3.0b1] - 2026-08-13

### Added

- Installable `frame2frame` package with a CLI, a focused Python configuration
  entry point, and MediaPipe, Hopenet, and 6DRepNet pose adapters.
- Deterministic, model-free temporal-filter regression benchmark covering
  synthetic jitter propagation, step response, aligned motion error, and
  filter-only throughput.
- Seeded synthetic pipeline demo using a scripted pose source; it does not
  exercise a pose backend.
- Architecture, benchmark, privacy, security, citation, and third-party notices.
- A usage guide covering supported configuration, estimator ownership,
  timestamps, display, audio, and output publication.
- Optional source-audio preservation through `ffmpeg`.

### Changed

- Project identity and repository links now point to `jasperzhao05`.
- MediaPipe documentation now describes the Face Landmarker transformation
  matrix path rather than the retired solvePnP description.
- The default-install dependency graph and Hopenet optional extra are declared
  and packaging-smoke-tested; 6DRepNet remains an isolated experimental
  adapter because its published OpenCV dependency conflicts with MediaPipe's.
- Project-managed model downloads are checksum-verified and atomic.
- Package metadata and installed artifacts now cover Python 3.9–3.13 and include
  the default MediaPipe backend.
- README commands now identify when a source checkout is required, and the
  contributor guide maps each change area to focused and full validation.
- README demo assets use repository-relative paths and documentation links use
  canonical repository URLs for the GitHub-first release surface.
- Pipeline orchestration, temporal buffering, video I/O, validation, rendering,
  and download flows now have explicit single-purpose owners with less
  duplicated state and branching.
- Contributor checks enforce a McCabe complexity ceiling of 10 to keep future
  control flow readable.
- FIR state is primed at the first sample; face centre and size advance through
  the same temporal path as rotation; offline rendering compensates known FIR
  group delay; projected corners retain subpixel precision through the warp.

### Fixed

- Pose dropouts no longer misalign the FIR observation queue and delayed frames.
- Readers, internally created estimators, writers, and temporary audio files
  are released on initialisation and processing failures.
- Audio preservation fails explicitly when `ffmpeg` cannot remux the source.
- The built-in Hopenet definition includes the checkpoint-required vestigial
  fine-tuning head. This is a checkpoint-compatibility change, not a
  reproduction of the paper's reported metrics.
- One Euro derivative estimation now uses consecutive raw observations; the
  deterministic benchmark protocol is versioned as schema 2 with refreshed
  reference values.

### Removed

- Committed third-party processed footage and its derived angle plot. The
  repository now uses an owned synthetic visual and reproducible scripts.

[Unreleased]: https://github.com/jasperzhao05/frame2frame-virtual-screen/compare/v0.3.0b2...HEAD
[0.3.0b2]: https://github.com/jasperzhao05/frame2frame-virtual-screen/releases/tag/v0.3.0b2
[0.3.0b1]: https://github.com/jasperzhao05/frame2frame-virtual-screen/releases/tag/v0.3.0b1
