# Security policy

## Supported versions

Security fixes are applied to the latest released minor version and the default
branch. Older snapshots may receive a fix only when backporting is practical.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use
[GitHub's private security advisory form](https://github.com/jasperzhao05/frame2frame-virtual-screen/security/advisories/new)
and include:

- affected version or commit;
- operating system and Python version;
- minimal reproduction or proof of concept;
- expected impact;
- whether a third-party model, media file, or executable is required.

Avoid attaching private face footage. A synthetic reproducer is strongly
preferred. If real media is essential, describe that fact first and wait for a
private transfer method.

## Security boundaries

- Model assets managed by this project are pinned by SHA-256, downloaded to a
  temporary file, verified, and atomically moved into the cache.
- A caller-supplied local weight or model path is trusted input.
- Video/image parsing is delegated to OpenCV and its media stack. Treat
  untrusted media as potentially hostile and keep those dependencies patched.
- `--preserve-audio` executes the `ffmpeg` binary resolved from `PATH`. Use a
  trusted installation and avoid running with elevated privileges.
- The experimental 6DRepNet package controls its own model acquisition and has
  conflicting OpenCV metadata. It is not an automatic extra; review and isolate
  that upstream package before use in a sensitive environment.
- This is research-oriented rendering software, not a hardened biometric,
  authentication, safety, or surveillance system.

Dependency-only problems should also be reported upstream, but please notify
this project when a version constraint or mitigation is needed here.

## Build and release trust

- GitHub Actions workflows grant the repository token only the permissions each
  workflow needs. External actions are pinned to full commit SHAs and kept
  current through reviewed Dependabot pull requests.
- Dependency review rejects pull requests that introduce runtime dependencies
  with a known high- or critical-severity vulnerability. Repository
  administrators should keep CodeQL default setup, Dependabot alerts and
  security updates, private vulnerability reporting, secret scanning, and push
  protection enabled; these settings are not all represented by files in the
  repository.
- CI builds and tests candidate wheel and source archives, verifies that a
  release tag matches the project version, and records archive checksums. It
  does not publish to PyPI or automatically create a GitHub release.
- A `SHA256SUMS` file detects a changed download when compared with the copy on
  the GitHub release. It is not a maintainer signature or independent build
  provenance; use the tagged source and CI run when stronger audit evidence is
  required.
