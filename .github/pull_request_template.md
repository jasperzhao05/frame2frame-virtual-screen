## What changed

Describe the user-visible behavior, the invariant being protected, and why this
design was chosen. For an internal-only change, explain why behavior is expected
to remain identical.

## Validation

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy`
- [ ] `pytest -q --cov=frame2frame --cov-report=term-missing`

List the focused tests and their results. For filter, timing, rendering, or
pipeline behavior, also report:

- [ ] `python -m scripts.benchmark_smoothing --check`, or **N/A** with reason
- [ ] short synthetic demo, or **N/A** with reason

For packaging, audio, backend, or performance changes, include the additional
commands, seed, Python version, environment, and result required by
`CONTRIBUTING.md`.

## Safety and assets

- [ ] Frame ordering and filter-delay alignment are preserved, or **N/A** with reason.
- [ ] New downloads are size- and checksum-pinned, atomic, and documented, or **N/A**.
- [ ] New third-party assets or dependencies are recorded in `THIRD_PARTY_NOTICES.md`, or **N/A**.
- [ ] No secrets, private footage, model weights, cache files, or generated outputs are committed.

## Documentation

- [ ] User-facing behavior, configuration, limitations, and benchmark changes are documented, or **N/A**.
- [ ] Source-checkout-only commands are labeled and run from the repository root.
