# Benchmark protocol

`scripts/benchmark_smoothing.py` compares five frozen temporal policies without
measuring a pose model at the same time. It is deterministic, uses no camera,
downloads nothing, and resets every filter at the start of each workload.

## Experimental design

The default protocol is a balanced `2 × 4` stress matrix:

- two clean motions: a smooth multi-axis sweep and a stop-and-reverse trace;
- four seeded, non-periodic observation-error regimes: independent noise,
  temporally correlated noise, bounded decaying bursts, and sample-and-hold
  error.

Each of the eight cells contains 225 frames at 30 fps. The four noise traces
have the same per-axis standard deviations—1.25° yaw, 1.0° pitch, and 1.6°
roll—and each noise realization is reused across both motions. This balances
the factors; it does not claim that the four errors occur equally often in real
tracking systems.

Each policy runs on paired inputs:

```text
clean motion ───────────────▶ policy A ──▶ distance-4 screen ──┐
                                                               ├─ residual
same motion + seeded error ─▶ policy B ──▶ distance-4 screen ──┘
```

Both paths use fresh policy instances, identical parameters, and synchronized
frame indices. Their paired difference measures the causal effect of injected
observation error, including any noise-induced divergence in adaptive state.
Intentional smoothing of the clean motion is measured separately rather than
mislabeled as noise reduction.

Projection uses a fixed 480×320 carrier centered at `(240, 160)`, face size 40
px, and screen distance 4; only attitude changes. The default evaluation drops
the first 30 frames of each cell as shared warm-up.

## Frozen policies

- raw pass-through;
- benchmark-only EMA with a 2.5 Hz cutoff;
- the runtime FIR angular defaults: 2.5 Hz yaw, 1.25 Hz pitch, 0.25 Hz roll,
  5 Hz transition, and 60 dB Kaiser target;
- the runtime One Euro angular defaults: minimum cutoff 1.0, beta 0.3, and
  derivative cutoff 1.0;
- Kalman A100: acceleration standard deviation 100.0 and measurement standard
  deviation 1.0.

Translation smoothing is disabled for every policy. Parameters are frozen
across cells; the experiment does not choose a separate best setting for each
noise or motion.

## Metrics

For each motion-by-noise cell, the report computes:

- **screen jitter RMS** — RMS corresponding-corner distance between paired
  noisy- and clean-input outputs after projection;
- **RMS ratio** — filtered screen jitter divided by raw screen jitter in that
  same cell;
- **p95 ratio** — the same ratio using the 95th percentile of per-frame corner
  error;
- **reduction** — `100 × (1 − ratio)`; negative means the policy amplified the
  error.

The report also compares each noisy filtered screen with the **unfiltered
clean screen at the same frame**. `screen_current_rmse_px` and
`screen_current_p95_px` include residual noise, intentional-motion distortion,
and lag together, without Screen Age alignment. The macro is the equal-cell
mean RMS; the tail reports the worst cell's p95. It is a geometric error for
causal output, not a perceptual rating or an offline compensated-render score.

The headline noise score is the unweighted mean of the eight cell ratios. The
reported worst cell is the minimum reduction over those cells. Ratios are
averaged instead of pooling frames, so each stress condition has equal weight
even when projected raw amplitudes differ slightly.

Clean-motion fidelity is evaluated once for each of the two unique motions:

- **Screen Age** — one global integer offset per clean motion, chosen by
  minimizing pooled corner error over all retained frames;
- **age-aligned motion error** — corresponding-corner RMS after that alignment.

Their macro values average two motions, not eight repeated noise cells. The
macro age can therefore be fractional even though each motion-level age is an
integer; it is not a sub-frame estimate. The JSON receipt also includes
angle-space residuals, a 50% yaw-step response, and three-axis attitude-update
throughput. Throughput excludes translation, decoding, a neural model,
rendering, plotting, and encoding.

## Default reference

```bash
python -m scripts.benchmark_smoothing --check
```

Default deterministic result at 30 fps with seed `20260730`:

| Policy | Macro RMS reduction | Worst-cell reduction | Macro p95 reduction | Screen Age macro [range] | Age-aligned error |
|---|---:|---:|---:|---:|---:|
| Raw (`none`) | 0.0% | 0.0% | 0.0% | 0.0 frames | 0.000 px |
| EMA 2.5 Hz | 31.3% | 9.9% | 33.0% | 2.0 frames | 1.704 px |
| FIR angular default | 42.5% | 14.5% | 46.9% | 11.0 frames | 1.854 px |
| One Euro angular default | 14.7% | 3.7% | 15.4% | 0.5 frames [0–1] | 2.024 px |
| Kalman A100 | 17.2% | −1.1% | 19.3% | 0.5 frames [0–1] | 3.646 px |

The raw input averages 6.662 px screen-jitter RMS and 12.719 px p95 across the
eight cells. FIR rejects the most error on average and in the weakest cell, but
EMA has lower age-aligned motion error and substantially lower Screen Age.
Kalman slightly amplifies the temporally correlated-noise regime.

The noise marginal below averages the two motions within each error regime:

| Policy | Independent | Correlated | Burst | Sample-and-hold |
|---|---:|---:|---:|---:|
| Raw (`none`) | 0.0% | 0.0% | 0.0% | 0.0% |
| EMA 2.5 Hz | 53.1% | 10.0% | 34.9% | 27.2% |
| FIR angular default | 65.7% | 14.8% | 51.1% | 38.2% |
| One Euro angular default | 32.8% | 4.0% | 10.8% | 11.4% |
| Kalman A100 | 43.7% | −0.8% | 19.3% | 6.8% |

The clean-motion receipt shows why noise rejection alone is insufficient:

| Policy | Smooth: age / error | Reversal: age / error |
|---|---:|---:|
| Raw (`none`) | 0 / 0.000 px | 0 / 0.000 px |
| EMA 2.5 Hz | 2 / 0.451 px | 2 / 2.957 px |
| FIR angular default | 11 / 0.368 px | 11 / 3.341 px |
| One Euro angular default | 1 / 0.946 px | 0 / 3.102 px |
| Kalman A100 | 0 / 0.937 px | 1 / 6.355 px |

These fixed configurations are reference points, not tuned winners. The
reference above is synthetic, attitude-only, and single-seed. It does not measure pose
accuracy, translation, dropout recovery, camera-to-display latency, subjective
quality, or the distribution of errors in a deployment.

## Repeated noise experiment

```bash
python -m scripts.benchmark_repeated_smoothing --output output/smoothing-repeated.json
```

The frozen repeated run uses 20 consecutive seeds, `20260730–20260749`, with
the same two motions, four noise regimes, 225 frames per cell, and five policy
configurations. It runs 160 cells and 36,000 noisy frames per policy. Noise is
paired across motions within a seed, so these are 20 seed-level repetitions,
not 160 independent population samples. Clean-motion ages and errors do not
change with the noise seed and are not treated as repeated evidence.

Results below are mean ± **sample standard deviation** (`ddof=1`) across seeds,
not confidence intervals. Worst cells range over every seed and workload.

| Policy | RMS reduction % | Seed range % | Worst cell % | Same-frame RMS px |
|---|---:|---:|---:|---:|
| Raw | 0.00 ± 0.00 | 0.00–0.00 | 0.00 | 6.52 ± 0.10 |
| EMA | 31.64 ± 1.15 | 29.39–34.31 | 6.53 | 10.29 ± 0.33 |
| FIR | 41.94 ± 1.84 | 38.38–46.30 | 7.06 | 48.25 ± 0.37 |
| One Euro | 14.49 ± 0.75 | 12.67–16.03 | 0.96 | 6.01 ± 0.20 |
| Kalman | 18.07 ± 1.45 | 15.72–21.32 | −5.07 | 7.13 ± 0.20 |

FIR wins macro noise reduction in 20/20 seeds. One Euro has the smallest
same-frame RMS for these configurations. Neither finding is a universal family
ranking: parameters are frozen reference settings, not operating points matched
for latency or stabilization strength. Equal tuning budgets and a development /
held-out comparison of the resulting trade-off curves are a separate study.

The [compact receipt](smoothing-repeated-data.json) retains protocol, source-file
hashes, environment, and aggregate results. The command writes those fields plus
every seed's cell and motion rows, allowing independent re-aggregation. Speed
timing is omitted from this scientific receipt. Arithmetic checks apply to every
trial; seed-specific quality thresholds from the single-seed regression do not
reject or censor adverse trials.

## Regression contract

`--check` verifies the matrix registry, frame accounting, macro arithmetic, raw
identity, finite outputs, and broad regression envelopes:

| Contract | Accepted envelope |
|---|---:|
| Raw | identity in every cell; zero Screen Age |
| EMA | macro RMS reduction ≥ 20%; no worsened cell; aligned error ≤ 5.5 px |
| FIR | macro RMS reduction ≥ 30%; worst cell ≥ 5%; Screen Age within one frame of designed delay; aligned error ≤ 6.5 px |
| One Euro | macro RMS reduction ≥ 5%; aligned error ≤ 4.0 px |
| Kalman | macro RMS reduction ≥ 5%; worst cell ≥ −5%; aligned error ≤ 7.0 px |
| EMA / One Euro / Kalman step response | 50% crossing within two frames |

These are deliberately broad software-regression guards, not promised quality
for arbitrary signals. `--check` is restricted to the frozen 1800-frame,
30 fps, seed-`20260730` protocol; custom settings remain available as
exploratory reports without `--check`. Schema version 5 adds same-frame geometric
error while preserving version 4's inputs and existing metrics. Version 4
replaced version 3's single trace and periodic wobble with the balanced
non-periodic matrix; version 3 quality numbers are not directly comparable.

Screen Age here is a synthetic clean-motion alignment. The separate BIWI
protocol selects one age on development subjects and freezes it before held-out
evaluation. Neither quantity is capture-to-display latency.

The FIR delay is a causal property. Offline frame-delay compensation aligns the
rendered pose with its source frame; live preview remains causal.

## Machine-readable output

```bash
python -m scripts.benchmark_smoothing \
  --format json \
  --output output/smoothing-benchmark.json
```

The JSON receipt includes:

- protocol `schema_version`, seed, total frames, frames per cell, frame rate,
  warm-up, and screen carrier;
- ordered motion-by-noise registry and metric definitions;
- exact policy parameters;
- per-cell raw and residual RMS/p95 values and ratios;
- per-motion Screen Age and age-aligned error;
- an explicit flag when a custom run's Screen Age reaches its search boundary;
- macro, worst-cell, angle-space, step-response, and machine-dependent
  throughput fields;
- Python, platform, and NumPy versions.

For a pull request, attach the JSON as a workflow artifact or paste a concise
before/after table. Do not commit local benchmark output.

## Comparing performance responsibly

Throughput is sensitive to CPU, power state, Python, NumPy/SciPy build, and
background load. Compare two commits on the same machine and environment:

```bash
python -m scripts.benchmark_smoothing --format json --speed-samples 50000
```

Run each commit multiple times for a serious speed claim. Deterministic quality
and temporal metrics are the primary cross-machine regression signals.

End-to-end FPS is a separate measurement because pose backend, resolution,
codec, preview, and storage dominate it. When reporting an end-to-end run,
include at least:

- exact commit and command;
- input resolution, frame count, and source fps;
- backend, device, and model asset;
- filter and delay-compensation setting;
- whether output, plotting, display, and audio preservation were enabled;
- mean plus a tail statistic such as p95, not mean alone.

## Changing the protocol

Keep the default seed, workloads, definitions, and thresholds stable within a
minor release. If a metric or workload changes:

1. increment `schema_version`;
2. document why it changed;
3. regenerate the reference tables;
4. avoid comparing values across schema versions as though they share a
   baseline.
