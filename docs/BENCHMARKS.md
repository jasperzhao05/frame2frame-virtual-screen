# Benchmark protocol

`scripts/benchmark_smoothing.py` measures temporal filters without measuring a
pose model at the same time. It is deterministic, does not use a camera, and
does not download anything.

## What it measures

The script generates a 60-second, 30 fps three-axis head-motion trace from
low-frequency sinusoids. With seed `20260730`, it adds independent Gaussian
detector noise and a small high-frequency wobble.

Each filter runs twice:

```text
clean synthetic motion ───────────▶ filter A ──┐
                                               ├─ difference ─▶ residual jitter
same motion + seeded detector noise ▶ filter B ─┘
```

Because A and B have identical configuration and state progression, their
difference isolates propagated observation noise. Intentional smoothing of the
clean head motion is not mislabeled as jitter reduction.

The report contains:

- **residual jitter RMS** — root mean square of `filter(noisy) - filter(clean)`;
- **jitter reduction** — reduction from unfiltered input jitter over the same
  evaluation window;
- **50% latency** — samples from a 20-degree yaw step to the first output at or
  above 10 degrees;
- **aligned motion RMSE** — clean filtered output compared with clean input
  after shifting by measured step latency;
- **throughput** — complete six-channel observations updated per second.

The speed loop includes yaw, pitch, roll, center-x, center-y, and face size. It
does not include decoding, a neural model, rendering, plotting, or encoding.

## Default reference

```bash
python -m scripts.benchmark_smoothing --check
```

Deterministic quality/latency result:

| Filter | Residual jitter RMS | Reduction | 50% latency | Aligned motion RMSE |
|---|---:|---:|---:|---:|
| `none` | 1.329° | 0.0% | 0 frames / 0.0 ms | 0.000° |
| `fir` | 0.383° | 71.2% | 11 frames / 366.7 ms | 0.072° |
| `oneeuro` | 0.867° | 34.7% | 0 frames / 0.0 ms | 0.387° |

The default input jitter is approximately 1.329° RMS. Minor last-decimal
differences can result from numerical-library versions; `--check` encodes the
intended regression bounds.

The self-check uses envelopes rather than exact floating-point equality:

| Contract | Accepted envelope |
|---|---:|
| Pass-through jitter reduction | within 0.01 percentage point of 0% |
| Pass-through latency | exactly 0 frames |
| Pass-through aligned motion RMSE | at most `1e-12` degrees |
| FIR jitter reduction | at least 70% |
| FIR measured latency | within 1 frame of designed group delay |
| FIR aligned motion RMSE | at most 0.15° |
| One Euro jitter reduction | at least 20% |
| One Euro 50% latency | at most 2 frames |
| One Euro aligned motion RMSE | at most 0.75° |

Every reported numeric metric must also be finite. These are regression guards,
not claims that every signal or deployment will meet the same quality values.

This reference is protocol schema version 2. It corrects the One Euro
derivative to use consecutive raw observations, as specified by the original
algorithm, instead of comparing each observation with the smoothed output.

The One Euro row crosses 50% within the first updated sample for this large,
fast step because its cutoff adapts to speed. That does **not** mean it has zero
phase lag for every motion; its aligned motion error captures part of that
trade-off.

The FIR delay is a causal property. Offline frame-delay compensation aligns the
rendered pose with the corresponding source frame, while live preview remains
causal.

## Machine-readable output

```bash
python -m scripts.benchmark_smoothing \
  --format json \
  --output output/smoothing-benchmark.json
```

JSON includes:

- protocol `schema_version`;
- seed, frame count, frame rate, and metric definitions;
- Python, platform, and NumPy versions;
- aggregate and per-axis jitter;
- designed FIR group delay;
- machine-dependent throughput.

For a pull request, attach the JSON as a workflow artifact or paste a concise
before/after table. Do not commit local benchmark output.

## Comparing performance responsibly

Throughput is sensitive to CPU, power state, Python, NumPy/SciPy build, and
background load. Compare two commits on the same machine and environment:

```bash
python -m scripts.benchmark_smoothing --format json --speed-samples 50000
```

Run each commit multiple times for a serious speed claim. Quality and latency
are the primary cross-machine regression metrics.

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

Keep the default seed, trace, definitions, and thresholds stable within a minor
release. If a metric or workload changes:

1. increment `schema_version`;
2. document why it changed;
3. regenerate the reference table;
4. avoid comparing values across schema versions as though they share a
   baseline.
