# Screen Age protocol

Screen Age measures the effective time shift introduced by a causal attitude
filter in the geometry that reaches the image. It complements conventional
angle-space metrics by asking what moment the rendered screen most closely
represents.

## Definition

At output frame `t`, let `C(t)` be the current screen carrier: its projected
center and apparent scale. For candidate age `k`, construct a counterfactual
screen

```text
Q_k(t) = project(GT attitude at t-k, C(t)).
```

The filter-response screen is

```text
F(t) = project(filter(GT attitude) at t, C(t)).
```

The effective Screen Age is the smallest `k` that minimizes the pooled squared
distance between corresponding corners of `F(t)` and `Q_k(t)`. Holding `C(t)`
fixed prevents face translation and apparent-scale changes from being
misidentified as attitude-filter delay.

Each configuration selects one global age on development subjects. That age is
then frozen before held-out evaluation; a separate best-matching age on the
held-out split is reported only as a calibration check, never as a selection
input.

## Frozen protocol

| Item | Value |
|---|---:|
| Source frame rate | 30 fps |
| Subjects | 20 total; 10 development, 10 held out |
| Eligible continuous segments | 45 total; 25 development, 20 held out |
| Minimum segment length | 75 frames |
| Edge trim | 15 frames |
| Candidate Screen Age | 0–20 frames |
| Retained output frames | 5,203 development; 4,179 held out |
| Bootstrap audit | 2,000 equal-subject resamples |

The carrier uses calibrated BIWI ground-truth position and depth-derived scale.
Only attitude is filtered. Missing-observation recovery is excluded so that the
experiment isolates temporal filtering.

## FIR calibration

The six registered linear-phase FIR designs collapse to three analytic group
delays. Every development estimate and held-out diagnostic recovers that delay;
each bootstrap interval is exactly `[k, k]`.

| FIR design | Analytic delay | Development estimate | Held-out diagnostic |
|---|---:|---:|---:|
| `fir-fc-1-tw-3` | 19 | 19 [19, 19] | 19 [19, 19] |
| `fir-fc-1.5-tw-5` | 11 | 11 [11, 11] | 11 [11, 11] |
| `fir-fc-2.5` | 11 | 11 [11, 11] | 11 [11, 11] |
| `fir-project-default` | 11 | 11 [11, 11] | 11 [11, 11] |
| `fir-fc-4-tw-8` | 7 | 7 [7, 7] | 7 [7, 7] |
| `fir-fc-6-tw-8` | 7 | 7 [7, 7] | 7 [7, 7] |

This calibration establishes that the projected-corner metric recovers a known
causal delay. It does **not** establish that FIR is universally the best
filter, that a filtered screen is perceptually preferred, or that the measured
age is capture-to-display latency.

## Rebuild the figure

The committed receipt contains only aggregate, non-identifying values needed
for the calibration panel. It contains no images, per-frame coordinates,
subject identifiers, or local filesystem paths. Panel A separately embeds one
checksum-pinned held-out BIWI RGB frame and frozen projected-screen geometry
for the mechanically selected event. The current carrier fixes center and
scale; only ground-truth attitude changes across the counterfactual trail. The
ten thin solid rectangles in Panel A are the consecutive age-1 through age-10
counterfactual screens; they are neither interpolated nor decorative geometry.

```bash
python scripts/make_screen_age_figure.py \
  --data docs/screen-age-data.json \
  --scene docs/screen-age-scene.jpg \
  --out docs/screen-age.svg
```

The generator uses only the Python standard library and emits a deterministic,
standalone SVG with no scripts or external resources. It verifies the still by
SHA-256 before embedding it. The README uses a 2× PNG preview for consistent
browser rendering; the preview links to the vector source.

## Evidence boundary and data source

The receipt and mechanism event were produced from the
[BIWI Kinect Head Pose Database](https://vision.ee.ethz.ch/datsets.html), which
ETH Zurich makes available for non-commercial research and education. The
repository includes only the single checksum-pinned RGB frame used by Panel A,
not the database, depth frames, pose files, or subject identifiers. Work using
BIWI should cite:

> G. Fanelli, M. Dantone, J. Gall, A. Fossati, and L. Van Gool, “Random Forests
> for Real Time 3D Face Analysis,” *International Journal of Computer Vision*,
> 101(3), 437–458, 2013.

BIWI provides a geometric reference, not a human perceptual preference label.
Panel A is one held-out mechanism example; Panel B is the aggregate calibration
result. The single event is not a claim of representative pose accuracy or
general perceptual quality.

The result does not evaluate MediaPipe accuracy, eye gaze, headset telemetry,
SLAM, world anchoring, missing-observation recovery, or end-to-end system
latency.
