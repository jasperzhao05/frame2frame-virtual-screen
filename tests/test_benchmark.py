import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts._filter_comparison import (
    EMAFilter,
    benchmark_workloads,
    policy_receipt,
    workload_receipt,
)
from scripts.benchmark_smoothing import (
    _screen_age_search_limit,
    _self_check,
    main,
    run_benchmark,
)

_EXPECTED_POLICIES = [
    {"key": "none", "label": "RAW OBSERVATION", "parameters": {}},
    {"key": "ema", "label": "EMA 2.5 HZ", "parameters": {"cutoff_hz": 2.5}},
    {
        "key": "fir",
        "label": "FIR / ANGULAR DEFAULT",
        "parameters": {
            "cutoff_hz": 2.5,
            "transition_hz": 5.0,
            "ripple_db": 60.0,
            "pitch_cutoff_scale": 0.5,
            "roll_cutoff_scale": 0.1,
        },
    },
    {
        "key": "oneeuro",
        "label": "ONE EURO / ANGULAR DEFAULT",
        "parameters": {"min_cutoff": 1.0, "beta": 0.3, "d_cutoff": 1.0},
    },
    {
        "key": "kalman",
        "label": "KALMAN A100",
        "parameters": {"acceleration_std": 100.0, "measurement_std": 1.0},
    },
]


@pytest.fixture(scope="module")
def benchmark_report():
    return run_benchmark(speed_samples=1)


def test_frozen_policy_registry_matches_the_public_comparison():
    assert policy_receipt() == _EXPECTED_POLICIES


def test_benchmark_matrix_is_balanced_deterministic_and_scale_matched():
    first = benchmark_workloads(120, 30.0, 20260730)
    second = benchmark_workloads(120, 30.0, 20260730)

    assert [item.key for item in first] == [item["key"] for item in workload_receipt()]
    assert [(item.motion.key, item.noise.key) for item in first] == [
        (motion, noise)
        for motion in ("smooth", "reversal")
        for noise in ("white", "correlated", "burst", "sample-hold")
    ]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.clean, right.clean)
        np.testing.assert_array_equal(left.noisy, right.noisy)
        np.testing.assert_allclose(
            np.std(left.noisy - left.clean, axis=0),
            (1.25, 1.0, 1.6),
            atol=1e-12,
        )

    # Noise identity is held constant across motions, leaving motion as the
    # only changed factor within each paired noise regime.
    for noise_index in range(4):
        np.testing.assert_allclose(
            first[noise_index].noisy - first[noise_index].clean,
            first[noise_index + 4].noisy - first[noise_index + 4].clean,
            atol=1e-12,
        )

    signals = {item.noise.key: item.noisy - item.clean for item in first[:4]}
    lag_one = {
        key: np.array([np.corrcoef(values[:-1, axis], values[1:, axis])[0, 1] for axis in range(3)])
        for key, values in signals.items()
    }
    assert np.max(np.abs(lag_one["white"])) < 0.15
    assert np.min(lag_one["correlated"]) > 0.75

    burst_magnitude = np.linalg.norm(signals["burst"], axis=1)
    assert np.max(burst_magnitude) / np.percentile(burst_magnitude, 75) > 5.0

    held_steps = np.linalg.norm(np.diff(signals["sample-hold"], axis=0), axis=1)
    assert np.mean(held_steps < 0.5) > 0.6
    assert not np.allclose(first[0].clean, first[4].clean)


def test_benchmark_receipt_is_self_consistent(benchmark_report):
    expected_keys = [policy["key"] for policy in _EXPECTED_POLICIES]

    assert benchmark_report["schema_version"] == 5
    assert benchmark_report["method"]["policies"] == _EXPECTED_POLICIES
    assert [row["filter"] for row in benchmark_report["results"]] == expected_keys
    assert benchmark_report["method"]["screen_carrier"]["distance"] == 4.0
    assert benchmark_report["method"]["workload_count"] == 8
    assert benchmark_report["method"]["frames_per_workload"] == 225
    assert benchmark_report["method"]["evaluation_warmup_frames_per_workload"] == 30
    assert _self_check(benchmark_report) == []

    raw = benchmark_report["results"][0]
    assert len(raw["workloads"]) == 8
    assert len(raw["motions"]) == 2
    assert raw["macro_screen_jitter_reduction_pct"] == pytest.approx(0.0)
    assert raw["macro_screen_jitter_rms_px"] == pytest.approx(
        benchmark_report["method"]["macro_input_screen_jitter_rms_px"]
    )


def test_readme_comparison_matches_repeated_receipt_and_clean_motion(benchmark_report):
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    repeated = json.loads((root / "docs/smoothing-repeated-data.json").read_text(encoding="utf-8"))
    repetitions = {row["filter"]: row for row in repeated["results"]}
    labels = {
        "none": "Raw",
        "ema": "EMA 2.5 Hz",
        "fir": "FIR angular default",
        "oneeuro": "One Euro angular default",
        "kalman": "Kalman A100",
    }
    for row in benchmark_report["results"]:
        repetition = repetitions[row["filter"]]
        assert len(repetition["clean_motions"]) == len(row["motions"])
        continuous_metrics = {
            "motion_rmse_deg_aligned",
            "screen_age_ms",
            "screen_motion_rmse_px_aligned",
        }
        for recorded, current in zip(repetition["clean_motions"], row["motions"]):
            assert recorded.keys() == current.keys()
            for metric, value in recorded.items():
                if metric in continuous_metrics:
                    # Saved macOS results and Linux math libraries can differ
                    # in the last float bits. Keep discrete metadata exact.
                    assert current[metric] == pytest.approx(value, rel=0, abs=1e-12)
                else:
                    assert current[metric] == value
        noise = repetition["rms_reduction_pct"]
        age_min, age_max = row["screen_age_range_frames"]
        if age_min != age_max:
            age = f"{row['macro_screen_age_frames']:.1f} [{age_min}–{age_max}]"
        else:
            age = str(age_min)
        expected = (
            f"| {labels[row['filter']]} | "
            f"{noise['mean']:.2f} ± {noise['sample_std']:.2f}% | {age} | "
            f"{row['macro_screen_motion_rmse_px_aligned']:.3f} px |"
        )
        assert expected in readme


def test_custom_runs_expand_age_search_but_cannot_use_the_default_check(capsys):
    assert _screen_age_search_limit(60.0, {"none": 0, "fir": 22}) == 40

    with pytest.raises(SystemExit):
        main(["--frames", "1920", "--fps", "60", "--check"])
    assert "--check validates only the frozen default protocol" in capsys.readouterr().err


def test_current_frame_error_keeps_fir_lag_visible(benchmark_report):
    rows = {row["filter"]: row for row in benchmark_report["results"]}
    for cell in rows["none"]["workloads"]:
        assert cell["screen_current_rmse_px"] == pytest.approx(cell["input_screen_jitter_rms_px"])
    # Same-time geometry and noise rejection answer different questions. This
    # catches accidentally age-aligning the new current-frame metric.
    assert (
        rows["fir"]["macro_screen_current_rmse_px"] > rows["none"]["macro_screen_current_rmse_px"]
    )
    assert rows["fir"]["macro_screen_jitter_reduction_pct"] > 0


def test_seed_aggregation_checks_protocol_and_retains_each_trial(benchmark_report):
    from scripts.benchmark_repeated_smoothing import _statistics, summarize

    assert _statistics([1.0, 3.0]) == pytest.approx(
        {"mean": 2.0, "sample_std": math.sqrt(2), "min": 1.0, "max": 3.0}
    )
    second = run_benchmark(seed=20260731, speed_samples=1)
    trials = [benchmark_report, second]
    result = summarize(trials)
    assert result[2]["macro_noise_wins_including_ties"] == 2
    worst = min(
        cell["screen_jitter_reduction_pct"]
        for trial in trials
        for cell in trial["results"][4]["workloads"]
    )
    assert result[4]["worst_cell"]["rms_reduction_pct"] == worst
    with pytest.raises(ValueError, match="distinct seeds"):
        summarize([benchmark_report, benchmark_report])
    second["method"]["fps"] = 60
    with pytest.raises(ValueError, match="protocols differ"):
        summarize(trials)


def test_repeated_trial_validation_rejects_bad_aggregates(benchmark_report):
    report = deepcopy(benchmark_report)
    report["results"][2]["macro_screen_current_rmse_px"] = 0.0
    failures = _self_check(report, reference_thresholds=False)
    assert any("current-frame error" in failure for failure in failures)


def test_benchmark_self_check_rejects_real_report_regressions(benchmark_report):
    report = deepcopy(benchmark_report)
    rows = {row["filter"]: row for row in report["results"]}
    rows["fir"]["macro_screen_jitter_ratio"] = 0.99
    rows["ema"]["macro_screen_motion_rmse_px_aligned"] = 20.0
    rows["ema"]["macro_axis_jitter_rms_deg"]["yaw"] = float("nan")
    rows["oneeuro"]["latency_frames_50pct"] = 20
    rows["kalman"]["worst_screen_jitter_reduction_pct"] = -20.0
    rows["fir"]["macro_screen_jitter_p95_reduction_pct"] = 999.0
    report["method"]["macro_input_screen_jitter_rms_px"] = 999.0

    failures = _self_check(report)

    assert any("fir macro screen-jitter ratio" in failure for failure in failures)
    assert any("EMA aligned screen error" in failure for failure in failures)
    assert any("macro_axis_jitter_rms_deg.yaw" in failure for failure in failures)
    assert any("One Euro 50% step latency" in failure for failure in failures)
    assert any("kalman worst-cell reduction" in failure for failure in failures)
    assert any("Kalman worst-cell RMS reduction" in failure for failure in failures)
    assert any("fir macro p95 reduction" in failure for failure in failures)
    assert any("method macro input-screen RMS" in failure for failure in failures)


def test_benchmark_only_ema_uses_the_declared_cutoff_and_timestep():
    filt = EMAFilter(30.0, cutoff_hz=2.5)
    assert filt.update(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

    dt = 0.1
    alpha = dt / (1.0 / (2.0 * math.pi * 2.5) + dt)
    assert filt.update(10.0, -5.0, 2.0, dt=dt) == pytest.approx(
        (10.0 * alpha, -5.0 * alpha, 2.0 * alpha)
    )
