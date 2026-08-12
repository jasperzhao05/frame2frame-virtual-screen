from copy import deepcopy

from scripts.benchmark_smoothing import _self_check


def _passing_report():
    common = {
        "jitter_rms_deg": 0.5,
        "latency_ms_50pct": 0.0,
        "throughput_observations_s": 1.0,
    }
    return {
        "results": [
            {
                **common,
                "filter": "none",
                "jitter_reduction_pct": 0.0,
                "latency_frames_50pct": 0,
                "designed_group_delay_frames": 0,
                "motion_rmse_deg_aligned": 0.0,
            },
            {
                **common,
                "filter": "fir",
                "jitter_reduction_pct": 75.0,
                "latency_frames_50pct": 11,
                "designed_group_delay_frames": 11,
                "motion_rmse_deg_aligned": 0.1,
            },
            {
                **common,
                "filter": "oneeuro",
                "jitter_reduction_pct": 30.0,
                "latency_frames_50pct": 1,
                "designed_group_delay_frames": 0,
                "motion_rmse_deg_aligned": 0.5,
            },
        ]
    }


def test_benchmark_self_check_accepts_the_quality_envelope():
    assert _self_check(_passing_report()) == []


def test_benchmark_self_check_rejects_motion_or_latency_regressions():
    report = deepcopy(_passing_report())
    rows = {row["filter"]: row for row in report["results"]}
    rows["fir"]["motion_rmse_deg_aligned"] = 1.0
    rows["oneeuro"]["latency_frames_50pct"] = 20
    rows["oneeuro"]["motion_rmse_deg_aligned"] = 2.0

    failures = _self_check(report)

    assert any("FIR aligned motion" in failure for failure in failures)
    assert any("One Euro 50% step latency" in failure for failure in failures)
    assert any("One Euro aligned motion" in failure for failure in failures)
