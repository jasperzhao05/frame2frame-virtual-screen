"""Repeat the frozen filter workload across noise seeds without retuning.

Run ``python -m scripts.benchmark_repeated_smoothing --output report.json``.
The default is the 20-seed protocol originally reviewed in the project chat.
Raw per-cell measurements are retained; speed-loop timing is excluded from
the scientific receipt. Standard deviations describe seed variation, not a
population confidence interval or a perceptual preference estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from copy import deepcopy
from importlib import metadata
from pathlib import Path

import numpy as np

from scripts.benchmark_smoothing import _self_check, run_benchmark

DEFAULT_SEED = 20260730
DEFAULT_REPEATS = 20
_ROOT = Path(__file__).resolve().parents[1]
_METRICS = {
    "rms_reduction_pct": "macro_screen_jitter_reduction_pct",
    "p95_reduction_pct": "macro_screen_jitter_p95_reduction_pct",
    "current_rmse_px": "macro_screen_current_rmse_px",
}


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("seed statistics require at least two finite values")
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize(trials: list[dict]) -> list[dict]:
    """Aggregate matched policy/cell trials, preserving negative worst cases."""
    seeds = [trial["method"]["seed"] for trial in trials]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two distinct seeds are required")
    baseline = dict(trials[0]["method"])
    for key in (
        "seed",
        "macro_input_jitter_rms_deg",
        "macro_input_screen_jitter_rms_px",
        "macro_input_screen_jitter_p95_px",
    ):
        baseline.pop(key)
    for trial in trials:
        method = {key: trial["method"][key] for key in baseline}
        if method != baseline:
            raise ValueError("trial protocols differ beyond their noise seeds")
        failures = _self_check(trial, reference_thresholds=False)
        if failures:
            raise ValueError("invalid trial: " + "; ".join(failures))

    results = []
    for index, first in enumerate(trials[0]["results"]):
        rows = [trial["results"][index] for trial in trials]
        if any(row["motions"] != first["motions"] for row in rows):
            raise ValueError("clean-motion metrics changed with the noise seed")
        worst_trial, worst_cell = min(
            ((trial, cell) for trial, row in zip(trials, rows) for cell in row["workloads"]),
            key=lambda pair: pair[1]["screen_jitter_reduction_pct"],
        )
        wins = sum(
            math.isclose(
                row["macro_screen_jitter_reduction_pct"],
                max(item["macro_screen_jitter_reduction_pct"] for item in trial["results"]),
                rel_tol=0,
                abs_tol=1e-10,
            )
            for trial, row in zip(trials, rows)
        )
        results.append(
            {
                "filter": first["filter"],
                **{
                    label: _statistics([row[key] for row in rows])
                    for label, key in _METRICS.items()
                },
                "worst_cell": {
                    "seed": worst_trial["method"]["seed"],
                    "workload": worst_cell["workload"],
                    "rms_reduction_pct": worst_cell["screen_jitter_reduction_pct"],
                },
                "macro_noise_wins_including_ties": wins,
                "clean_motions": deepcopy(first["motions"]),
            }
        )
    return results


def _provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        revision = {
            "head": git("rev-parse", "HEAD"),
            "worktree_dirty": bool(git("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = {"head": None, "worktree_dirty": None}
    paths = sorted((_ROOT / "frame2frame").rglob("*.py")) + [
        _ROOT / "scripts" / name
        for name in (
            "_filter_comparison.py",
            "benchmark_smoothing.py",
            "benchmark_repeated_smoothing.py",
        )
    ]
    return {
        **revision,
        "source_sha256": {
            path.relative_to(_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "versions": {
            name: metadata.version(name) for name in ("numpy", "scipy", "opencv-contrib-python")
        },
    }


def run_repeated(*, seed: int = DEFAULT_SEED, repeats: int = DEFAULT_REPEATS) -> dict:
    if seed < 0 or repeats < 2:
        raise ValueError("seed must be non-negative and repeats must be at least two")
    trials = []
    for current in range(seed, seed + repeats):
        trial = run_benchmark(seed=current, speed_samples=1)
        trials.append(trial)
        print(f"completed seed {current} ({len(trials)}/{repeats})", file=sys.stderr)
    results = summarize(trials)
    # Timing does not belong in deterministic science rows. All measurements
    # used by summarize remain available for independent re-aggregation.
    for trial in trials:
        for row in trial["results"]:
            row.pop("throughput_attitudes_s")
    return {
        "schema_version": 1,
        "protocol": {
            "base_schema_version": trials[0]["schema_version"],
            "seeds": list(range(seed, seed + repeats)),
            "trials": repeats,
            "cells": repeats * trials[0]["method"]["workload_count"],
            "noisy_frames_per_policy": repeats * trials[0]["method"]["total_frames"],
            "std_definition": "sample standard deviation across seeds (ddof=1)",
            "selection": "consecutive fixed seeds; identical policies; no per-cell retuning",
            "scope": (
                "synthetic attitude-only noise sensitivity; no population or preference inference"
            ),
        },
        "provenance": _provenance(),
        "results": results,
        "trials": trials,
    }


def _table(report: dict) -> str:
    lines = ["policy       noise RMS reduction     worst cell    current-frame error"]
    for row in report["results"]:
        noise = row["rms_reduction_pct"]
        current = row["current_rmse_px"]
        lines.append(
            f"{row['filter']:<12} {noise['mean']:6.2f} +/- {noise['sample_std']:4.2f}%"
            f"       {row['worst_cell']['rms_reduction_pct']:6.2f}%"
            f"       {current['mean']:6.2f} +/- {current['sample_std']:.2f} px"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seed < 0 or args.repeats < 2:
        parser.error("--seed must be non-negative; --repeats must be at least two")
    report = run_repeated(seed=args.seed, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(_table(report))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
