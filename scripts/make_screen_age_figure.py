"""Build the static Screen Age research figure.

The calibration panel reads only the compact aggregate receipt committed under
``docs/``.  The mechanism panel embeds one checksum-pinned BIWI frame and the
fixed projected-screen geometry for a held-out event.  Maintainers can
explicitly refresh the aggregate receipt from the canonical local Screen Age
result with ``--from-canonical``; the public receipt remains aggregate-only.

    python -m scripts.make_screen_age_figure
"""

# SVG markup stays literal so the emitted artifact is easy to audit.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = _ROOT / "docs/screen-age-data.json"
_DEFAULT_OUTPUT = _ROOT / "docs/screen-age.svg"
_DEFAULT_SCENE = _ROOT / "docs/screen-age-scene.jpg"
_SCENE_SHA256 = "35fe82ba46e1dd926e36c1301711101bb91842e78af49de4561b9d1cd3ffc461"
_FPS = 30.0
_SCREEN_DISTANCE = 4.0
_BOOTSTRAP_RESAMPLES = 2_000
_PLOT_LEFT = 1200.0
_PLOT_TOP = 270.0
_PLOT_SIZE = 500.0
_EXPECTED_DELAYS = {
    "fir-fc-1-tw-3": 19,
    "fir-fc-1.5-tw-5": 11,
    "fir-fc-2.5": 11,
    "fir-project-default": 11,
    "fir-fc-4-tw-8": 7,
    "fir-fc-6-tw-8": 7,
}
_FORBIDDEN_RECEIPT_KEYS = {
    "image_data",
    "image_path",
    "local_path",
    "per_subject",
    "record_id",
    "segment_id",
    "sequence_id",
    "source_path",
    "subject_id",
    "frame_id",
    "frame_index",
    "frame_path",
    "corner_coordinates",
    "landmarks",
}
_SUBJECT_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9])[MF]\d{2}(?![A-Za-z0-9])")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the publish-safe static Screen Age SVG.")
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--scene",
        type=Path,
        default=_DEFAULT_SCENE,
        help="Checksum-pinned BIWI frame used by the mechanism panel.",
    )
    parser.add_argument(
        "--from-canonical",
        type=Path,
        help="Explicitly rebuild the compact receipt from canonical effective-age.json.",
    )
    return parser.parse_args(argv)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _scene_data_uri(path: Path) -> str:
    """Return the one verified, attributed scene as an embedded JPEG."""

    payload = path.read_bytes()
    digest = _sha256(payload)
    if digest != _SCENE_SHA256:
        raise ValueError(f"Screen Age scene changed: expected {_SCENE_SHA256}, received {digest}")
    if not payload.startswith(b"\xff\xd8\xff"):
        raise ValueError("Screen Age scene must remain a JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def _exact_ci(value: int, payload: Any, *, label: str) -> list[int]:
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError(f"{label} must be a two-value interval")
    interval = [int(item) for item in payload]
    if any(float(item) != float(value) for item in payload) or interval != [value, value]:
        raise ValueError(f"{label} must recover the analytic delay exactly")
    return interval


def _strict_int(payload: Any, *, label: str) -> int:
    if type(payload) is not int:
        raise ValueError(f"{label} must be an integer")
    return payload


def _canonical_fir_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in source["results"] if row.get("filter_family") == "fir"]
    if len(rows) != len(_EXPECTED_DELAYS):
        raise ValueError("canonical FIR registry must contain exactly six designs")
    by_name = {row["filter_name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("canonical FIR registry contains duplicate designs")
    if set(by_name) != set(_EXPECTED_DELAYS):
        raise ValueError("canonical FIR registry does not match the frozen six designs")
    return [by_name[name] for name in _EXPECTED_DELAYS]


def _validate_canonical_protocol(source: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "kind": "filter-response-effective-age",
        "role": "primary corrected response-age analysis",
        "eligible_segments": 45,
        "minimum_segment_frames": 75,
        "trim_frames": 15,
        "selection_split": "development",
        "evaluation_split": "test",
    }
    mismatches = {
        key: source.get(key)
        for key, expected_value in expected.items()
        if source.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"canonical Screen Age protocol changed: {mismatches}")
    if source.get("candidate_age_frames") != list(range(21)):
        raise ValueError("canonical candidate-age grid must remain 0 through 20")


def _compact_design(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row["filter_name"])
    delay = _EXPECTED_DELAYS[name]
    development = row["splits"]["development"]
    test = row["splits"]["test"]
    selected = int(row["development_selected_effective_age_frames"])
    development_age = int(development["oracle_age_frames_diagnostic"])
    held_out_age = int(test["oracle_age_frames_diagnostic"])
    if (selected, development_age, held_out_age) != (delay, delay, delay):
        raise ValueError(f"{name} no longer recovers analytic delay {delay}")
    if row.get("development_selected_at_lookback_boundary") is not False:
        raise ValueError(f"{name} selected at the candidate-age boundary")
    bootstrap_counts = {
        int(row["development_selection_bootstrap"]["resamples"]),
        int(development["oracle_age_bootstrap_diagnostic"]["resamples"]),
        int(test["oracle_age_bootstrap_diagnostic"]["resamples"]),
    }
    if bootstrap_counts != {_BOOTSTRAP_RESAMPLES}:
        raise ValueError(f"{name} must use {_BOOTSTRAP_RESAMPLES} bootstrap resamples")

    return {
        "analytic_delay_frames": delay,
        "development_ci95_frames": _exact_ci(
            delay,
            development["oracle_age_bootstrap_diagnostic"]["age_ci95_frames"],
            label=f"{name} development CI",
        ),
        "development_estimate_frames": development_age,
        "filter_name": name,
        "held_out_ci95_frames": _exact_ci(
            delay,
            test["oracle_age_bootstrap_diagnostic"]["age_ci95_frames"],
            label=f"{name} held-out CI",
        ),
        "held_out_diagnostic_frames": held_out_age,
        "project_default": bool(row["project_default"]),
    }


def _consistent_split_count(rows: list[dict[str, Any]], split: str) -> int:
    counts = {int(row["splits"][split]["subjects"]) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"inconsistent aggregate count for {split}")
    return counts.pop()


def _public_receipt(canonical_path: Path) -> dict[str, Any]:
    source_bytes = canonical_path.read_bytes()
    source = json.loads(source_bytes)
    _validate_canonical_protocol(source)
    rows = _canonical_fir_rows(source)
    designs = [_compact_design(row) for row in rows]
    development_subjects = _consistent_split_count(rows, "development")
    test_subjects = _consistent_split_count(rows, "test")
    support = source["fixed_support_receipt"]

    return {
        "benchmark": {
            "name": "BIWI Kinect Head Pose Database",
            "publication_scope": "aggregate measurements only; no source media or record-level rows",
        },
        "figure_claim": (
            "Across six registered causal FIR designs, recovered Screen Age equals "
            "analytic group delay in development and held-out diagnostics."
        ),
        "fir_designs": designs,
        "hero": {
            "filter_name": "fir-project-default",
            "screen_age_frames": 11,
            "screen_age_ms": round(11 / _FPS * 1000.0, 1),
        },
        "kind": "screen-age-public-aggregate-receipt",
        "protocol": {
            "attitude_only": True,
            "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
            "candidate_age_range_frames": [0, 20],
            "evaluation_split": "test",
            "fps": _FPS,
            "minimum_segment_frames": int(source["minimum_segment_frames"]),
            "reference": "past raw ground-truth attitude projected with the shared current carrier",
            "screen_distance": _SCREEN_DISTANCE,
            "selection": (
                "smallest exact argmin on the development equal-subject MSE curve; "
                "no cross-filter selection"
            ),
            "selection_split": "development",
            "test_role": "evaluation-only oracle diagnostic; no test selection",
            "trim_frames": int(source["trim_frames"]),
        },
        "schema_version": 1,
        "source": {
            "artifact_kind": str(source["kind"]),
            "sha256": _sha256(source_bytes),
        },
        "support": {
            "eligible_segments": {
                "development": int(source["eligible_segments_by_split"]["development"]),
                "test": int(source["eligible_segments_by_split"]["test"]),
                "total": int(source["eligible_segments"]),
            },
            "retained_output_frames": {
                "development": int(support["development"]["retained_output_frames"]),
                "test": int(support["test"]["retained_output_frames"]),
                "total": int(
                    support["development"]["retained_output_frames"]
                    + support["test"]["retained_output_frames"]
                ),
            },
            "subjects": {
                "development": development_subjects,
                "test": test_subjects,
                "total": development_subjects + test_subjects,
            },
        },
    }


def _walk(value: Any) -> list[tuple[str | None, Any]]:
    rows: list[tuple[str | None, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.append((str(key), item))
            rows.extend(_walk(item))
    elif isinstance(value, list):
        for item in value:
            rows.append((None, item))
            rows.extend(_walk(item))
    return rows


def _validate_public_design(row: dict[str, Any]) -> None:
    name = str(row["filter_name"])
    delay = _EXPECTED_DELAYS[name]
    if _strict_int(row.get("analytic_delay_frames"), label=f"{name} analytic delay") != delay:
        raise ValueError(f"{name} analytic delay must remain {delay}")
    if (
        _strict_int(row.get("development_estimate_frames"), label=f"{name} development estimate")
        != delay
    ):
        raise ValueError(f"{name} development estimate must recover {delay}")
    if (
        _strict_int(
            row.get("held_out_diagnostic_frames"),
            label=f"{name} held-out diagnostic",
        )
        != delay
    ):
        raise ValueError(f"{name} held-out diagnostic must recover {delay}")
    _exact_ci(
        delay,
        row.get("development_ci95_frames"),
        label=f"{name} development CI",
    )
    _exact_ci(
        delay,
        row.get("held_out_ci95_frames"),
        label=f"{name} held-out CI",
    )
    if row.get("project_default") != (name == "fir-project-default"):
        raise ValueError(f"{name} has an invalid project-default designation")


def _validate_public_designs(receipt: dict[str, Any]) -> None:
    designs = receipt.get("fir_designs")
    if not isinstance(designs, list) or len(designs) != 6:
        raise ValueError("public receipt must contain exactly six FIR designs")
    if not all(isinstance(row, dict) for row in designs):
        raise ValueError("every public FIR design must be an object")
    names = {row.get("filter_name") for row in designs}
    if names != set(_EXPECTED_DELAYS):
        raise ValueError("public receipt FIR registry changed")
    for row in designs:
        _validate_public_design(row)


def _validate_public_protocol(receipt: dict[str, Any]) -> None:
    protocol = receipt.get("protocol")
    expected_protocol = {
        "attitude_only": True,
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        "candidate_age_range_frames": [0, 20],
        "evaluation_split": "test",
        "fps": _FPS,
        "minimum_segment_frames": 75,
        "screen_distance": _SCREEN_DISTANCE,
        "selection_split": "development",
        "trim_frames": 15,
    }
    if not isinstance(protocol, dict):
        raise ValueError("public receipt requires a protocol object")
    protocol_mismatches = {
        key: protocol.get(key)
        for key, expected in expected_protocol.items()
        if protocol.get(key) != expected
    }
    if protocol_mismatches:
        raise ValueError(f"public Screen Age protocol changed: {protocol_mismatches}")


def _validate_public_support(receipt: dict[str, Any]) -> None:
    expected_support = {
        "eligible_segments": {"development": 25, "test": 20, "total": 45},
        "retained_output_frames": {"development": 5203, "test": 4179, "total": 9382},
        "subjects": {"development": 10, "test": 10, "total": 20},
    }
    if receipt.get("support") != expected_support:
        raise ValueError("public Screen Age support counts changed")


def _validate_public_hero(receipt: dict[str, Any]) -> None:
    expected_hero = {
        "filter_name": "fir-project-default",
        "screen_age_frames": 11,
        "screen_age_ms": 366.7,
    }
    if receipt.get("hero") != expected_hero:
        raise ValueError("public Screen Age hero changed")


def _validate_public_privacy(receipt: dict[str, Any]) -> None:
    for key, value in _walk(receipt):
        if key is not None and key.lower() in _FORBIDDEN_RECEIPT_KEYS:
            raise ValueError(f"record-level field is not publish-safe: {key}")
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        if "/users/" in lowered or "data:image" in lowered or "frame_" in lowered:
            raise ValueError("public receipt contains a local or media reference")
        if _SUBJECT_IDENTIFIER.search(value):
            raise ValueError("public receipt contains a record-level identifier")


def _validate_public_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema_version") != 1:
        raise ValueError("public receipt must use schema_version 1")
    if receipt.get("kind") != "screen-age-public-aggregate-receipt":
        raise ValueError("unexpected public receipt kind")
    _validate_public_designs(receipt)
    _validate_public_protocol(receipt)
    _validate_public_support(receipt)
    _validate_public_hero(receipt)
    _validate_public_privacy(receipt)


def _path(points: list[tuple[float, float]]) -> str:
    return (
        " ".join(
            f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}" for index, (x, y) in enumerate(points)
        )
        + " Z"
    )


def _x_scale(value: float) -> float:
    return _PLOT_LEFT + (value - 5.0) / 16.0 * _PLOT_SIZE


def _y_scale(value: float) -> float:
    return _PLOT_TOP + _PLOT_SIZE - (value - 5.0) / 16.0 * _PLOT_SIZE


def _parity_grid() -> str:
    plot_right = _PLOT_LEFT + _PLOT_SIZE
    plot_bottom = _PLOT_TOP + _PLOT_SIZE
    markup: list[str] = []
    for value in (7, 11, 15, 19):
        x = _x_scale(float(value))
        y = _y_scale(float(value))
        markup.extend(
            [
                f'      <line x1="{_PLOT_LEFT:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" stroke="#dedfdd" stroke-width="1.5"/>',
                f'      <line x1="{x:.1f}" y1="{_PLOT_TOP:.1f}" x2="{x:.1f}" y2="{plot_bottom:.1f}" stroke="#dedfdd" stroke-width="1.5"/>',
                f'      <line x1="{x:.1f}" y1="{plot_bottom:.1f}" x2="{x:.1f}" y2="{plot_bottom + 9:.1f}" stroke="#8d8f90" stroke-width="2"/>',
                f'      <text x="{x:.1f}" y="{plot_bottom + 40:.1f}" text-anchor="middle" fill="#1d1f21" font-size="22">{value}</text>',
                f'      <line x1="{_PLOT_LEFT - 9:.1f}" y1="{y:.1f}" x2="{_PLOT_LEFT:.1f}" y2="{y:.1f}" stroke="#8d8f90" stroke-width="2"/>',
                f'      <text x="{_PLOT_LEFT - 22:.1f}" y="{y + 8:.1f}" text-anchor="end" fill="#1d1f21" font-size="22">{value}</text>',
            ]
        )
    return "\n".join(markup)


def _calibration_marks(designs: list[dict[str, Any]]) -> str:
    counts = Counter(int(row["analytic_delay_frames"]) for row in designs)
    if counts != Counter({7: 2, 11: 3, 19: 1}):
        raise ValueError("expected three FIR calibration loci with multiplicities 2, 3, and 1")
    labels = {
        7: (_x_scale(7.0) + 16.0, _y_scale(7.0) - 15.0, "start"),
        11: (_x_scale(11.0) + 16.0, _y_scale(11.0) - 15.0, "start"),
        19: (_x_scale(19.0) - 12.0, _y_scale(19.0) + 40.0, "end"),
    }
    output: list[str] = []
    for row in designs:
        analytic = float(row["analytic_delay_frames"])
        development = float(row["development_estimate_frames"])
        held_out = float(row["held_out_diagnostic_frames"])
        x = _x_scale(analytic)
        development_y = _y_scale(development)
        held_out_y = _y_scale(held_out)
        output.extend(
            [
                (
                    f'      <circle cx="{x:.1f}" cy="{development_y:.1f}" r="13" '
                    'fill="#fbfaf7" stroke="#27ef75" stroke-width="3.5"/>'
                ),
                (
                    f'      <path d="M{x:.1f},{held_out_y - 8:.1f} '
                    f"L{x + 8:.1f},{held_out_y:.1f} L{x:.1f},{held_out_y + 8:.1f} "
                    f'L{x - 8:.1f},{held_out_y:.1f} Z" fill="#17212b" '
                    'stroke="#fbfaf7" stroke-width="1.5"/>'
                ),
            ]
        )

    for value in (7, 11, 19):
        label_x, label_y, anchor = labels[value]
        label = {
            7: "2 designs",
            11: "3 designs · default",
            19: "1 design",
        }[value]
        output.append(
            f'      <text x="{label_x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="{anchor}" fill="#1d1f21" font-size="22" font-weight="500">'
            f"{label}</text>"
        )
    return "\n".join(output)


def _render_svg(
    receipt: dict[str, Any],
    scene_path: Path = _DEFAULT_SCENE,
) -> str:
    _validate_public_receipt(receipt)
    designs = receipt["fir_designs"]
    hero = receipt["hero"]
    screen_age = int(hero["screen_age_frames"])
    screen_age_ms = float(hero["screen_age_ms"])
    if hero.get("filter_name") != "fir-project-default" or screen_age != 11:
        raise ValueError("hero must remain the project-default 11-frame FIR")

    scene_data_uri = _scene_data_uri(scene_path)
    photo_x = 64.0
    photo_y = 225.0
    photo_scale = 2.5
    crop_x = 128.0
    crop_y = 112.0

    def project_path(points: list[tuple[float, float]]) -> str:
        return _path(
            [
                (
                    photo_x + (x - crop_x) * photo_scale,
                    photo_y + (y - crop_y) * photo_scale,
                )
                for x, y in points
            ]
        )

    # One held-out BIWI event, mechanically selected by the registered Screen
    # Age procedure. Center and scale remain fixed to the current carrier; only
    # attitude changes across the counterfactual ground-truth screens.
    gt_now = [(289.94, 173.36), (462.71, 149.24), (481.93, 232.30), (301.22, 257.91)]
    gt_age_11 = [(206.23, 230.78), (364.12, 228.46), (364.98, 324.74), (207.26, 313.99)]
    fir_now = [(208.54, 226.52), (367.59, 221.31), (368.87, 317.52), (209.30, 310.01)]
    # Ten consecutive intermediate ages make the screen worldline visible.
    # Every outline is a mechanically derived counterfactual with the current
    # center and scale, not interpolated or decorative geometry.
    history_quads = [
        (10, [(192.86, 233.55), (339.67, 223.18), (344.34, 320.85), (196.52, 315.82)]),
        (9, [(184.78, 234.47), (322.30, 218.55), (329.04, 317.12), (189.62, 316.06)]),
        (8, [(180.05, 232.86), (310.43, 210.79), (318.38, 309.80), (185.16, 313.99)]),
        (7, [(177.14, 229.66), (301.97, 203.37), (309.74, 302.59), (181.52, 310.40)]),
        (6, [(178.27, 230.06), (305.08, 197.71), (314.99, 296.02), (183.85, 310.77)]),
        (5, [(183.12, 224.25), (316.75, 191.39), (325.87, 288.71), (187.65, 305.30)]),
        (4, [(190.38, 220.77), (332.53, 184.07), (343.58, 279.52), (195.89, 302.18)]),
        (3, [(205.30, 212.80), (359.81, 174.98), (372.57, 267.80), (211.60, 294.91)]),
        (2, [(230.84, 197.86), (397.72, 163.84), (411.52, 253.28), (237.29, 280.88)]),
        (1, [(258.24, 186.25), (430.65, 154.86), (447.78, 240.86), (267.37, 269.96)]),
    ]
    ghost_paths = "\n".join(
        f'      <path class="screen-worldline-history" data-age-frames="{age}" '
        f'd="{project_path(points)}" fill="none" stroke="#1d1f21" '
        f'stroke-width="2.2" opacity="{0.12 + (10 - age) * (0.22 / 9):.2f}"/>'
        for age, points in history_quads
    )
    gt_now_path = project_path(gt_now)
    gt_age_11_path = project_path(gt_age_11)
    fir_now_path = project_path(fir_now)
    parity_grid = _parity_grid()
    calibration_marks = _calibration_marks(designs)
    plot_right = _PLOT_LEFT + _PLOT_SIZE
    plot_bottom = _PLOT_TOP + _PLOT_SIZE
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1800 1080" role="img" aria-labelledby="screen-age-title screen-age-desc">
  <title id="screen-age-title">The rendered screen carries a clock</title>
  <desc id="screen-age-desc">Panel A shows a held-out BIWI frame with the current ground-truth screen, an eleven-frame-old counterfactual ground-truth screen, the current causal FIR response, and ten intermediate counterfactual screens. Panel B is a smaller aggregate calibration plot showing exact recovery of the registered seven-, eleven-, and nineteen-frame FIR delays.</desc>
  <rect width="1800" height="1080" fill="#fbfaf7"/>
  <defs>
    <clipPath id="screen-age-event-crop">
      <rect x="64" y="225" width="1000" height="700"/>
    </clipPath>
  </defs>
  <g font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" fill="#1d1f21">
    <text x="64" y="72" font-size="54" font-weight="500" letter-spacing="-1.2">The rendered screen carries a clock.</text>

    <g id="mechanism">
      <text x="64" y="142" font-size="27" font-weight="500">A<tspan fill="#909294" font-weight="400"> · Mechanism</tspan></text>

      <line x1="84" y1="192" x2="149" y2="192" stroke="#1d1f21" stroke-width="3"/>
      <text x="167" y="202" font-size="26" font-weight="400">GT(t)</text>
      <line x1="324" y1="192" x2="389" y2="192" stroke="#1d1f21" stroke-width="3" stroke-dasharray="13 10"/>
      <text x="407" y="202" font-size="26" font-weight="400">GT(t−11)</text>
      <line x1="674" y1="192" x2="739" y2="192" stroke="#27ef75" stroke-width="4"/>
      <text x="757" y="202" font-size="26" font-weight="400">FIR(t)</text>

      <g clip-path="url(#screen-age-event-crop)">
        <image href="{scene_data_uri}" x="-256" y="-55" width="1600" height="1200" preserveAspectRatio="xMidYMid meet"/>
{ghost_paths}
        <path d="{gt_age_11_path}" fill="none" stroke="#fbfaf7" stroke-width="8" stroke-dasharray="15 11"/>
        <path d="{gt_age_11_path}" fill="none" stroke="#1d1f21" stroke-width="3.5" stroke-dasharray="15 11"/>
        <path d="{gt_now_path}" fill="none" stroke="#fbfaf7" stroke-width="10"/>
        <path d="{gt_now_path}" fill="none" stroke="#1d1f21" stroke-width="4.5"/>
        <path d="{fir_now_path}" fill="#27ef75" fill-opacity="0.10" stroke="#27ef75" stroke-width="5"/>
      </g>
      <rect x="64" y="225" width="1000" height="700" fill="none" stroke="#c7c8c6" stroke-width="2"/>

      <text x="564" y="978" text-anchor="middle" font-size="34" font-weight="500"><tspan fill="#909294" font-weight="400">Screen Age</tspan><tspan> · {screen_age} frames · {screen_age_ms:.0f} ms</tspan></text>
      <line x1="214" y1="1018" x2="914" y2="1018" stroke="#27ef75" stroke-width="3.5"/>
      <line x1="214" y1="1005" x2="214" y2="1031" stroke="#27ef75" stroke-width="3.5"/>
      <line x1="914" y1="1005" x2="914" y2="1031" stroke="#27ef75" stroke-width="3.5"/>
      <text x="214" y="1062" text-anchor="middle" font-size="24">t−11</text>
      <text x="914" y="1062" text-anchor="middle" font-size="24">t</text>
    </g>

    <g id="calibration">
      <text x="1128" y="142" font-size="27" font-weight="500">B<tspan fill="#909294" font-weight="400"> · Calibration</tspan></text>

      <circle cx="1160" cy="192" r="10" fill="#fbfaf7" stroke="#27ef75" stroke-width="3"/>
      <text x="1184" y="202" font-size="22">development</text>
      <path d="M1460,182 L1470,192 L1460,202 L1450,192 Z" fill="#1d1f21" stroke="#fbfaf7" stroke-width="1.5"/>
      <text x="1484" y="202" font-size="22">held-out</text>

      <rect x="{_PLOT_LEFT:.1f}" y="{_PLOT_TOP:.1f}" width="{_PLOT_SIZE:.1f}" height="{_PLOT_SIZE:.1f}" fill="#ffffff" stroke="#bfc1c0" stroke-width="2"/>
{parity_grid}
      <line x1="{_PLOT_LEFT:.1f}" y1="{plot_bottom:.1f}" x2="{plot_right:.1f}" y2="{_PLOT_TOP:.1f}" stroke="#1d1f21" stroke-width="2.5" stroke-dasharray="10 8" opacity="0.62"/>
{calibration_marks}

      <text x="1450" y="860" text-anchor="middle" font-size="22" font-weight="400">Known FIR delay (frames)</text>
      <text x="1140" y="520" text-anchor="middle" font-size="22" font-weight="400" transform="rotate(-90 1140 520)">Measured Screen Age (frames)</text>
      <text x="1450" y="958" text-anchor="middle" fill="#909294" font-size="22">6 / 6 exact · development and held-out</text>
    </g>
  </g>
</svg>
'''


def _validate_history_paths(svg: str) -> None:
    history_ages = [
        int(value)
        for value in re.findall(
            r'class="screen-worldline-history" data-age-frames="(\d+)"',
            svg,
        )
    ]
    if history_ages != list(range(10, 0, -1)):
        raise ValueError("SVG must contain the ten registered intermediate screen ages")


def _validate_svg(svg: str) -> None:
    lowered = svg.lower()
    forbidden = (
        "<script",
        "<foreignobject",
        "xlink:href",
        "/users/",
        "file:",
    )
    found = [token for token in forbidden if token in lowered]
    if found:
        raise ValueError(f"SVG is not publish-safe: {found}")
    clip_urls = re.findall(r"url\(([^)]+)\)", svg)
    if clip_urls != ["#screen-age-event-crop"]:
        raise ValueError(f"SVG contains an unexpected URL reference: {clip_urls}")
    root = ET.fromstring(svg)
    if root.attrib.get("viewBox") != "0 0 1800 1080":
        raise ValueError("SVG viewBox must remain 1800x1080")
    if root.find("{http://www.w3.org/2000/svg}title") is None:
        raise ValueError("SVG requires an accessible title")
    if root.find("{http://www.w3.org/2000/svg}desc") is None:
        raise ValueError("SVG requires an accessible description")
    _validate_history_paths(svg)
    images = re.findall(r'<image href="(data:image/jpeg;base64,[^"]+)"', svg)
    if len(images) != 1:
        raise ValueError("SVG must contain exactly one embedded, verified JPEG scene")
    hrefs = re.findall(r'\shref="([^"]+)"', svg)
    if hrefs != images:
        raise ValueError("SVG must not reference external resources")
    image_payload = base64.b64decode(images[0].partition(",")[2], validate=True)
    if _sha256(image_payload) != _SCENE_SHA256:
        raise ValueError("embedded Screen Age scene changed")
    font_sizes = [int(value) for value in re.findall(r'font-size="(\d+)"', svg)]
    if not font_sizes or min(font_sizes) < 21:
        raise ValueError("all SVG text must remain at least 21 canvas pixels")


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.from_canonical is not None:
        receipt = _public_receipt(args.from_canonical)
        _validate_public_receipt(receipt)
        _atomic_write(
            args.data,
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
    else:
        receipt = json.loads(args.data.read_text(encoding="utf-8"))

    svg = _render_svg(receipt, args.scene)
    _validate_svg(svg)
    _atomic_write(args.out, svg)
    print(f"wrote {args.out} ({len(svg.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
