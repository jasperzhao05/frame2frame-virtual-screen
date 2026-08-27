"""Build the publish-safe static Screen Age research figure.

The default path reads only the compact aggregate receipt committed under
``docs/``.  Maintainers can explicitly refresh that receipt from the canonical
local Screen Age result with ``--from-canonical``; source media and record-level
rows are never copied.

    python -m scripts.make_screen_age_figure
"""

# SVG markup stays literal so the emitted artifact is easy to audit.
# ruff: noqa: E501

from __future__ import annotations

import argparse
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
_FPS = 30.0
_SCREEN_DISTANCE = 4.0
_BOOTSTRAP_RESAMPLES = 2_000
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
    return 1150.0 + (value - 5.0) / 16.0 * 350.0


def _y_scale(value: float) -> float:
    return 540.0 - (value - 5.0) / 16.0 * 350.0


def _parity_grid() -> str:
    markup: list[str] = []
    for value in (7, 11, 15, 19):
        x = _x_scale(float(value))
        y = _y_scale(float(value))
        markup.extend(
            [
                f'      <line x1="1150" y1="{y:.1f}" x2="1500" y2="{y:.1f}" stroke="#e3e6e8" stroke-width="1.5"/>',
                f'      <line x1="{x:.1f}" y1="190" x2="{x:.1f}" y2="540" stroke="#e3e6e8" stroke-width="1.5"/>',
                f'      <line x1="{x:.1f}" y1="540" x2="{x:.1f}" y2="549" stroke="#7b858f" stroke-width="2"/>',
                f'      <text x="{x:.1f}" y="576" text-anchor="middle" fill="#5e6974" font-size="20">{value}</text>',
                f'      <line x1="1141" y1="{y:.1f}" x2="1150" y2="{y:.1f}" stroke="#7b858f" stroke-width="2"/>',
                f'      <text x="1128" y="{y + 7:.1f}" text-anchor="end" fill="#5e6974" font-size="20">{value}</text>',
            ]
        )
    return "\n".join(markup)


def _calibration_marks(designs: list[dict[str, Any]]) -> str:
    counts = Counter(int(row["analytic_delay_frames"]) for row in designs)
    if counts != Counter({7: 2, 11: 3, 19: 1}):
        raise ValueError("expected three FIR calibration loci with multiplicities 2, 3, and 1")
    labels = {
        7: (1218.0, 487.0, "start"),
        11: (1305.0, 400.0, "start"),
        19: (1438.0, 270.0, "start"),
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
                    f'      <circle cx="{x:.1f}" cy="{development_y:.1f}" r="14" '
                    'fill="#fbfaf7" stroke="#2f6fdf" stroke-width="3"/>'
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
            f'text-anchor="{anchor}" fill="#17212b" font-size="20" font-weight="650">'
            f"{label}</text>"
        )
    return "\n".join(output)


def _render_svg(receipt: dict[str, Any]) -> str:
    _validate_public_receipt(receipt)
    designs = receipt["fir_designs"]
    hero = receipt["hero"]
    screen_age = int(hero["screen_age_frames"])
    screen_age_ms = float(hero["screen_age_ms"])
    if hero.get("filter_name") != "fir-project-default" or screen_age != 11:
        raise ValueError("hero must remain the project-default 11-frame FIR")

    trajectories = [
        [(254, 353), (447, 319), (463, 430), (265, 465)],
        [(333, 323), (531, 284), (552, 396), (346, 438)],
        [(420, 289), (624, 245), (650, 359), (437, 407)],
        [(512, 252), (722, 203), (752, 320), (533, 371)],
        [(607, 213), (824, 160), (856, 280), (628, 336)],
    ]
    filtered = [(258, 349), (451, 317), (467, 432), (269, 468)]
    ghost_paths = "\n".join(
        (
            f'      <path d="{_path(points)}" fill="none" stroke="#7b858f" '
            f'stroke-width="2.5" opacity="{opacity:.2f}"/>'
        )
        for points, opacity in zip(trajectories[1:-1], (0.22, 0.30, 0.40))
    )
    past_path = _path(trajectories[0])
    filtered_path = _path(filtered)
    current_path = _path(trajectories[-1])
    parity_grid = _parity_grid()
    calibration_marks = _calibration_marks(designs)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 680" role="img" aria-labelledby="screen-age-title screen-age-desc">
  <title id="screen-age-title">Screen Age mechanism and causal FIR calibration</title>
  <desc id="screen-age-desc">A project-authored geometric schematic shows a causal FIR response closest to an eleven-frame-old ground-truth counterfactual, with a small visible mismatch between them. A square parity plot shows that six FIR designs recover their analytic delays at seven, eleven, and nineteen frames in development and held-out diagnostics. No person media is included.</desc>
  <rect width="1600" height="680" fill="#fbfaf7"/>
  <g font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" fill="#17212b">
    <line x1="1040" y1="44" x2="1040" y2="640" stroke="#d6dce1" stroke-width="2"/>

    <g id="mechanism">
      <text x="64" y="58" font-size="20" font-weight="700" letter-spacing="2.2" fill="#5e6974">A · MECHANISM</text>
      <text x="64" y="106" font-size="34" font-weight="650" letter-spacing="-0.5">A causal filter can represent an older moment.</text>
      <text x="64" y="140" font-size="20" fill="#5e6974">Past attitudes are reprojected on the same current carrier.</text>

      <path d="M66,175 L96,170 L99,188 L69,193 Z" fill="none" stroke="#17212b" stroke-width="3"/>
      <text x="112" y="190" font-size="20" font-weight="600">GT(t)</text>
      <path d="M226,175 L256,170 L259,188 L229,193 Z" fill="none" stroke="#6f7782" stroke-width="3" stroke-dasharray="8 6"/>
      <text x="272" y="190" font-size="20" font-weight="600">GT(t−11)</text>
      <path d="M446,175 L476,170 L479,188 L449,193 Z" fill="#2f6fdf" fill-opacity="0.12" stroke="#2f6fdf" stroke-width="3"/>
      <text x="492" y="190" font-size="20" font-weight="600">FIR[GT](t)</text>

      <path d="M352,404 C445,354 555,303 730,247" fill="none" stroke="#aeb8c0" stroke-width="2.5" stroke-dasharray="5 9"/>
      <path d="M730,247 L713,242 L719,258 Z" fill="#aeb8c0"/>
{ghost_paths}
      <path d="{filtered_path}" fill="#2f6fdf" fill-opacity="0.10" stroke="#2f6fdf" stroke-width="4"/>
      <path d="{past_path}" fill="none" stroke="#6f7782" stroke-width="2.5" stroke-dasharray="11 8"/>
      <path d="{current_path}" fill="none" stroke="#17212b" stroke-width="4"/>

      <line x1="304" y1="586" x2="741" y2="586" stroke="#2f6fdf" stroke-width="3"/>
      <line x1="304" y1="574" x2="304" y2="598" stroke="#2f6fdf" stroke-width="3"/>
      <line x1="741" y1="574" x2="741" y2="598" stroke="#2f6fdf" stroke-width="3"/>
      <text x="522.5" y="563" text-anchor="middle" font-size="25" font-weight="700" fill="#2f6fdf">Screen Age · {screen_age} frames · {screen_age_ms:.0f} ms</text>
      <text x="304" y="626" text-anchor="middle" font-size="20" fill="#5e6974">t−11</text>
      <text x="741" y="626" text-anchor="middle" font-size="20" fill="#5e6974">t</text>
    </g>

    <g id="calibration">
      <text x="1080" y="58" font-size="20" font-weight="700" letter-spacing="2.2" fill="#5e6974">B · CALIBRATION</text>
      <text x="1080" y="106" font-size="30" font-weight="650" letter-spacing="-0.4">Screen Age recovers known FIR delay.</text>

      <circle cx="1092" cy="148" r="11" fill="#fbfaf7" stroke="#2f6fdf" stroke-width="3"/>
      <text x="1116" y="155" font-size="20">development estimate</text>
      <path d="M1370,137 L1381,148 L1370,159 L1359,148 Z" fill="#17212b" stroke="#fbfaf7" stroke-width="1.5"/>
      <text x="1393" y="155" font-size="20">held-out diagnostic</text>

      <rect x="1150" y="190" width="350" height="350" fill="#ffffff" stroke="#aeb8c0" stroke-width="2"/>
{parity_grid}
      <line x1="1150" y1="540" x2="1500" y2="190" stroke="#17212b" stroke-width="2.5" stroke-dasharray="10 8" opacity="0.66"/>
{calibration_marks}
      <text x="1172" y="224" font-size="20" font-weight="650" fill="#5e6974">6 / 6 ages recovered</text>

      <text x="1325" y="623" text-anchor="middle" font-size="20" font-weight="600">Analytic FIR group delay (frames)</text>
      <text x="1087" y="365" text-anchor="middle" font-size="20" font-weight="600" transform="rotate(-90 1087 365)">Recovered Screen Age (frames)</text>
    </g>
  </g>
</svg>
'''


def _validate_svg(svg: str) -> None:
    lowered = svg.lower()
    forbidden = (
        "<image",
        "<script",
        "<foreignobject",
        "data:image",
        "xlink:href",
        " href=",
        "/users/",
        "url(",
    )
    found = [token for token in forbidden if token in lowered]
    if found:
        raise ValueError(f"SVG is not publish-safe: {found}")
    root = ET.fromstring(svg)
    if root.attrib.get("viewBox") != "0 0 1600 680":
        raise ValueError("SVG viewBox must remain 1600x680")
    if root.find("{http://www.w3.org/2000/svg}title") is None:
        raise ValueError("SVG requires an accessible title")
    if root.find("{http://www.w3.org/2000/svg}desc") is None:
        raise ValueError("SVG requires an accessible description")
    font_sizes = [int(value) for value in re.findall(r'font-size="(\d+)"', svg)]
    if not font_sizes or min(font_sizes) < 20:
        raise ValueError("all SVG text must remain at least 20 canvas pixels")


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

    svg = _render_svg(receipt)
    _validate_svg(svg)
    _atomic_write(args.out, svg)
    print(f"wrote {args.out} ({len(svg.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
