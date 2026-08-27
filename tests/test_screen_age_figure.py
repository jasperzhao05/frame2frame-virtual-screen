import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.make_screen_age_figure import (
    _calibration_marks,
    _canonical_fir_rows,
    _render_svg,
    _validate_public_receipt,
    _x_scale,
    _y_scale,
)

_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT_PATH = _ROOT / "docs/screen-age-data.json"
_SVG_PATH = _ROOT / "docs/screen-age.svg"


def _receipt():
    return json.loads(_RECEIPT_PATH.read_text(encoding="utf-8"))


def test_committed_screen_age_figure_is_reproducible_and_publish_safe():
    receipt = _receipt()

    _validate_public_receipt(receipt)
    svg = _render_svg(receipt)

    assert svg == _SVG_PATH.read_text(encoding="utf-8")
    assert "<image" not in svg
    assert "<script" not in svg
    assert "/Users/" not in svg


def test_public_receipt_rejects_a_false_recovered_age_claim():
    receipt = deepcopy(_receipt())
    default = next(row for row in receipt["fir_designs"] if row["project_default"])
    default["development_estimate_frames"] = 3
    default["held_out_diagnostic_frames"] = 20
    default["development_ci95_frames"] = [0, 20]

    with pytest.raises(ValueError, match="development estimate must recover"):
        _validate_public_receipt(receipt)


def test_public_receipt_rejects_fractional_age_and_changed_hero():
    fractional = deepcopy(_receipt())
    default = next(row for row in fractional["fir_designs"] if row["project_default"])
    default["development_estimate_frames"] = 11.9

    with pytest.raises(ValueError, match="development estimate must be an integer"):
        _validate_public_receipt(fractional)

    changed_hero = deepcopy(_receipt())
    changed_hero["hero"]["screen_age_ms"] = 366.8

    with pytest.raises(ValueError, match="hero changed"):
        _validate_public_receipt(changed_hero)


def test_calibration_marks_are_driven_by_recovered_values():
    designs = _receipt()["fir_designs"]
    default = next(row for row in designs if row["project_default"])
    default["development_estimate_frames"] = 10
    default["held_out_diagnostic_frames"] = 12

    markup = _calibration_marks(designs)
    x = _x_scale(11)
    development_y = _y_scale(10)
    held_out_y = _y_scale(12)

    assert f'cx="{x:.1f}" cy="{development_y:.1f}"' in markup
    assert f"M{x:.1f},{held_out_y - 8:.1f}" in markup


def test_canonical_registry_rejects_duplicate_fir_designs():
    rows = [
        {"filter_family": "fir", "filter_name": name}
        for name in (
            "fir-fc-1-tw-3",
            "fir-fc-1.5-tw-5",
            "fir-fc-2.5",
            "fir-project-default",
            "fir-fc-4-tw-8",
            "fir-fc-6-tw-8",
        )
    ]
    rows.append(deepcopy(rows[0]))

    with pytest.raises(ValueError, match="exactly six"):
        _canonical_fir_rows({"results": rows})


def test_public_receipt_names_candidate_age_as_a_range():
    protocol = _receipt()["protocol"]

    assert protocol["candidate_age_range_frames"] == [0, 20]
    assert "candidate_age_frames" not in protocol


def test_source_manifest_includes_screen_age_publication_assets():
    manifest = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    docs_rule = next(
        line for line in manifest.splitlines() if line.startswith("recursive-include docs")
    )

    assert all(pattern in docs_rule.split() for pattern in ("*.json", "*.png", "*.svg"))
