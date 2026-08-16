from copy import deepcopy

import pytest

from scripts.benchmark_pipeline import _markdown, _self_check, run_benchmark


@pytest.fixture(scope="module")
def receipt():
    return run_benchmark(frames=60, runs=2)


def test_short_pipeline_receipt_covers_the_reliability_contracts(receipt):
    assert receipt["status"] == "pass"
    assert receipt["failures"] == []
    assert receipt["method"]["dropout_schedule_exercised"] is True
    assert receipt["aggregate"]["output_digests_stable"] is True
    assert receipt["environment"]["source_revision_status"] in {
        "clean_commit",
        "dirty_worktree",
        "unavailable",
    }

    frames = receipt["method"]["source_frames_decoded"]
    assert [run["summary_frames"] for run in receipt["runs"]] == [frames, frames]
    assert [run["output"]["frames"] for run in receipt["runs"]] == [frames, frames]


def test_self_check_reports_frame_digest_and_dropout_schedule_regressions(receipt):
    broken = deepcopy(receipt)
    broken["runs"][0]["summary_frames"] -= 1
    broken["runs"][1]["output"]["decoded_blake2b"] = "drift"
    broken["method"]["dropout_schedule_exercised"] = False

    failures = _self_check(broken)

    assert any("frame conservation" in failure for failure in failures)
    assert any("digest drifted" in failure for failure in failures)
    assert any("dropout schedule" in failure for failure in failures)


def test_markdown_receipt_keeps_the_throughput_boundary_visible(receipt):
    rendered = _markdown(receipt)

    assert "not neural-backend FPS" in rendered
    assert "not camera-to-display latency" in rendered
    assert "no pass/fail floor" in rendered
