from copy import deepcopy

import pytest

from scripts import benchmark_dynamic_content as benchmark


@pytest.fixture(scope="module")
def receipt():
    return benchmark.run_benchmark(frames=6, runs=2)


def test_dynamic_content_receipt_checks_structural_contracts(receipt):
    assert receipt["status"] == "pass"
    assert receipt["failures"] == []
    assert all(receipt["contracts"].values())
    assert receipt["aggregate"]["frame_mapping_stable"] is True
    assert receipt["aggregate"]["output_digest_stable"] is True
    assert [run["frames"] for run in receipt["runs"]] == [6, 6]
    assert all(run["mapping_exact"] for run in receipt["runs"])
    assert all(run["zero_copy_every_frame"] for run in receipt["runs"])


def test_dynamic_content_receipt_reports_timings_without_thresholds(receipt):
    for stage in ("sample", "prepare", "composite"):
        timing = receipt["aggregate"]["timings"][stage]
        assert timing["samples"] == 12
        assert timing["p50_ms"] >= 0
        assert timing["p95_ms"] >= timing["p50_ms"]
    assert all("threshold" not in name for name in receipt["aggregate"])
    assert "not portable pass/fail thresholds" in " ".join(receipt["evidence_boundaries"])


def test_self_check_reports_contract_mapping_and_digest_regressions(receipt):
    broken = deepcopy(receipt)
    broken["contracts"]["latest_wins"] = False
    broken["runs"][0]["mapping_exact"] = False
    broken["aggregate"]["output_digest_stable"] = False

    failures = benchmark._self_check(broken)

    assert any("latest_wins" in failure for failure in failures)
    assert any("frame mapping drifted" in failure for failure in failures)
    assert any("output digest" in failure for failure in failures)


def test_markdown_keeps_timing_evidence_boundary_visible(receipt):
    rendered = benchmark._markdown(receipt)

    assert "p50 (ms)" in rendered
    assert "p95 (ms)" in rendered
    assert "not CI timing gates" in rendered
    assert "latest_wins" in rendered


@pytest.mark.parametrize(("frames", "runs"), [(0, 1), (1, 0), (True, 1)])
def test_dynamic_content_benchmark_rejects_invalid_workloads(frames, runs):
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.run_benchmark(frames=frames, runs=runs)
