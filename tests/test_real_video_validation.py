import hashlib
import json
from pathlib import Path

import pytest

from frame2frame import PipelineConfig, RunSummary
from scripts import validate_real_video as validation
from scripts.fetch_examples import BASE, CLIPS


def _media(path: str, *, frames: int = 96, width: int = 768, height: int = 432):
    return validation.MediaFacts(
        path=path,
        sha256="a" * 64,
        size_bytes=1234,
        decoded_frames=frames,
        fps=30.0,
        width_px=width,
        height_px=height,
        decoded_duration_seconds=frames / 30.0,
    )


def _provenance():
    return validation.SourceProvenance(
        uri="https://example.test/clip.mp4",
        license="CC BY 4.0",
        attribution="Example author",
        registry_match=None,
    )


def _receipt(*, input_media=None, output_media=None, summary=None):
    return validation.build_receipt(
        config=PipelineConfig(
            input="input.mp4",
            output="output.mp4",
            plot_path=None,
        ),
        provenance=_provenance(),
        input_media=input_media or _media("input.mp4"),
        output_media=output_media or _media("output.mp4"),
        summary=summary
        or RunSummary(
            frames=96,
            faces=72,
            fps=30.0,
            mean_inference_ms=8.5,
            output="output.mp4",
        ),
        pipeline_wall_seconds=4.0,
        command=["python", "-m", "scripts.validate_real_video"],
        project_state={"commit": "1" * 40, "worktree_dirty": False},
        environment={"python": "3.12.0"},
    )


def test_pinned_example_provenance_is_resolved_by_content_digest():
    name = "head-pose-face-detection-female.mp4"

    source = validation.resolve_provenance(
        Path("renamed.mp4"),
        CLIPS[name].sha256,
        source_uri=None,
        source_license=None,
        source_attribution=None,
    )

    assert source.uri == f"{BASE}/{name}"
    assert source.registry_match.endswith(f"::{name}")
    assert "Creative Commons Attribution 4.0" in source.license


def test_unknown_source_requires_a_complete_provenance_note():
    with pytest.raises(ValueError, match="provide --source-uri"):
        validation.resolve_provenance(
            Path("unknown.mp4"),
            "f" * 64,
            source_uri="owned-capture:one",
            source_license=None,
            source_attribution="Owner",
        )


def test_receipt_reports_only_direct_operational_measurements():
    receipt = _receipt()

    assert receipt["status"] == "pass"
    assert receipt["project"]["source_revision_status"] == "clean_commit"
    assert receipt["run"] == {
        "frames_processed": 96,
        "fresh_detection_frames": 72,
        "fresh_detection_frame_rate_pct": 75.0,
        "source_fps": 30.0,
        "mean_estimator_inference_ms": 8.5,
        "pipeline_wall_seconds": 4.0,
        "processed_frames_per_wall_second": 24.0,
        "audio_remuxed": False,
    }
    non_claims = " ".join(receipt["scope"]["does_not_measure"])
    assert "ground-truth" in non_claims
    assert "perceptual" in non_claims
    assert "production readiness" in non_claims
    assert all(check["passed"] for check in receipt["checks"])


def test_dirty_revision_is_disclosed_without_relabelling_operational_checks():
    receipt = validation.build_receipt(
        config=PipelineConfig(input="input.mp4", output="output.mp4", plot_path=None),
        provenance=_provenance(),
        input_media=_media("input.mp4"),
        output_media=_media("output.mp4"),
        summary=RunSummary(96, 72, 30.0, 8.5, "output.mp4"),
        pipeline_wall_seconds=4.0,
        command=["python", "-m", "scripts.validate_real_video"],
        project_state={"commit": "1" * 40, "worktree_dirty": True},
        environment={"python": "3.12.0"},
    )

    assert receipt["status"] == "pass"
    assert receipt["project"]["source_revision_status"] == "dirty_worktree"


def test_receipt_fails_frame_conservation_without_calling_it_accuracy():
    receipt = _receipt(output_media=_media("output.mp4", frames=95))

    assert receipt["status"] == "fail"
    failed = [check["name"] for check in receipt["checks"] if not check["passed"]]
    assert "output preserved the processed frame count" in failed
    assert "output duration stayed within one frame" not in failed
    assert "accuracy" not in " ".join(check["name"] for check in receipt["checks"])


def test_probe_video_hashes_bytes_and_counts_decoded_frames(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake-but-hashable-video")

    class FakeReader:
        fps = 25.0
        size = (640, 360)

        def __init__(self, received):
            assert received == path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def __iter__(self):
            return iter((object(), object(), object()))

    monkeypatch.setattr(validation, "VideoReader", FakeReader)

    facts = validation.probe_video(path)

    assert facts.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert facts.decoded_frames == 3
    assert facts.decoded_duration_seconds == 3 / 25.0
    assert (facts.width_px, facts.height_px) == (640, 360)


def test_main_writes_stable_json_from_mocked_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    receipt_path = tmp_path / "receipt.json"
    source.write_bytes(b"owned input")
    output.write_bytes(b"processed output")
    inputs = _media(str(source), frames=10, width=320, height=180)
    outputs = _media(str(output), frames=10, width=320, height=180)

    monkeypatch.setattr(
        validation,
        "probe_video",
        lambda path: inputs if path == source else outputs,
    )
    monkeypatch.setattr(
        validation,
        "run",
        lambda config: RunSummary(10, 8, 30.0, 4.25, str(output)),
    )
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr(validation.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        validation,
        "_git_state",
        lambda: {"commit": "2" * 40, "worktree_dirty": False},
    )
    monkeypatch.setattr(validation, "_environment", lambda: {"python": "3.12.0"})

    result = validation.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--receipt",
            str(receipt_path),
            "--source-uri",
            "owned-capture:one",
            "--source-license",
            "owned",
            "--source-attribution",
            "Capture owner",
        ]
    )

    assert result == 0
    raw = receipt_path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert raw == json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    assert parsed["status"] == "pass"
    assert parsed["source"]["uri"] == "owned-capture:one"
    assert parsed["run"]["pipeline_wall_seconds"] == 2.0
    assert parsed["configuration"]["backend"] == "mediapipe"
    assert parsed["configuration"]["filter"]["kind"] == "fir"
    assert parsed["configuration"]["preserve_audio"] is False
    assert parsed["configuration"]["plot_path"] is None
