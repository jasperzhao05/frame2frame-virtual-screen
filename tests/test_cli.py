from types import SimpleNamespace
from urllib.error import URLError

import pytest

from frame2frame import cli


def test_help_does_not_initialize_a_pose_backend(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    assert "Render a head-locked virtual screen" in capsys.readouterr().out


def test_cli_translates_flags_to_pipeline_config(monkeypatch, tmp_path):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return SimpleNamespace(
            frames=12,
            faces=10,
            fps=24.0,
            mean_inference_ms=4.5,
            output=config.output,
        )

    monkeypatch.setattr(cli, "run", fake_run)
    output = tmp_path / "render.mp4"

    assert (
        cli.main(
            [
                "--input",
                "input.mp4",
                "--output",
                str(output),
                "--preserve-audio",
                "--backend",
                "hopenet",
                "--filter",
                "oneeuro",
                "--screen-distance",
                "6.5",
                "--screen-width",
                "3.0",
                "--screen-height",
                "1.5",
                "--cutoff",
                "3.25",
                "--no-smooth-translation",
                "--no-delay-compensation",
                "--dropout-hold",
                "0.15",
                "--dropout-reset",
                "0.4",
                "--no-screen",
                "--axis",
                "--bbox",
                "--plot",
                "",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert config.input == "input.mp4"
    assert config.output == str(output)
    assert config.preserve_audio is True
    assert config.backend == "hopenet"
    assert config.filter.kind == "oneeuro"
    assert config.filter.cutoff_hz == 3.25
    assert config.filter.smooth_translation is False
    assert config.compensate_delay is False
    assert config.dropout_hold_seconds == 0.15
    assert config.dropout_reset_seconds == 0.4
    assert config.screen.distance_mul == 6.5
    assert config.screen.width_mul == 3.0
    assert config.screen.height_mul == 1.5
    assert config.draw_screen is False
    assert config.draw_axis is True
    assert config.draw_bbox is True
    assert config.plot_path is None


def test_cli_requires_exactly_one_input_source():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "clip.mp4", "--webcam", "0"])


def test_version_does_not_require_an_input(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"frame2frame {cli.__version__}"


def test_empty_output_disables_video_writing():
    args = cli.build_parser().parse_args(["--webcam", "0", "--output", ""])

    assert cli.config_from_args(args).output is None


def test_runtime_error_is_concise_unless_debug_is_enabled(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run",
        lambda config: (_ for _ in ()).throw(FileNotFoundError("missing clip")),
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(["--input", "missing.mp4"])
    assert exc.value.code == 2
    assert capsys.readouterr().err == "frame2frame: error: missing clip\n"

    with pytest.raises(FileNotFoundError, match="missing clip"):
        cli.main(["--input", "missing.mp4", "--debug"])


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(URLError("offline"), id="network-unavailable"),
        pytest.param(
            PermissionError("output is not writable"),
            id="output-permission-denied",
        ),
    ],
)
def test_os_errors_are_concise_without_debug(monkeypatch, capsys, error):
    monkeypatch.setattr(cli, "run", lambda config: (_ for _ in ()).throw(error))

    with pytest.raises(SystemExit) as exc:
        cli.main(["--input", "clip.mp4"])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert str(error) in stderr
    assert "Traceback" not in stderr
