from types import SimpleNamespace
from urllib.error import URLError

import pytest

from frame2frame import cli


def test_help_does_not_initialize_a_pose_backend(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    assert "Render a head-relative virtual screen" in capsys.readouterr().out


def test_cli_translates_flags_to_pipeline_config(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return SimpleNamespace(
            frames=12,
            faces=10,
            fps=24.0,
            mean_inference_ms=4.5,
            mean_content_ms=0.02,
            mean_render_ms=0.8,
            content_samples=12,
            render_samples=10,
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
                "--focal-length-px",
                "812.5",
                "--screen-video",
                "interface.mp4",
                "--screen-video-end",
                "loop",
                "--screen-fit",
                "contain",
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
    assert config.screen.focal_length == 812.5
    assert config.screen.video_path == "interface.mp4"
    assert config.screen.video_end == "loop"
    assert config.screen.content_fit == "contain"
    assert config.draw_screen is False
    assert config.draw_axis is True
    assert config.draw_bbox is True
    assert config.plot_path is None
    stdout = capsys.readouterr().out
    assert "content: 0.020 ms/sample" in stdout
    assert "render: 0.800 ms/attempt" in stdout
    assert "attempts: 10" in stdout


def test_cli_screen_defaults_match_the_configuration_contract():
    args = cli.build_parser().parse_args(["--input", "input.mp4"])
    config = cli.config_from_args(args)

    assert config.screen == cli.ScreenConfig()


def test_cli_kalman_selects_the_attitude_only_registered_design():
    args = cli.build_parser().parse_args(["--input", "input.mp4", "--filter", "kalman"])
    config = cli.config_from_args(args)

    assert config.filter.kind == "kalman"
    assert config.filter.acceleration_std == 100.0
    assert config.filter.measurement_std == 1.0
    assert config.filter.smooth_translation is False
    config.validate(fps=30.0)


def test_cli_requires_exactly_one_input_source():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "clip.mp4", "--webcam", "0"])


def test_cli_rejects_two_screen_content_files():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--input", "clip.mp4", "--texture", "screen.png", "--screen-video", "ui.mp4"]
        )


def test_version_does_not_require_an_input(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"frame2frame {cli.__version__}"


def test_doctor_does_not_require_or_initialize_an_input(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_doctor", lambda: called.append("doctor") or 0)
    monkeypatch.setattr(cli, "run", lambda config: (_ for _ in ()).throw(AssertionError(config)))

    assert cli.main(["--doctor"]) == 0
    assert called == ["doctor"]


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
