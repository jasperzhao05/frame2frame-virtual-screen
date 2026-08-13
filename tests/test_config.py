import os

import pytest

from frame2frame.config import FilterConfig, PipelineConfig, ScreenConfig


def _minimal_config(**kwargs):
    defaults = dict(
        input="input.mp4",
        output=None,
        plot_path=None,
        draw_screen=False,
        filter=FilterConfig(kind="none"),
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def test_symlink_alias_is_rejected(tmp_path):
    source = tmp_path / "source.mp4"
    alias = tmp_path / "alias.mp4"
    source.write_bytes(b"source")
    alias.symlink_to(source)

    with pytest.raises(ValueError, match="different paths"):
        _minimal_config(input=str(source), output=str(alias)).validate()


def test_hard_link_alias_is_rejected(tmp_path):
    source = tmp_path / "source.mp4"
    alias = tmp_path / "alias.mp4"
    source.write_bytes(b"source")
    os.link(source, alias)

    with pytest.raises(ValueError, match="different paths"):
        _minimal_config(input=str(source), output=str(alias)).validate()


@pytest.mark.parametrize(
    "collision",
    [
        pytest.param("input", id="matches-input"),
        pytest.param("output", id="matches-output"),
    ],
)
def test_plot_path_cannot_overwrite_input_or_output(tmp_path, collision):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    plot = source if collision == "input" else output

    with pytest.raises(ValueError, match="plot_path"):
        _minimal_config(
            input=str(source),
            output=str(output),
            plot_path=str(plot),
        ).validate()


@pytest.mark.parametrize(
    "collision",
    [
        pytest.param("input", id="matches-input"),
        pytest.param("output", id="matches-output"),
        pytest.param("plot", id="matches-plot"),
    ],
)
def test_texture_path_cannot_be_overwritten_or_used_as_video_input(tmp_path, collision):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    plot = tmp_path / "plot.png"
    texture = {"input": source, "output": output, "plot": plot}[collision]

    with pytest.raises(ValueError, match="screen.texture_path"):
        _minimal_config(
            input=str(source),
            output=str(output),
            plot_path=str(plot),
            screen=ScreenConfig(texture_path=str(texture)),
        ).validate()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param(
            PipelineConfig(input="a", webcam=0),
            "exactly one",
            id="both-file-and-webcam",
        ),
        pytest.param(
            PipelineConfig(webcam=-1),
            "webcam",
            id="negative-webcam-index",
        ),
        pytest.param(
            _minimal_config(max_plot_samples=0),
            "max_plot_samples",
            id="zero-max-plot-samples",
        ),
        pytest.param(
            _minimal_config(plot_path="plot-without-extension"),
            "image suffix",
            id="plot-without-image-suffix",
        ),
        pytest.param(
            _minimal_config(display=1),
            "boolean",
            id="integer-display-flag",
        ),
        pytest.param(
            _minimal_config(screen=ScreenConfig(width_mul=True)),
            "finite",
            id="boolean-screen-width",
        ),
        pytest.param(
            _minimal_config(screen=ScreenConfig(border_thickness=True)),
            "border_thickness",
            id="boolean-border-thickness",
        ),
        pytest.param(
            _minimal_config(screen=ScreenConfig(texture_path="")),
            "texture_path",
            id="empty-texture-path",
        ),
        pytest.param(
            _minimal_config(screen=ScreenConfig(alpha=1.1)),
            "alpha",
            id="alpha-above-one",
        ),
        pytest.param(
            _minimal_config(filter=FilterConfig(kind="bogus")),
            "filter",
            id="unknown-filter-kind",
        ),
        pytest.param(
            _minimal_config(filter=FilterConfig(kind="none", beta=False)),
            "finite",
            id="boolean-oneeuro-beta",
        ),
        pytest.param(
            _minimal_config(filter=FilterConfig(kind="fir", cutoff_hz=float("nan"))),
            "finite",
            id="nonfinite-fir-cutoff",
        ),
        pytest.param(
            _minimal_config(dropout_hold_seconds=-0.1),
            "dropout_hold_seconds",
            id="negative-dropout-hold",
        ),
        pytest.param(
            _minimal_config(dropout_reset_seconds=0),
            "dropout_reset_seconds",
            id="zero-dropout-reset",
        ),
        pytest.param(
            _minimal_config(dropout_hold_seconds=0.6),
            "dropout_hold_seconds",
            id="hold-exceeds-reset",
        ),
    ],
)
def test_invalid_configuration_ranges_are_rejected(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_fir_limits_are_checked_against_source_fps():
    cfg = _minimal_config(filter=FilterConfig(kind="fir", cutoff_hz=6.0, transition_hz=2.0))

    with pytest.raises(ValueError, match="Nyquist"):
        cfg.validate(fps=10.0)


def test_wide_fir_transition_is_valid_for_low_frame_rate():
    cfg = _minimal_config(filter=FilterConfig(kind="fir", cutoff_hz=2.5, transition_hz=5.0))

    cfg.validate(fps=6.0)


def test_audio_preservation_requires_file_input_and_output():
    with pytest.raises(ValueError, match="file input"):
        PipelineConfig(webcam=0, preserve_audio=True).validate()
    with pytest.raises(ValueError, match="output"):
        _minimal_config(preserve_audio=True).validate()
