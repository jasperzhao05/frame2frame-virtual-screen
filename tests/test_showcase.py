import numpy as np
import pytest

from scripts.make_showcase import (
    _FRAMES,
    _LABEL_HEIGHT,
    _OUTPUT_SIZE,
    _PANEL_SIZE,
    _comparison_frame,
    _showcase_traces,
)


@pytest.fixture(scope="module")
def showcase_traces():
    return _showcase_traces()


def test_showcase_renders_one_meaningful_six_panel_frame(showcase_traces):
    frame = _comparison_frame(_FRAMES // 2, showcase_traces)

    assert frame.shape == (_OUTPUT_SIZE[1], _OUTPUT_SIZE[0], 3)

    width, height = _PANEL_SIZE
    clean = frame[_LABEL_HEIGHT:height, 0:width]
    raw = frame[_LABEL_HEIGHT:height, width : 2 * width]
    assert np.any(clean != raw)


def test_showcase_loop_has_no_boundary_jump(showcase_traces):
    for trace in showcase_traces.values():
        ordinary_steps = np.linalg.norm(np.diff(trace, axis=0), axis=1)
        loop_step = np.linalg.norm(trace[0] - trace[-1])

        assert loop_step <= 1.5 * np.percentile(ordinary_steps, 95)
