import numpy as np
import pytest

from frame2frame.geometry import head_forward
from scripts.make_demo import _observation_source


@pytest.mark.parametrize("frame_index", [15, 30, 75, 90])
def test_synthetic_face_looks_toward_its_lateral_excursion(frame_index):
    frames = 120
    width, height = 480, 320
    observation = _observation_source(
        frames,
        (width, height),
        noise_deg=0.0,
        noise_px=0.0,
    )(frame_index, np.empty((0, 0, 3), dtype=np.uint8))

    face_offset_x = observation.center[0] - width / 2
    gaze_x = head_forward(observation.pose.yaw, observation.pose.pitch)[0]

    assert np.sign(gaze_x) == np.sign(face_offset_x)
