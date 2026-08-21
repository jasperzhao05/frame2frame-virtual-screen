import cv2
import numpy as np
import pytest

from frame2frame import geometry
from frame2frame.pose._canonical_pose import _canonical_vertices, solve_renderer_rotation
from frame2frame.pose.mediapipe_face import MediaPipeEstimator

_CANONICAL_TO_CAMERA = np.diag([1.0, -1.0, -1.0])
_FRAME_SHAPE = (480, 640, 3)
_FOCAL_LENGTH = 640.0


def _project_face(
    yaw: float,
    pitch: float,
    roll: float,
    translation: tuple[float, float, float] = (0.0, 0.0, 80.0),
) -> np.ndarray:
    renderer_rotation = geometry.head_rotation(yaw, pitch, roll)
    canonical_rotation = renderer_rotation @ _CANONICAL_TO_CAMERA
    rotation_vector, _ = cv2.Rodrigues(canonical_rotation)
    intrinsic = geometry.camera_matrix(_FOCAL_LENGTH, 320.0, 240.0)
    points, _ = cv2.projectPoints(
        _canonical_vertices(),
        rotation_vector,
        np.asarray(translation, dtype=np.float64),
        intrinsic,
        None,
    )
    return points.reshape(-1, 2)


@pytest.mark.parametrize(
    ("yaw", "pitch", "roll"),
    [
        pytest.param(0.0, 0.0, 0.0, id="neutral"),
        pytest.param(30.0, 15.0, 7.0, id="left-up-clockwise"),
        pytest.param(-40.0, -20.0, -12.0, id="right-down-counterclockwise"),
        pytest.param(45.0, 30.0, 20.0, id="operating-envelope-boundary"),
    ],
)
def test_canonical_fit_recovers_renderer_rotation(yaw, pitch, roll):
    landmarks = _project_face(yaw, pitch, roll)

    recovered = solve_renderer_rotation(
        landmarks,
        _FRAME_SHAPE,
        focal_length=_FOCAL_LENGTH,
    )

    assert recovered is not None
    assert np.allclose(recovered, geometry.head_rotation(yaw, pitch, roll), atol=1e-6)


@pytest.mark.parametrize(
    "translation",
    [
        pytest.param((0.0, 0.0, 80.0), id="centered"),
        pytest.param((12.0, -7.0, 95.0), id="upper-right-and-farther"),
        pytest.param((-10.0, 8.0, 65.0), id="lower-left-and-closer"),
    ],
)
def test_canonical_fit_rotation_is_independent_of_face_translation(translation):
    landmarks = _project_face(-32.0, 18.0, -9.0, translation)

    recovered = solve_renderer_rotation(
        landmarks,
        _FRAME_SHAPE,
        focal_length=_FOCAL_LENGTH,
    )

    assert recovered is not None
    assert np.allclose(recovered, geometry.head_rotation(-32.0, 18.0, -9.0), atol=1e-6)


def test_canonical_fit_uses_only_the_stable_468_face_vertices():
    landmarks = _project_face(22.0, -11.0, 5.0)
    landmarks_with_iris = np.vstack([landmarks, np.full((10, 2), 1_000_000.0)])

    recovered = solve_renderer_rotation(
        landmarks_with_iris,
        _FRAME_SHAPE,
        focal_length=_FOCAL_LENGTH,
    )

    assert recovered is not None
    assert np.allclose(recovered, geometry.head_rotation(22.0, -11.0, 5.0), atol=1e-6)


@pytest.mark.parametrize(
    "landmarks",
    [
        pytest.param(np.zeros((467, 2)), id="too-few"),
        pytest.param(np.zeros((468, 1)), id="one-coordinate"),
        pytest.param(np.full((468, 2), np.nan), id="nonfinite"),
    ],
)
def test_canonical_fit_rejects_invalid_landmark_sets(landmarks):
    assert solve_renderer_rotation(landmarks, _FRAME_SHAPE) is None


@pytest.mark.parametrize("focal_length", [0.0, -1.0, float("nan"), True])
def test_canonical_fit_rejects_invalid_focal_length(focal_length):
    with pytest.raises(ValueError, match="focal_length"):
        solve_renderer_rotation(
            _project_face(0.0, 0.0, 0.0),
            _FRAME_SHAPE,
            focal_length=focal_length,
        )


def test_mediapipe_adapter_preserves_the_public_angle_signs():
    landmarks = _project_face(25.0, -14.0, 8.0)

    class Detector:
        def process(self, frame, timestamp_ms):
            assert timestamp_ms is None
            return landmarks

    estimator = MediaPipeEstimator.__new__(MediaPipeEstimator)
    estimator._detector = Detector()
    estimator._signs = (1.0, -1.0, 1.0)
    estimator._focal_length = _FOCAL_LENGTH

    observation = estimator.estimate(np.zeros(_FRAME_SHAPE, dtype=np.uint8))

    assert observation is not None
    assert observation.pose.yaw == pytest.approx(25.0, abs=1e-6)
    assert observation.pose.pitch == pytest.approx(-14.0, abs=1e-6)
    assert observation.pose.roll == pytest.approx(8.0, abs=1e-6)
