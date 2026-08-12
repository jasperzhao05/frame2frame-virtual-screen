import numpy as np
import pytest

from frame2frame.pose import available_backends, create_estimator
from frame2frame.pose._facemesh import (
    FaceMeshDetector,
    _detected_face_crop,
    _resolve_model,
    _VideoTimestampSequence,
    bbox_from_points,
)
from frame2frame.pose.base import FaceObservation, HeadPose
from frame2frame.pose.hopenet import _resolve_weights


def test_custom_face_model_path_must_exist(tmp_path):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"caller-supplied model")

    assert _resolve_model(model) == model
    with pytest.raises(FileNotFoundError, match="model not found"):
        _resolve_model(tmp_path / "missing.task")


def test_custom_hopenet_weights_must_be_a_file(tmp_path):
    weights = tmp_path / "weights.pkl"
    weights.write_bytes(b"caller-supplied weights")

    assert _resolve_weights(weights, None) == weights
    with pytest.raises(FileNotFoundError, match="not a file"):
        _resolve_weights(tmp_path, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"num_faces": 0}, id="zero-face-count"),
        pytest.param({"num_faces": True}, id="boolean-face-count"),
        pytest.param(
            {"min_detection_confidence": -0.1},
            id="negative-detection-confidence",
        ),
        pytest.param(
            {"min_tracking_confidence": 1.1},
            id="tracking-confidence-above-one",
        ),
        pytest.param({"fps": 0}, id="zero-fps"),
        pytest.param({"fps": float("nan")}, id="nonfinite-fps"),
    ],
)
def test_face_detector_rejects_invalid_options_before_backend_initialization(kwargs):
    with pytest.raises(ValueError):
        FaceMeshDetector(**kwargs)


def test_video_timestamps_remain_strict_after_integer_rounding():
    timestamps = _VideoTimestampSequence(2000.0)

    assert [timestamps.next(), timestamps.next(), timestamps.next()] == [0, 1, 2]


def test_video_timestamps_reject_a_source_clock_that_moves_backwards():
    timestamps = _VideoTimestampSequence(30.0)
    timestamps.next(10.0)

    with pytest.raises(ValueError, match="backwards"):
        timestamps.next(9.0)


def test_bbox_is_clipped_to_the_frame():
    points = np.array([[-4.0, 3.0], [30.0, 25.0]])

    assert bbox_from_points(points, margin=2, frame_shape=(20, 24, 3)) == (0, 1, 23, 19)


def test_deep_backends_share_one_landmark_crop_contract():
    class Detector:
        def landmarks(self, frame, timestamp_ms):
            assert timestamp_ms == 125.0
            return np.array([[2.0, 3.0], [8.0, 9.0]])

    frame = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
    detected = _detected_face_crop(Detector(), frame, margin=1, timestamp_ms=125.0)

    assert detected is not None
    assert detected.bbox == (1, 2, 9, 10)
    assert np.array_equal(detected.image, frame[2:10, 1:9])


def test_deep_backend_crop_propagates_missing_detection():
    class Detector:
        def landmarks(self, frame, timestamp_ms):
            return None

    frame = np.zeros((12, 14, 3), np.uint8)

    assert _detected_face_crop(Detector(), frame, margin=1, timestamp_ms=None) is None


def test_observation_derives_shared_geometry_from_bbox():
    pose = HeadPose(1.0, 2.0, 3.0)

    observation = FaceObservation.from_bbox(pose, (10, 20, 30, 60))

    assert observation.center == (20, 40)
    assert observation.size == 20.0


def test_observation_preserves_half_pixel_bbox_center():
    observation = FaceObservation.from_bbox(HeadPose(0, 0, 0), (10, 20, 31, 61))

    assert observation.center == (20.5, 40.5)


def test_observation_remains_mutable_for_pipeline_smoothing():
    observation = FaceObservation.from_bbox(HeadPose(0, 0, 0), (10, 20, 30, 60))

    observation.pose.yaw = 12.0
    observation.center = (21.0, 41.0)

    assert observation.pose.yaw == 12.0
    assert observation.center == (21.0, 41.0)


def test_pose_registry_lists_builtin_backend_names():
    assert available_backends() == ("mediapipe", "hopenet", "6drepnet")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("unknown", id="unknown"),
    ],
)
def test_pose_registry_rejects_invalid_names_cleanly(name):
    with pytest.raises(ValueError, match="backend"):
        create_estimator(name)
