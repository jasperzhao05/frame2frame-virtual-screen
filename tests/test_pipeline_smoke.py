import cv2
import numpy as np

from frame2frame.config import FilterConfig, PipelineConfig, ScreenConfig
from frame2frame.pipeline import run
from frame2frame.pose.base import FaceObservation, HeadPose
from frame2frame.pose.scripted import ScriptedEstimator
from frame2frame.video import VideoWriter

SIZE = (320, 240)


def _make_clip(path, frames, fps=30.0):
    w, h = SIZE
    with VideoWriter(str(path), fps, SIZE) as writer:
        for _ in range(frames):
            img = np.full((h, w, 3), 30, np.uint8)
            cv2.circle(img, (w // 2, h // 2), 30, (200, 180, 160), -1)
            writer.write(img)


def _moving_face(i, frame):
    w, h = SIZE
    pose = HeadPose(20 * np.sin(i / 10), 10 * np.cos(i / 10), 5.0)
    c = (w // 2, h // 2)
    return FaceObservation(pose, c, 30.0, (c[0] - 30, c[1] - 30, c[0] + 30, c[1] + 30))


def test_pipeline_runs_and_writes_output(tmp_path):
    frames = 24
    inp, out, plot = tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path / "angles.png"
    _make_clip(inp, frames)

    cfg = PipelineConfig(
        input=str(inp),
        output=str(out),
        filter=FilterConfig(kind="fir"),
        screen=ScreenConfig(border_thickness=2),
        draw_axis=True,
        draw_bbox=True,
        plot_path=str(plot),
    )
    summary = run(cfg, estimator=ScriptedEstimator(_moving_face))

    assert summary.frames == frames
    assert summary.faces == frames
    assert out.exists() and out.stat().st_size > 0
    assert plot.exists()

    cap = cv2.VideoCapture(str(out))
    written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert written >= frames - 2  # container frame count can be off by one


def test_frames_without_a_face_pass_through(tmp_path):
    frames = 10
    inp, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_clip(inp, frames)

    cfg = PipelineConfig(input=str(inp), output=str(out), plot_path=None)
    summary = run(cfg, estimator=ScriptedEstimator(lambda i, frame: None))

    assert summary.faces == 0
    assert summary.frames == frames
    assert out.exists()
