import numpy as np

from frame2frame.config import FilterConfig, PipelineConfig
from frame2frame.pose.base import FaceObservation, HeadPose


class EmptyReader:
    fps = 30.0
    size = (20, 12)

    def __init__(self):
        self.released = False

    def __iter__(self):
        return iter(())

    def release(self):
        self.released = True


class ClosingEstimator:
    def __init__(self, error=None):
        self.error = error
        self.closed = False

    def estimate(self, frame):
        if self.error:
            raise self.error
        return None

    def close(self):
        self.closed = True


class SequenceReader:
    fps = 10.0
    size = (8, 6)

    def __init__(self, count):
        self.count = count
        self.released = False

    def __iter__(self):
        for index in range(self.count):
            yield np.full((6, 8, 3), index, np.uint8)

    def release(self):
        self.released = True


class RecordingWriter:
    instances = []

    def __init__(self, path, fps, size):
        self.path = str(path)
        self.frames = []
        self.released = False
        self.__class__.instances.append(self)

    def write(self, frame):
        self.frames.append(int(frame[0, 0, 0]))

    def release(self):
        self.released = True


def make_observation(value):
    return FaceObservation(
        HeadPose(float(value * 10), 0.0, 0.0),
        (4, 3),
        2.0,
        (2, 1, 6, 5),
    )


def minimal_pipeline_config(**kwargs):
    defaults = dict(
        input="input.mp4",
        output=None,
        plot_path=None,
        draw_screen=False,
        filter=FilterConfig(kind="none"),
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def timing_pipeline_config(**kwargs):
    defaults = dict(
        input="input.mp4",
        output="output.mp4",
        plot_path=None,
        draw_screen=False,
        draw_axis=True,
        filter=FilterConfig(kind="none", smooth_translation=False),
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)
