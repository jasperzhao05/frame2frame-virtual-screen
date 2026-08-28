"""Exercise the real-time screen-content seam with generated or video frames.

The producer deliberately runs independently from pose estimation.  It publishes
either sequence/timestamp cards or a looping video into a capacity-one
LatestFrameSource, so a slow consumer never receives an internal queue of stale
content.  Press Escape in the preview window to stop.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

from frame2frame import (
    FilterConfig,
    LatestFrameSource,
    PipelineConfig,
    ScreenConfig,
    run,
)


def make_live_card(sequence: int, elapsed_seconds: float, width: int, height: int) -> np.ndarray:
    """Build an asymmetric card whose sequence and orientation are easy to audit."""
    image = np.full((height, width, 3), (22, 24, 30), np.uint8)
    corner = max(18, min(width, height) // 12)
    image[:corner, :corner] = (30, 80, 245)  # red, top-left
    image[:corner, -corner:] = (40, 220, 60)  # green, top-right
    image[-corner:, -corner:] = (235, 90, 35)  # blue, bottom-right
    image[-corner:, :corner] = (30, 220, 235)  # yellow, bottom-left

    scale = max(0.7, width / 640.0)
    thickness = max(1, round(2 * scale))
    cv2.putText(
        image,
        f"LIVE  {sequence:06d}",
        (round(width * 0.08), round(height * 0.43)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"t = {elapsed_seconds:8.3f} s",
        (round(width * 0.08), round(height * 0.66)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.72,
        (175, 185, 205),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    scan_x = int((elapsed_seconds * 160.0) % max(width, 1))
    cv2.line(image, (scan_x, 0), (scan_x, height - 1), (255, 255, 255), thickness)
    return image


def _produce(
    source: LatestFrameSource,
    stop: threading.Event,
    fps: float,
    size: tuple[int, int],
) -> None:
    period = 1.0 / fps
    origin = time.monotonic()
    deadline = origin
    sequence = 0
    while not stop.is_set():
        now = time.monotonic()
        source.publish(
            make_live_card(sequence, now - origin, size[0], size[1]),
            copy=False,
        )
        sequence += 1
        after_publish = time.monotonic()
        deadline = _next_deadline(deadline, after_publish, period)
        stop.wait(deadline - after_publish)


def _produce_video(
    source: LatestFrameSource,
    stop: threading.Event,
    path: str,
    fps_override: float | None,
    ready: threading.Event,
    errors: list[BaseException],
) -> None:
    """Loop a video on an independent clock and publish only its latest frame."""
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open content video: {path}")
        native_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = native_fps if fps_override is None else fps_override
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError("content video must report a positive finite frame rate")

        period = 1.0 / fps
        deadline = time.monotonic()
        published = False
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok:
                if not published:
                    raise RuntimeError(f"content video contains no decodable frames: {path}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"could not restart content video: {path}")

            source.publish(frame, copy=False)
            if not published:
                published = True
                ready.set()
            after_publish = time.monotonic()
            deadline = _next_deadline(deadline, after_publish, period)
            stop.wait(deadline - after_publish)
    except BaseException as error:
        errors.append(error)
        ready.set()
    finally:
        capture.release()


def _next_deadline(previous: float, now: float, period: float) -> float:
    """Advance once, skipping obsolete producer ticks after a stall."""
    scheduled = previous + period
    return scheduled if scheduled > now else now + period


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webcam", type=int, default=0, help="person-camera device index")
    parser.add_argument(
        "--content-video",
        help="loop a video through the asynchronous latest-frame producer",
    )
    parser.add_argument(
        "--producer-fps",
        type=float,
        help="producer rate override; defaults to 60 for cards or the video's native rate",
    )
    parser.add_argument("--content-width", type=int, default=640)
    parser.add_argument("--content-height", type=int, default=360)
    parser.add_argument("--output", default="output/live-content.mp4")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.webcam < 0:
        raise ValueError("--webcam must be non-negative")
    if args.producer_fps is not None and (
        args.producer_fps <= 0 or not np.isfinite(args.producer_fps)
    ):
        raise ValueError("--producer-fps must be finite and greater than zero")
    if args.content_width <= 0 or args.content_height <= 0:
        raise ValueError("content dimensions must be greater than zero")

    source = LatestFrameSource()
    stop = threading.Event()
    ready = threading.Event()
    producer_errors: list[BaseException] = []
    if args.content_video:
        producer = threading.Thread(
            target=_produce_video,
            args=(
                source,
                stop,
                args.content_video,
                args.producer_fps,
                ready,
                producer_errors,
            ),
            name="frame2frame-live-video",
        )
    else:
        producer = threading.Thread(
            target=_produce,
            args=(
                source,
                stop,
                60.0 if args.producer_fps is None else float(args.producer_fps),
                (args.content_width, args.content_height),
            ),
            name="frame2frame-live-content",
        )
    producer.start()
    if args.content_video:
        if not ready.wait(timeout=5.0):
            stop.set()
            producer.join(timeout=2.0)
            raise RuntimeError("content video producer did not become ready")
        if producer_errors:
            producer.join(timeout=2.0)
            raise RuntimeError("content video producer failed") from producer_errors[0]
    try:
        summary = run(
            PipelineConfig(
                webcam=args.webcam,
                output=args.output or None,
                plot_path=None,
                display=True,
                filter=FilterConfig(kind="kalman", smooth_translation=False),
                screen=ScreenConfig(
                    distance_mul=4.0,
                    width_mul=4.0,
                    height_mul=2.25,
                    alpha=0.9,
                    content_fit="contain",
                ),
            ),
            content_source=source,
        )
    finally:
        stop.set()
        producer.join(timeout=2.0)
        if producer.is_alive():
            raise RuntimeError("live content producer did not stop")
    if producer_errors:
        raise RuntimeError("content video producer failed") from producer_errors[0]

    print(
        f"content sample {summary.mean_content_ms:.3f} ms; "
        f"screen render {summary.mean_render_ms:.3f} ms; "
        f"{summary.content_samples} content samples; "
        f"{summary.render_samples} render attempts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
