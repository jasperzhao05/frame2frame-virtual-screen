"""Create a fully synthetic, reproducible pipeline integration demo.

The generated input, observations, and virtual-screen texture are all produced
locally. No real face, model weight, download, or network access is involved.
Seeded detector noise makes the stabilisation visible in the angle plot.

    python -m scripts.make_demo --out output/demo.mp4
    python -m scripts.make_demo --filter oneeuro --seed 7
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from frame2frame.config import FilterConfig, PipelineConfig, ScreenConfig
from frame2frame.pipeline import run
from frame2frame.pose.base import FaceObservation, HeadPose
from frame2frame.pose.scripted import ScriptedEstimator
from frame2frame.video import VideoWriter

_HORIZONTAL_EXCURSION = 0.18


def _background(width: int, height: int) -> np.ndarray:
    yy = np.linspace(0, 1, height)[:, None]
    img = np.empty((height, width, 3), np.uint8)
    img[..., 0] = (30 + 40 * yy).astype(np.uint8)
    img[..., 1] = (25 + 25 * yy).astype(np.uint8)
    img[..., 2] = (20 + 20 * yy).astype(np.uint8)
    return img


def synthetic_clip(
    path: str | Path,
    frames: int = 120,
    size: tuple[int, int] = (640, 480),
    fps: float = 30.0,
) -> str | Path:
    width, height = size
    radius = min(width, height) // 8
    with VideoWriter(path, fps, size) as writer:
        for frame_index in range(frames):
            image = _background(width, height)
            phase = frame_index / frames
            horizontal_motion = np.sin(2 * np.pi * phase)
            cx = int(width * (0.5 + _HORIZONTAL_EXCURSION * horizontal_motion))
            cy = int(height * 0.5 + height * 0.08 * np.sin(4 * np.pi * phase))
            cv2.circle(image, (cx, cy), radius, (200, 180, 160), -1)
            cv2.circle(image, (cx, cy), radius, (60, 60, 60), 2)
            # A small profile marker makes the scripted look direction visible.
            # Positive image-x is right; the pose contract below therefore uses
            # negative yaw for the same motion.
            profile_tip = (cx + int(radius * 1.15 * horizontal_motion), cy)
            cv2.line(image, (cx, cy), profile_tip, (60, 60, 60), 3)
            cv2.circle(image, profile_tip, max(2, radius // 12), (60, 60, 60), -1)
            writer.write(image)
    return path


def _observation_source(
    frames: int,
    size: tuple[int, int],
    seed: int = 20260730,
    noise_deg: float = 1.5,
    noise_px: float = 2.0,
) -> Callable[[int, np.ndarray], FaceObservation]:
    width, height = size
    radius = min(width, height) // 8
    rng = np.random.default_rng(seed)
    angle_noise = rng.normal(0.0, noise_deg, size=(frames, 3))
    position_noise = rng.normal(0.0, noise_px, size=(frames, 3))

    def observation_at(frame_index: int, _frame: np.ndarray) -> FaceObservation:
        phase = frame_index / frames
        cx = width * (0.5 + _HORIZONTAL_EXCURSION * np.sin(2 * np.pi * phase))
        cy = height * 0.5 + height * 0.08 * np.sin(4 * np.pi * phase)
        clean_angles = np.array(
            [
                # Positive yaw looks image-left, so negate the shared motion
                # signal to make the face look toward its lateral excursion.
                -30 * np.sin(2 * np.pi * phase),
                12 * np.sin(4 * np.pi * phase),
                8 * np.sin(2 * np.pi * phase + 1.0),
            ]
        )
        yaw, pitch, roll = clean_angles + angle_noise[frame_index]
        noisy_cx, noisy_cy = np.array([cx, cy]) + position_noise[frame_index, :2]
        noisy_radius = max(1.0, radius + position_noise[frame_index, 2])
        pose = HeadPose(yaw=float(yaw), pitch=float(pitch), roll=float(roll))
        bbox = (
            int(noisy_cx - noisy_radius),
            int(noisy_cy - noisy_radius),
            int(noisy_cx + noisy_radius),
            int(noisy_cy + noisy_radius),
        )
        return FaceObservation(
            pose,
            (float(noisy_cx), float(noisy_cy)),
            float(noisy_radius),
            bbox,
        )

    return observation_at


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic frame2frame demo.")
    parser.add_argument("--input", default="output/demo_input.mp4")
    parser.add_argument("--out", default="output/demo.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--filter", choices=("fir", "oneeuro", "none"), default="fir")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--no-axis",
        action="store_true",
        help="omit the diagnostic pose axis from rendered frames",
    )
    parser.add_argument(
        "--noise-deg",
        type=float,
        default=1.5,
        help="standard deviation of seeded pose noise (default: %(default)s)",
    )
    parser.add_argument(
        "--noise-px",
        type=float,
        default=2.0,
        help="standard deviation of seeded centre/size noise (default: %(default)s)",
    )
    parser.add_argument(
        "--plot",
        default="output/demo_angles.png",
        help="raw-vs-smoothed angle plot (empty to skip)",
    )
    args = parser.parse_args(argv)

    if args.frames < 1 or args.width < 64 or args.height < 64 or args.fps <= 0:
        parser.error("frames and fps must be positive; width and height must be at least 64")
    if args.noise_deg < 0 or args.noise_px < 0:
        parser.error("noise values cannot be negative")

    size = (args.width, args.height)
    synthetic_clip(args.input, args.frames, size, args.fps)

    config = PipelineConfig(
        input=args.input,
        output=args.out,
        filter=FilterConfig(kind=args.filter),
        screen=ScreenConfig(border_thickness=2),
        draw_axis=not args.no_axis,
        plot_path=args.plot or None,
    )
    estimator = ScriptedEstimator(
        _observation_source(
            args.frames,
            size,
            seed=args.seed,
            noise_deg=args.noise_deg,
            noise_px=args.noise_px,
        )
    )
    summary = run(config, estimator=estimator)
    print(
        f"wrote {summary.output} ({summary.frames} frames, filter={args.filter}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
