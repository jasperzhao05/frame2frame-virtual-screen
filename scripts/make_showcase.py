"""Build the deterministic five-policy comparison used in the README.

The visual is fully project-authored. Every panel shares one clean motion,
one seeded noisy observation stream, one carrier, and one distance-four
renderer. Only the causal attitude policy changes; FIR is not delay-aligned.

    python -m scripts.make_showcase --out docs/demo-comparison.gif
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from frame2frame import render
from frame2frame.config import ScreenConfig
from frame2frame.pose.base import FaceObservation, HeadPose
from frame2frame.video import VideoWriter
from scripts._filter_comparison import (
    POLICIES,
    create_comparison_filter,
    showcase_trace,
)

_FRAMES = 120
_FPS = 30.0
_GIF_FPS = 15
_SEED = 20260730
_SOURCE_SIZE = (480, 320)
_PANEL_SIZE = (300, 200)
_OUTPUT_SIZE = (900, 400)
_MAX_BYTES = 3_000_000
_PREWARM_CYCLES = 20
_LABEL_HEIGHT = 27


def _background(width: int, height: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (29 + 38 * yy).astype(np.uint8)
    image[..., 1] = (26 + 25 * yy).astype(np.uint8)
    image[..., 2] = (24 + 18 * yy).astype(np.uint8)
    return image


def _source_frame(
    frame_index: int,
    clean_angles: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float], float]:
    width, height = _SOURCE_SIZE
    phase = frame_index / _FRAMES
    center = (
        width * (0.5 + 0.11 * np.sin(2.0 * np.pi * phase)),
        height * (0.52 + 0.045 * np.sin(4.0 * np.pi * phase)),
    )
    radius = min(width, height) * 0.12
    image = _background(width, height)
    center_px = tuple(int(round(value)) for value in center)
    radius_px = int(round(radius))

    cv2.circle(image, center_px, radius_px, (192, 176, 158), -1, cv2.LINE_AA)
    cv2.circle(image, center_px, radius_px, (76, 82, 88), 2, cv2.LINE_AA)

    yaw = float(clean_angles[0])
    profile = float(np.clip(-yaw / 24.0, -1.0, 1.0))
    tip = (
        int(round(center[0] + radius * 1.12 * profile)),
        int(round(center[1] - radius * 0.08)),
    )
    cv2.line(image, center_px, tip, (62, 68, 74), 3, cv2.LINE_AA)
    cv2.circle(image, tip, 4, (62, 68, 74), -1, cv2.LINE_AA)
    return image, center, radius


def _render_screen(
    source: np.ndarray,
    angles: np.ndarray,
    center: tuple[float, float],
    size: float,
    config: ScreenConfig,
    texture: np.ndarray,
) -> np.ndarray:
    observation = FaceObservation(
        pose=HeadPose(*angles),
        center=center,
        size=size,
        bbox=(
            center[0] - size,
            center[1] - size,
            center[0] + size,
            center[1] + size,
        ),
    )
    return render.draw_virtual_screen(source.copy(), observation, config, texture)


def _panel(frame: np.ndarray, label: str) -> np.ndarray:
    width, height = _PANEL_SIZE
    output = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(output, (0, 0), (width, _LABEL_HEIGHT), (18, 20, 22), -1)
    cv2.putText(
        output,
        label,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (242, 244, 246),
        1,
        cv2.LINE_AA,
    )
    return output


def _showcase_traces() -> dict[str, np.ndarray]:
    clean, noisy = showcase_trace(_FRAMES, _SEED)
    filters = {policy.key: create_comparison_filter(policy.key, _FPS) for policy in POLICIES}
    for _ in range(_PREWARM_CYCLES):
        for noisy_angles in noisy:
            for filt in filters.values():
                filt.update(*noisy_angles, dt=1.0 / _FPS)

    outputs = {
        key: np.asarray(
            [filt.update(*angles, dt=1.0 / _FPS) for angles in noisy],
            dtype=np.float64,
        )
        for key, filt in filters.items()
    }
    return {"clean": clean, **outputs}


def _screen_config() -> ScreenConfig:
    return ScreenConfig(
        distance_mul=4.0,
        alpha=0.84,
        border_color=(0, 220, 255),
        border_thickness=2,
    )


def _comparison_frame(
    frame_index: int,
    traces: dict[str, np.ndarray],
    *,
    config: ScreenConfig | None = None,
    texture: np.ndarray | None = None,
) -> np.ndarray:
    clean_angles = traces["clean"][frame_index]
    labels = {policy.key: policy.label for policy in POLICIES}
    active_config = config or _screen_config()
    active_texture = render.default_texture() if texture is None else texture
    source, center, size = _source_frame(frame_index, clean_angles)
    panels = {
        "clean": _panel(
            _render_screen(
                source,
                clean_angles,
                center,
                size,
                active_config,
                active_texture,
            ),
            "CLEAN REFERENCE",
        ),
        **{
            key: _panel(
                _render_screen(
                    source,
                    traces[key][frame_index],
                    center,
                    size,
                    active_config,
                    active_texture,
                ),
                labels[key],
            )
            for key in labels
        },
    }
    top = np.concatenate((panels["clean"], panels["none"], panels["ema"]), axis=1)
    bottom = np.concatenate((panels["fir"], panels["oneeuro"], panels["kalman"]), axis=1)
    frame = np.concatenate((top, bottom), axis=0)
    if frame.shape[1::-1] != _OUTPUT_SIZE:
        raise RuntimeError(f"unexpected comparison frame size: {frame.shape[1::-1]}")
    return frame


def _comparison_frames() -> Iterator[np.ndarray]:
    traces = _showcase_traces()
    config = _screen_config()
    texture = render.default_texture()
    for frame_index in range(_FRAMES):
        yield _comparison_frame(frame_index, traces, config=config, texture=texture)


def _encode_gif(ffmpeg: str, source: Path, output: Path) -> None:
    filters = (
        f"fps={_GIF_FPS},split[palette_source][gif_source];"
        "[palette_source]palettegen=max_colors=144:stats_mode=diff[palette];"
        "[gif_source][palette]paletteuse=dither=bayer:bayer_scale=3:"
        "diff_mode=rectangle"
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filters,
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the five-policy README comparison GIF.")
    parser.add_argument("--out", default="docs/demo-comparison.gif")
    args = parser.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required to build the showcase GIF")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frame2frame-showcase-") as directory:
        work = Path(directory)
        source = work / "comparison.mp4"
        with VideoWriter(source, _FPS, _OUTPUT_SIZE) as writer:
            for frame in _comparison_frames():
                writer.write(frame)
        encoded = work / "demo-comparison.gif"
        _encode_gif(ffmpeg, source, encoded)
        size = encoded.stat().st_size
        if size > _MAX_BYTES:
            raise RuntimeError(f"showcase GIF is {size:,} bytes; limit is {_MAX_BYTES:,}")
        encoded.replace(output)

    print(
        f"wrote {output} ({_OUTPUT_SIZE[0]}x{_OUTPUT_SIZE[1]}, "
        f"{_FRAMES / _FPS:.1f}s, {_GIF_FPS} fps, {size:,} bytes, seed={_SEED})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
