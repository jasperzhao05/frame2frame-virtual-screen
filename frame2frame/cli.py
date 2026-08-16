"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging

from . import __version__
from ._doctor import run_doctor
from .config import FilterConfig, PipelineConfig, ScreenConfig
from .pipeline import run
from .pose import available_backends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frame2frame",
        description="Render a head-locked virtual screen from a video or webcam.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", help="input video path")
    source.add_argument("--webcam", type=int, metavar="INDEX", help="webcam device index")
    source.add_argument(
        "--doctor",
        action="store_true",
        help="check runtime readiness without downloading models or opening a source",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="output/processed.mp4",
        help="output video path; pass an empty value to skip writing (default: %(default)s)",
    )
    parser.add_argument(
        "--preserve-audio",
        action="store_true",
        help="copy source audio into the output with ffmpeg (file inputs only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show a Python traceback instead of a concise runtime error",
    )
    parser.add_argument(
        "--backend",
        default="mediapipe",
        choices=available_backends(),
        help="pose backend (default: %(default)s)",
    )
    parser.add_argument(
        "--filter",
        default="fir",
        choices=["fir", "oneeuro", "none"],
        help="temporal smoothing (default: %(default)s)",
    )

    parser.add_argument("--texture", help="image to show on the virtual screen")
    parser.add_argument(
        "--screen-distance",
        type=float,
        default=5.0,
        help="screen distance as a multiple of the configured depth scale",
    )
    parser.add_argument(
        "--screen-width", type=float, default=4.0, help="screen width in face-size units"
    )
    parser.add_argument(
        "--screen-height", type=float, default=2.0, help="screen height in face-size units"
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        help="calibrated camera focal length in pixels (default: max frame dimension)",
    )
    parser.add_argument("--cutoff", type=float, default=2.5, help="FIR low-pass cutoff in Hz")
    parser.add_argument(
        "--no-smooth-translation",
        action="store_true",
        help="smooth only the angles, not the face centre/size",
    )
    parser.add_argument(
        "--no-delay-compensation",
        action="store_true",
        help="composite onto current frames even with the FIR's group delay",
    )
    parser.add_argument(
        "--dropout-hold",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="hold the last screen across brief detection misses (default: %(default)s)",
    )
    parser.add_argument(
        "--dropout-reset",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="reset tracking after a sustained miss (default: %(default)s)",
    )

    parser.add_argument("--no-screen", action="store_true", help="skip the virtual screen")
    parser.add_argument("--axis", action="store_true", help="draw the pose axis overlay")
    parser.add_argument("--bbox", action="store_true", help="draw the face box")
    parser.add_argument("--display", action="store_true", help="show a live preview window")
    parser.add_argument(
        "--plot",
        default="output/angle_processed.png",
        help="where to save the angle plot (empty to skip)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input=args.input,
        webcam=args.webcam,
        output=args.output or None,
        backend=args.backend,
        filter=FilterConfig(
            kind=args.filter,
            cutoff_hz=args.cutoff,
            smooth_translation=not args.no_smooth_translation,
        ),
        screen=ScreenConfig(
            distance_mul=args.screen_distance,
            width_mul=args.screen_width,
            height_mul=args.screen_height,
            focal_length=args.focal_length_px,
            texture_path=args.texture,
        ),
        compensate_delay=not args.no_delay_compensation,
        dropout_hold_seconds=args.dropout_hold,
        dropout_reset_seconds=args.dropout_reset,
        draw_screen=not args.no_screen,
        draw_axis=args.axis,
        draw_bbox=args.bbox,
        display=args.display,
        plot_path=args.plot or None,
        preserve_audio=args.preserve_audio,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.doctor:
        return run_doctor()
    try:
        summary = run(config_from_args(args))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if args.debug:
            raise
        parser.exit(2, f"{parser.prog}: error: {error}\n")
    print(
        f"frames: {summary.frames}  faces: {summary.faces}  "
        f"fps: {summary.fps:.1f}  inference: {summary.mean_inference_ms:.1f} ms/frame"
    )
    if summary.output:
        print(f"saved: {summary.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
