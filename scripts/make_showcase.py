"""Rebuild the deterministic synthetic comparison used at the top of README.

The two panes use the same generated clip, scripted pose stream, and random
seed. Only the temporal filter changes. Local ``ffmpeg`` is required to encode
the compact, browser-friendly GIF.

    python -m scripts.make_showcase --out docs/demo-comparison.gif
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from scripts.make_demo import main as make_demo

_FRAMES = 120
_WIDTH = 480
_HEIGHT = 320
_FPS = 30
_GIF_FPS = 15
_SEED = 20260730
_MAX_BYTES = 2_000_000


def _render_variant(work: Path, kind: str) -> Path:
    output = work / f"{kind}.mp4"
    result = make_demo(
        [
            "--input",
            str(work / "input.mp4"),
            "--out",
            str(output),
            "--plot",
            "",
            "--filter",
            kind,
            "--frames",
            str(_FRAMES),
            "--width",
            str(_WIDTH),
            "--height",
            str(_HEIGHT),
            "--fps",
            str(_FPS),
            "--seed",
            str(_SEED),
            "--no-axis",
        ]
    )
    if result != 0:
        raise RuntimeError(f"synthetic {kind} render failed with exit code {result}")
    return output


def _encode_gif(ffmpeg: str, raw: Path, fir: Path, output: Path) -> None:
    pane = (
        f"fps={_GIF_FPS},scale=360:-2:flags=lanczos,"
        "drawbox=x=0:y=0:w=iw:h=36:color=black@0.72:t=fill"
    )
    filters = (
        f"[0:v]{pane},drawtext=text='RAW POSE':fontcolor=white:fontsize=16:"
        "x=(w-text_w)/2:y=9[left];"
        f"[1:v]{pane},drawtext=text='DELAY-ALIGNED FIR':fontcolor=white:fontsize=16:"
        "x=(w-text_w)/2:y=9[right];"
        "[left][right]hstack=inputs=2[stack];"
        "[stack]split[for_palette][for_gif];"
        "[for_palette]palettegen=max_colors=128:stats_mode=diff[palette];"
        "[for_gif][palette]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle[out]"
    )
    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(raw.parent / "font-cache")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw),
            "-i",
            str(fir),
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-loop",
            "0",
            str(output),
        ],
        check=True,
        env=environment,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the README comparison GIF.")
    parser.add_argument("--out", default="docs/demo-comparison.gif")
    args = parser.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required to build the showcase GIF")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frame2frame-showcase-") as directory:
        work = Path(directory)
        raw = _render_variant(work, "none")
        fir = _render_variant(work, "fir")
        encoded = work / "demo-comparison.gif"
        _encode_gif(ffmpeg, raw, fir, encoded)
        size = encoded.stat().st_size
        if size > _MAX_BYTES:
            raise RuntimeError(f"showcase GIF is {size:,} bytes; limit is {_MAX_BYTES:,}")
        encoded.replace(output)

    print(
        f"wrote {output} (720x240, {_FRAMES / _FPS:.1f}s, {_GIF_FPS} fps, "
        f"{size:,} bytes, seed={_SEED})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
