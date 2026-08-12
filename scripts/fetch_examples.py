"""Download attributed public head-pose clips and optionally process them.

Source: Intel IoT DevKit sample-videos, licensed CC BY 4.0. Every payload is
verified against a pinned SHA-256 digest and remains in Git-ignored directories.
Processing needs a pose backend; downloading alone does not.

    python scripts/fetch_examples.py                  # default mediapipe backend
    python scripts/fetch_examples.py --backend hopenet
    python scripts/fetch_examples.py --limit 1        # just the first clip
    python scripts/fetch_examples.py --download-only
    python scripts/fetch_examples.py --list
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from frame2frame._downloads import ensure_download
from frame2frame.config import PipelineConfig, ScreenConfig
from frame2frame.pipeline import run
from frame2frame.pose import available_backends

REPOSITORY = "https://github.com/intel-iot-devkit/sample-videos"
BASE = f"{REPOSITORY}/raw/master"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"


class Clip(NamedTuple):
    sha256: str
    size_bytes: int


CLIPS: dict[str, Clip] = {
    "head-pose-face-detection-female.mp4": Clip(
        "e9290821ac0e0a186e8f5cae5e3b56e8062921642d46a5f65f1ca5b311811fc5",
        15_628_037,
    ),
    "head-pose-face-detection-male.mp4": Clip(
        "994ca625f091d1422b93d97a4bd67c4de1e5ed7064c9079a68cd7526c96bfd68",
        15_522_596,
    ),
    "head-pose-face-detection-female-and-male.mp4": Clip(
        "650166430c4bf9ddc470ac17a86d1fcbd6d76c64e60ed73675fdc6b3e3d3af38",
        16_788_193,
    ),
    "face-demographics-walking.mp4": Clip(
        "91af68da819a9c0caab06c6c21414e6ebdb378ff28fee90e1f937165bc1007c6",
        6_406_124,
    ),
    "face-demographics-walking-and-pause.mp4": Clip(
        "d88ab9aa03634f66f8815db3dc940e1cdd80b098440effb20882e814fd206bf5",
        9_406_029,
    ),
}


def download(name: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    clip = CLIPS[name]
    return ensure_download(
        f"{BASE}/{name}",
        dest / name,
        sha256=clip.sha256,
        expected_size=clip.size_bytes,
    )


def write_source_note(dest: Path, names: Sequence[str]) -> None:
    """Leave attribution beside downloads so it follows copied local assets."""
    lines = [
        "# Source and attribution",
        "",
        "These files were downloaded by frame2frame for local evaluation.",
        "",
        f"- Source: {REPOSITORY}",
        f"- License: Creative Commons Attribution 4.0 International ({LICENSE})",
        "- Attribution: Intel IoT DevKit `sample-videos` contributors",
        (
            "- Changes in processed outputs: virtual-screen/diagnostic overlays "
            "and video re-encoding"
        ),
        "",
        "## Verified files",
        "",
        "| File | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| `{name}` | {CLIPS[name].size_bytes} | `{CLIPS[name].sha256}` |" for name in names
    )
    lines.append("")
    (dest / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and process example clips.")
    parser.add_argument("--backend", default="mediapipe", choices=available_backends())
    parser.add_argument("--indir", default="examples/inputs")
    parser.add_argument("--outdir", default="examples/outputs")
    parser.add_argument("--limit", type=int, default=len(CLIPS))
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="verify/download inputs without loading a pose backend",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list attributed files and pinned digests without downloading",
    )
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit cannot be negative")

    names = list(CLIPS)[: args.limit]
    if args.list:
        for name in names:
            clip = CLIPS[name]
            print(f"{clip.sha256}  {clip.size_bytes:>8} bytes  {name}")
        return 0

    indir, outdir = Path(args.indir), Path(args.outdir)
    for name in names:
        src = download(name, indir)
        print(f"verified {src}")
        if args.download_only:
            continue
        stem = Path(name).stem
        config = PipelineConfig(
            input=str(src),
            output=str(outdir / f"{stem}.mp4"),
            backend=args.backend,
            screen=ScreenConfig(border_thickness=2),
            plot_path=str(outdir / f"{stem}_angles.png"),
        )
        summary = run(config)
        print(
            f"  {stem}: {summary.frames} frames, {summary.faces} with a face, "
            f"{summary.mean_inference_ms:.1f} ms/frame -> {summary.output}"
        )
    if names:
        write_source_note(indir, names)
        print(f"attribution: {indir / 'SOURCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
