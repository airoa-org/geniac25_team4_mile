#!/usr/bin/env python3
"""
Convert OXE Fractal dataset videos from AV1 to H.264 into a mirrored folder structure.

Input layout (root):
  videos/chunk-XYZ/observation.images.image/episode_XXXXXXXX.mp4   # AV1 (source)

Output layout (root):
  videos_h264/chunk-XYZ/observation.images.image/episode_XXXXXXXX.mp4  # H.264 (target)

Usage:
  python tools/convert_oxe_av1_to_h264.py --data_root /path/to/fractal20220817_data_lerobot \
      --crf 23 --preset veryfast --workers 4

Notes:
  - Requires ffmpeg in PATH
  - Skips files that already exist unless --overwrite is set
"""

import argparse
import concurrent.futures
import subprocess
from pathlib import Path


def convert_one(src: Path, dst: Path, crf: int, preset: str, overwrite: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return True
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-v", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--camera_dir", default="observation.images.image", type=str,
                    help="camera subdir under videos/")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", type=str, default="veryfast")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    src_root = root / "videos"
    dst_root = root / "videos_h264"

    sources = list(src_root.glob(f"chunk-*/{args.camera_dir}/episode_*.mp4"))
    print(f"Found {len(sources)} source videos under {src_root}")

    def task(srcp: Path):
        rel = srcp.relative_to(src_root)
        dstp = dst_root / rel
        ok = convert_one(srcp, dstp, args.crf, args.preset, args.overwrite)
        print(("✅", dstp) if ok else ("❌", dstp))
        return ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(task, sources))

    print(f"Done. Success: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()


