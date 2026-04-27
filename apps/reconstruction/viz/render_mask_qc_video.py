"""Render a QC video showing SAM3 mask overlays for every (frame, cam) pair.

For each frame F, composes a grid of all camera views (each showing the
cow overlaid in green by its mask), saves the grid as a JPG. After all
frames are rendered, run ffmpeg externally to encode the JPGs into a
30-fps MP4 < 100 MB.

This script is split into a render step (per-chunk, parallelisable across
SLURM array tasks) and a final encode step. Both phases live here, but
typical use is::

    # phase 1 (per-chunk): render frame JPGs for a sub-range
    python -m apps.reconstruction.viz.render_mask_qc_video <data_root> \
        --frames 0:144 --output_dir /scratch/<...>/qc_frames

    # phase 2 (after all chunks done): encode to MP4
    python -m apps.reconstruction.viz.render_mask_qc_video <data_root> \
        --encode_only --output_dir /scratch/<...>/qc_frames \
        --output_video /scratch/<...>/mask_qc.mp4

Layout choices: 11 cams in 4 cols × 3 rows of 480×270 → 1920×810 final.
At 30 fps for 1434 frames (~48 s) with CRF 28, output is ~40–60 MB.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _parse_frames(s: str) -> list[int]:
    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 2:
            return list(range(parts[0], parts[1]))
        return list(range(parts[0], parts[1], parts[2]))
    return [int(p) for p in s.split(",") if p.strip()]


def _compose_overlay(img_bgr: np.ndarray, mask_u8: np.ndarray,
                     color: tuple = (0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    out = img_bgr.copy()
    m = mask_u8 > 0
    if m.any():
        out[m] = ((1 - alpha) * out[m] +
                  alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


def _render_grid(data_root: Path, frame: int, cam_names: list[str],
                 tile_w: int, tile_h: int, cols: int) -> np.ndarray:
    tiles = []
    for cam in cam_names:
        img_p = data_root / "images" / cam / f"{frame:06d}.jpg"
        msk_p = data_root / "masks"  / cam / f"{frame:06d}.png"
        img = cv2.imread(str(img_p))
        if img is None:
            tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            continue
        msk = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.zeros(img.shape[:2], dtype=np.uint8)
        elif msk.shape[:2] != img.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        thumb = cv2.resize(_compose_overlay(img, msk), (tile_w, tile_h),
                           interpolation=cv2.INTER_AREA)
        # Cam label (black stroke under white text)
        for color, thick in [((0, 0, 0), 4), ((255, 255, 255), 1)]:
            cv2.putText(thumb, f"cam {cam}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick, cv2.LINE_AA)
        tiles.append(thumb)

    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:                       # pad to full grid
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                      for r in range(rows)])

    # Frame label at bottom
    label = f"frame {frame:06d}"
    for color, thick in [((0, 0, 0), 6), ((255, 255, 255), 2)]:
        cv2.putText(grid, label, (12, grid.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, thick, cv2.LINE_AA)
    return grid


def _render_phase(data_root: Path, frames: list[int], cam_names: list[str],
                  out_dir: Path, tile_w: int, tile_h: int, cols: int,
                  jpeg_quality: int) -> int:
    written = 0
    for f in frames:
        out_path = out_dir / f"{f:06d}.jpg"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        grid = _render_grid(data_root, f, cam_names, tile_w, tile_h, cols)
        cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        written += 1
        if written % 25 == 0:
            print(f"[render] wrote {written}/{len(frames)}", flush=True)
    return written


def _encode_phase(out_dir: Path, output_video: Path, fps: int, crf: int,
                  max_mb: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("[encode] ffmpeg not on PATH — module load ffmpeg/7.1.1")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    # Use -start_number to handle non-zero starting frame
    jpgs = sorted(out_dir.glob("*.jpg"))
    if not jpgs:
        raise SystemExit(f"[encode] no JPGs in {out_dir}")
    start = int(jpgs[0].stem)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", str(start),
        "-i", str(out_dir / "%06d.jpg"),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_video),
    ]
    print(f"[encode] $ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)
    size_mb = output_video.stat().st_size / 1e6
    print(f"[encode] {output_video} = {size_mb:.1f} MB"
          + (" — over budget!" if size_mb > max_mb else " — under budget"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render mask-QC video: green-overlay grid per frame")
    p.add_argument("data_root", type=str)
    p.add_argument("--frames", type=str, default=None,
                   help="frame range 'START:END[:STEP]' or 'N1,N2,...'. "
                        "Required for the render phase.")
    p.add_argument("--output_dir", required=True,
                   help="where to read/write per-frame grid JPGs")
    p.add_argument("--output_video", default=None,
                   help="if set + --encode_only, write MP4 here")
    p.add_argument("--encode_only", action="store_true",
                   help="skip rendering; just ffmpeg the existing JPGs")
    p.add_argument("--tile_w", type=int, default=480)
    p.add_argument("--tile_h", type=int, default=270)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=28,
                   help="x264 quality knob (higher=smaller; 28 ~ 40-60 MB "
                        "for 1434 frames at 1920x810)")
    p.add_argument("--max_mb", type=float, default=100.0,
                   help="warn if output video exceeds this size")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.encode_only:
        if args.frames is None:
            raise SystemExit("[render] --frames required (use --encode_only "
                             "to skip render)")
        cam_names = sorted(d.name for d in (data_root / "images").iterdir()
                           if d.is_dir())
        frames = _parse_frames(args.frames)
        print(f"[render] {len(frames)} frames, {len(cam_names)} cams "
              f"({args.cols} cols × {(len(cam_names) + args.cols - 1) // args.cols} rows "
              f"of {args.tile_w}×{args.tile_h}) → "
              f"{args.cols * args.tile_w}×{((len(cam_names)+args.cols-1)//args.cols)*args.tile_h}")
        n = _render_phase(data_root, frames, cam_names, out_dir,
                          args.tile_w, args.tile_h, args.cols, args.jpeg_quality)
        print(f"[render] done, wrote {n} new frames; total in dir: "
              f"{len(list(out_dir.glob('*.jpg')))}")

    if args.output_video:
        _encode_phase(out_dir, Path(args.output_video),
                      args.fps, args.crf, args.max_mb)


if __name__ == "__main__":
    main()
