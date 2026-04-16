"""
Build a single preview video: all camera folders under a session ``raw/`` root are
tiled into one grid per frame, with the frame id drawn at the top-left, then encoded
to one MP4. Each view is downscaled before tiling to keep memory and encoder limits sane.

Typical layout (11 cameras): ``--rows 3 --cols 4`` (12 slots, one empty pad cell).

Example::

    python scripts/postprocess/multiview_grid_video.py /path/to/cow_1_board/raw \\
        -o /path/to/cow_1_board/multiview_11cams.mp4 --fps 30 --max-cell-side 720
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Import grid helpers from sibling module (same directory as this file).
_POST_DIR = Path(__file__).resolve().parent
if str(_POST_DIR) not in sys.path:
    sys.path.insert(0, str(_POST_DIR))

from concat_images_grid import build_grid, parse_pad_bgr  # noqa: E402

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Match img2vid.py: keep frames within typical H.264 / OpenCV limits.
MAX_OUTPUT_DIM = 4096


def natural_key_stem(stem: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", stem)]


def find_camera_image_dirs(raw_root: str) -> list[str]:
    """
    ``raw_root/01/images``, ``02/images``, ... sorted by folder name.
    """
    root = os.path.abspath(raw_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Not a directory: {root}")

    dirs: list[str] = []
    for name in sorted(os.listdir(root)):
        if re.fullmatch(r"\d{2}", name):
            img_dir = os.path.join(root, name, "images")
            if os.path.isdir(img_dir):
                dirs.append(img_dir)

    if not dirs:
        raise FileNotFoundError(
            f"No .../NN/images/ directories found under {root} (expected e.g. 01/images, 02/images, …)"
        )
    return dirs


def stems_in_dir(img_dir: str) -> set[str]:
    out: set[str] = set()
    for f in os.listdir(img_dir):
        if f.lower().endswith(ALLOWED_EXTENSIONS):
            out.add(os.path.splitext(f)[0])
    return out


def resolve_frame_path(img_dir: str, stem: str) -> str | None:
    for ext in ALLOWED_EXTENSIONS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def common_frame_stems_sorted(img_dirs: list[str]) -> list[str]:
    sets = [stems_in_dir(d) for d in img_dirs]
    if not sets:
        return []
    common = set.intersection(*sets)
    return sorted(common, key=natural_key_stem)


def infer_rows_cols(
    n_views: int, rows: int | None, cols: int | None
) -> tuple[int, int]:
    """
    Choose a rows×cols grid that fits ``n_views`` panels. Prefer a slightly wide grid
    (more columns than a square root) so phone footage reads left-to-right, top-to-bottom.
    """
    if n_views < 1:
        raise ValueError("Need at least one camera images/ directory.")

    if rows is not None and cols is not None:
        if rows * cols < n_views:
            raise ValueError(
                f"Grid too small: rows×cols={rows}×{cols}={rows*cols} < n_views={n_views}"
            )
        return rows, cols

    if rows is not None:
        cols = int(math.ceil(n_views / rows))
        return rows, cols

    if cols is not None:
        rows = int(math.ceil(n_views / cols))
        return rows, cols

    c = max(1, int(math.ceil(math.sqrt(n_views * 1.15))))
    r = int(math.ceil(n_views / c))
    return r, c


def _compute_output_size(width: int, height: int, max_dim: int) -> tuple[int, int]:
    if width <= max_dim and height <= max_dim:
        return width, height
    scale = min(max_dim / width, max_dim / height)
    out_w = int(round(width * scale))
    out_h = int(round(height * scale))
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)
    return max(2, out_w), max(2, out_h)


def put_label_top_left(
    image_bgr: np.ndarray,
    text: str,
    font_scale: float = 1.0,
    thickness: int = 2,
    margin: int = 12,
) -> None:
    """In-place: dark outline + bright text for readability on varied frames."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    outline = thickness + 2
    org = (margin, margin + int(28 * font_scale))
    cv2.putText(image_bgr, text, org, font, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Concatenate all cameras under a raw session into a grid per frame, "
            "label frame id, and write one MP4."
        )
    )
    p.add_argument(
        "raw_root",
        type=str,
        help="Session root containing 01/images, 02/images, … (e.g. .../cow_1_board/raw).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output MP4 path (default: <raw_root>/multiview_grid_<n>cams.mp4).",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Output FPS (default: 30).")
    p.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Grid rows (optional; if omitted with --cols, rows are inferred).",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Grid columns (optional; if omitted with --rows, cols are inferred).",
    )
    p.add_argument(
        "--max-cell-side",
        type=int,
        default=720,
        help=(
            "Each view is letterboxed into a square cell of this side length (pixels) "
            "before tiling. Lower = smaller memory / faster (default: 720)."
        ),
    )
    p.add_argument(
        "--max-output-dim",
        type=int,
        default=MAX_OUTPUT_DIM,
        help=f"Max width or height of final encoded frame (default: {MAX_OUTPUT_DIM}).",
    )
    p.add_argument(
        "--pad-color",
        type=str,
        default="black",
        help="Pad color for letterboxing / empty grid slots (see concat_images_grid).",
    )
    p.add_argument(
        "--label-prefix",
        type=str,
        default="frame ",
        help="Text before the frame stem in the top-left label (default: 'frame ').",
    )
    p.add_argument(
        "--font-scale",
        type=float,
        default=1.1,
        help="OpenCV font scale for the top-left frame label.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=None,
        help="0-based index into the sorted common frame list (inclusive).",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="0-based end index into the sorted common frame list (exclusive).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = os.path.abspath(args.raw_root)
    pad_bgr = parse_pad_bgr(args.pad_color)

    img_dirs = find_camera_image_dirs(raw_root)
    n_cams = len(img_dirs)
    stems = common_frame_stems_sorted(img_dirs)
    if not stems:
        raise SystemExit(
            f"[multiview_grid_video] No common frame files across all {n_cams} cameras under {raw_root}"
        )

    if args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end if args.end is not None else len(stems)
        stems = stems[s:e]

    rows, cols = infer_rows_cols(n_cams, args.rows, args.cols)
    cell_side = max(32, int(args.max_cell_side))

    out_path = args.output
    if not out_path:
        slug = os.path.basename(raw_root.rstrip(os.sep)) or "session"
        out_path = os.path.join(raw_root, f"multiview_grid_{n_cams}cams.mp4")
    else:
        out_path = os.path.abspath(out_path)
        od = os.path.dirname(out_path)
        if od:
            os.makedirs(od, exist_ok=True)

    # First frame: determine video size after optional final downscale.
    first_paths: list[str] = []
    for d in img_dirs:
        p0 = resolve_frame_path(d, stems[0])
        if p0 is None:
            raise FileNotFoundError(f"Missing frame {stems[0]!r} under {d}")
        first_paths.append(p0)

    grid0 = build_grid(
        first_paths,
        cols=cols,
        rows=rows,
        pad_bgr=pad_bgr,
        cell_w=cell_side,
        cell_h=cell_side,
    )
    put_label_top_left(
        grid0,
        f"{args.label_prefix}{stems[0]}",
        font_scale=args.font_scale,
    )
    oh, ow = grid0.shape[:2]
    out_w, out_h = _compute_output_size(ow, oh, args.max_output_dim)
    if (out_w, out_h) != (ow, oh):
        print(
            f"[multiview_grid_video] Final resize for encoder: {ow}x{oh} -> {out_w}x{out_h} "
            f"(max_output_dim={args.max_output_dim})"
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    def write_frame(grid: np.ndarray) -> None:
        if grid.shape[1] != out_w or grid.shape[0] != out_h:
            grid = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(grid)

    write_frame(grid0)

    for stem in tqdm(stems[1:], desc="frames", unit="fr"):
        paths = []
        for d in img_dirs:
            pp = resolve_frame_path(d, stem)
            if pp is None:
                raise FileNotFoundError(f"Missing frame {stem!r} under {d}")
            paths.append(pp)
        grid = build_grid(
            paths,
            cols=cols,
            rows=rows,
            pad_bgr=pad_bgr,
            cell_w=cell_side,
            cell_h=cell_side,
        )
        put_label_top_left(
            grid,
            f"{args.label_prefix}{stem}",
            font_scale=args.font_scale,
        )
        write_frame(grid)

    writer.release()
    print(
        f"[multiview_grid_video] Wrote {out_path}  cameras={n_cams}  grid={rows}x{cols}  "
        f"frames={len(stems)}  fps={args.fps}  cell={cell_side}px"
    )


if __name__ == "__main__":
    main()
