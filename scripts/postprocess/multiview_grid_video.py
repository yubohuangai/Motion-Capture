"""
Tile every camera's ``images/`` into one grid per frame and encode a single MP4.

**Input layout** (session root is usually ``…/<dataset>/raw``)::

    raw/
      01/images/<files>.jpg
      02/images/<files>.jpg
      …

Camera folders are two-digit names ``01``, ``02``, … Each ``images/`` folder holds frames
for that view. This script does **not** read timestamps, CSVs, or sync outputs: it sorts
files in each ``images/`` (natural order on the filename stem) and pairs **by index** —
frame ``i`` of the video uses the ``i``-th sorted file from each camera. The number of
frames is ``min`` of per-camera counts (a warning is printed if counts differ).

Typical grid for 11 cameras: ``--rows 3 --cols 4`` (12 slots, one empty pad cell).

Example::

    python scripts/postprocess/multiview_grid_video.py /path/to/cow_1_board/raw \\
        -o /path/to/multiview_11cams.mp4 --fps 30 --max-cell-side 720

**Speed (typical bottlenecks: JPEG decode × 11 cams, then encode):**

- ``--read-scale 2`` (or ``4``): decode at 1/2 or 1/4 resolution in ``imread`` (much faster for 4K sources when cells are small).
- ``--encoder ffmpeg --ffmpeg-nvenc``: NVIDIA GPU H.264 encode (needs ``ffmpeg`` + driver); otherwise ``--encoder ffmpeg`` uses fast CPU x264.
- Lower ``--max-cell-side`` (e.g. 540) and/or ``--fps``.

OpenCV resize/letterboxing stays on **CPU**; GPU helps mainly via **NVENC** when you choose FFmpeg + NVENC.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_POST_DIR = Path(__file__).resolve().parent
if str(_POST_DIR) not in sys.path:
    sys.path.insert(0, str(_POST_DIR))

from concat_images_grid import build_grid, parse_pad_bgr  # noqa: E402

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

MAX_OUTPUT_DIM = 4096

# OpenCV: decode smaller than full resolution (big win for 4K → small cells).
_READ_SCALE_TO_IMREAD: dict[int, int] = {
    1: int(cv2.IMREAD_COLOR),
    2: int(cv2.IMREAD_REDUCED_COLOR_2),
    4: int(cv2.IMREAD_REDUCED_COLOR_4),
    8: int(cv2.IMREAD_REDUCED_COLOR_8),
}


def natural_key_stem(stem: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", stem)]


def find_camera_image_dirs(raw_root: str) -> list[str]:
    """``raw_root/01/images``, ``02/images``, … sorted by folder name."""
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


def sorted_stems_in_dir(img_dir: str) -> list[str]:
    stems: list[str] = []
    for f in os.listdir(img_dir):
        if f.lower().endswith(ALLOWED_EXTENSIONS):
            stems.append(os.path.splitext(f)[0])
    return sorted(stems, key=natural_key_stem)


def build_frame_plans_by_index(img_dirs: list[str]) -> list[list[str]]:
    """
    One row per output frame: [stem_cam0, stem_cam1, …] using the i-th sorted stem per camera.
    """
    per_cam = [sorted_stems_in_dir(d) for d in img_dirs]
    lengths = [len(x) for x in per_cam]
    if not lengths or min(lengths) == 0:
        raise SystemExit("[multiview_grid_video] At least one camera images/ folder has no images.")

    n = min(lengths)
    if max(lengths) != min(lengths):
        print(
            f"[multiview_grid_video] Warning: per-camera frame counts differ {lengths}; "
            f"using first {n} frames only."
        )

    return [[per_cam[j][i] for j in range(len(img_dirs))] for i in range(n)]


def resolve_frame_path(img_dir: str, stem: str) -> str | None:
    for ext in ALLOWED_EXTENSIONS:
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def infer_rows_cols(
    n_views: int, rows: int | None, cols: int | None
) -> tuple[int, int]:
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


def truncate_stem(s: str, max_len: int = 32) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max(1, max_len - 1)] + "…"


def put_label_top_left(
    image_bgr: np.ndarray,
    text: str,
    font_scale: float = 1.0,
    thickness: int = 2,
    margin: int = 12,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    outline = thickness + 2
    org = (margin, margin + int(28 * font_scale))
    cv2.putText(image_bgr, text, org, font, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


class _FfmpegBgrPipe:
    """Stream raw BGR frames to ffmpeg (optionally NVENC)."""

    def __init__(self, out_path: str, w: int, h: int, fps: float, nvenc: bool) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found in PATH. Install ffmpeg or use --encoder opencv."
            )
        cmd: list[str] = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
        ]
        if nvenc:
            cmd += [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-cq",
                "28",
                "-pix_fmt",
                "yuv420p",
            ]
        else:
            cmd += [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
            ]
        cmd.append(out_path)
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.w = w
        self.h = h

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            raise ValueError("frame size mismatch for ffmpeg pipe")
        b = frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)
        assert self._p.stdin is not None
        self._p.stdin.write(b.tobytes())

    def close(self) -> None:
        if self._p.stdin:
            self._p.stdin.close()
        self._p.wait()
        if self._p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {self._p.returncode}")


def format_frame_label(
    row_index: int,
    total: int,
    stems: list[str],
    label_prefix: str,
    label_cam_index: int,
) -> str:
    col = max(0, min(label_cam_index, len(stems) - 1))
    stem_show = truncate_stem(stems[col])
    body = f"#{row_index + 1:05d}/{total}  {stem_show}"
    if label_prefix:
        return f"{label_prefix} {body}".strip()
    return body


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "From raw/NN/images/: build one tiled grid per frame (sorted order per camera) and encode one MP4."
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
        "--read-scale",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help=(
            "JPEG decode scale: 1=full res, 2=half, 4=quarter, 8=eighth (OpenCV IMREAD_REDUCED_*). "
            "Use 2 or 4 with large inputs for much faster loads (default: 1)."
        ),
    )
    p.add_argument(
        "--no-parallel-load",
        action="store_true",
        help="Disable parallel image loading per frame (default: parallel on).",
    )
    p.add_argument(
        "--encoder",
        choices=("opencv", "ffmpeg"),
        default="opencv",
        help=(
            "opencv: cv2.VideoWriter (simple). ffmpeg: pipe raw BGR to ffmpeg (often faster; "
            "use with --ffmpeg-nvenc on NVIDIA)."
        ),
    )
    p.add_argument(
        "--ffmpeg-nvenc",
        action="store_true",
        help="With --encoder ffmpeg, use h264_nvenc (NVIDIA). Requires ffmpeg with NVENC support.",
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
        default="",
        help="Optional text prepended to the top-left label (default: empty).",
    )
    p.add_argument(
        "--label-cam",
        type=int,
        default=0,
        help="Which camera’s stem to show in the label (0 = first camera folder, i.e. 01).",
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
        help="0-based frame index to start from (inclusive).",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="0-based end frame index (exclusive).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = os.path.abspath(args.raw_root)
    pad_bgr = parse_pad_bgr(args.pad_color)

    img_dirs = find_camera_image_dirs(raw_root)
    n_cams = len(img_dirs)
    plans_all = build_frame_plans_by_index(img_dirs)
    full_total = len(plans_all)

    slice_start = args.start or 0
    slice_end = args.end if args.end is not None else full_total
    plans = plans_all[slice_start:slice_end]
    n_out = len(plans)
    if n_out == 0:
        raise SystemExit("[multiview_grid_video] No frames to encode after slicing (--start/--end).")

    rows, cols = infer_rows_cols(n_cams, args.rows, args.cols)
    cell_side = max(32, int(args.max_cell_side))

    out_path = args.output
    if not out_path:
        out_path = os.path.join(raw_root, f"multiview_grid_{n_cams}cams.mp4")
    else:
        out_path = os.path.abspath(out_path)
        od = os.path.dirname(out_path)
        if od:
            os.makedirs(od, exist_ok=True)

    def paths_for_stems(stems: list[str]) -> list[str]:
        out: list[str] = []
        for j, d in enumerate(img_dirs):
            pp = resolve_frame_path(d, stems[j])
            if pp is None:
                raise FileNotFoundError(f"Missing image for stem {stems[j]!r} under {d}")
            out.append(pp)
        return out

    imread_flag = _READ_SCALE_TO_IMREAD[args.read_scale]
    parallel_load = not args.no_parallel_load
    if args.read_scale != 1:
        print(
            f"[multiview_grid_video] Decoding images at 1/{args.read_scale} linear size "
            f"(IMREAD_REDUCED); use --read-scale 1 for full-resolution decode."
        )
    if args.encoder == "ffmpeg" and args.ffmpeg_nvenc:
        print("[multiview_grid_video] Using ffmpeg with h264_nvenc (GPU encode if driver supports it).")
    elif args.encoder == "ffmpeg":
        print("[multiview_grid_video] Using ffmpeg with libx264 (CPU).")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cv_writer: cv2.VideoWriter | None = None
    ff_writer: _FfmpegBgrPipe | None = None
    out_w = out_h = 0

    def resize_for_sink(grid: np.ndarray) -> np.ndarray:
        if grid.shape[1] != out_w or grid.shape[0] != out_h:
            return cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return grid

    for i, stems in enumerate(tqdm(plans, desc="frames", unit="fr")):
        paths = paths_for_stems(stems)
        grid = build_grid(
            paths,
            cols=cols,
            rows=rows,
            pad_bgr=pad_bgr,
            cell_w=cell_side,
            cell_h=cell_side,
            imread_flag=imread_flag,
            parallel_imread=parallel_load,
        )
        global_row = slice_start + i
        put_label_top_left(
            grid,
            format_frame_label(
                global_row,
                full_total,
                stems,
                args.label_prefix,
                args.label_cam,
            ),
            font_scale=args.font_scale,
        )

        if cv_writer is None and ff_writer is None:
            oh, ow = grid.shape[:2]
            out_w, out_h = _compute_output_size(ow, oh, args.max_output_dim)
            if (out_w, out_h) != (ow, oh):
                print(
                    f"[multiview_grid_video] Final resize for encoder: {ow}x{oh} -> {out_w}x{out_h} "
                    f"(max_output_dim={args.max_output_dim})"
                )
            if args.encoder == "ffmpeg":
                ff_writer = _FfmpegBgrPipe(
                    out_path, out_w, out_h, float(args.fps), args.ffmpeg_nvenc
                )
            else:
                cv_writer = cv2.VideoWriter(
                    out_path, fourcc, float(args.fps), (out_w, out_h)
                )
                if not cv_writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

        out = resize_for_sink(grid)
        if ff_writer is not None:
            ff_writer.write(out)
        else:
            assert cv_writer is not None
            cv_writer.write(out)

    if ff_writer is not None:
        ff_writer.close()
    elif cv_writer is not None:
        cv_writer.release()

    enc = args.encoder
    if args.encoder == "ffmpeg" and args.ffmpeg_nvenc:
        enc = "ffmpeg+nvenc"
    print(
        f"[multiview_grid_video] Wrote {out_path}  cameras={n_cams}  grid={rows}x{cols}  "
        f"frames={n_out} (indices {slice_start}..{slice_start + n_out - 1} of {full_total})  "
        f"fps={args.fps}  cell={cell_side}px  read_scale={args.read_scale}  encoder={enc}"
    )


if __name__ == "__main__":
    main()
