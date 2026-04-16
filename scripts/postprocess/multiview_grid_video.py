"""
Build a single preview **video** from **images** only: every ``NN/images/`` folder under a
session ``raw/`` root is tiled into one grid per time step, the frame label is drawn at the
top-left, and the result is encoded to one MP4. Each view is downscaled before tiling.

**Pairing frames across cameras:** The script does not “do CSV work”. It only needs to know,
for each output frame, which **image filename** to take from camera ``01``, ``02``, … If every
camera uses the **same** stem (e.g. ``000001.jpg`` everywhere), use ``--align intersection``.
After ``sync.py --extract``, stems are usually **different per camera** (timestamp names); then
pairing comes from **sync’s output** — the same per-camera stem table that
``move_unmatched.py`` reads (on disk it is named ``matched.csv`` under ``output/exp/...``).
That file is just a table of stems; omit ``--sync-matched`` to use the default repo path.

Typical layout (11 cameras): ``--rows 3 --cols 4`` (12 slots, one empty pad cell).

Example::

    python scripts/postprocess/multiview_grid_video.py /path/to/cow_1_board/raw \\
        -o /path/to/cow_1_board/multiview_11cams.mp4 --fps 30 --max-cell-side 720

If the pairing file is not under ``<repo>/output/exp/...``, pass it::

    python scripts/postprocess/multiview_grid_video.py /path/to/raw \\
        --sync-matched /path/to/matched.csv
    # ``--csv`` is accepted as an alias for ``--sync-matched``.
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
import pandas as pd
from tqdm import tqdm

# Import grid helpers from sibling module (same directory as this file).
_POST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _POST_DIR.parent.parent
if str(_POST_DIR) not in sys.path:
    sys.path.insert(0, str(_POST_DIR))

from concat_images_grid import build_grid, parse_pad_bgr  # noqa: E402

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# Match img2vid.py: keep frames within typical H.264 / OpenCV limits.
MAX_OUTPUT_DIM = 4096


def natural_key_stem(stem: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", stem)]


def raw_session_slug(data_root: str) -> str:
    """
    Same slug as ``move_unmatched.py`` / ``sync.py`` output folder naming:
    ``.../data/0411/raw`` → ``0411``; ``.../my_dataset`` (no ``raw``) → ``my_dataset``.
    """
    p = Path(data_root).resolve()
    name = p.name
    if name.lower() == "raw" and p.parent != p:
        slug = p.parent.name
    else:
        slug = name
    slug = re.sub(r"[^\w\-]+", "_", slug, flags=re.ASCII).strip("_") or "session"
    return slug[:80]


def default_matched_csv_path(raw_root: str, exp_threshold: str) -> Path:
    """``<repo>/output/exp/<slug>_<threshold>/matched.csv`` (same as ``move_unmatched.py``)."""
    slug = raw_session_slug(raw_root)
    return _REPO_ROOT / "output" / "exp" / f"{slug}_{exp_threshold}" / "matched.csv"


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


def load_matched_frame_plans(pairing_path: str, n_cams: int) -> list[list[str]]:
    """
    Load sync’s pairing table (no header; one row per sync; column ``i`` = filename stem for
    camera ``i`` in ``01``, ``02``, … order). On disk this is often ``matched.csv``; values are
    read as strings to preserve long integer timestamps.
    """
    path = Path(pairing_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pairing file not found: {path}")

    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, engine="python")
    if df.shape[1] < n_cams:
        raise ValueError(
            f"{path.name} has {df.shape[1]} columns; need {n_cams} for {n_cams} cameras under raw."
        )
    if df.shape[1] > n_cams:
        df = df.iloc[:, :n_cams]

    plans: list[list[str]] = []
    for _, row in df.iterrows():
        stems = [str(row[j]).strip() for j in range(n_cams)]
        if any(s == "" or s.lower() == "nan" for s in stems):
            continue
        plans.append(stems)

    if not plans:
        raise ValueError(f"No valid rows in {path}")
    return plans


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
    """In-place: dark outline + bright text for readability on varied frames."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    outline = thickness + 2
    org = (margin, margin + int(28 * font_scale))
    cv2.putText(image_bgr, text, org, font, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(image_bgr, text, org, font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "From raw session images/: build one tiled grid per frame and encode a single MP4. "
            "Uses sync’s pairing table when filenames differ per camera (see --align / --sync-matched)."
        )
    )
    p.add_argument(
        "raw_root",
        type=str,
        help="Session root containing 01/images, 02/images, … (e.g. .../cow_1_board/raw).",
    )
    p.add_argument(
        "--align",
        choices=("matched", "intersection"),
        default="matched",
        help=(
            "How to pair frames across cameras. "
            "'matched' uses sync.py’s pairing table (default file: matched.csv under output/exp/…; "
            "required when each camera uses different timestamp stems). "
            "'intersection' pairs by identical stems in every images/ (only when names match)."
        ),
    )
    p.add_argument(
        "--sync-matched",
        "--csv",
        dest="sync_matched_path",
        default=None,
        metavar="PATH",
        help=(
            "Path to sync.py’s frame pairing table (same stems as move_unmatched.py; usually named "
            "matched.csv). Not a separate CSV pipeline—only tells this script which image stem to "
            "load per camera per row. Default: <repo>/output/exp/<slug>_<exp-threshold>/matched.csv."
        ),
    )
    p.add_argument(
        "--exp-threshold",
        default="16ms",
        metavar="LABEL",
        help="Folder segment in output/exp/ when using the default --sync-matched path (default: 16ms).",
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
        default="",
        help="Optional text prepended to the top-left label (default: empty).",
    )
    p.add_argument(
        "--label-stem-column",
        type=int,
        default=0,
        help="When using --align matched, which camera column’s stem to show in the label (0 = cam 01).",
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
        help="0-based index into the frame list / matched rows (inclusive).",
    )
    p.add_argument(
        "--end",
        type=int,
        default=None,
        help="0-based end index into the frame list / matched rows (exclusive).",
    )
    return p.parse_args()


def build_frame_plans(
    args: argparse.Namespace, raw_root: str, img_dirs: list[str], n_cams: int
) -> tuple[list[list[str]], str]:
    """
    Returns (plans, align_mode) where each plan row is [stem_cam0, stem_cam1, …].
    """
    if args.align == "intersection":
        stems = common_frame_stems_sorted(img_dirs)
        if not stems:
            raise SystemExit(
                f"[multiview_grid_video] No identical filenames across all {n_cams} cameras under {raw_root}. "
                f"After sync extraction, stems are per-camera timestamps — use default --align matched "
                f"and sync’s pairing file (matched.csv) from sync.py."
            )
        return [[s] * n_cams for s in stems], "intersection"

    pairing_path = args.sync_matched_path
    if not pairing_path:
        pairing_path = str(default_matched_csv_path(raw_root, args.exp_threshold))
    else:
        pairing_path = str(Path(pairing_path).expanduser().resolve())

    if not Path(pairing_path).is_file():
        raise SystemExit(
            f"[multiview_grid_video] Sync pairing file not found:\n  {pairing_path}\n"
            f"Run sync.py so output/exp/.../matched.csv exists, or pass --sync-matched PATH "
            f"(alias: --csv). See --exp-threshold (default {args.exp_threshold!r}) for the default folder name."
        )

    plans = load_matched_frame_plans(pairing_path, n_cams)
    print(f"[multiview_grid_video] Loaded {len(plans)} paired rows from:\n  {pairing_path}")
    return plans, "matched"


def format_frame_label(
    align: str,
    row_index: int,
    total: int,
    stems: list[str],
    label_prefix: str,
    label_stem_column: int,
) -> str:
    col = max(0, min(label_stem_column, len(stems) - 1))
    stem_show = truncate_stem(stems[col])
    if align == "intersection":
        t = f"{label_prefix}{stems[0]}".strip()
        return t or str(stems[0])

    # row_index: 0-based index into full pairing table (before --start/--end slice)
    body = f"#{row_index + 1:05d}/{total}  {stem_show}"
    if label_prefix:
        return f"{label_prefix} {body}".strip()
    return body


def main() -> None:
    args = parse_args()
    raw_root = os.path.abspath(args.raw_root)
    pad_bgr = parse_pad_bgr(args.pad_color)

    img_dirs = find_camera_image_dirs(raw_root)
    n_cams = len(img_dirs)
    plans_all, align_mode = build_frame_plans(args, raw_root, img_dirs, n_cams)
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter | None = None
    out_w = out_h = 0

    def write_frame(grid: np.ndarray) -> None:
        assert writer is not None
        if grid.shape[1] != out_w or grid.shape[0] != out_h:
            grid = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(grid)

    for i, stems in enumerate(tqdm(plans, desc="frames", unit="fr")):
        paths = paths_for_stems(stems)
        grid = build_grid(
            paths,
            cols=cols,
            rows=rows,
            pad_bgr=pad_bgr,
            cell_w=cell_side,
            cell_h=cell_side,
        )
        global_row = slice_start + i
        put_label_top_left(
            grid,
            format_frame_label(
                align_mode,
                global_row,
                full_total,
                stems,
                args.label_prefix,
                args.label_stem_column,
            ),
            font_scale=args.font_scale,
        )

        if writer is None:
            oh, ow = grid.shape[:2]
            out_w, out_h = _compute_output_size(ow, oh, args.max_output_dim)
            if (out_w, out_h) != (ow, oh):
                print(
                    f"[multiview_grid_video] Final resize for encoder: {ow}x{oh} -> {out_w}x{out_h} "
                    f"(max_output_dim={args.max_output_dim})"
                )
            writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

        write_frame(grid)

    assert writer is not None
    writer.release()
    print(
        f"[multiview_grid_video] Wrote {out_path}  align={align_mode}  cameras={n_cams}  "
        f"grid={rows}x{cols}  frames={n_out} (rows {slice_start}..{slice_start + n_out - 1} of {full_total})  "
        f"fps={args.fps}  cell={cell_side}px"
    )


if __name__ == "__main__":
    main()
