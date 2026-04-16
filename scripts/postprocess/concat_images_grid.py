"""
Filepath: scripts/postprocess/concat_images_grid.py

Concatenate images from a directory into a single grid (e.g. 4 columns × 3 rows).
Useful for mask / visualization dumps with one panel per camera.

Empty slots (when n_images < rows*cols) are filled with a solid pad color.
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np


def natural_key(path: str):
    """Sort key so cam2 comes before cam10."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def collect_images(image_dir: str) -> list[str]:
    """Sorted list of image paths; if none at top level, gather from immediate subdirs."""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    imgs = sorted(
        [
            p
            for p in glob(os.path.join(image_dir, "*.*"))
            if p.lower().endswith(exts)
        ],
        key=natural_key,
    )
    if imgs:
        return imgs

    subdirs = [d for d in glob(os.path.join(image_dir, "*")) if os.path.isdir(d)]
    combined: list[str] = []
    for sd in sorted(subdirs, key=natural_key):
        combined.extend(
            sorted(
                [
                    p
                    for p in glob(os.path.join(sd, "*.*"))
                    if p.lower().endswith(exts)
                ],
                key=natural_key,
            )
        )
    return combined


def parse_pad_bgr(s: str) -> tuple[int, int, int]:
    s = s.strip().lower()
    if s in ("black", "0"):
        return (0, 0, 0)
    if s in ("white", "255"):
        return (255, 255, 255)
    parts = s.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "pad_color: use 'black', 'white', or 'B G R' three integers (OpenCV order)"
        )
    return tuple(int(x) for x in parts)


def letterbox_to_cell(
    img_bgr: np.ndarray, cell_w: int, cell_h: int, pad_bgr: tuple[int, int, int]
) -> np.ndarray:
    """Resize with aspect preserved; center on cell_w x cell_h canvas."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty image array")
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((cell_h, cell_w, 3), pad_bgr, dtype=np.uint8)
    y0 = (cell_h - new_h) // 2
    x0 = (cell_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def build_grid(
    paths: list[str],
    cols: int,
    rows: int,
    pad_bgr: tuple[int, int, int],
    cell_w: int | None,
    cell_h: int | None,
    *,
    imread_flag: int = cv2.IMREAD_COLOR,
    parallel_imread: bool = True,
    fit: str = "letterbox",
) -> np.ndarray:
    """
    ``imread_flag``: pass ``cv2.IMREAD_REDUCED_COLOR_2`` (etc.) to decode smaller than full-res.
    ``parallel_imread``: load paths in parallel (faster for many panels).
    ``fit``:
      - ``letterbox``: preserve aspect, pad to cell_w×cell_h (default).
      - ``uniform``: preserve aspect, scale so max(w,h) equals max(cell_w, cell_h); no padding
        (all panels share the same WxH; best when every source has the same resolution).
      - ``stretch``: resize exactly to cell_w×cell_h (may distort if aspect differs).
    """
    if fit not in ("letterbox", "uniform", "stretch"):
        raise ValueError(f"fit must be letterbox|uniform|stretch, got {fit!r}")
    slots = cols * rows
    if len(paths) > slots:
        raise ValueError(
            f"[concat_images_grid] {len(paths)} images but grid is only {cols}x{rows}={slots} slots"
        )

    def _read_one(p: str) -> np.ndarray:
        im = cv2.imread(p, imread_flag)
        if im is None:
            raise FileNotFoundError(f"failed to read image: {p}")
        return im

    if parallel_imread and len(paths) > 1:
        max_workers = min(32, len(paths))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            loaded = list(ex.map(_read_one, paths))
    else:
        loaded = [_read_one(p) for p in paths]

    if not loaded:
        raise ValueError("no images to concatenate")

    if fit == "letterbox":
        if cell_w is None or cell_h is None:
            max_w = max(im.shape[1] for im in loaded)
            max_h = max(im.shape[0] for im in loaded)
            cell_w = cell_w or max_w
            cell_h = cell_h or max_h
        cells = [letterbox_to_cell(im, int(cell_w), int(cell_h), pad_bgr) for im in loaded]
        tw, th = int(cell_w), int(cell_h)
    elif fit == "uniform":
        if cell_w is None or cell_h is None:
            raise ValueError("uniform fit requires cell_w and cell_h (use max side for both)")
        cap = max(int(cell_w), int(cell_h))
        im0 = loaded[0]
        h0, w0 = im0.shape[:2]
        scale = cap / max(w0, h0)
        tw = max(1, int(round(w0 * scale)))
        th = max(1, int(round(h0 * scale)))
        cells = [cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA) for im in loaded]
    else:
        if cell_w is None or cell_h is None:
            raise ValueError("stretch fit requires cell_w and cell_h")
        tw, th = int(cell_w), int(cell_h)
        cells = [cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA) for im in loaded]

    n_pad = slots - len(cells)
    blank = np.full((th, tw, 3), pad_bgr, dtype=np.uint8)
    cells.extend([blank.copy() for _ in range(n_pad)])

    row_tiles = []
    for r in range(rows):
        row_cells = cells[r * cols : (r + 1) * cols]
        row_tiles.append(np.hstack(row_cells))
    return np.vstack(row_tiles)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate images from a directory into a rows×cols grid."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing images (or per-camera subfolders with one image each).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns (default: 4).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of rows (default: 3).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output image path (default: <input_dir>/grid_{cols}x{rows}.png).",
    )
    parser.add_argument(
        "--pad-color",
        type=str,
        default="black",
        help="Pad color: black, white, or 'B G R' (OpenCV). Used for letterboxing and empty cells.",
    )
    parser.add_argument(
        "--cell-w",
        type=int,
        default=None,
        help="Fixed cell width in pixels (default: max image width).",
    )
    parser.add_argument(
        "--cell-h",
        type=int,
        default=None,
        help="Fixed cell height in pixels (default: max image height).",
    )
    args = parser.parse_args()
    pad_bgr = parse_pad_bgr(args.pad_color)

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"[concat_images_grid] not a directory: {input_dir}")

    paths = collect_images(input_dir)
    if not paths:
        raise SystemExit(f"[concat_images_grid] no images found under {input_dir}")

    out_path = args.output
    if not out_path:
        out_path = os.path.join(input_dir, f"grid_{args.cols}x{args.rows}.png")
    else:
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    grid = build_grid(
        paths,
        cols=args.cols,
        rows=args.rows,
        pad_bgr=pad_bgr,
        cell_w=args.cell_w,
        cell_h=args.cell_h,
    )

    ok = cv2.imwrite(out_path, grid)
    if not ok:
        raise SystemExit(f"[concat_images_grid] failed to write: {out_path}")

    h, w = grid.shape[:2]
    print(f"[concat_images_grid] images: {len(paths)}  grid: {args.cols}x{args.rows}  out: {w}x{h}")
    print(f"[concat_images_grid] wrote {out_path}")


if __name__ == "__main__":
    main()
