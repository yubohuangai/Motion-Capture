"""
Generate a printable ChArUco board PNG aligned with apps/calibration/detect_calibration_board.py (charuco) presets.

Default preset `8x5_7x4_inner`: DICT_4X4_50, (8,5) squares / (7,4) inner corners, marker/square 0.7.
PNG matches the full sheet (PRINT_PAGE_INCHES_WH at PRINT_DPI); the ChArUco pattern is scaled to
fit inside PRINT_SAFE_PATTERN_METERS_WH and centered for post-print edge cutting.
Print at 100% scale with no distortion.

After saving, a scaled OpenCV window opens by default; use --no-preview to skip (e.g. SSH/CI).

Override margins with --margin-width / --margin-height (replaces preset borders for that run).

Default PNG and JSON metadata (<name>.json) go under apps/calibration/generated_charuco_boards/
(unless you pass -o with another directory). Use --no-write-meta to skip the JSON.

Example:
  python3 apps/calibration/generate_charuco_board.py
  python3 apps/calibration/generate_charuco_board.py --no-preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))

import cv2
import cv2.aruco as aruco
import numpy as np

from charuco_board_presets import (
    BOARD_PRESETS,
    DEFAULT_CHARUCO_PRESET,
    get_aruco_dictionary,
    preset_marker_length,
    preset_physical_meters,
    resolve_border_px,
)

DEFAULT_PRESET = DEFAULT_CHARUCO_PRESET
MARGIN_SIZE_PX = 10
# Max long side (px) for on-screen preview; full-resolution PNG is still written.
PREVIEW_MAX_SIDE_PX = 1400
# Default output folder (PNG + optional .json) relative to this package.
DEFAULT_OUTPUT_SUBDIR = "generated_charuco_boards"


def default_output_dir() -> Path:
    return _CALIB_DIR / DEFAULT_OUTPUT_SUBDIR


def default_output_path(preset_output: str, dict_name: str) -> str:
    p = Path(preset_output)
    safe = dict_name.replace(" ", "_")
    name = f"{p.stem}_{safe}{p.suffix}"
    return str(default_output_dir() / name)


def generate_png(
    preset: dict,
    dict_name: str,
    margin_size: int = MARGIN_SIZE_PX,
    margin_width: int | None = None,
    margin_height: int | None = None,
) -> np.ndarray:
    squares = preset["squares"]
    square_length = float(preset.get("square_length", 800))
    marker_length = preset_marker_length(preset)
    image_size = preset["image_size"]
    w, h = int(image_size[0]), int(image_size[1])

    dict_obj = get_aruco_dictionary(dict_name)
    board = aruco.CharucoBoard(squares, square_length, marker_length, dict_obj)

    img = np.ones((h, w), dtype=np.uint8) * 255
    img = board.generateImage((w, h), marginSize=margin_size, img=img)

    top, bottom, left, right = resolve_border_px(preset, margin_width, margin_height)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return img


def show_preview(gray: np.ndarray, title: str = "ChArUco board") -> None:
    """Resize if needed, then cv2.imshow until a key is pressed."""
    h, w = gray.shape[:2]
    m = max(h, w)
    if m > PREVIEW_MAX_SIDE_PX:
        scale = PREVIEW_MAX_SIDE_PX / m
        vis = cv2.resize(
            gray,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        vis = gray
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.imshow(title, vis)
    print("Preview window: press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Generate one ChArUco board PNG (matches detect_calibration_board charuco presets).")
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        choices=sorted(BOARD_PRESETS.keys()),
        help=f"Board layout (default: {DEFAULT_PRESET})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output PNG path (default: apps/calibration/generated_charuco_boards/"
            "<preset_stem>_<DICT_NAME>.png)"
        ),
    )
    parser.add_argument(
        "--dict",
        dest="dictionary",
        default=None,
        help="Override cv2.aruco dictionary name (e.g. DICT_4X4_50)",
    )
    parser.add_argument(
        "--margin-width",
        type=int,
        default=None,
        help="White border pixels on left and right (overrides preset for that axis)",
    )
    parser.add_argument(
        "--margin-height",
        type=int,
        default=None,
        help="White border pixels on top and bottom (overrides preset for that axis)",
    )
    parser.add_argument(
        "--no-write-meta",
        action="store_false",
        dest="write_meta",
        help="Skip writing <output>.json next to the PNG (default: write metadata)",
    )
    parser.set_defaults(write_meta=True)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip opening a window after saving (default: show scaled preview)",
    )
    args = parser.parse_args()

    preset = BOARD_PRESETS[args.preset]
    dict_name = args.dictionary or preset["dictionary"]

    if args.output is not None:
        out_path = args.output
    else:
        out_path = default_output_path(preset["output"], dict_name)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    img = generate_png(
        preset,
        dict_name,
        margin_width=args.margin_width,
        margin_height=args.margin_height,
    )
    top, bottom, left, right = resolve_border_px(
        preset, args.margin_width, args.margin_height
    )

    print(f"preset={args.preset!r}  dictionary={dict_name!r}")
    print(f"{out_path}  ndarray shape: {img.shape}  (H, W)")
    phys = preset_physical_meters(preset, (top, bottom, left, right))
    if phys is not None:
        pw, ph = phys["paper_width_m"], phys["paper_height_m"]
        aw, ah = phys["pattern_width_m"], phys["pattern_height_m"]
        sqm = phys["chessboard_square_side_m"]
        mm = phys["marker_side_m"]
        print(
            "Physical (1:1 print vs print_page_inches_wh, meters): "
            f"paper {pw:.3f} x {ph:.3f} m; "
            f"pattern {aw:.3f} x {ah:.3f} m; "
            f"chessboard_square {sqm:.3f} m; "
            f"marker_square {mm:.3f} m"
        )
    else:
        print(
            "Physical meters: N/A (preset has no print_page_inches_wh; "
            "only pixel geometry is defined)."
        )
    cv2.imwrite(out_path, img)

    if args.write_meta:
        sq = float(preset.get("square_length", 800))
        meta = {
            "preset": args.preset,
            "dictionary": dict_name,
            "squares": [int(preset["squares"][0]), int(preset["squares"][1])],
            "marker_ratio": float(preset["marker_ratio"]),
            "square_length": sq,
            "marker_length": float(preset_marker_length(preset)),
            "inner_corners_pattern": [
                int(preset["squares"][0] - 1),
                int(preset["squares"][1] - 1),
            ],
            "image_size_wh": [int(preset["image_size"][0]), int(preset["image_size"][1])],
            "border_copyMakeBorder_top_bottom_left_right": [top, bottom, left, right],
            "final_image_wh": [
                int(preset["image_size"][0]) + left + right,
                int(preset["image_size"][1]) + top + bottom,
            ],
            "detect_charuco_hint": (
                f"python3 apps/calibration/detect_calibration_board.py <data> --preset {args.preset}"
                + (f" --dict {dict_name}" if args.dictionary else "")
            ),
        }
        if "print_page_inches_wh" in preset:
            meta["print_page_inches_wh"] = preset["print_page_inches_wh"]
        if "print_safe_pattern_meters_wh" in preset:
            meta["print_safe_pattern_meters_wh"] = preset["print_safe_pattern_meters_wh"]
        if "print_dpi" in preset:
            meta["print_dpi"] = preset["print_dpi"]
        if phys is not None:
            meta["physical_meters"] = {k: float(v) for k, v in phys.items()}
        meta_path = Path(out_path).with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote {meta_path}")

    if not args.no_preview:
        show_preview(img)


if __name__ == "__main__":
    main()
