"""
Shared ChArUco board definitions for detect_calibration_board.py (charuco mode) and generate_charuco_board.py.

Detection fields (required): squares, marker_ratio, dictionary.
Generator fields: square_length, image_size (W, H), border_px OR margin_width_px + margin_height_px, output.
"""

from __future__ import annotations

import cv2.aruco as aruco

# Inch → meter (international inch).
INCH_TO_M = 0.0254

# ---------------------------------------------------------------------------
# Defaults for this project (8×5 squares = 7×4 inner corners; DICT_4X4_50; ratio 0.7)
# ---------------------------------------------------------------------------
DEFAULT_CHARUCO_PRESET = "8x5_7x4_inner"
DEFAULT_CHARUCO_DICTIONARY = "DICT_4X4_50"
DEFAULT_CHARUCO_SQUARES = (8, 5)
DEFAULT_CHARUCO_INNER_CORNERS = (7, 4)
DEFAULT_CHARUCO_MARKER_RATIO = 0.7

# Full sheet size (PNG / print), inches × DPI → pixels.
# Max ChArUco pattern extent on paper (meters, width × height); centered on the sheet after trim.
PRINT_PAGE_INCHES_WH = (36.0, 24.0)
PRINT_SAFE_PATTERN_METERS_WH = (0.83, 0.5)
PRINT_DPI = 300
PRINT_PAGE_WH_PX = (
    int(round(PRINT_PAGE_INCHES_WH[0] * PRINT_DPI)),
    int(round(PRINT_PAGE_INCHES_WH[1] * PRINT_DPI)),
)


def _meters_to_print_px(m_w: float, m_h: float, dpi: int) -> tuple[int, int]:
    """Physical meters on the printed page → pixels at dpi (dots per inch)."""
    px_per_m = dpi / INCH_TO_M
    return int(round(m_w * px_per_m)), int(round(m_h * px_per_m))


def _fit_rect_inside(
    max_w: int, max_h: int, aspect_w: int, aspect_h: int
) -> tuple[int, int]:
    """Largest WxH with W/H = aspect_w/aspect_h that fits in max_w x max_h."""
    h_if_w = int(round(max_w * aspect_h / aspect_w))
    if h_if_w <= max_h:
        return max_w, h_if_w
    w_if_h = int(round(max_h * aspect_w / aspect_h))
    return w_if_h, max_h


def _symmetric_border_px(
    inner_wh: tuple[int, int], outer_wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) so inner + border matches outer (W,H)."""
    iw, ih = inner_wh
    ow, oh = outer_wh
    dw, dh = ow - iw, oh - ih
    if dw < 0 or dh < 0:
        raise ValueError("outer_wh must be >= inner_wh on both axes")
    mt, mb = dh // 2, dh - dh // 2
    ml, mr = dw // 2, dw - dw // 2
    return (mt, mb, ml, mr)


# Inner ChArUco raster: 8:5, max extent PRINT_SAFE_PATTERN_METERS_WH; centered on full sheet.
_SAFE_W, _SAFE_H = _meters_to_print_px(
    PRINT_SAFE_PATTERN_METERS_WH[0], PRINT_SAFE_PATTERN_METERS_WH[1], PRINT_DPI
)
_sx, _sy = DEFAULT_CHARUCO_SQUARES[0], DEFAULT_CHARUCO_SQUARES[1]
CHARUCO_8X5_BOARD_IMAGE_WH = _fit_rect_inside(_SAFE_W, _SAFE_H, _sx, _sy)
CHARUCO_8X5_PAGE_BORDER_PX = _symmetric_border_px(
    CHARUCO_8X5_BOARD_IMAGE_WH, PRINT_PAGE_WH_PX
)

# Preset name -> config. Extra keys are ignored by detection; print keys are ignored by detection logic.
BOARD_PRESETS: dict[str, dict] = {
    "5x7_a4": {
        "squares": (5, 7),
        "marker_ratio": 25.5 / 30,
        "dictionary": "DICT_4X4_250",
        "square_length": 200,
        "image_size": (1050, 1485),
        "border_px": (100, 100, 90, 90),
        "output": "charuco_board_5x7.png",
    },
    "6x9_a4": {
        "squares": (6, 9),
        "marker_ratio": 560 / 800,
        "dictionary": "DICT_6X6_250",
        "square_length": 800,
        "image_size": (4800, 7200),
        "border_px": (610, 600, 570, 570),
        "output": "charuco_board_6x9.png",
    },
    "4x6_a4": {
        "squares": (4, 6),
        "marker_ratio": 560 / 800,
        "dictionary": "DICT_6X6_250",
        "square_length": 800,
        "image_size": (3200, 4800),
        "border_px": (570, 570, 500, 500),
        "output": "charuco_board_4x6.png",
    },
    "8x5_7x4_inner": {
        "squares": DEFAULT_CHARUCO_SQUARES,
        "marker_ratio": DEFAULT_CHARUCO_MARKER_RATIO,
        "dictionary": DEFAULT_CHARUCO_DICTIONARY,
        "square_length": 800,
        "image_size": CHARUCO_8X5_BOARD_IMAGE_WH,
        "border_px": CHARUCO_8X5_PAGE_BORDER_PX,
        "output": "charuco_board_8x5_7x4_inner.png",
        "print_page_inches_wh": list(PRINT_PAGE_INCHES_WH),
        "print_safe_pattern_meters_wh": list(PRINT_SAFE_PATTERN_METERS_WH),
        "print_dpi": PRINT_DPI,
    },
}


def get_aruco_dictionary(name: str):
    if not hasattr(aruco, name):
        raise ValueError(
            f"Unknown ArUco dictionary name: {name!r} (expected cv2.aruco.DICT_* name)"
        )
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def preset_marker_length(preset: dict) -> float:
    sq = float(preset.get("square_length", 800))
    return sq * float(preset["marker_ratio"])


def preset_physical_meters(
    preset: dict, border_tb_lr: tuple[int, int, int, int]
) -> dict[str, float] | None:
    """
    Map pixel layout to meters when the preset defines print_page_inches_wh (full sheet size).
    Assumes the PNG is printed at 1:1 with that sheet aspect (no non-uniform scaling).

    Returns paper size, ChArUco pattern footprint on paper, and one chessboard square side (m).
    All distances are rounded to 3 decimal places (millimeter resolution).
    """
    if "print_page_inches_wh" not in preset:
        return None
    sw_in, sh_in = float(preset["print_page_inches_wh"][0]), float(
        preset["print_page_inches_wh"][1]
    )
    paper_w_m = sw_in * INCH_TO_M
    paper_h_m = sh_in * INCH_TO_M

    iw, ih = int(preset["image_size"][0]), int(preset["image_size"][1])
    top, bottom, left, right = border_tb_lr
    ow = iw + left + right
    oh = ih + top + bottom
    if ow <= 0 or oh <= 0:
        return None

    sqx, sqy = int(preset["squares"][0]), int(preset["squares"][1])
    square_from_w_m = (iw / sqx) * (paper_w_m / ow)
    square_from_h_m = (ih / sqy) * (paper_h_m / oh)
    square_side_m = 0.5 * (square_from_w_m + square_from_h_m)

    pattern_w_m = iw * paper_w_m / ow
    pattern_h_m = ih * paper_h_m / oh
    marker_side_m = square_side_m * float(preset["marker_ratio"])

    out = {
        "paper_width_m": paper_w_m,
        "paper_height_m": paper_h_m,
        "pattern_width_m": pattern_w_m,
        "pattern_height_m": pattern_h_m,
        "chessboard_square_side_m": square_side_m,
        "chessboard_square_side_from_x_m": square_from_w_m,
        "chessboard_square_side_from_y_m": square_from_h_m,
        "marker_side_m": marker_side_m,
    }
    return {k: round(float(v), 3) for k, v in out.items()}


def resolve_border_px(
    preset: dict,
    margin_width: int | None = None,
    margin_height: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Return (top, bottom, left, right) for cv2.copyMakeBorder.
    If margin_width or margin_height is set, symmetric borders use those; the other axis falls back to preset.
    If preset has border_px and no CLI margin args, use border_px.
    Otherwise use margin_width_px / margin_height_px symmetrically.
    """
    if margin_width is not None or margin_height is not None:
        w = (
            int(margin_width)
            if margin_width is not None
            else int(preset.get("margin_width_px", 0))
        )
        h = (
            int(margin_height)
            if margin_height is not None
            else int(preset.get("margin_height_px", 0))
        )
        return (h, h, w, w)
    if "border_px" in preset:
        t, b, l, r = preset["border_px"]
        return (int(t), int(b), int(l), int(r))
    w = int(preset.get("margin_width_px", 0))
    h = int(preset.get("margin_height_px", 0))
    return (h, h, w, w)
