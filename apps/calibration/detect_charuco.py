'''
  Batch ChArUco corner detection for EasyMocap-style datasets.

  Writes the same chessboard/*.json format as detect_chessboard.py (keypoints3d,
  keypoints2d, visited, grid_size, pattern) so apps/calibration/calib_intri.py
  can consume annotations unchanged.

  Requires OpenCV 4.7+ aruco API (CharucoBoard + CharucoDetector), same as
  CalibCam/detect_charuco_image.py style boards.

  Presets are defined in charuco_board_presets.py (shared with generate_charuco_board.py).

  Default preset is 8x5_7x4_inner (DICT_4X4_50, 7×4 inner corners, marker ratio 0.7).

  Example (dataset root with images under <path>/images/):
    python3 apps/calibration/detect_charuco.py /path/to/data
    python3 apps/calibration/detect_charuco.py /path/to/data --preset 5x7_a4

  Single image (writes chessboard/<name>.json next to the image, viz under <parent>/output/calibration/):
    python3 apps/calibration/detect_charuco.py /path/to/one_image.jpg

  Print a matching board PNG:
    python3 apps/calibration/generate_charuco_board.py
'''
from __future__ import annotations

import argparse
import os
import sys
import threading
from os.path import join
from pathlib import Path

_CALIB_DIR = Path(__file__).resolve().parent
if str(_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_CALIB_DIR))

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from easymocap.annotator import ImageFolder
from easymocap.annotator.file_utils import getFileList, read_json, save_json
from easymocap.mytools.debug_utils import mywarn

from charuco_board_presets import (
    BOARD_PRESETS,
    DEFAULT_CHARUCO_DICTIONARY,
    DEFAULT_CHARUCO_PRESET,
    get_aruco_dictionary,
)

# Script-level board profiles for easy switching without CLI flags.
# Change ACTIVE_BOARD_PROFILE to swap defaults.
BOARD_DEFAULT_PROFILES = {
    # Previous default behavior in this script.
    "legacy_default": {
        "preset": DEFAULT_CHARUCO_PRESET,
        "dictionary": None,  # None -> use preset dictionary
        "grid_m": 0.033,
    },
    # New board: charuco_board_8x5_7x4_inner_DICT_4X4_50.json
    # physical_meters.chessboard_square_side_m = 0.1
    "board_8x5_7x4_dict4x4_50_36x24": {
        "preset": "8x5_7x4_inner",
        "dictionary": "DICT_4X4_50",
        "grid_m": 0.1,
    },
}
ACTIVE_BOARD_PROFILE = "board_8x5_7x4_dict4x4_50_36x24"


def build_board(squares_xy: tuple[int, int], marker_ratio: float, dict_name: str):
    """squares_xy = (squares_x, squares_y) as in cv2.aruco.CharucoBoard."""
    dict_obj = get_aruco_dictionary(dict_name)
    square_length = 1.0
    marker_length = square_length * float(marker_ratio)
    board = aruco.CharucoBoard(squares_xy, square_length, marker_length, dict_obj)
    charuco_params = aruco.CharucoParameters()
    detector_params = aruco.DetectorParameters()
    refine_params = aruco.RefineParameters()
    detector = aruco.CharucoDetector(board, charuco_params, detector_params, refine_params)
    return board, detector


def board_to_keypoints3d(board, grid_size_m: float) -> np.ndarray:
    """3D template in meters; XY order matches easymocap CharucoBoard (swap of board X/Y)."""
    obj = board.getChessboardCorners()
    obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    # Same convention as easymocap/annotator/chessboard.py CharucoBoard
    obj = obj[:, [1, 0, 2]] * float(grid_size_m)
    return obj


def create_charuco_templates(
    path: str,
    image_sub: str,
    keypoints3d: np.ndarray,
    pattern: tuple[int, int],
    grid_size: float,
    ext: str,
    overwrite: bool,
):
    """Create or refresh chessboard/*.json templates parallel to images."""
    sx, sy = pattern
    template = {
        "keypoints3d": keypoints3d.tolist(),
        "keypoints2d": np.zeros((keypoints3d.shape[0], 3)).tolist(),
        "pattern": [int(sx), int(sy)],
        "grid_size": grid_size,
        "visited": False,
    }
    imgnames = getFileList(join(path, image_sub), ext=ext)
    for rel in tqdm(imgnames, desc="create template charuco"):
        annname = join(path, "chessboard", rel.replace(ext, ".json"))
        if os.path.exists(annname) and overwrite:
            data = read_json(annname)
            data["keypoints3d"] = template["keypoints3d"]
            data["pattern"] = template["pattern"]
            data["grid_size"] = template["grid_size"]
            save_json(annname, data)
        elif os.path.exists(annname) and not overwrite:
            continue
        else:
            save_json(annname, template)


def detect_one_image(
    img_bgr: np.ndarray,
    detector,
    board,
    n_corners: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Returns (visualization_bgr_or_none, keypoints2d Nx3).
    keypoints2d uses confidence in last channel (1 = detected).
    """
    ch_corners, ch_ids, _, _ = detector.detectBoard(img_bgr)
    k2d = np.zeros((n_corners, 3), dtype=np.float64)
    if ch_ids is None or ch_corners is None:
        return None, k2d
    ids = np.asarray(ch_ids, dtype=np.int32).reshape(-1)
    pts = np.asarray(ch_corners, dtype=np.float64).reshape(-1, 2)
    n = min(len(ids), len(pts))
    for i in range(n):
        cid = int(ids[i])
        if 0 <= cid < n_corners:
            k2d[cid, 0] = pts[i, 0]
            k2d[cid, 1] = pts[i, 1]
            k2d[cid, 2] = 1.0
    vis = aruco.drawDetectedCornersCharuco(
        img_bgr.copy(), ch_corners, ch_ids, cornerColor=(0, 0, 255)
    )
    return vis, k2d


def _rel_under_image_root(imgname: str, path: str, image_sub: str) -> str:
    root = os.path.normpath(join(path, image_sub))
    rel = os.path.relpath(imgname, root)
    if rel.startswith(".."):
        return os.path.basename(imgname)
    return rel


def _detect_charuco_worker(datas, path: str, image_sub: str, out: str, args, detector, board, n_corners: int):
    for imgname, annotname in datas:
        img = cv2.imread(imgname)
        if img is None:
            mywarn(f"[detect_charuco] Cannot read {imgname}")
            continue
        annots = read_json(annotname)
        annots["visited"] = True
        show, k2d = detect_one_image(img, detector, board, n_corners)
        annots["keypoints2d"] = k2d.tolist()
        save_json(annotname, annots)
        if show is None:
            if args.debug:
                mywarn(f"[detect_charuco] No ChArUco in {imgname}")
            continue
        rel = _rel_under_image_root(imgname, path, image_sub)
        outname = join(out, rel)
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, show)


def detect_charuco_batch(path: str, image_sub: str, out: str, args, board, detector, n_corners: int):
    create_charuco_templates(
        path,
        image_sub,
        board_to_keypoints3d(board, args.grid),
        pattern=((args.squares[0] - 1), (args.squares[1] - 1)),
        grid_size=args.grid,
        ext=args.ext,
        overwrite=args.overwrite3d,
    )
    dataset = ImageFolder(path, image=image_sub, annot="chessboard", ext=args.ext)
    dataset.isTmp = False
    trange = list(range(len(dataset)))
    threads = []
    for i in range(args.mp):
        ranges = trange[i :: args.mp]
        datas = [dataset[t] for t in ranges]
        t = threading.Thread(
            target=_detect_charuco_worker,
            args=(datas, path, image_sub, out, args, detector, board, n_corners),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def _normalize_single_image_arg(path_arg: str) -> tuple[str, str | None]:
    """
    If path_arg is an image file, return (parent directory as dataset root, absolute image path).
    Otherwise return (path_arg unchanged, None).
    """
    ap = os.path.abspath(path_arg)
    if os.path.isfile(ap) and ap.lower().endswith((".jpg", ".jpeg", ".png")):
        return os.path.dirname(ap), ap
    return path_arg, None


def detect_charuco_one_file(
    image_path: str,
    data_root: str,
    out: str,
    args,
    board,
    detector,
    n_corners: int,
):
    """Detect on one image; writes chessboard/<stem>.json under data_root and visualization under out."""
    rel = os.path.basename(image_path)
    annname = join(data_root, "chessboard", str(Path(rel).with_suffix(".json")))
    keypoints3d = board_to_keypoints3d(board, args.grid)
    pattern = ((args.squares[0] - 1), (args.squares[1] - 1))
    sx, sy = pattern
    template = {
        "keypoints3d": keypoints3d.tolist(),
        "keypoints2d": np.zeros((keypoints3d.shape[0], 3)).tolist(),
        "pattern": [int(sx), int(sy)],
        "grid_size": args.grid,
        "visited": False,
    }
    if not os.path.exists(annname):
        save_json(annname, template)
    elif args.overwrite3d:
        data = read_json(annname)
        data["keypoints3d"] = template["keypoints3d"]
        data["pattern"] = template["pattern"]
        data["grid_size"] = template["grid_size"]
        save_json(annname, data)

    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"[detect_charuco] Cannot read {image_path}")
    annots = read_json(annname)
    annots["visited"] = True
    show, k2d = detect_one_image(img, detector, board, n_corners)
    annots["keypoints2d"] = k2d.tolist()
    save_json(annname, annots)
    if show is None:
        if args.debug:
            mywarn(f"[detect_charuco] No ChArUco in {image_path}")
        print(f"[detect_charuco] No ChArUco detected; annotation saved: {annname}")
        return
    outname = join(out, rel)
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    cv2.imwrite(outname, show)
    print(f"[detect_charuco] wrote {outname}")


def main():
    if ACTIVE_BOARD_PROFILE not in BOARD_DEFAULT_PROFILES:
        raise SystemExit(
            f"Unknown ACTIVE_BOARD_PROFILE={ACTIVE_BOARD_PROFILE!r}; "
            f"available: {sorted(BOARD_DEFAULT_PROFILES.keys())}"
        )
    profile = BOARD_DEFAULT_PROFILES[ACTIVE_BOARD_PROFILE]

    parser = argparse.ArgumentParser(description="Batch ChArUco detection (EasyMocap chessboard JSON layout)")
    parser.add_argument(
        "path",
        type=str,
        help="Dataset root (contains images/ and chessboard/) or a single .jpg/.png image path",
    )
    parser.add_argument("--image", type=str, default="images", help="Subfolder of path with images")
    parser.add_argument("--out", type=str, default=None, help="Visualization output root")
    parser.add_argument("--ext", type=str, default=".jpg", choices=[".jpg", ".png"])
    parser.add_argument(
        "--preset",
        type=str,
        default=profile["preset"],
        choices=sorted(BOARD_PRESETS.keys()),
    )
    parser.add_argument(
        "--manual",
        nargs=3,
        metavar=("SX", "SY", "MARKER_RATIO"),
        default=None,
        help="Override preset: squares_x squares_y marker_ratio (marker side / square side)",
    )
    parser.add_argument(
        "--dict",
        dest="dictionary",
        default=profile["dictionary"],
        help="cv2.aruco dictionary name, e.g. DICT_4X4_250 (overrides preset)",
    )
    parser.add_argument(
        "--grid",
        type=float,
        default=float(profile["grid_m"]),
        help="Physical square size in meters (scales keypoints3d for calibration)",
    )
    parser.add_argument("--mp", type=int, default=4, help="Thread count")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite3d", action="store_true", help="Refresh keypoints3d in existing JSON")

    args = parser.parse_args()
    data_root, single_image = _normalize_single_image_arg(args.path)
    args.path = data_root
    if single_image is not None:
        args.ext = ".jpg" if single_image.lower().endswith((".jpg", ".jpeg")) else ".png"

    if args.out is None:
        args.out = join(args.path, "output", "calibration")

    if args.manual is not None:
        sx_s, sy_s, r_s = args.manual
        args.squares = (int(sx_s), int(sy_s))
        marker_ratio = float(r_s)
        dict_name = args.dictionary or DEFAULT_CHARUCO_DICTIONARY
        preset_label = "manual"
    else:
        cfg = BOARD_PRESETS[args.preset]
        args.squares = tuple(cfg["squares"])
        marker_ratio = float(cfg["marker_ratio"])
        dict_name = args.dictionary or cfg["dictionary"]
        preset_label = args.preset

    if min(args.squares) <= 0 or not (0.0 < marker_ratio < 1.0):
        raise SystemExit("Invalid squares or marker_ratio")

    board, detector = build_board(args.squares, marker_ratio, dict_name)
    n_corners = int(np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3).shape[0])

    print(
        f"[detect_charuco] preset={preset_label!r} squares={args.squares} "
        f"dict={dict_name!r} marker_ratio={marker_ratio:.6f} n_corners={n_corners} grid_m={args.grid}"
    )

    if single_image is not None:
        print(f"[detect_charuco] single image mode: {single_image}")
        detect_charuco_one_file(
            single_image, args.path, args.out, args, board, detector, n_corners
        )
    else:
        detect_charuco_batch(args.path, args.image, args.out, args, board, detector, n_corners)


if __name__ == "__main__":
    main()
