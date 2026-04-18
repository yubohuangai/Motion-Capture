"""
  Unified calibration board detection for EasyMocap-style datasets (chessboard/ JSON).

  Modes (--mode):
    charuco — ChArUco (cv2.aruco CharucoDetector); default.
    chessboard — classical chessboard corners (findChessboardCorners).

  Writes the same chessboard/*.json format for calib_intri.py / calib_extri.py.

  Examples:
    python apps/calibration/detect_calibration_board.py /path/to/data
    python apps/calibration/detect_calibration_board.py --mode chessboard /path/to/data

  Shims (same flags as before): detect_charuco.py, detect_chessboard.py
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import threading
import time
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
from easymocap.annotator.chessboard import findChessboardCorners
from easymocap.annotator.file_utils import getFileList, read_json, save_json
from easymocap.mytools.debug_utils import log, mywarn
import func_timeout
import threading

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

# --- Classical chessboard (OpenCV findChessboardCorners) -----------------------------
CHESSBOARD_BOARD_PROFILES = {
    "legacy_9x6_0p111": {
        "pattern": (9, 6),
        "grid_m": 0.111,
    },
    "inner_7x4_0p1": {
        "pattern": (7, 4),
        "grid_m": 0.1,
    },
}
ACTIVE_CHESSBOARD_PROFILE = "inner_7x4_0p1"


def _parse_pattern_opt(s):
    if s is None:
        return None
    a, b = s.split(",")
    return (int(a.strip()), int(b.strip()))


def getChessboard3d(pattern, gridSize, axis="yx"):
    template = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    object_points = np.zeros((pattern[1] * pattern[0], 3), np.float32)
    if axis == "xz":
        object_points[:, 0] = template[:, 0]
        object_points[:, 2] = template[:, 1]
    elif axis == "yx":
        object_points[:, 0] = template[:, 1]
        object_points[:, 1] = template[:, 0]
    else:
        raise NotImplementedError
    object_points = object_points * gridSize
    return object_points


def create_chessboard_templates_classic(
    path, image, pattern, gridSize, ext, overwrite, axis="yx"
):
    print("Create chessboard {}".format(pattern))
    keypoints3d = getChessboard3d(pattern, gridSize=gridSize, axis=axis)
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(join(path, image), ext=ext)
    template = {
        "keypoints3d": keypoints3d.tolist(),
        "keypoints2d": keypoints2d.tolist(),
        "pattern": pattern,
        "grid_size": gridSize,
        "visited": False,
    }
    for imgname in tqdm(imgnames, desc="create template chessboard"):
        annname = imgname.replace(ext, ".json")
        annname = join(path, "chessboard", annname)
        if os.path.exists(annname) and overwrite:
            data = read_json(annname)
            data["keypoints3d"] = template["keypoints3d"]
            save_json(annname, data)
        elif os.path.exists(annname) and not overwrite:
            continue
        else:
            save_json(annname, template)


def _detect_chessboard_worker(datas, path, image, out, pattern, args, thread_idx, per_thread_detected):
    detected = 0
    for imgname, annotname in datas:
        img = cv2.imread(imgname)
        if img is None:
            mywarn(f"[detect_chessboard] Cannot read {imgname}")
            continue
        annots = read_json(annotname)
        try:
            show = findChessboardCorners(
                img, annots, pattern, fix_orientation=args.fix_orientation
            )
        except func_timeout.exceptions.FunctionTimedOut:
            show = None
        save_json(annotname, annots)
        if show is None:
            if args.debug:
                mywarn("[Info] Cannot find chessboard in {}".format(imgname))
            continue
        detected += 1
        outname = join(out, imgname.replace(path + "/{}/".format(image), ""))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        if isinstance(show, np.ndarray):
            cv2.imwrite(outname, show)
    per_thread_detected[thread_idx] = detected


def detect_chessboard_batch(path, image, out, pattern, gridSize, args):
    t_batch0 = time.perf_counter()
    create_chessboard_templates_classic(
        path,
        image,
        pattern,
        gridSize,
        ext=args.ext,
        overwrite=args.overwrite3d,
        axis=args.axis,
    )
    t_after_templates = time.perf_counter()
    views = _discover_views(path, image, args.ext)
    labels = ", ".join(_view_label(v) for v in views)
    print(
        f"[detect_chessboard] detection pass: {len(views)} view(s) [{labels}] "
        f"(templates {t_after_templates - t_batch0:.1f}s; threads={args.mp})",
        flush=True,
    )

    per_view: list[tuple[str, int, int, float]] = []
    for vi, view in enumerate(views):
        label = _view_label(view)
        t0 = time.perf_counter()
        print(
            f"[detect_chessboard] view {vi + 1}/{len(views)} '{label}': start "
            f"(elapsed since batch start {t0 - t_batch0:.1f}s)",
            flush=True,
        )
        if view is None:
            dataset = ImageFolder(path, image=image, annot="chessboard", ext=args.ext)
        else:
            dataset = ImageFolder(path, view, image=image, annot="chessboard", ext=args.ext)
        dataset.isTmp = False
        n = len(dataset)
        if n == 0:
            per_view.append((label, 0, 0, 0.0))
            print(
                f"[detect_chessboard] view {vi + 1}/{len(views)} '{label}': done "
                f"Chessboard detected in 0/0 frames in 0.0s",
                flush=True,
            )
            continue

        trange = list(range(n))
        per_thread_detected = [0] * args.mp
        threads = []
        for ti in range(args.mp):
            ranges = trange[ti :: args.mp]
            datas = [dataset[t] for t in ranges]
            thread = threading.Thread(
                target=_detect_chessboard_worker,
                args=(datas, path, image, out, pattern, args, ti, per_thread_detected),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        n_detected = int(sum(per_thread_detected))
        dt = time.perf_counter() - t0
        per_view.append((label, n, n_detected, dt))
        print(
            f"[detect_chessboard] view {vi + 1}/{len(views)} '{label}': done "
            f"Chessboard detected in {n_detected}/{n} frames in {dt:.1f}s",
            flush=True,
        )

    t_total = time.perf_counter() - t_batch0
    print(
        f"[detect_chessboard] all views done in {t_total:.1f}s wall time (including template creation)",
        flush=True,
    )
    for label, n_frames, n_detected, dt in per_view:
        print(
            f"  - view {label!r}: {n_detected}/{n_frames} frames with board, detection pass {dt:.1f}s",
            flush=True,
        )


def _detect_by_search(path, image, out, pattern, sub, args):
    dataset = ImageFolder(path, sub=sub, annot="chessboard", ext=args.ext)
    dataset.isTmp = False
    nFrames = len(dataset)
    found = np.zeros(nFrames, dtype=bool)
    visited = np.zeros(nFrames, dtype=bool)
    proposals = []
    init_step = args.max_step
    min_step = args.min_step
    for nf in range(0, nFrames, init_step):
        if nf + init_step < len(dataset):
            proposals.append([nf, nf + init_step])
    while len(proposals) > 0:
        left, right = proposals.pop(0)
        print(
            "[detect] {} {:4.1f}% Check [{:5d}, {:5d}]".format(
                sub, visited.sum() / visited.shape[0] * 100, left, right
            ),
            end=" ",
        )
        for nf in [left, right]:
            if not visited[nf]:
                visited[nf] = True
                imgname, annotname = dataset[nf]
                img = cv2.imread(imgname)
                annots = read_json(annotname)
                try:
                    show = findChessboardCorners(
                        img, annots, pattern, fix_orientation=args.fix_orientation
                    )
                except func_timeout.exceptions.FunctionTimedOut:
                    show = None
                save_json(annotname, annots)
                if show is None:
                    if args.debug:
                        print("[Info] Cannot find chessboard in {}".format(imgname))
                    found[nf] = False
                    continue
                found[nf] = True
                outname = join(
                    out, imgname.replace(path + "{}{}{}".format(os.sep, image, os.sep), "")
                )
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                if isinstance(show, np.ndarray):
                    cv2.imwrite(outname, show)
        print("{}-{}".format("o" if found[left] else "x", "o" if found[right] else "x"))
        if not found[left] and not found[right]:
            visited[left:right] = True
            continue
        mid = (left + right) // 2
        if mid == left or mid == right:
            continue
        if mid - left > min_step:
            proposals.append((left, mid))
        if right - mid > min_step:
            proposals.append((mid, right))


def detect_chessboard_sequence(path, image, out, pattern, gridSize, args):
    t0 = time.perf_counter()
    create_chessboard_templates_classic(
        path,
        image,
        pattern,
        gridSize,
        ext=args.ext,
        overwrite=args.overwrite3d,
        axis=args.axis,
    )
    t_after_templates = time.perf_counter()
    print(
        f"[detect_chessboard] seq mode: templates {t_after_templates - t0:.1f}s; "
        f"max_step={args.max_step} min_step={args.min_step}",
        flush=True,
    )
    subs = sorted(os.listdir(join(path, image)))
    subs = [s for s in subs if os.path.isdir(join(path, image, s))]
    if len(subs) == 0:
        subs = [None]
    from multiprocessing import Process

    tasks = []
    for sub in subs:
        task = Process(target=_detect_by_search, args=(path, image, out, pattern, sub, args))
        task.start()
        tasks.append(task)
    for task in tasks:
        task.join()
    t_after_search = time.perf_counter()
    for sub in subs:
        dataset = ImageFolder(path, sub=sub, annot="chessboard", ext=args.ext)
        dataset.isTmp = False
        count, visited = 0, 0
        for nf in range(len(dataset)):
            imgname, annotname = dataset[nf]
            annots = read_json(annotname)
            if annots["visited"]:
                visited += 1
            if annots["keypoints2d"][0][-1] > 0.01:
                count += 1
        label = _view_label(sub)
        log(f"[detect_chessboard] view {label!r}: found {count:4d}/{visited:4d} visited frames")
    t_total = time.perf_counter() - t0
    print(
        f"[detect_chessboard] seq mode done in {t_total:.1f}s wall time "
        f"(search phase {time.perf_counter() - t_after_templates:.1f}s)",
        flush=True,
    )


def check_chessboard_views(path, out, image_sub="images"):
    subs_notvalid = []
    for sub in sorted(os.listdir(join(path, image_sub))):
        if os.path.exists(join(out, sub)):
            continue
        subs_notvalid.append(sub)
    print(subs_notvalid)
    mywarn("Cannot find chessboard in view {}".format(subs_notvalid))
    mywarn("Please annot them manually:")
    mywarn(
        "python3 apps/annotation/annot_calib.py {} --mode chessboard --annot chessboard --sub {}".format(
            path, " ".join(subs_notvalid)
        )
    )


# Populated in each process by _mp_pool_init (multiprocessing).
_MP_BOARD = None
_MP_DETECTOR = None
_MP_N_CORNERS = 0


def _effective_cpu_count() -> int:
    """CPUs visible to this process (respects Slurm cgroup on Linux when available)."""
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 4


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


def _detect_charuco_worker(
    datas,
    path: str,
    image_sub: str,
    out: str,
    args,
    detector,
    board,
    n_corners: int,
    thread_idx: int,
    per_thread_detected: list[int],
):
    detected = 0
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
        detected += 1
        rel = _rel_under_image_root(imgname, path, image_sub)
        outname = join(out, rel)
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, show)
    per_thread_detected[thread_idx] = detected


def _discover_views(path: str, image_sub: str, ext: str) -> list[str | None]:
    """Return ordered camera/view folder names, or [None] for a flat images/ tree."""
    root = join(path, image_sub)
    if not os.path.isdir(root):
        return [None]
    rels = getFileList(root, ext=ext)
    if not rels:
        return [None]
    views: set[str | None] = set()
    for rel in rels:
        p = Path(rel)
        if len(p.parts) == 1:
            views.add(None)
        else:
            views.add(p.parts[0])
    ordered: list[str | None] = [v for v in sorted(v for v in views if v is not None)]
    if None in views:
        ordered.append(None)
    return ordered if ordered else [None]


def _view_label(view: str | None) -> str:
    return "flat" if view is None else str(view)


def _run_detection_threads(
    dataset: ImageFolder,
    path: str,
    image_sub: str,
    out: str,
    args,
    detector,
    board,
    n_corners: int,
) -> tuple[int, int]:
    """Returns (frames_processed, frames_with_detection)."""
    dataset.isTmp = False
    n = len(dataset)
    if n == 0:
        return 0, 0
    trange = list(range(n))
    per_thread_detected = [0] * args.mp
    threads = []
    for i in range(args.mp):
        ranges = trange[i :: args.mp]
        datas = [dataset[t] for t in ranges]
        t = threading.Thread(
            target=_detect_charuco_worker,
            args=(
                datas,
                path,
                image_sub,
                out,
                args,
                detector,
                board,
                n_corners,
                i,
                per_thread_detected,
            ),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return n, int(sum(per_thread_detected))


def _mp_pool_init(squares_xy: tuple[int, int], marker_ratio: float, dict_name: str, n_corners: int):
    global _MP_BOARD, _MP_DETECTOR, _MP_N_CORNERS
    # One OpenCV thread per process avoids oversubscription when running many workers.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    _MP_BOARD, _MP_DETECTOR = build_board(squares_xy, marker_ratio, dict_name)
    _MP_N_CORNERS = int(n_corners)


def _mp_process_chunk(
    payload: tuple[str, str, str, bool, list[tuple[str, str]]],
) -> tuple[int, int]:
    """
    Process a chunk of (imgname, annotname) pairs in a worker process.
    Returns (detected_count, frame_count).
    """
    path, image_sub, out, debug, chunk = payload
    board = _MP_BOARD
    detector = _MP_DETECTOR
    n_corners = _MP_N_CORNERS
    detected = 0
    for imgname, annotname in chunk:
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
            if debug:
                mywarn(f"[detect_charuco] No ChArUco in {imgname}")
            continue
        detected += 1
        rel = _rel_under_image_root(imgname, path, image_sub)
        outname = join(out, rel)
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, show)
    return detected, len(chunk)


def _split_exactly_n_chunks(pairs: list[tuple[str, str]], n_chunks: int) -> list[list[tuple[str, str]]]:
    """Split pairs into exactly n_chunks non-empty lists (as even as possible)."""
    if not pairs:
        return []
    n_chunks = max(1, min(n_chunks, len(pairs)))
    k, m = divmod(len(pairs), n_chunks)
    out: list[list[tuple[str, str]]] = []
    start = 0
    for i in range(n_chunks):
        end = start + k + (1 if i < m else 0)
        if start < end:
            out.append(pairs[start:end])
        start = end
    return out


def _mp_context():
    """Prefer fork on Linux/HPC (fast, inherits imports). Fall back to spawn if fork is unavailable."""
    if sys.platform == "win32":
        return mp.get_context("spawn")
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


def _run_detection_processes(
    dataset: ImageFolder,
    path: str,
    image_sub: str,
    out: str,
    args,
    n_workers: int,
    squares_xy: tuple[int, int],
    marker_ratio: float,
    dict_name: str,
    n_corners: int,
    view_label: str,
) -> tuple[int, int]:
    """Returns (frames_processed, frames_with_detection)."""
    dataset.isTmp = False
    n = len(dataset)
    if n == 0:
        return 0, 0
    pairs: list[tuple[str, str]] = [dataset[i] for i in range(n)]
    n_workers = max(1, min(n_workers, n, _effective_cpu_count()))
    chunks = _split_exactly_n_chunks(pairs, n_workers)
    payloads = [(path, image_sub, out, args.debug, ch) for ch in chunks]
    ctx = _mp_context()
    initargs = (squares_xy, marker_ratio, dict_name, n_corners)
    n_procs = len(payloads)
    detected_total = 0
    with ctx.Pool(processes=n_procs, initializer=_mp_pool_init, initargs=initargs) as pool:
        for det, _n in tqdm(
            pool.imap_unordered(_mp_process_chunk, payloads, chunksize=1),
            total=len(payloads),
            desc=f"detect charuco [{view_label}]",
            leave=True,
        ):
            detected_total += det
    return n, int(detected_total)


def _resolve_worker_count(workers_flag: int) -> int:
    if workers_flag <= 0:
        return max(1, min(8, _effective_cpu_count()))
    return workers_flag


def detect_charuco_batch(path: str, image_sub: str, out: str, args, board, detector, n_corners: int):
    t_batch0 = time.perf_counter()
    create_charuco_templates(
        path,
        image_sub,
        board_to_keypoints3d(board, args.grid),
        pattern=((args.squares[0] - 1), (args.squares[1] - 1)),
        grid_size=args.grid,
        ext=args.ext,
        overwrite=args.overwrite3d,
    )
    t_after_templates = time.perf_counter()
    views = _discover_views(path, image_sub, args.ext)
    labels = ", ".join(_view_label(v) for v in views)
    if args.workers == 1:
        par_info = f"threads={args.mp} (single process)"
    else:
        n_proc_hint = _resolve_worker_count(args.workers)
        par_info = (
            f"processes<={n_proc_hint} per view (fork pool; "
            f"--workers 1 --mp N for thread-only on macOS/spawn)"
        )
    print(
        f"[detect_charuco] detection pass: {len(views)} view(s) [{labels}] "
        f"(templates {t_after_templates - t_batch0:.1f}s; {par_info})",
        flush=True,
    )
    per_view: list[tuple[str, int, int, float]] = []
    for vi, view in enumerate(views):
        label = _view_label(view)
        t0 = time.perf_counter()
        print(
            f"[detect_charuco] view {vi + 1}/{len(views)} '{label}': start "
            f"(elapsed since batch start {t0 - t_batch0:.1f}s)",
            flush=True,
        )
        if view is None:
            dataset = ImageFolder(path, image=image_sub, annot="chessboard", ext=args.ext)
        else:
            dataset = ImageFolder(path, view, image=image_sub, annot="chessboard", ext=args.ext)
        if args.workers == 1:
            n_frames, n_detected = _run_detection_threads(
                dataset, path, image_sub, out, args, detector, board, n_corners
            )
        else:
            n_proc = _resolve_worker_count(args.workers)
            n_frames, n_detected = _run_detection_processes(
                dataset,
                path,
                image_sub,
                out,
                args,
                n_proc,
                tuple(args.squares),
                float(args.marker_ratio),
                str(args.dict_name),
                n_corners,
                label,
            )
        dt = time.perf_counter() - t0
        per_view.append((label, n_frames, n_detected, dt))
        print(
            f"[detect_charuco] view {vi + 1}/{len(views)} '{label}': done "
            f"ChArUco detected in {n_detected}/{n_frames} frames in {dt:.1f}s",
            flush=True,
        )
    t_total = time.perf_counter() - t_batch0
    print(f"[detect_charuco] all views done in {t_total:.1f}s wall time (including template creation)", flush=True)
    for label, n_frames, n_detected, dt in per_view:
        print(f"  - view {label!r}: {n_detected}/{n_frames} frames with board, detection pass {dt:.1f}s", flush=True)


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
    _ch_profile = BOARD_DEFAULT_PROFILES[ACTIVE_BOARD_PROFILE]

    parser = argparse.ArgumentParser(
        description="Calibration board detection → chessboard/*.json (classical chessboard or ChArUco)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("chessboard", "charuco"),
        default="charuco",
        help="chessboard: findChessboardCorners; charuco: CharucoDetector (default).",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Dataset root, or a single image path (charuco mode only).",
    )
    parser.add_argument("--image", type=str, default="images", help="Image subfolder under dataset root")
    parser.add_argument("--out", type=str, default=None, help="Visualization output root")
    parser.add_argument("--ext", type=str, default=".jpg", choices=[".jpg", ".png"])
    parser.add_argument(
        "--grid",
        type=float,
        default=None,
        help="Square size in meters; if omitted, uses the active profile for the selected mode.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite3d", action="store_true", help="Refresh keypoints3d in existing JSON")
    parser.add_argument(
        "--mp",
        type=int,
        default=4,
        help="Thread count (chessboard batch, or charuco when --workers 1).",
    )

    gcb = parser.add_argument_group("chessboard mode")
    gcb.add_argument(
        "--board-profile",
        type=str,
        default=ACTIVE_CHESSBOARD_PROFILE,
        choices=sorted(CHESSBOARD_BOARD_PROFILES.keys()),
        help="Inner corners + grid preset (used when --grid/--pattern omitted).",
    )
    gcb.add_argument(
        "--pattern",
        type=_parse_pattern_opt,
        default=None,
        help="Inner corners W,H e.g. 7,4. Default: from --board-profile.",
    )
    gcb.add_argument("--max_step", type=int, default=50)
    gcb.add_argument("--min_step", type=int, default=0)
    gcb.add_argument("--axis", type=str, default="yx")
    gcb.add_argument("--fix_orientation", action="store_true")
    gcb.add_argument("--seq", action="store_true", help="Binary search per view (multiprocessing)")
    gcb.add_argument("--check", action="store_true", help="Report views missing viz output")

    gch = parser.add_argument_group("charuco mode")
    gch.add_argument(
        "--preset",
        type=str,
        default=_ch_profile["preset"],
        choices=sorted(BOARD_PRESETS.keys()),
    )
    gch.add_argument(
        "--manual",
        nargs=3,
        metavar=("SX", "SY", "MARKER_RATIO"),
        default=None,
        help="Override preset: squares_x squares_y marker_ratio",
    )
    gch.add_argument(
        "--dict",
        dest="dictionary",
        default=_ch_profile["dictionary"],
        help="cv2.aruco dictionary name (overrides preset)",
    )
    gch.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0=auto processes; 1=threads only (--mp); N=N processes",
    )

    args = parser.parse_args()

    if args.mode == "chessboard":
        prof = CHESSBOARD_BOARD_PROFILES[args.board_profile]
        if args.pattern is None:
            args.pattern = prof["pattern"]
        if args.grid is None:
            args.grid = prof["grid_m"]
        print(
            "[detect_calibration_board] mode=chessboard board_profile={!r} pattern={} grid_m={}".format(
                args.board_profile, args.pattern, args.grid
            )
        )
        if args.out is None:
            args.out = os.path.join(args.path, "output", "calibration")
        if os.path.isfile(os.path.abspath(args.path)):
            raise SystemExit("chessboard mode expects a dataset root, not a single image path.")
        if args.seq:
            detect_chessboard_sequence(
                args.path, args.image, args.out, args.pattern, args.grid, args
            )
        else:
            detect_chessboard_batch(
                args.path, args.image, args.out, args.pattern, args.grid, args
            )
        if args.check:
            check_chessboard_views(args.path, args.out, args.image)
        return

    # --- charuco ---
    data_root, single_image = _normalize_single_image_arg(args.path)
    args.path = data_root
    if single_image is not None:
        args.ext = ".jpg" if single_image.lower().endswith((".jpg", ".jpeg")) else ".png"

    if args.grid is None:
        args.grid = float(_ch_profile["grid_m"])

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

    args.marker_ratio = float(marker_ratio)
    args.dict_name = str(dict_name)

    board, detector = build_board(args.squares, marker_ratio, dict_name)
    n_corners = int(np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3).shape[0])

    print(
        "[detect_calibration_board] mode=charuco preset={!r} squares={} dict={!r} "
        "marker_ratio={:.6f} n_corners={} grid_m={}".format(
            preset_label, args.squares, dict_name, marker_ratio, n_corners, args.grid
        )
    )

    if single_image is not None:
        print(f"[detect_calibration_board] single image: {single_image}")
        detect_charuco_one_file(single_image, args.path, args.out, args, board, detector, n_corners)
    else:
        detect_charuco_batch(args.path, args.image, args.out, args, board, detector, n_corners)


if __name__ == "__main__":
    main()
