"""
File: apps/calibration/refine_extri_global.py

Global refinement (bundle adjustment) for multi-camera extrinsics.

Input:
  - intri.yml (OpenCV FileStorage YAML, like your example)
  - extri.yml (OpenCV FileStorage YAML, like your example)
  - chessboard/<cam>/*.json (keypoints3d, keypoints2d [x,y,conf])

Output:
  - extri_refined.yml (OpenCV FileStorage YAML, same style)

Notes:
  - Does NOT require frames visible in all cameras.
  - Uses frames where >=2 cameras have valid detections.
  - Fixes cam0 as the world to remove gauge freedom.
  - Keeps intrinsics fixed.
"""

import os
import json
from glob import glob
from os.path import join
import numpy as np
import cv2

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


# ------------------ IO: OpenCV YAML ------------------

def _fs_read_mat(fs: cv2.FileStorage, key: str):
    node = fs.getNode(key)
    if node.empty():
        return None
    return node.mat()

def read_intri_opencv_yaml(path: str):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)

    names_node = fs.getNode("names")
    names = []
    for i in range(names_node.size()):
        names.append(names_node.at(i).string())

    intri = {}
    for cam in names:
        K = _fs_read_mat(fs, f"K_{cam}")
        dist = _fs_read_mat(fs, f"dist_{cam}")
        if K is None or dist is None:
            raise RuntimeError(f"Missing intrinsics for cam {cam} in {path}")
        intri[cam] = {"K": K, "dist": dist}
    fs.release()
    return names, intri

def read_extri_opencv_yaml(path: str):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)

    names_node = fs.getNode("names")
    names = []
    for i in range(names_node.size()):
        names.append(names_node.at(i).string())

    extri = {}
    for cam in names:
        rvec = _fs_read_mat(fs, f"R_{cam}")       # (3,1)
        R = _fs_read_mat(fs, f"Rot_{cam}")        # (3,3)
        t = _fs_read_mat(fs, f"T_{cam}")          # (3,1)
        if rvec is None and R is not None:
            rvec, _ = cv2.Rodrigues(R)
        if R is None and rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
        if rvec is None or R is None or t is None:
            raise RuntimeError(f"Missing extrinsics for cam {cam} in {path}")
        extri[cam] = {"Rvec": rvec, "R": R, "T": t}
    fs.release()
    return names, extri

def write_extri_opencv_yaml(path: str, names: list, extri: dict):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open for write: {path}")

    fs.write("names", names)
    for cam in names:
        R = extri[cam]["R"]
        T = extri[cam]["T"]
        rvec, _ = cv2.Rodrigues(R)
        fs.write(f"R_{cam}", rvec)
        fs.write(f"Rot_{cam}", R)
        fs.write(f"T_{cam}", T)
    fs.release()


# ------------------ helpers ------------------

def sample_list(lst, step):
    return lst if step <= 1 else lst[::step]

def rt_inv(R, t):
    Rinv = R.T
    tinv = -Rinv @ t
    return Rinv, tinv

def rt_compose(R2, t2, R1, t1):
    return R2 @ R1, R2 @ t1 + t2

def board2cam_to_cam0_to_cam(extri_board2cam: dict, cam0: str, cam: str):
    """
    Convert board->cam and board->cam0 to cam0->cam
    Xc = Rc*Xb + tc
    X0 = R0*Xb + t0
    => Xc = (Rc*R0^T)*X0 + (tc - (Rc*R0^T)*t0)
    """
    R0 = extri_board2cam[cam0]["R"]
    t0 = extri_board2cam[cam0]["T"]
    Rc = extri_board2cam[cam]["R"]
    tc = extri_board2cam[cam]["T"]
    R_c0 = Rc @ R0.T
    t_c0 = tc - R_c0 @ t0
    return R_c0, t_c0

def project_points(K, dist, R_c0, t_c0, R_b0, t_b0, X_b):
    # X0 = R_b0*X + t_b0
    X0 = (R_b0 @ X_b.T + t_b0).T
    rvec_c0, _ = cv2.Rodrigues(R_c0)
    img, _ = cv2.projectPoints(X0, rvec_c0, t_c0, K, dist)
    return img.reshape(-1, 2)


def print_view_histogram(obs, prefix="[BA]"):
    """
    obs: list of frame_obs dict(cam -> (X,u))
    Prints:
      N frames can be seen in K views
    """
    counts = {}
    for frame_obs in obs:
        nv = len(frame_obs)
        counts[nv] = counts.get(nv, 0) + 1

    # Print in descending #views for readability
    for nv in sorted(counts.keys(), reverse=True):
        print(f"{prefix} {counts[nv]} frames can be seen in {nv} views at the same time")

def compute_reprojection_stats(residuals_1d: np.ndarray):
    """
    residuals_1d: stacked residual vector [dx0, dy0, dx1, dy1, ...]
    Returns mean L2 pixel error, RMS pixel error, and number of points.
    """
    r = residuals_1d.reshape(-1, 2)
    per_pt = np.linalg.norm(r, axis=1)  # pixel L2 per point
    mean = float(per_pt.mean()) if per_pt.size > 0 else 0.0
    rms = float(np.sqrt((per_pt ** 2).mean())) if per_pt.size > 0 else 0.0
    return mean, rms, int(per_pt.size)


def pack_rt(rvec, tvec):
    return np.hstack([rvec.reshape(3), tvec.reshape(3)])

def unpack_rt(x):
    rvec = x[:3].reshape(3, 1)
    tvec = x[3:6].reshape(3, 1)
    return rvec, tvec


def read_chess_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    X = np.array(data["keypoints3d"], dtype=np.float64)
    u = np.array(data["keypoints2d"], dtype=np.float64)
    if X.shape[0] != u.shape[0]:
        L = min(X.shape[0], u.shape[0])
        X = X[:L]
        u = u[:L]
    valid = u[:, 2] > 0
    X = X[valid]
    u = u[valid, :2]
    return X, u


# ------------------ BA core ------------------

def global_refine(
    root: str,
    intri_path: str,
    extri_path: str,
    chess_dir: str = "chessboard",
    step: int = 6,
    max_frames: int = -1,
    loss: str = "huber",
    f_scale: float = 3.0,
    out_path: str = "extri_refined.yml",
):
    if least_squares is None:
        raise ImportError("scipy is required: pip install scipy")

    names_intri, intri = read_intri_opencv_yaml(intri_path)
    names_extri, extri_init = read_extri_opencv_yaml(extri_path)

    # Use camera order from extri.yml (usually consistent)
    camnames = names_extri
    assert len(camnames) >= 2, "Need >= 2 cameras"

    cam0 = camnames[0]

    # ---- build union frame set ----
    cam_to_map = {}
    all_names = set()
    for cam in camnames:
        paths = sorted(glob(join(root, chess_dir, cam, "*.json")))
        paths = sample_list(paths, step)
        cmap = {os.path.basename(p): p for p in paths}
        cam_to_map[cam] = cmap
        all_names |= set(cmap.keys())

    all_names = sorted(all_names)
    if max_frames and max_frames > 0:
        all_names = all_names[:max_frames]

    # ---- build sparse observations, keep frames with >=2 cams ----
    obs = []          # list of dict(cam -> (X,u))
    kept = []
    drop = 0
    total_points = 0

    for name in all_names:
        frame_obs = {}
        for cam in camnames:
            p = cam_to_map[cam].get(name, None)
            if p is None:
                continue
            try:
                X, u = read_chess_json(p)
            except Exception as e:
                continue
            if X.shape[0] < 4:
                continue
            frame_obs[cam] = (X, u)
            total_points += X.shape[0]

        if len(frame_obs) < 2:
            drop += 1
            continue

        obs.append(frame_obs)
        kept.append(name)

    if len(obs) == 0:
        raise RuntimeError("No frames with >=2 cameras see the board; cannot BA.")

    print(f"[BA] cameras={len(camnames)} frames_kept={len(obs)} dropped={drop} points={total_points}")
    print_view_histogram(obs, prefix="[BA]")
    avg_views = np.mean([len(o) for o in obs])
    print(f"[BA] avg views per kept frame: {avg_views:.2f}")
    
    # ---- init cam0->cam from board->cam extri_init ----
    cam_rts_init = {cam0: (np.eye(3), np.zeros((3, 1), dtype=np.float64))}
    for cam in camnames[1:]:
        R_c0, t_c0 = board2cam_to_cam0_to_cam(extri_init, cam0, cam)
        cam_rts_init[cam] = (R_c0.astype(np.float64), t_c0.astype(np.float64))

    # ---- init per-frame board->cam0 via PnP (prefer cam0) ----
    board_rts_init = []
    for frame_obs in obs:
        if cam0 in frame_obs:
            X, u = frame_obs[cam0]
            ok, rvec, tvec = cv2.solvePnP(
                X.astype(np.float32), u.astype(np.float32),
                intri[cam0]["K"], intri[cam0]["dist"],
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            Rb, _ = cv2.Rodrigues(rvec)
            board_rts_init.append((Rb.astype(np.float64), tvec.astype(np.float64)))
        else:
            # fallback: use any cam then map into cam0
            cam_any = next(iter(frame_obs.keys()))
            X, u = frame_obs[cam_any]
            ok, rvec, tvec = cv2.solvePnP(
                X.astype(np.float32), u.astype(np.float32),
                intri[cam_any]["K"], intri[cam_any]["dist"],
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            Rb_cam, _ = cv2.Rodrigues(rvec)
            tb_cam = tvec.astype(np.float64)

            R_c0, t_c0 = cam_rts_init[cam_any]
            R_0c, t_0c = rt_inv(R_c0, t_c0)
            Rb_0, tb_0 = rt_compose(R_0c, t_0c, Rb_cam.astype(np.float64), tb_cam)
            board_rts_init.append((Rb_0, tb_0))

    # ---- pack variables: cams(1..)+frames ----
    cam_order = camnames[1:]
    x0 = []
    for cam in cam_order:
        R, t = cam_rts_init[cam]
        rvec, _ = cv2.Rodrigues(R)
        x0.append(pack_rt(rvec, t))
    for (Rb, tb) in board_rts_init:
        rvec, _ = cv2.Rodrigues(Rb)
        x0.append(pack_rt(rvec, tb))
    x0 = np.concatenate(x0, axis=0)

    # ---- residual function ----
    def fun(x):
        idx = 0
        cam_rts = {cam0: (np.eye(3), np.zeros((3, 1), dtype=np.float64))}
        for cam in cam_order:
            rvec, tvec = unpack_rt(x[idx:idx+6])
            R, _ = cv2.Rodrigues(rvec)
            cam_rts[cam] = (R, tvec)
            idx += 6

        board_rts = []
        for _ in range(len(obs)):
            rvec, tvec = unpack_rt(x[idx:idx+6])
            Rb, _ = cv2.Rodrigues(rvec)
            board_rts.append((Rb, tvec))
            idx += 6

        res = []
        for f, frame_obs in enumerate(obs):
            Rb0, tb0 = board_rts[f]
            for cam, (X, u) in frame_obs.items():
                if cam == cam0:
                    R_c0 = np.eye(3)
                    t_c0 = np.zeros((3, 1), dtype=np.float64)
                else:
                    R_c0, t_c0 = cam_rts[cam]
                uhat = project_points(intri[cam]["K"], intri[cam]["dist"], R_c0, t_c0, Rb0, tb0, X)
                res.append((uhat - u).reshape(-1))

        return np.concatenate(res, axis=0)

    # --- reprojection stats BEFORE optimization ---
    res0 = fun(x0)
    mean0, rms0, npts0 = compute_reprojection_stats(res0)
    print(f"[BA] reprojection BEFORE: mean={mean0:.3f}px  rms={rms0:.3f}px  points={npts0}")

    print(f"[BA] optimize: cams={len(cam_order)} (cam0 fixed), frames={len(obs)}, loss={loss}, f_scale={f_scale}")
    result = least_squares(fun, x0, method="trf", loss=loss, f_scale=f_scale, verbose=2)
    print(f"[BA] success={result.success} cost={result.cost:.3f} msg={result.message}")
    # --- reprojection stats AFTER optimization ---
    res1 = fun(result.x)
    mean1, rms1, npts1 = compute_reprojection_stats(res1)
    print(f"[BA] reprojection AFTER : mean={mean1:.3f}px  rms={rms1:.3f}px  points={npts1}")


    # ---- unpack optimized ----
    x = result.x
    idx = 0
    cam_rts_opt = {cam0: (np.eye(3), np.zeros((3, 1), dtype=np.float64))}
    for cam in cam_order:
        rvec, tvec = unpack_rt(x[idx:idx+6])
        R, _ = cv2.Rodrigues(rvec)
        cam_rts_opt[cam] = (R, tvec)
        idx += 6

    board_rts_opt = []
    for _ in range(len(obs)):
        rvec, tvec = unpack_rt(x[idx:idx+6])
        Rb, _ = cv2.Rodrigues(rvec)
        board_rts_opt.append((Rb, tvec))
        idx += 6

    # Reference board pose = first kept frame (consistent with “board/world” in extri.yml)
    Rb_ref, tb_ref = board_rts_opt[0]

    refined = {}
    for cam in camnames:
        R_c0, t_c0 = cam_rts_opt[cam]  # cam0->cam
        R_bc, t_bc = rt_compose(R_c0, t_c0, Rb_ref, tb_ref)  # board->cam
        refined[cam] = {"R": R_bc, "T": t_bc}
        center = -R_bc.T @ t_bc
        print(f"[BA] {cam} center => {center.squeeze()}")

    out_full = out_path if out_path.startswith("/") else join(root, out_path)
    write_extri_opencv_yaml(out_full, camnames, refined)
    print(f"[BA] wrote: {out_full}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="root folder (contains chessboard/)")
    parser.add_argument("--intri", type=str, required=True, help="intri.yml (OpenCV YAML)")
    parser.add_argument("--extri", type=str, required=True, help="extri.yml (OpenCV YAML)")
    parser.add_argument("--chess", type=str, default="chessboard", help="chessboard folder name")
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--loss", type=str, default="huber",
                        choices=["linear", "huber", "soft_l1", "cauchy", "arctan"])
    parser.add_argument("--f_scale", type=float, default=3.0)
    parser.add_argument("--out", type=str, default="extri_refined.yml")
    args = parser.parse_args()

    global_refine(
        root=args.path,
        intri_path=args.intri,
        extri_path=args.extri,
        chess_dir=args.chess,
        step=args.step,
        max_frames=args.max_frames,
        loss=args.loss,
        f_scale=args.f_scale,
        out_path=args.out,
    )