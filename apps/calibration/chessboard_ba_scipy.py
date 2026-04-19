"""
Bundle adjustment on chessboard corner observations using scipy (no COLMAP binary required).

Drop-in alternative to chessboard_ba_colmap.py for environments where the COLMAP
binary is not available (e.g. Digital Research Alliance of Canada HPC clusters).
Uses scipy.optimize.least_squares (Trust-Region / Levenberg-Marquardt) with a
sparse Jacobian for efficiency.

Pipeline:
  1) Read camera parameters from intri.yml / extri.yml
  2) Read chessboard/<cam>/*.json corner detections
  3) Build multi-view tracks: each (frame, corner_id) becomes one 3D point
  4) Triangulate initial 3D points from camera pairs
  5) Run sparse bundle adjustment (scipy, no external binary)
  6) Write refined yml files

Inputs:
  - intri.yml / extri.yml
  - chessboard/<cam>/*.json

Outputs:
  - refined intri / extri yaml (intri_scipy_ba.yml, extri_scipy_ba.yml)
  - optimized 3D points npz (output/points_chess_scipy_ba.npz, key 'xyz')
    compatible with: python apps/calibration/vis_chess_sfm_ba.py <path>

Requires: numpy, scipy, opencv-python  (no COLMAP binary needed).
"""

import os
from collections import defaultdict
from glob import glob
from os.path import basename, join

import cv2
import numpy as np

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from easymocap.mytools import read_json
from easymocap.mytools.camera_utils import read_camera, write_extri, write_intri

# Reuse all shared utility functions from the COLMAP variant to avoid duplication.
# Importing does NOT require the colmap binary — only chessboard_ba_colmap.main() does.
from chessboard_ba_colmap import (
    align_world_to_camera,
    compute_reprojection_error,
    parse_keypoints2d,
    resolve_path,
    sample_frames_balanced,
    sample_list,
    select_vis_image_auto,
    triangulate_track,
    undistort_uv,
    visualize_reprojection,
)

# ---------------------------------------------------------------------------
# Sparse bundle adjustment (scipy)
# ---------------------------------------------------------------------------

def run_bundle_adjustment(
    cams_init,
    camnames,
    points_init,
    observations,
    refine_intri=True,
    refine_distortion=False,
    max_iter=500,
    func_tol=1e-6,
):
    """Sparse Levenberg-Marquardt bundle adjustment via scipy.

    observations: list of (cam_idx, pt_idx, u, v, conf)
    Returns: cams_opt (same structure as cams_init), points3d_opt (N, 3)
    """
    try:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
    except ImportError:
        raise SystemExit(
            "[BA] scipy is required. Install with: pip install scipy"
        )

    n_cams = len(camnames)
    n_pts  = points_init.shape[0]
    n_obs  = len(observations)

    # Fixed intrinsics / distortion arrays  shape (n_cams, 4)
    K_fixed    = np.zeros((n_cams, 4), dtype=np.float64)
    dist_fixed = np.zeros((n_cams, 4), dtype=np.float64)
    for ci, c in enumerate(camnames):
        K = cams_init[c]["K"]
        K_fixed[ci] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        d = np.asarray(cams_init[c]["dist"], dtype=np.float64).reshape(-1)
        dist_fixed[ci, : min(4, d.size)] = d[: min(4, d.size)]

    # Parameter block per camera: rvec(3) tvec(3) [intri(4)] [dist(4)]
    cam_param_size = 6 + (4 if refine_intri else 0) + (4 if refine_distortion else 0)
    n_cam_params   = n_cams * cam_param_size

    cam_x0 = np.zeros((n_cams, cam_param_size), dtype=np.float64)
    for ci, c in enumerate(camnames):
        cam_x0[ci, 0:3] = np.asarray(cams_init[c]["Rvec"], dtype=np.float64).reshape(3)
        cam_x0[ci, 3:6] = np.asarray(cams_init[c]["T"],    dtype=np.float64).reshape(3)
        off = 6
        if refine_intri:
            cam_x0[ci, off : off + 4] = K_fixed[ci]
            off += 4
        if refine_distortion:
            cam_x0[ci, off : off + 4] = dist_fixed[ci]

    x0 = np.concatenate([cam_x0.reshape(-1), points_init.reshape(-1)])

    obs_cam = np.array([o[0] for o in observations], dtype=np.int32)
    obs_pt  = np.array([o[1] for o in observations], dtype=np.int32)
    obs_u   = np.array([o[2] for o in observations], dtype=np.float64)
    obs_v   = np.array([o[3] for o in observations], dtype=np.float64)

    def residuals(x):
        cam_x = x[:n_cam_params].reshape(n_cams, cam_param_size)
        pts   = x[n_cam_params:].reshape(n_pts, 3)
        res   = np.empty(2 * n_obs, dtype=np.float64)
        for ci in range(n_cams):
            mask = obs_cam == ci
            if not np.any(mask):
                continue
            rvec = cam_x[ci, 0:3].reshape(3, 1)
            tvec = cam_x[ci, 3:6].reshape(3, 1)
            off  = 6
            if refine_intri:
                fx, fy, cx, cy = cam_x[ci, off : off + 4]
                K   = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                off += 4
            else:
                fx, fy, cx, cy = K_fixed[ci]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            if refine_distortion:
                d4 = cam_x[ci, off : off + 4]
            else:
                d4 = dist_fixed[ci]
            dist = np.array([[d4[0], d4[1], d4[2], d4[3], 0.0]], dtype=np.float64)

            idx  = np.where(mask)[0]
            proj, _ = cv2.projectPoints(pts[obs_pt[idx]], rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            res[idx * 2]     = proj[:, 0] - obs_u[idx]
            res[idx * 2 + 1] = proj[:, 1] - obs_v[idx]
        return res

    # Sparsity: residual pair i depends on one camera block + one point block
    sparsity = lil_matrix((2 * n_obs, len(x0)), dtype=np.int8)
    for i in range(n_obs):
        ci, pi = int(obs_cam[i]), int(obs_pt[i])
        sparsity[2 * i,     ci * cam_param_size : (ci + 1) * cam_param_size] = 1
        sparsity[2 * i + 1, ci * cam_param_size : (ci + 1) * cam_param_size] = 1
        pt_off = n_cam_params + pi * 3
        sparsity[2 * i,     pt_off : pt_off + 3] = 1
        sparsity[2 * i + 1, pt_off : pt_off + 3] = 1

    print(
        f"[scipy-BA] bundle adjustment: n_cams={n_cams} n_pts={n_pts} "
        f"n_obs={n_obs} n_params={len(x0)} "
        f"refine_intri={refine_intri} refine_distortion={refine_distortion}"
    )

    result = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        ftol=func_tol,
        xtol=1e-8,
        gtol=0,
        max_nfev=max_iter * 4,
        verbose=1,
    )

    x_opt   = result.x
    cam_x   = x_opt[:n_cam_params].reshape(n_cams, cam_param_size)
    pts_opt = x_opt[n_cam_params:].reshape(n_pts, 3)

    cams_opt = {}
    for ci, c in enumerate(camnames):
        rvec = cam_x[ci, 0:3].reshape(3, 1)
        tvec = cam_x[ci, 3:6].reshape(3, 1)
        off  = 6
        if refine_intri:
            fx, fy, cx, cy = cam_x[ci, off : off + 4]
            K   = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            off += 4
        else:
            K = cams_init[c]["K"].copy().astype(np.float64)
        if refine_distortion:
            d4   = cam_x[ci, off : off + 4]
            dist = np.array([[d4[0], d4[1], d4[2], d4[3], 0.0]], dtype=np.float64)
        else:
            dist = np.asarray(cams_init[c]["dist"], dtype=np.float64).copy()
        R, _ = cv2.Rodrigues(rvec)
        cams_opt[c] = {
            "K":    K,
            "dist": dist,
            "R":    R.astype(np.float64),
            "Rvec": rvec,
            "T":    tvec,
        }

    return cams_opt, pts_opt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    root       = args.path
    intri_path = resolve_path(root, args.intri)
    extri_path = resolve_path(root, args.extri)
    chess_root = resolve_path(root, args.chess)
    out_extri  = resolve_path(root, args.out_extri)
    out_intri  = resolve_path(root, args.out_intri)
    out_points = resolve_path(root, args.out_points)

    cams     = read_camera(intri_path, extri_path)
    camnames = [c for c in cams["basenames"] if os.path.isdir(join(chess_root, c))]
    cams.pop("basenames")
    if len(camnames) < 2:
        raise RuntimeError("Need >= 2 cameras with chessboard detections.")

    print(f"[scipy-BA] cameras: {camnames}")

    # ------------------------------------------------------------------
    # Build tracks from chessboard detections
    # ------------------------------------------------------------------
    cam_to_map = {}
    all_frames = set()
    for cam in camnames:
        chess_list = sorted(glob(join(chess_root, cam, "*.json")))
        chess_list = sample_list(chess_list, args.step)
        m = {basename(p): p for p in chess_list}
        cam_to_map[cam] = m
        all_frames |= set(m.keys())

    all_frames = sorted(all_frames)
    if not all_frames:
        raise RuntimeError("No chessboard json files found.")

    frame_k2d_cache = {}
    frame_valid_cams = {}
    bad_json = 0
    for fr in all_frames:
        frame_k2d_cache[fr] = {}
        valid_cams = set()
        for cam in camnames:
            p = cam_to_map[cam].get(fr)
            if p is None:
                continue
            data = read_json(p)
            k2d  = parse_keypoints2d(data)
            if k2d is None:
                bad_json += 1
                continue
            if float(np.max(k2d[:, 2])) < args.conf:
                continue
            frame_k2d_cache[fr][cam] = k2d
            valid_cams.add(cam)
        frame_valid_cams[fr] = valid_cams

    candidate_frames = [fr for fr in all_frames if len(frame_valid_cams[fr]) >= args.min_views]
    if not candidate_frames:
        raise RuntimeError("No frames with valid chessboard detections in enough views.")

    if args.max_frames > 0 and len(candidate_frames) > args.max_frames:
        used_frames = sample_frames_balanced(
            candidate_frames=candidate_frames,
            frame_valid_cams=frame_valid_cams,
            camnames=camnames,
            max_frames=args.max_frames,
        )
    else:
        used_frames = candidate_frames

    cam_frame_counts = {
        cam: int(sum(cam in frame_valid_cams[fr] for fr in used_frames)) for cam in camnames
    }
    print(
        f"[scipy-BA] frame selection: total_json_frames={len(all_frames)} "
        f"valid_frames={len(candidate_frames)} used_frames={len(used_frames)} "
        f"(max_frames={args.max_frames}) bad_or_invalid_json={bad_json}"
    )
    print(f"[scipy-BA] frame coverage per camera: {cam_frame_counts}")

    tracks      = []
    kept_frames = 0
    for fr in used_frames:
        frame_k2d = frame_k2d_cache[fr]
        if len(frame_k2d) < args.min_views:
            continue
        kept_frames += 1
        max_pid = max(v.shape[0] for v in frame_k2d.values())
        for pid in range(max_pid):
            obs = []
            for cam, k2d in frame_k2d.items():
                if pid >= k2d.shape[0]:
                    continue
                u, v, conf = float(k2d[pid, 0]), float(k2d[pid, 1]), float(k2d[pid, 2])
                if conf < args.conf:
                    continue
                obs.append((cam, u, v, conf))
            if len(obs) >= args.min_views:
                tracks.append({"frame": fr, "pid": pid, "obs": obs})

    if not tracks:
        raise RuntimeError("No valid tracks with enough observations.")

    if args.max_points > 0 and len(tracks) > args.max_points:
        rng  = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(tracks), size=args.max_points, replace=False))
        tracks = [tracks[i] for i in keep]

    # ------------------------------------------------------------------
    # Triangulate initial 3D points
    # ------------------------------------------------------------------
    cam_centers = {}
    for cam in camnames:
        R = cams[cam]["R"].astype(np.float64)
        T = cams[cam]["T"].astype(np.float64)
        cam_centers[cam] = (-R.T @ T).reshape(3)

    cam_to_idx   = {c: i for i, c in enumerate(camnames)}
    points_init  = []
    observations = []   # (cam_idx, pt_idx, u, v, conf)
    dropped      = 0
    for tr in tracks:
        X = triangulate_track(tr["obs"], cams, cam_centers)
        if X is None:
            dropped += 1
            continue
        pidx = len(points_init)
        points_init.append(X)
        for cam, u, v, conf in tr["obs"]:
            observations.append((cam_to_idx[cam], pidx, float(u), float(v), float(conf)))

    if not points_init:
        raise RuntimeError("Triangulation failed for all tracks.")

    points_init = np.asarray(points_init, dtype=np.float64)
    print(
        f"[scipy-BA] frames_used={len(used_frames)} kept={kept_frames} "
        f"tracks={len(tracks)} triangulated={points_init.shape[0]} dropped={dropped} "
        f"observations={len(observations)}"
    )

    # ------------------------------------------------------------------
    # Reprojection error BEFORE
    # ------------------------------------------------------------------
    errs_before = compute_reprojection_error(cams, camnames, points_init, observations)
    print(
        f"[scipy-BA] reprojection BEFORE: "
        f"mean={errs_before.mean():.3f}px rms={np.sqrt((errs_before**2).mean()):.3f}px "
        f"n={len(errs_before)}"
    )

    # ------------------------------------------------------------------
    # Run scipy bundle adjustment
    # ------------------------------------------------------------------
    cams_opt, points3d_opt = run_bundle_adjustment(
        cams_init=cams,
        camnames=camnames,
        points_init=points_init,
        observations=observations,
        refine_intri=args.refine_intri,
        refine_distortion=args.refine_distortion,
        max_iter=args.max_iter,
        func_tol=args.func_tol,
    )

    # Merge H/W and any keys the optimizer doesn't touch
    for cam in camnames:
        if cam in cams_opt:
            for key in ("H", "W"):
                if key in cams[cam]:
                    cams_opt[cam][key] = cams[cam][key]

    errs_after = compute_reprojection_error(cams_opt, camnames, points3d_opt, observations)
    print(
        f"[scipy-BA] reprojection AFTER : "
        f"mean={errs_after.mean():.3f}px rms={np.sqrt((errs_after**2).mean()):.3f}px "
        f"n={len(errs_after)}"
    )

    # ------------------------------------------------------------------
    # Align world so origin_cam is at origin
    # ------------------------------------------------------------------
    if args.origin_cam:
        if args.origin_cam not in camnames:
            raise RuntimeError(
                f"--origin_cam '{args.origin_cam}' not in cameras. Available: {camnames}"
            )
        cams_opt, points3d_opt = align_world_to_camera(
            cams_opt, points3d_opt, args.origin_cam
        )
        print(f"[scipy-BA] aligned world to camera '{args.origin_cam}' (origin)")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    write_intri(out_intri, cams_opt)
    write_extri(out_extri, cams_opt)
    os.makedirs(os.path.dirname(out_points) or ".", exist_ok=True)
    np.savez_compressed(out_points, xyz=points3d_opt)
    print(f"[scipy-BA] wrote intrinsics : {out_intri}")
    print(f"[scipy-BA] wrote extrinsics : {out_extri}")
    print(f"[scipy-BA] wrote points3d   : {out_points} (xyz, N={points3d_opt.shape[0]})")
    print(f"[scipy-BA] visualize: python apps/calibration/vis_chess_sfm_ba.py {root}")

    # ------------------------------------------------------------------
    # Optional: visualize reprojection on a selected image
    # ------------------------------------------------------------------
    if args.vis_image:
        vis_cam, vis_frame = None, None
        if args.vis_image.strip().lower() == "auto":
            vis_cam, vis_frame, n_vis, mean_e0 = select_vis_image_auto(
                observations, tracks, camnames, errs_before
            )
            if vis_cam is None or vis_frame is None:
                print("[vis] No observations to visualize.")
            else:
                print(
                    f"[vis] Auto-selected worst pre-BA reproj view: cam={vis_cam} "
                    f"frame={vis_frame} (n={n_vis}, mean err before BA={mean_e0:.3f}px)"
                )
        else:
            parts = args.vis_image.replace(",", " ").split()
            if len(parts) != 2:
                print(
                    f"[vis] --vis_image expects 'auto' or 'cam,frame' e.g. '01,000000'. "
                    f"Got: {args.vis_image}"
                )
            else:
                vis_cam, vis_frame = parts[0].strip(), parts[1].strip()
                if vis_frame.endswith(".json"):
                    vis_frame = vis_frame[:-5]
                used_frames_norm = {f[:-5] if f.endswith(".json") else f for f in used_frames}
                if vis_frame not in used_frames_norm:
                    print(
                        f"[vis] Frame '{vis_frame}' not in used_frames. "
                        f"First/last: {used_frames[0]}, {used_frames[-1]}"
                    )
                    vis_cam, vis_frame = None, None

        if vis_cam is not None and vis_frame is not None:
            try:
                visualize_reprojection(
                    root=root,
                    cam=vis_cam,
                    frame=vis_frame,
                    cams=cams,
                    cams_opt=cams_opt,
                    points_init=points_init,
                    points3d_opt=points3d_opt,
                    observations=observations,
                    tracks=tracks,
                    camnames=camnames,
                    out_path=resolve_path(root, args.vis_out) if args.vis_out else None,
                )
            except Exception as e:
                print(f"[vis] Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bundle adjustment on chessboard corners (scipy, no COLMAP binary required)."
    )
    parser.add_argument("path", type=str, help="dataset root")
    parser.add_argument("--intri",      type=str, default="intri.yml")
    parser.add_argument("--extri",      type=str, default="extri.yml")
    parser.add_argument("--chess",      type=str, default="chessboard")
    parser.add_argument("--out_intri",  type=str, default="intri_scipy_ba.yml")
    parser.add_argument("--out_extri",  type=str, default="extri_scipy_ba.yml")
    parser.add_argument("--out_points", type=str, default="output/points_chess_scipy_ba.npz")

    parser.add_argument(
        "--origin_cam", type=str, default="",
        help="camera to use as world origin (identity R, zero T); '' to keep extri frame",
    )

    parser.add_argument("--conf",       type=float, default=0.1,  help="min keypoint confidence")
    parser.add_argument("--min_views",  type=int,   default=2,    help="min cameras per track")
    parser.add_argument("--step",       type=int,   default=1,    help="sample frames by step")
    parser.add_argument("--max_frames", type=int,   default=300,  help="limit number of frames")
    parser.add_argument("--max_points", type=int,   default=-1,   help="cap tracks (-1 for all)")
    parser.add_argument("--seed",       type=int,   default=42)

    parser.add_argument("--refine_intri",    dest="refine_intri", action="store_true")
    parser.add_argument("--no-refine_intri", dest="refine_intri", action="store_false")
    parser.add_argument(
        "--refine_distortion", action="store_true",
        help="also refine k1,k2,p1,p2; default keeps input distortion fixed",
    )
    parser.add_argument("--max_iter",  type=int,   default=500,  help="BA max iterations")
    parser.add_argument("--func_tol",  type=float, default=1e-6, help="BA function tolerance")

    parser.add_argument(
        "--vis_image", nargs="?", const="auto", default="auto", metavar="cam,frame",
        help="reprojection vis: 'auto' picks highest pre-BA error view; or pass 'cam,frame'",
    )
    parser.add_argument(
        "--no-vis_image", dest="vis_image", action="store_const", const="",
        help="skip reprojection visualization",
    )
    parser.add_argument(
        "--vis_out", type=str, default="",
        help="output path for reprojection vis (default: output/reproj_vis_<cam>_<frame>.jpg)",
    )

    parser.set_defaults(refine_intri=True)
    main(parser.parse_args())
