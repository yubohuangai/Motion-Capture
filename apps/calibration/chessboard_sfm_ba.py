"""
COLMAP-style SfM bundle adjustment for multi-camera chessboard captures.

This script treats each (frame, corner_id) as one 3D point track observed by
multiple cameras at the same timestamp:
  1) Build observations from chessboard/<cam>/*.json
  2) (Optional) RANSAC filtering for each track
  3) Initialize 3D points by pair triangulation with current cameras
  3) Jointly optimize camera extrinsics + 3D points by reprojection BA

Inputs:
  - intri.yml
  - extri.yml
  - chessboard/<cam>/*.json

Outputs:
  - refined extrinsics yaml
  - optimized 3D points npz
"""

import os
from glob import glob
from os.path import basename, join

import cv2
import numpy as np

from easymocap.mytools import read_json
from easymocap.mytools.camera_utils import read_camera, write_extri, write_intri

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None

try:
    from scipy.sparse import lil_matrix
except Exception:
    lil_matrix = None


def sample_list(lst, step):
    if step <= 1:
        return lst
    return lst[::step]


def resolve_path(root, path_or_name):
    if os.path.isabs(path_or_name):
        return path_or_name
    return join(root, path_or_name)


def pack_rt(rvec, tvec):
    return np.hstack([rvec.reshape(3), tvec.reshape(3)])


def unpack_rt(x):
    rvec = x[:3].reshape(3, 1)
    tvec = x[3:6].reshape(3, 1)
    return rvec, tvec


def undistort_uv(uv, K, dist):
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    return out


def triangulate_pair(c0, c1, uv0, uv1, cams):
    cam0 = cams[c0]
    cam1 = cams[c1]
    uv0_u = undistort_uv(np.asarray([uv0], dtype=np.float64), cam0["K"], cam0["dist"])[0]
    uv1_u = undistort_uv(np.asarray([uv1], dtype=np.float64), cam1["K"], cam1["dist"])[0]

    P0 = np.hstack([cam0["R"], cam0["T"]]).astype(np.float64)
    P1 = np.hstack([cam1["R"], cam1["T"]]).astype(np.float64)
    Xh = cv2.triangulatePoints(P0, P1, uv0_u.reshape(2, 1), uv1_u.reshape(2, 1)).reshape(4)
    if abs(float(Xh[3])) < 1e-12:
        return None
    X = (Xh[:3] / Xh[3]).reshape(3)

    z0 = (cam0["R"] @ X.reshape(3, 1) + cam0["T"])[2, 0]
    z1 = (cam1["R"] @ X.reshape(3, 1) + cam1["T"])[2, 0]
    if z0 <= 1e-6 or z1 <= 1e-6:
        return None
    return X


def project_one(cam, X):
    img, _ = cv2.projectPoints(
        X.reshape(1, 3),
        cam["Rvec"],
        cam["T"],
        cam["K"],
        cam["dist"],
    )
    return img.reshape(2)


def triangulate_track(track_obs, cams, cam_centers):
    # track_obs: list[(cam_name, u, v, w)]
    n = len(track_obs)
    if n < 2:
        return None

    best = None
    best_score = 1e18
    for i in range(n):
        ci, ui, vi, _ = track_obs[i]
        for j in range(i + 1, n):
            cj, uj, vj, _ = track_obs[j]
            baseline = np.linalg.norm(cam_centers[ci] - cam_centers[cj])
            if baseline < 1e-6:
                continue
            X = triangulate_pair(ci, cj, (ui, vi), (uj, vj), cams)
            if X is None:
                continue

            errs = []
            for c, u, v, _w in track_obs:
                uv_hat = project_one(cams[c], X)
                errs.append(np.linalg.norm(uv_hat - np.array([u, v], dtype=np.float64)))
            score = float(np.mean(errs))
            score = score / baseline
            if score < best_score:
                best_score = score
                best = X
    return best


def triangulate_track_ransac(track_obs, cams, cam_centers, reproj_thresh, min_inliers):
    # track_obs: list[(cam_name, u, v, w)]
    n = len(track_obs)
    if n < 2:
        return None, []

    # Generate candidate 3D points from all camera pairs.
    candidates = []
    for i in range(n):
        ci, ui, vi, _ = track_obs[i]
        for j in range(i + 1, n):
            cj, uj, vj, _ = track_obs[j]
            baseline = np.linalg.norm(cam_centers[ci] - cam_centers[cj])
            if baseline < 1e-6:
                continue
            X = triangulate_pair(ci, cj, (ui, vi), (uj, vj), cams)
            if X is None:
                continue
            candidates.append(X)

    if len(candidates) == 0:
        return None, []

    best_X = None
    best_inliers = []
    best_err = 1e18

    for X in candidates:
        inliers = []
        errs = []
        for k, (cam, u, v, _w) in enumerate(track_obs):
            uv_hat = project_one(cams[cam], X)
            err = float(np.linalg.norm(uv_hat - np.array([u, v], dtype=np.float64)))
            if err <= reproj_thresh:
                inliers.append(k)
                errs.append(err)

        if len(inliers) < min_inliers:
            continue

        mean_err = float(np.mean(errs)) if len(errs) > 0 else 1e18
        if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and mean_err < best_err):
            best_X = X
            best_inliers = inliers
            best_err = mean_err

    if best_X is None:
        return None, []

    # Refine using only inliers by the existing baseline-weighted strategy.
    inlier_obs = [track_obs[i] for i in best_inliers]
    X_refined = triangulate_track(inlier_obs, cams, cam_centers)
    if X_refined is None:
        X_refined = best_X
    return X_refined, best_inliers


def compute_reprojection_stats(residuals_1d):
    r = residuals_1d.reshape(-1, 2)
    per_pt = np.linalg.norm(r, axis=1)
    mean = float(per_pt.mean()) if per_pt.size > 0 else 0.0
    rms = float(np.sqrt((per_pt ** 2).mean())) if per_pt.size > 0 else 0.0
    return mean, rms, int(per_pt.size)


def make_jacobian_sparsity(n_rows, n_cols):
    if lil_matrix is None:
        return None
    A = lil_matrix((n_rows, n_cols), dtype=np.uint8)
    return A


def pack_intri(K, dist, refine_dist):
    params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
    if refine_dist:
        d = np.asarray(dist, dtype=np.float64).reshape(-1)
        k1 = float(d[0]) if d.size > 0 else 0.0
        k2 = float(d[1]) if d.size > 1 else 0.0
        params += [k1, k2]
    return np.array(params, dtype=np.float64)


def unpack_intri(params, base_dist, refine_dist):
    fx, fy, cx, cy = [float(v) for v in params[:4]]
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    d = np.asarray(base_dist, dtype=np.float64).reshape(-1)
    if d.size == 0:
        d = np.zeros((5,), dtype=np.float64)
    if refine_dist:
        if d.size < 2:
            dd = np.zeros((max(2, d.size),), dtype=np.float64)
            dd[: d.size] = d
            d = dd
        d[0] = float(params[4])
        d[1] = float(params[5])
    return K, d.reshape(1, -1)


def run_ba(
    cams,
    camnames,
    ref_cam,
    points_init,
    observations,
    loss,
    f_scale,
    max_nfev,
    cam_sigma_r,
    cam_sigma_t,
    refine_intri,
    refine_dist,
    intri_sigma_f,
    intri_sigma_c,
    dist_sigma,
):
    # observations: list[(cam_idx, point_idx, u, v, w)]
    cam_to_idx = {c: i for i, c in enumerate(camnames)}
    opt_cams = [c for c in camnames if c != ref_cam]
    n_cam_opt = len(opt_cams)
    n_pts = points_init.shape[0]
    n_obs = len(observations)

    if n_pts == 0 or n_obs == 0:
        raise RuntimeError("No points/observations for BA.")

    cam_init = {}
    x_cam = []
    for cam in opt_cams:
        rvec = cams[cam]["Rvec"].astype(np.float64)
        tvec = cams[cam]["T"].astype(np.float64)
        cam_init[cam] = np.hstack([rvec.reshape(3), tvec.reshape(3)])
        x_cam.append(pack_rt(rvec, tvec))
    x_cam = np.concatenate(x_cam, axis=0) if len(x_cam) > 0 else np.zeros((0,), dtype=np.float64)

    intri_opt_cams = list(camnames) if refine_intri else []
    intri_dim = 4 + (2 if refine_dist else 0)
    intri_init = {}
    x_intri = []
    for cam in intri_opt_cams:
        p = pack_intri(cams[cam]["K"], cams[cam]["dist"], refine_dist=refine_dist)
        intri_init[cam] = p.copy()
        x_intri.append(p)
    x_intri = np.concatenate(x_intri, axis=0) if len(x_intri) > 0 else np.zeros((0,), dtype=np.float64)

    x0 = np.concatenate([x_cam, x_intri, points_init.reshape(-1)], axis=0)

    obs_cam = np.array([o[0] for o in observations], dtype=np.int32)
    obs_pt = np.array([o[1] for o in observations], dtype=np.int32)
    obs_uv = np.array([[o[2], o[3]] for o in observations], dtype=np.float64)
    obs_w = np.array([o[4] for o in observations], dtype=np.float64)

    cam_param_offset = 0
    intri_param_offset = n_cam_opt * 6
    pt_param_offset = intri_param_offset + len(intri_opt_cams) * intri_dim
    opt_cam_to_local = {cam_to_idx[c]: i for i, c in enumerate(opt_cams)}
    opt_intri_to_local = {cam_to_idx[c]: i for i, c in enumerate(intri_opt_cams)}

    # Jacobian sparsity
    n_prior_rows = n_cam_opt * 6 + len(intri_opt_cams) * intri_dim
    n_rows = 2 * n_obs + n_prior_rows
    n_cols = n_cam_opt * 6 + len(intri_opt_cams) * intri_dim + n_pts * 3
    J = make_jacobian_sparsity(n_rows, n_cols)
    if J is not None:
        for i in range(n_obs):
            r0 = 2 * i
            r1 = r0 + 1
            c_global = int(obs_cam[i])
            if c_global in opt_cam_to_local:
                c_local = opt_cam_to_local[c_global]
                cb = cam_param_offset + c_local * 6
                J[r0, cb : cb + 6] = 1
                J[r1, cb : cb + 6] = 1
            if c_global in opt_intri_to_local:
                i_local = opt_intri_to_local[c_global]
                ib = intri_param_offset + i_local * intri_dim
                J[r0, ib : ib + intri_dim] = 1
                J[r1, ib : ib + intri_dim] = 1
            pb = pt_param_offset + int(obs_pt[i]) * 3
            J[r0, pb : pb + 3] = 1
            J[r1, pb : pb + 3] = 1

        rb = 2 * n_obs
        for i in range(n_cam_opt * 6):
            J[rb + i, i] = 1
        rb += n_cam_opt * 6
        for i in range(len(intri_opt_cams) * intri_dim):
            J[rb + i, intri_param_offset + i] = 1

    def unpack_all(x):
        cam_rt = {}
        idx = 0
        for cam in opt_cams:
            rvec, tvec = unpack_rt(x[idx : idx + 6])
            R, _ = cv2.Rodrigues(rvec)
            cam_rt[cam] = {
                "Rvec": rvec,
                "R": R,
                "T": tvec,
            }
            idx += 6
        intri_opt = {}
        for cam in intri_opt_cams:
            p = x[idx : idx + intri_dim]
            K, dist = unpack_intri(p, cams[cam]["dist"], refine_dist=refine_dist)
            intri_opt[cam] = {"K": K, "dist": dist}
            idx += intri_dim
        points = x[pt_param_offset:].reshape(-1, 3)
        return cam_rt, intri_opt, points

    def fun(x):
        cam_rt, intri_opt, points = unpack_all(x)

        res = []
        for i in range(n_obs):
            cidx = int(obs_cam[i])
            pidx = int(obs_pt[i])
            u, v = obs_uv[i]
            w = float(obs_w[i])
            cam_name = camnames[cidx]

            if cam_name == ref_cam:
                rvec = cams[cam_name]["Rvec"]
                tvec = cams[cam_name]["T"]
            else:
                rvec = cam_rt[cam_name]["Rvec"]
                tvec = cam_rt[cam_name]["T"]
            if cam_name in intri_opt:
                K_use = intri_opt[cam_name]["K"]
                dist_use = intri_opt[cam_name]["dist"]
            else:
                K_use = cams[cam_name]["K"]
                dist_use = cams[cam_name]["dist"]

            uv_hat, _ = cv2.projectPoints(
                points[pidx].reshape(1, 3),
                rvec,
                tvec,
                K_use,
                dist_use,
            )
            duv = (uv_hat.reshape(2) - np.array([u, v], dtype=np.float64)) * np.sqrt(max(w, 1e-6))
            res.append(duv)

        # camera parameter priors to keep solution stable
        sigma_r = max(float(cam_sigma_r), 1e-8)
        sigma_t = max(float(cam_sigma_t), 1e-8)
        for cam in opt_cams:
            p0 = cam_init[cam]
            p = np.hstack([cam_rt[cam]["Rvec"].reshape(3), cam_rt[cam]["T"].reshape(3)])
            dp = p - p0
            prior = np.hstack([dp[:3] / sigma_r, dp[3:] / sigma_t])
            res.append(prior)

        # intrinsics priors (keep close to calibration result)
        sigma_f = max(float(intri_sigma_f), 1e-8)
        sigma_c = max(float(intri_sigma_c), 1e-8)
        sigma_d = max(float(dist_sigma), 1e-8)
        for cam in intri_opt_cams:
            p0 = intri_init[cam]
            p = pack_intri(intri_opt[cam]["K"], intri_opt[cam]["dist"], refine_dist=refine_dist)
            dp = p - p0
            prior = [dp[0] / sigma_f, dp[1] / sigma_f, dp[2] / sigma_c, dp[3] / sigma_c]
            if refine_dist:
                prior += [dp[4] / sigma_d, dp[5] / sigma_d]
            res.append(np.asarray(prior, dtype=np.float64))

        return np.concatenate(res, axis=0)

    print(
        f"[SFM-BA] start: cams={len(camnames)} ref={ref_cam} "
        f"cams_opt={n_cam_opt} intri_opt={len(intri_opt_cams)} "
        f"points={n_pts} observations={n_obs}"
    )
    res0 = fun(x0)
    m0, r0, n0 = compute_reprojection_stats(res0[: 2 * n_obs])
    print(f"[SFM-BA] reprojection BEFORE: mean={m0:.3f}px rms={r0:.3f}px points={n0}")

    use_sparse = J is not None
    if use_sparse:
        print("[SFM-BA] using sparse Jacobian")
    else:
        print("[SFM-BA] sparse Jacobian unavailable, fallback to dense FD")

    result = least_squares(
        fun,
        x0,
        method="trf",
        jac_sparsity=J if use_sparse else None,
        loss=loss,
        f_scale=float(f_scale),
        max_nfev=int(max_nfev),
        verbose=2,
    )
    print(f"[SFM-BA] done: success={result.success}, cost={result.cost:.4f}")

    cam_rt_opt, intri_opt, points_opt = unpack_all(result.x)
    res1 = fun(result.x)
    m1, r1, n1 = compute_reprojection_stats(res1[: 2 * n_obs])
    print(f"[SFM-BA] reprojection AFTER : mean={m1:.3f}px rms={r1:.3f}px points={n1}")

    cams_out = {}
    for cam in camnames:
        c = dict(cams[cam])
        if cam in cam_rt_opt:
            c["Rvec"] = cam_rt_opt[cam]["Rvec"]
            c["R"] = cam_rt_opt[cam]["R"]
            c["T"] = cam_rt_opt[cam]["T"]
        if cam in intri_opt:
            c["K"] = intri_opt[cam]["K"]
            c["dist"] = intri_opt[cam]["dist"]
        cams_out[cam] = c

    return {
        "cameras": cams_out,
        "points3d": points_opt,
        "result": result,
    }


def main(args):
    if least_squares is None:
        raise ImportError("scipy is required: pip install scipy")
    if args.refine_dist and not args.refine_intri:
        raise ValueError("--refine_dist requires --refine_intri")

    root = args.path
    intri_path = resolve_path(root, args.intri)
    extri_path = resolve_path(root, args.extri)
    chess_root = resolve_path(root, args.chess)
    out_extri = resolve_path(root, args.out_extri)
    out_intri = resolve_path(root, args.out_intri)
    out_points = resolve_path(root, args.out_points)

    cams = read_camera(intri_path, extri_path)
    camnames = [c for c in cams["basenames"] if os.path.isdir(join(chess_root, c))]
    cams.pop("basenames")
    if len(camnames) < 2:
        raise RuntimeError("Need at least 2 cameras with chessboard detections.")

    ref_cam = args.ref_cam if args.ref_cam else camnames[0]
    if ref_cam not in camnames:
        raise ValueError(f"ref_cam={ref_cam} is not in detected camera set: {camnames}")

    # Build per-camera frame maps.
    cam_to_map = {}
    all_frames = set()
    for cam in camnames:
        chess_list = sorted(glob(join(chess_root, cam, "*.json")))
        chess_list = sample_list(chess_list, args.step)
        m = {basename(p): p for p in chess_list}
        cam_to_map[cam] = m
        all_frames |= set(m.keys())

    all_frames = sorted(all_frames)
    if args.max_frames > 0:
        all_frames = all_frames[: args.max_frames]
    if len(all_frames) == 0:
        raise RuntimeError("No chessboard json files found.")

    # Build tracks: (frame, corner_id) with observations in >= min_views cameras.
    tracks = []
    kept_frames = 0
    for fr in all_frames:
        frame_k2d = {}
        for cam in camnames:
            p = cam_to_map[cam].get(fr, None)
            if p is None:
                continue
            data = read_json(p)
            k2d = np.array(data["keypoints2d"], dtype=np.float64)
            if k2d.ndim != 2 or k2d.shape[1] < 2:
                continue
            if k2d.shape[1] == 2:
                k2d = np.concatenate([k2d, np.ones((k2d.shape[0], 1), dtype=np.float64)], axis=1)
            frame_k2d[cam] = k2d

        if len(frame_k2d) < args.min_views:
            continue
        kept_frames += 1

        max_pid = max([v.shape[0] for v in frame_k2d.values()])
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

    if len(tracks) == 0:
        raise RuntimeError("No valid tracks with enough observations were found.")

    if args.max_points > 0 and len(tracks) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(tracks), size=args.max_points, replace=False)
        keep = np.sort(keep)
        tracks = [tracks[i] for i in keep]
        print(f"[SFM-BA] sampled tracks: {len(tracks)}")

    # Camera centers for baseline scoring.
    cam_centers = {}
    for cam in camnames:
        R = cams[cam]["R"].astype(np.float64)
        T = cams[cam]["T"].astype(np.float64)
        cam_centers[cam] = (-R.T @ T).reshape(3)

    cam_to_idx = {c: i for i, c in enumerate(camnames)}

    # Triangulate initial 3D points and flatten observations.
    points_init = []
    observations = []
    dropped = 0
    total_obs_before = 0
    total_obs_after = 0
    tracks_ransac_kept = 0
    tracks_ransac_dropped = 0
    for tr in tracks:
        total_obs_before += len(tr["obs"])
        if args.ransac:
            X, inlier_ids = triangulate_track_ransac(
                tr["obs"],
                cams,
                cam_centers,
                reproj_thresh=args.ransac_thresh,
                min_inliers=max(args.min_views, args.ransac_min_inliers),
            )
            if X is None or len(inlier_ids) < args.min_views:
                dropped += 1
                tracks_ransac_dropped += 1
                continue
            obs_used = [tr["obs"][i] for i in inlier_ids]
            tracks_ransac_kept += 1
        else:
            X = triangulate_track(tr["obs"], cams, cam_centers)
            obs_used = tr["obs"]
        if X is None:
            dropped += 1
            continue
        pidx = len(points_init)
        points_init.append(X)
        total_obs_after += len(obs_used)
        for cam, u, v, conf in obs_used:
            observations.append((cam_to_idx[cam], pidx, float(u), float(v), float(conf)))

    if len(points_init) == 0:
        raise RuntimeError("Triangulation failed for all tracks.")

    points_init = np.asarray(points_init, dtype=np.float64)
    print(
        f"[SFM-BA] frames_total={len(all_frames)} frames_kept={kept_frames} "
        f"tracks={len(tracks)} triangulated={points_init.shape[0]} dropped={dropped} "
        f"observations={len(observations)}"
    )
    if args.ransac:
        print(
            f"[SFM-BA] ransac: tracks_kept={tracks_ransac_kept} "
            f"tracks_dropped={tracks_ransac_dropped} "
            f"obs_before={total_obs_before} obs_after={total_obs_after} "
            f"thresh={args.ransac_thresh:.2f}px"
        )

    out = run_ba(
        cams=cams,
        camnames=camnames,
        ref_cam=ref_cam,
        points_init=points_init,
        observations=observations,
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
        cam_sigma_r=args.cam_sigma_r,
        cam_sigma_t=args.cam_sigma_t,
        refine_intri=args.refine_intri,
        refine_dist=args.refine_dist,
        intri_sigma_f=args.intri_sigma_f,
        intri_sigma_c=args.intri_sigma_c,
        dist_sigma=args.dist_sigma,
    )

    write_intri(out_intri, out["cameras"])
    write_extri(out_extri, out["cameras"])
    os.makedirs(os.path.dirname(out_points), exist_ok=True)
    np.savez_compressed(out_points, xyz=out["points3d"])
    print(f"[SFM-BA] wrote intrinsics: {out_intri}")
    print(f"[SFM-BA] wrote extrinsics: {out_extri}")
    print(f"[SFM-BA] wrote points3d : {out_points}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset root")
    parser.add_argument("--intri", type=str, default="intri.yml")
    parser.add_argument("--extri", type=str, default="extri.yml")
    parser.add_argument("--chess", type=str, default="chessboard")
    parser.add_argument("--out_intri", type=str, default="intri_sfm_ba.yml")
    parser.add_argument("--out_extri", type=str, default="extri_sfm_ba.yml")
    parser.add_argument("--out_points", type=str, default="output/points_chess_sfm_ba.npz")

    parser.add_argument("--ref_cam", type=str, default="", help="fixed reference camera (default first)")
    parser.add_argument("--conf", type=float, default=0.1, help="min keypoint confidence")
    parser.add_argument("--min_views", type=int, default=2, help="min cameras observing a track")
    parser.add_argument("--step", type=int, default=1, help="sample chessboard frames by step")
    parser.add_argument("--max_frames", type=int, default=-1, help="limit number of used frames")
    parser.add_argument("--max_points", type=int, default=20000, help="cap tracks used in BA (-1 all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ransac", dest="ransac", action="store_true", help="enable per-track multi-view RANSAC filtering")
    parser.add_argument("--no-ransac", dest="ransac", action="store_false", help="disable RANSAC filtering")
    parser.add_argument("--ransac_thresh", type=float, default=2.5, help="RANSAC inlier reprojection threshold (pixels)")
    parser.add_argument("--ransac_min_inliers", type=int, default=2, help="minimum inliers required by RANSAC per track")

    parser.add_argument("--loss", type=str, default="huber",
                        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--f_scale", type=float, default=3.0)
    parser.add_argument("--max_nfev", type=int, default=80)
    parser.add_argument("--cam_sigma_r", type=float, default=0.05, help="rotation prior sigma (rad)")
    parser.add_argument("--cam_sigma_t", type=float, default=50.0, help="translation prior sigma")
    parser.add_argument(
        "--refine_intri",
        dest="refine_intri",
        action="store_true",
        help="jointly optimize intrinsics (fx, fy, cx, cy)",
    )
    parser.add_argument(
        "--no-refine_intri",
        dest="refine_intri",
        action="store_false",
        help="disable intrinsic refinement",
    )
    parser.add_argument(
        "--refine_dist",
        dest="refine_dist",
        action="store_true",
        help="also optimize distortion k1,k2 (requires --refine_intri)",
    )
    parser.add_argument(
        "--no-refine_dist",
        dest="refine_dist",
        action="store_false",
        help="disable distortion refinement",
    )
    parser.add_argument("--intri_sigma_f", type=float, default=80.0, help="focal prior sigma (pixels)")
    parser.add_argument("--intri_sigma_c", type=float, default=40.0, help="principal-point prior sigma (pixels)")
    parser.add_argument("--dist_sigma", type=float, default=0.05, help="distortion prior sigma for k1,k2")

    parser.set_defaults(refine_intri=True, refine_dist=True, ransac=True)
    main(parser.parse_args())
