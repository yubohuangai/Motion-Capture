#!/usr/bin/env python3
"""
Bundle adjustment after mv1p triangulation.

This script jointly optimizes:
1) Camera extrinsics (R, T) for all cameras except one fixed reference camera.
2) 3D keypoints for each frame/joint.

Typical usage after triangulation:
python3 apps/demo/mv1p_bundle_adjustment.py /mnt/yubo/s2/seq1 \
  --triang_out /mnt/yubo/s2/seq1/mp \
  --annot annots-mp \
  --undis
"""

import argparse
import os
from glob import glob
from os.path import join

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from easymocap.mytools.camera_utils import read_camera, write_intri, write_extri
from easymocap.mytools.file_utils import read_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint BA for camera extrinsics and 3D keypoints."
    )
    parser.add_argument("path", type=str, help="Dataset root path")
    parser.add_argument(
        "--triang_out",
        type=str,
        required=True,
        help="Triangulation output directory that contains keypoints3d",
    )
    parser.add_argument(
        "--annot",
        type=str,
        default="annots",
        help="2D annotation root inside path (e.g. annots-mp)",
    )
    parser.add_argument(
        "--subs",
        type=str,
        nargs="+",
        default=None,
        help="Camera names to use, defaults to intri/extri names",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=0,
        help="Person ID to optimize; if not found, fallback to first annotation",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="2D confidence threshold",
    )
    parser.add_argument(
        "--conf_power",
        type=float,
        default=1.0,
        help="Residual weight is conf ** conf_power",
    )
    parser.add_argument(
        "--ref_cam",
        type=str,
        default=None,
        help="Reference camera to keep fixed (default: first camera in subs)",
    )
    parser.add_argument(
        "--undis",
        action="store_true",
        help="Undistort 2D points and optimize with pinhole projection (recommended if triangulation used --undis)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="huber",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        help="Robust loss for least squares",
    )
    parser.add_argument(
        "--f_scale",
        type=float,
        default=3.0,
        help="Robust loss scale",
    )
    parser.add_argument(
        "--max_nfev",
        type=int,
        default=80,
        help="Maximum function evaluations",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: <triang_out>/ba)",
    )
    return parser.parse_args()


def _select_person(annots, pid):
    for ann in annots:
        ann_pid = ann.get("id", ann.get("personID", None))
        if ann_pid == pid:
            return ann
    if len(annots) > 0:
        return annots[0]
    return None


def _load_2d_keypoints(annot_file, pid, n_joints):
    if not os.path.exists(annot_file):
        return None
    data = read_json(annot_file)
    if isinstance(data, dict):
        annots = data.get("annots", [])
    else:
        annots = data
    person = _select_person(annots, pid)
    if person is None:
        return None
    kpts = person.get("keypoints", person.get("keypoints2d", None))
    if kpts is None:
        return None
    kpts = np.asarray(kpts, dtype=np.float64)
    if kpts.shape[0] < n_joints:
        return None
    return kpts[:n_joints, :3]


def _load_3d_keypoints(k3d_file, pid):
    data = read_json(k3d_file)
    if len(data) == 0:
        return None
    person = None
    for ann in data:
        ann_pid = ann.get("id", ann.get("personID", None))
        if ann_pid == pid:
            person = ann
            break
    if person is None:
        person = data[0]
    k3d = np.asarray(person["keypoints3d"], dtype=np.float64)
    if k3d.shape[1] == 3:
        conf = np.ones((k3d.shape[0], 1), dtype=np.float64)
        k3d = np.hstack([k3d, conf])
    return k3d


def _project_point(X, K, R, T):
    Xc = R @ X + T[:, 0]
    z = max(Xc[2], 1e-8)
    u = K[0, 0] * (Xc[0] / z) + K[0, 2]
    v = K[1, 1] * (Xc[1] / z) + K[1, 2]
    return np.array([u, v], dtype=np.float64)


def build_problem(args):
    intri = join(args.path, "intri.yml")
    extri = join(args.path, "extri.yml")
    cameras = read_camera(intri, extri)
    camnames = cameras["basenames"]
    if args.subs is not None and len(args.subs) > 0:
        camnames = [cam for cam in args.subs if cam in camnames]
    if len(camnames) < 2:
        raise ValueError("Need at least two cameras for BA.")

    ref_cam = args.ref_cam if args.ref_cam is not None else camnames[0]
    if ref_cam not in camnames:
        raise ValueError(f"ref_cam {ref_cam} not in selected cameras: {camnames}")

    k3d_dir = join(args.triang_out, "keypoints3d")
    k3d_files = sorted(glob(join(k3d_dir, "*.json")))
    if len(k3d_files) == 0:
        raise FileNotFoundError(f"No keypoints3d files found in {k3d_dir}")

    frames = []
    points3d_init = []
    points3d_conf = []
    observations = []

    # Infer joint count from first valid keypoints3d file.
    first_k3d = None
    for f in k3d_files:
        first_k3d = _load_3d_keypoints(f, args.pid)
        if first_k3d is not None:
            break
    if first_k3d is None:
        raise RuntimeError("No valid keypoints3d found.")
    n_joints = first_k3d.shape[0]

    cam_idx = {cam: i for i, cam in enumerate(camnames)}

    for k3d_file in k3d_files:
        frame_id = int(os.path.basename(k3d_file).replace(".json", ""))
        k3d = _load_3d_keypoints(k3d_file, args.pid)
        if k3d is None:
            continue
        if k3d.shape[0] != n_joints:
            continue

        local_frame_idx = len(frames)
        frames.append(frame_id)
        points3d_init.append(k3d[:, :3])
        points3d_conf.append(k3d[:, 3])

        for cam in camnames:
            annot_file = join(args.path, args.annot, cam, f"{frame_id:06d}.json")
            k2d = _load_2d_keypoints(annot_file, args.pid, n_joints)
            if k2d is None:
                continue
            K = cameras[cam]["K"]
            dist = cameras[cam]["dist"]
            if args.undis:
                k2d = k2d.copy()
                k2d[:, :2] = cv2.undistortPoints(
                    k2d[:, None, :2], K, dist, P=K
                )[:, 0, :]
            for j in range(n_joints):
                conf2d = float(k2d[j, 2])
                conf3d = float(k3d[j, 3])
                if conf2d <= args.conf or conf3d <= 0:
                    continue
                weight = conf2d ** args.conf_power
                obs = {
                    "cam": cam_idx[cam],
                    "point": local_frame_idx * n_joints + j,
                    "xy": k2d[j, :2].copy(),
                    "weight": weight,
                }
                observations.append(obs)

    if len(frames) == 0:
        raise RuntimeError("No valid frames loaded.")
    if len(observations) < 20:
        raise RuntimeError(f"Too few observations ({len(observations)}). Check inputs.")

    points3d_init = np.stack(points3d_init).reshape(-1, 3)
    points3d_conf = np.stack(points3d_conf).reshape(-1)

    cam_params_init = []
    opt_cams = []
    for cam in camnames:
        if cam == ref_cam:
            continue
        rvec = cameras[cam]["Rvec"].reshape(3)
        tvec = cameras[cam]["T"].reshape(3)
        cam_params_init.append(np.hstack([rvec, tvec]))
        opt_cams.append(cam)
    cam_params_init = np.array(cam_params_init, dtype=np.float64).reshape(-1, 6)

    return {
        "cameras": cameras,
        "camnames": camnames,
        "ref_cam": ref_cam,
        "opt_cams": opt_cams,
        "frames": frames,
        "n_joints": n_joints,
        "points3d_init": points3d_init,
        "points3d_conf": points3d_conf,
        "cam_params_init": cam_params_init,
        "observations": observations,
    }


def pack_params(cam_params, points3d):
    return np.hstack([cam_params.reshape(-1), points3d.reshape(-1)])


def unpack_params(x, n_opt_cams, n_points):
    split = n_opt_cams * 6
    cam_params = x[:split].reshape(n_opt_cams, 6)
    points3d = x[split:].reshape(n_points, 3)
    return cam_params, points3d


def build_cam_state(problem, cam_params_opt):
    cameras = problem["cameras"]
    camnames = problem["camnames"]
    ref_cam = problem["ref_cam"]
    opt_cams = problem["opt_cams"]
    state = {}
    opt_map = {cam: i for i, cam in enumerate(opt_cams)}
    for cam in camnames:
        if cam == ref_cam:
            R = cameras[cam]["R"]
            T = cameras[cam]["T"]
        else:
            params = cam_params_opt[opt_map[cam]]
            rvec = params[:3].reshape(3, 1)
            tvec = params[3:].reshape(3, 1)
            R = cv2.Rodrigues(rvec)[0]
            T = tvec
        state[cam] = {
            "K": cameras[cam]["K"],
            "dist": cameras[cam]["dist"],
            "R": R,
            "T": T,
        }
    return state


def residuals_func(x, problem, use_distortion=False):
    n_opt_cams = len(problem["opt_cams"])
    n_points = problem["points3d_init"].shape[0]
    cam_params_opt, points3d = unpack_params(x, n_opt_cams, n_points)
    cam_state = build_cam_state(problem, cam_params_opt)
    camnames = problem["camnames"]
    res = np.zeros((len(problem["observations"]) * 2,), dtype=np.float64)

    for i, obs in enumerate(problem["observations"]):
        cam = camnames[obs["cam"]]
        point = points3d[obs["point"]]
        K = cam_state[cam]["K"]
        R = cam_state[cam]["R"]
        T = cam_state[cam]["T"]
        if use_distortion:
            rvec = cv2.Rodrigues(R)[0]
            proj = cv2.projectPoints(
                point.reshape(1, 1, 3), rvec, T, K, cam_state[cam]["dist"]
            )[0][0, 0]
        else:
            proj = _project_point(point, K, R, T)
        diff = (proj - obs["xy"]) * obs["weight"]
        res[2 * i : 2 * i + 2] = diff
    return res


def build_jac_sparsity(problem):
    n_obs = len(problem["observations"])
    n_opt_cams = len(problem["opt_cams"])
    n_points = problem["points3d_init"].shape[0]
    n_params = n_opt_cams * 6 + n_points * 3
    A = lil_matrix((2 * n_obs, n_params), dtype=np.int8)

    camnames = problem["camnames"]
    opt_cam_index = {cam: i for i, cam in enumerate(problem["opt_cams"])}

    for i, obs in enumerate(problem["observations"]):
        cam = camnames[obs["cam"]]
        point = obs["point"]
        row0 = 2 * i
        row1 = row0 + 1

        if cam in opt_cam_index:
            cidx = opt_cam_index[cam]
            cbase = cidx * 6
            A[row0, cbase : cbase + 6] = 1
            A[row1, cbase : cbase + 6] = 1

        pbase = n_opt_cams * 6 + point * 3
        A[row0, pbase : pbase + 3] = 1
        A[row1, pbase : pbase + 3] = 1

    return A


def compute_reproj_stats(problem, cam_params_opt, points3d, use_distortion=False):
    cam_state = build_cam_state(problem, cam_params_opt)
    camnames = problem["camnames"]
    diffs = []
    for obs in problem["observations"]:
        cam = camnames[obs["cam"]]
        point = points3d[obs["point"]]
        K = cam_state[cam]["K"]
        R = cam_state[cam]["R"]
        T = cam_state[cam]["T"]
        if use_distortion:
            rvec = cv2.Rodrigues(R)[0]
            proj = cv2.projectPoints(
                point.reshape(1, 1, 3), rvec, T, K, cam_state[cam]["dist"]
            )[0][0, 0]
        else:
            proj = _project_point(point, K, R, T)
        diffs.append(np.linalg.norm(proj - obs["xy"]))
    diffs = np.array(diffs, dtype=np.float64)
    return {
        "mean": float(np.mean(diffs)) if diffs.size > 0 else float("nan"),
        "median": float(np.median(diffs)) if diffs.size > 0 else float("nan"),
    }


def save_results(args, problem, cam_params_opt, points3d_opt):
    out_root = args.out if args.out is not None else join(args.triang_out, "ba")
    os.makedirs(out_root, exist_ok=True)

    # Save refined cameras (intri unchanged, extri updated).
    cam_state = build_cam_state(problem, cam_params_opt)
    cams_to_write = {}
    for cam in problem["camnames"]:
        cams_to_write[cam] = {
            "K": cam_state[cam]["K"],
            "dist": problem["cameras"][cam]["dist"],
            "R": cam_state[cam]["R"],
            "T": cam_state[cam]["T"],
            "Rvec": cv2.Rodrigues(cam_state[cam]["R"])[0],
        }
        if "H" in problem["cameras"][cam]:
            cams_to_write[cam]["H"] = problem["cameras"][cam]["H"]
        if "W" in problem["cameras"][cam]:
            cams_to_write[cam]["W"] = problem["cameras"][cam]["W"]

    write_intri(join(out_root, "intri.yml"), cams_to_write)
    write_extri(join(out_root, "extri.yml"), cams_to_write)

    # Save refined keypoints3d as EasyMocap JSON.
    out_k3d = join(out_root, "keypoints3d")
    os.makedirs(out_k3d, exist_ok=True)
    n_joints = problem["n_joints"]
    conf = problem["points3d_conf"].reshape(-1, 1)
    points4d = np.hstack([points3d_opt, conf])
    for fi, frame_id in enumerate(problem["frames"]):
        sl = slice(fi * n_joints, (fi + 1) * n_joints)
        data = [{"id": args.pid, "keypoints3d": points4d[sl].tolist()}]
        save_json(join(out_k3d, f"{frame_id:06d}.json"), data)

    return out_root


def main():
    args = parse_args()
    problem = build_problem(args)

    n_obs = len(problem["observations"])
    n_frames = len(problem["frames"])
    n_points = problem["points3d_init"].shape[0]
    n_opt_cams = len(problem["opt_cams"])
    print(
        f"[INFO] BA setup: frames={n_frames}, cams={len(problem['camnames'])}, "
        f"opt_cams={n_opt_cams}, joints/frame={problem['n_joints']}, obs={n_obs}"
    )
    print(f"[INFO] Reference camera fixed: {problem['ref_cam']}")

    x0 = pack_params(problem["cam_params_init"], problem["points3d_init"])
    jac_sparsity = build_jac_sparsity(problem)
    use_distortion = not args.undis

    init_stats = compute_reproj_stats(
        problem, problem["cam_params_init"], problem["points3d_init"], use_distortion
    )
    print(
        "[INFO] Before BA reprojection error: "
        f"mean={init_stats['mean']:.3f}px, median={init_stats['median']:.3f}px"
    )

    result = least_squares(
        residuals_func,
        x0,
        jac_sparsity=jac_sparsity,
        verbose=2,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
        args=(problem, use_distortion),
    )

    cam_params_opt, points3d_opt = unpack_params(x=result.x, n_opt_cams=n_opt_cams, n_points=n_points)
    final_stats = compute_reproj_stats(problem, cam_params_opt, points3d_opt, use_distortion)
    print(
        "[INFO] After  BA reprojection error: "
        f"mean={final_stats['mean']:.3f}px, median={final_stats['median']:.3f}px"
    )
    print(f"[INFO] least_squares status={result.status}, message={result.message}")

    out_root = save_results(args, problem, cam_params_opt, points3d_opt)
    print(f"[DONE] Saved BA results to: {out_root}")
    print(f"[DONE] Refined keypoints3d: {join(out_root, 'keypoints3d')}")
    print(f"[DONE] Refined cameras: {join(out_root, 'intri.yml')}, {join(out_root, 'extri.yml')}")


if __name__ == "__main__":
    main()

