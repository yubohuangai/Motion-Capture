"""
Extract sparse multi-view point clouds from RGB images using camera calibration.

Dataset layout (EasyMocap-style):
    {path}/images/{sub}/%06d.jpg
    {path}/masks/{sub}/%06d.png               (optional but recommended)
    {path}/annots/{sub}/%06d.json             (optional fallback for mask/bbox)
    {path}/intri.yml
    {path}/extri.yml

Output:
    {path}/{out}/points/%06d.npz              (xyz, rgb, frame)
    {path}/{out}/points/%06d.ply              (optional)
    {path}/{out}/pointclouds.pkl              (list of xyz arrays by frame)
    {path}/{out}/meta.pkl                     (frame names and stats)
"""
import os
import json
import pickle
import argparse
from os.path import join, exists

import cv2
import numpy as np
from tqdm import tqdm

from easymocap.mytools import read_camera


def load_subs(path, subs, image_dir):
    if len(subs) == 0:
        subs = sorted(os.listdir(join(path, image_dir)))
    subs = [sub for sub in subs if os.path.isdir(join(path, image_dir, sub))]
    if len(subs) == 0:
        subs = [""]
    return subs


def sample_list(lst, step):
    if step <= 1:
        return lst
    return lst[::step]


def list_frames(image_root, ext):
    if not exists(image_root):
        return []
    return sorted([n for n in os.listdir(image_root) if n.endswith(ext)])


def read_mask(maskname):
    mask = cv2.imread(maskname, 0)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def read_mask_from_annot(annotname, shape_hw):
    if not exists(annotname):
        return None
    try:
        with open(annotname, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    if isinstance(data, dict):
        annots = data.get("annots", [])
    elif isinstance(data, list):
        annots = data
    else:
        annots = []
    if len(annots) == 0:
        return None
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in annots:
        if "mask" in ann and len(ann["mask"]) > 0:
            pts = np.asarray(ann["mask"], dtype=np.int32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            mask[pts[:, 1], pts[:, 0]] = 1
            continue
        if "bbox" in ann and len(ann["bbox"]) >= 4:
            x1, y1, x2, y2 = ann["bbox"][:4]
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            if x2 > x1 and y2 > y1:
                mask[y1:y2 + 1, x1:x2 + 1] = 1
    if mask.sum() == 0:
        return None
    if mask.sum() < 200:
        # Sparse points from annot["mask"] are not enough for feature masking.
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    return mask


def write_ply_ascii(outname, xyz, rgb=None):
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    n = xyz.shape[0]
    with open(outname, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def read_keypoints3d_json(filename, conf_thres=0.1):
    if not exists(filename):
        return np.zeros((0, 3), dtype=np.float32)
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(data, dict):
        data = data.get("annots", [])
    if not isinstance(data, list):
        return np.zeros((0, 3), dtype=np.float32)

    pts_all = []
    for person in data:
        if "keypoints3d" not in person:
            continue
        k3d = np.asarray(person["keypoints3d"], dtype=np.float32)
        if k3d.ndim != 2 or k3d.shape[0] == 0:
            continue
        if k3d.shape[1] >= 4:
            valid = k3d[:, 3] > conf_thres
            k3d = k3d[valid, :3]
        else:
            k3d = k3d[:, :3]
        if k3d.shape[0] > 0:
            pts_all.append(k3d)
    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts_all, axis=0).astype(np.float32)


def make_pairs(cams, pair_mode):
    if len(cams) < 2:
        return []
    if pair_mode == "all":
        pairs = []
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                pairs.append((cams[i], cams[j]))
        return pairs
    # ring pairs: (0,1), (1,2), ..., (N-1,0)
    return [(cams[i], cams[(i + 1) % len(cams)]) for i in range(len(cams))]


def undistort_and_gray(img_bgr, K, dist):
    img_ud = cv2.undistort(img_bgr, K, dist)
    gray = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)
    return img_ud, gray


def filter_depth_and_reproj(X, cam0, cam1, pts0, pts1, reproj_thres):
    R0, T0, K0 = cam0["R"], cam0["T"], cam0["K"]
    R1, T1, K1 = cam1["R"], cam1["T"], cam1["K"]

    Xc0 = (R0 @ X.T + T0).T
    Xc1 = (R1 @ X.T + T1).T
    valid = (Xc0[:, 2] > 1e-6) & (Xc1[:, 2] > 1e-6)
    if valid.sum() == 0:
        return np.zeros((0,), dtype=bool)

    Xv = X[valid]
    p0 = pts0[valid]
    p1 = pts1[valid]
    # Features are extracted on undistorted images, so reprojection should use zero distortion.
    r0, _ = cv2.projectPoints(Xv, cv2.Rodrigues(R0)[0], T0, K0, None)
    r1, _ = cv2.projectPoints(Xv, cv2.Rodrigues(R1)[0], T1, K1, None)
    e0 = np.linalg.norm(r0.reshape(-1, 2) - p0, axis=1)
    e1 = np.linalg.norm(r1.reshape(-1, 2) - p1, axis=1)
    good_v = (e0 < reproj_thres) & (e1 < reproj_thres)

    good = np.zeros((X.shape[0],), dtype=bool)
    idx = np.where(valid)[0]
    good[idx[good_v]] = True
    return good


def triangulate_pair(
    kp0,
    des0,
    kp1,
    des1,
    cam0,
    cam1,
    img0_ud,
    match_crosscheck,
    reproj_thres,
    use_ransac=False,
    ransac_method="essential",
    ransac_thres=1.0,
    ransac_prob=0.999,
    ransac_iters=2000,
):
    if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            0,
            0,
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=match_crosscheck)
    matches = matcher.match(des0, des1)
    n_matches_raw = len(matches)
    if n_matches_raw < 8:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            n_matches_raw,
            n_matches_raw,
        )

    matches = sorted(matches, key=lambda m: m.distance)
    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])

    if use_ransac and pts0.shape[0] >= 8:
        mask = None
        if ransac_method == "fundamental":
            _, mask = cv2.findFundamentalMat(
                pts0,
                pts1,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=ransac_thres,
                confidence=ransac_prob,
                maxIters=ransac_iters,
            )
        else:
            E, mask = cv2.findEssentialMat(
                pts0,
                pts1,
                cameraMatrix=cam0["K"],
                method=cv2.RANSAC,
                prob=ransac_prob,
                threshold=ransac_thres,
                maxIters=ransac_iters,
            )
            if E is None:
                mask = None
        if mask is not None:
            inlier = mask.reshape(-1).astype(bool)
            if inlier.sum() >= 8:
                pts0 = pts0[inlier]
                pts1 = pts1[inlier]
    n_matches_used = int(pts0.shape[0])

    P0 = cam0["P"]
    P1 = cam1["P"]
    X_h = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T).T
    X = (X_h[:, :3] / np.clip(X_h[:, 3:4], 1e-8, None)).astype(np.float64)
    good = filter_depth_and_reproj(X, cam0, cam1, pts0, pts1, reproj_thres)
    if good.sum() == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            n_matches_raw,
            n_matches_used,
        )

    X = X[good].astype(np.float32)
    pts0 = pts0[good]
    colors = []
    h, w = img0_ud.shape[:2]
    for p in pts0:
        x = int(np.clip(round(float(p[0])), 0, w - 1))
        y = int(np.clip(round(float(p[1])), 0, h - 1))
        b, g, r = img0_ud[y, x]
        colors.append([r, g, b])
    colors = np.asarray(colors, dtype=np.uint8)
    return X, colors, n_matches_raw, n_matches_used


def triangulate_multiview_dlt(obs_uv, cams):
    # obs_uv: list of (camname, [u, v]) with undistorted pixel coordinates
    A = []
    for cam, uv in obs_uv:
        P = cams[cam]["P"]
        u, v = float(uv[0]), float(uv[1])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A, dtype=np.float64)
    if A.shape[0] < 4:
        return None
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-10:
        return None
    X = (Xh[:3] / Xh[3]).astype(np.float64)
    return X


def reproj_valid_multiview(X, obs_uv, cams, reproj_thres):
    X = np.asarray(X, dtype=np.float64).reshape(1, 3)
    for cam, uv in obs_uv:
        R, T, K = cams[cam]["R"], cams[cam]["T"], cams[cam]["K"]
        Xc = (R @ X.T + T).T
        if Xc[0, 2] <= 1e-6:
            return False
        uvhat, _ = cv2.projectPoints(X, cv2.Rodrigues(R)[0], T, K, None)
        err = np.linalg.norm(uvhat.reshape(2) - np.asarray(uv, dtype=np.float64))
        if err > reproj_thres:
            return False
    return True


def ransac_filter_points(pts0, pts1, cam0, args):
    if pts0.shape[0] < 8 or (not args.ransac):
        return np.ones((pts0.shape[0],), dtype=bool)
    if args.ransac_method == "fundamental":
        _, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=args.ransac_thres,
            confidence=args.ransac_prob,
            maxIters=args.ransac_iters,
        )
    else:
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            cameraMatrix=cam0["K"],
            method=cv2.RANSAC,
            prob=args.ransac_prob,
            threshold=args.ransac_thres,
            maxIters=args.ransac_iters,
        )
        if E is None:
            mask = None
    if mask is None:
        return np.ones((pts0.shape[0],), dtype=bool)
    inlier = mask.reshape(-1).astype(bool)
    if inlier.sum() < 8:
        return np.ones((pts0.shape[0],), dtype=bool)
    return inlier


def build_frame_cloud_global(data, cams_ok, feats, args):
    ref_cam = cams_ok[0]
    kp_ref = feats[ref_cam]["kp"]
    des_ref = feats[ref_cam]["des"]
    if des_ref is None or len(kp_ref) < 8:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), {
            "pairs": 0,
            "matches": 0,
            "ransac_inliers": 0,
            "tracks": 0,
            "points": 0,
        }

    tracks = {}
    for i_ref, kp in enumerate(kp_ref):
        tracks[i_ref] = {"obs": {ref_cam: np.asarray(kp.pt, dtype=np.float32)}, "best_dist": {}}

    matches_total = 0
    ransac_inliers_total = 0
    pairs_used = 0
    for cam in cams_ok[1:]:
        kp = feats[cam]["kp"]
        des = feats[cam]["des"]
        if des is None or len(kp) < 8:
            continue
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=args.crosscheck)
        matches = matcher.match(des_ref, des)
        if len(matches) < 8:
            continue
        pairs_used += 1
        matches_total += len(matches)
        matches = sorted(matches, key=lambda m: m.distance)

        pts0 = np.float32([kp_ref[m.queryIdx].pt for m in matches])
        pts1 = np.float32([kp[m.trainIdx].pt for m in matches])
        inlier = ransac_filter_points(pts0, pts1, data[ref_cam]["cam"], args)
        if args.ransac:
            ransac_inliers_total += int(inlier.sum())

        for keep, m in zip(inlier.tolist(), matches):
            if not keep:
                continue
            q = int(m.queryIdx)
            t = int(m.trainIdx)
            d = float(m.distance)
            rec = tracks[q]
            prev = rec["best_dist"].get(cam, 1e9)
            if d < prev:
                rec["obs"][cam] = np.asarray(kp[t].pt, dtype=np.float32)
                rec["best_dist"][cam] = d

    clouds = []
    colors = []
    img_ref = data[ref_cam]["img_ud"]
    h, w = img_ref.shape[:2]
    for _, tr in tracks.items():
        if len(tr["obs"]) < 2:
            continue
        obs_uv = [(cam, uv) for cam, uv in tr["obs"].items()]
        X = triangulate_multiview_dlt(obs_uv, {cam: data[cam]["cam"] for cam in cams_ok})
        if X is None:
            continue
        if not reproj_valid_multiview(X, obs_uv, {cam: data[cam]["cam"] for cam in cams_ok}, args.reproj_thres):
            continue
        u_ref, v_ref = tr["obs"][ref_cam]
        x = int(np.clip(round(float(u_ref)), 0, w - 1))
        y = int(np.clip(round(float(v_ref)), 0, h - 1))
        b, g, r = img_ref[y, x]
        clouds.append(X.astype(np.float32))
        colors.append([r, g, b])

    if len(clouds) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), {
            "pairs": pairs_used,
            "matches": matches_total,
            "ransac_inliers": ransac_inliers_total,
            "tracks": 0,
            "points": 0,
        }
    X = np.asarray(clouds, dtype=np.float32).reshape(-1, 3)
    C = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    return X, C, {
        "pairs": pairs_used,
        "matches": matches_total,
        "ransac_inliers": ransac_inliers_total,
        "tracks": int(len(clouds)),
        "points": int(X.shape[0]),
    }


def build_frame_cloud_pairwise(data, cams_ok, feats, args):
    pair_cams = make_pairs(cams_ok, args.pair_mode)
    clouds = []
    colors = []
    matches_total = 0
    ransac_inliers_total = 0
    for c0, c1 in pair_cams:
        X, C, n_raw, n_used = triangulate_pair(
            feats[c0]["kp"],
            feats[c0]["des"],
            feats[c1]["kp"],
            feats[c1]["des"],
            data[c0]["cam"],
            data[c1]["cam"],
            data[c0]["img_ud"],
            args.crosscheck,
            args.reproj_thres,
            use_ransac=args.ransac,
            ransac_method=args.ransac_method,
            ransac_thres=args.ransac_thres,
            ransac_prob=args.ransac_prob,
            ransac_iters=args.ransac_iters,
        )
        matches_total += n_raw
        if args.ransac:
            ransac_inliers_total += n_used
        if X.shape[0] == 0:
            continue
        clouds.append(X)
        colors.append(C)
    if len(clouds) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), {
            "pairs": len(pair_cams),
            "matches": matches_total,
            "ransac_inliers": ransac_inliers_total,
            "tracks": 0,
            "points": 0,
        }
    X = np.concatenate(clouds, axis=0)
    C = np.concatenate(colors, axis=0)
    return X, C, {
        "pairs": len(pair_cams),
        "matches": matches_total,
        "ransac_inliers": ransac_inliers_total,
        "tracks": int(X.shape[0]),
        "points": int(X.shape[0]),
    }


def build_frame_cloud(frame, cameras, args):
    data = {}
    cams_ok = []
    for cam in cameras:
        imgname = join(args.path, args.image, cam, frame)
        if not exists(imgname):
            continue
        img_bgr = cv2.imread(imgname)
        if img_bgr is None:
            continue
        mask = None
        if args.use_mask:
            maskname = join(args.path, args.mask, cam, frame.replace(args.ext, ".png"))
            if exists(maskname):
                mask = read_mask(maskname)
            elif args.mask_from_annot:
                annotname = join(args.path, args.annot, cam, frame.replace(args.ext, ".json"))
                mask = read_mask_from_annot(annotname, img_bgr.shape[:2])
        camdata = cameras[cam]
        img_ud, gray = undistort_and_gray(img_bgr, camdata["K"], camdata["dist"])
        data[cam] = {"img_ud": img_ud, "gray": gray, "mask": mask, "cam": camdata}
        cams_ok.append(cam)
    if len(cams_ok) < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), {
            "pairs": 0,
            "matches": 0,
            "points": 0,
        }

    orb = cv2.ORB_create(
        nfeatures=args.nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=args.fast_threshold,
    )
    feats = {}
    for cam in cams_ok:
        item = data[cam]
        kp, des = orb.detectAndCompute(item["gray"], item["mask"])
        feats[cam] = {"kp": kp, "des": des}
    if args.match_mode == "global":
        X, C, stats = build_frame_cloud_global(data, cams_ok, feats, args)
    else:
        X, C, stats = build_frame_cloud_pairwise(data, cams_ok, feats, args)

    if args.voxel_size > 0:
        vox = np.floor(X / args.voxel_size).astype(np.int64)
        _, uniq_idx = np.unique(vox, axis=0, return_index=True)
        X = X[uniq_idx]
        C = C[uniq_idx]

    if args.max_points > 0 and X.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        ids = rng.choice(X.shape[0], size=args.max_points, replace=False)
        X = X[ids]
        C = C[ids]

    stats["points"] = int(X.shape[0])
    return X, C, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset root (contains images/, intri.yml, extri.yml)")
    parser.add_argument("--subs", type=str, nargs="+", default=[], help="camera list; default all under images/")
    parser.add_argument("--image", type=str, default="images")
    parser.add_argument("--mask", type=str, default="masks")
    parser.add_argument("--annot", type=str, default="annots")
    parser.add_argument("--intri", type=str, default="intri.yml")
    parser.add_argument("--extri", type=str, default="extri.yml")
    parser.add_argument("--out", type=str, default="pointcloud")
    parser.add_argument("--ext", type=str, default=".jpg")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--match_mode", type=str, default="global", choices=["global", "pairwise"])
    parser.add_argument("--pair_mode", type=str, default="all", choices=["ring", "all"])
    parser.add_argument("--nfeatures", type=int, default=4000)
    parser.add_argument("--fast_threshold", type=int, default=10)
    parser.add_argument("--crosscheck", action="store_true")
    parser.add_argument("--ransac", action="store_true", help="apply RANSAC filtering on 2D matches before triangulation")
    parser.add_argument("--ransac_method", type=str, default="essential", choices=["essential", "fundamental"])
    parser.add_argument("--ransac_thres", type=float, default=4.0, help="RANSAC inlier threshold in pixels")
    parser.add_argument("--ransac_prob", type=float, default=0.999, help="RANSAC confidence")
    parser.add_argument("--ransac_iters", type=int, default=2000, help="RANSAC max iterations")
    parser.add_argument("--reproj_thres", type=float, default=4.0)
    parser.add_argument("--voxel_size", type=float, default=0.0, help="world unit dedup voxel size (0 disables)")
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_mask", action="store_true", default=True, help="use masks/{cam}/xxxxxx.png if available")
    parser.add_argument("--mask_from_annot", action="store_true", help="fallback: annots/{cam}/xxxxxx.json")
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--k3d", type=str, default="", help="3D keypoints folder (relative to path or absolute)")
    parser.add_argument("--k3d_conf", type=float, default=0.1, help="confidence threshold for keypoints3d[:,3]")
    parser.add_argument("--k3d_color", type=int, nargs=3, default=[0, 255, 0], help="RGB color for inserted 3D keypoints")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    intri_path = join(args.path, args.intri)
    extri_path = join(args.path, args.extri)
    cameras = read_camera(intri_path, extri_path)
    cams_calib = cameras["basenames"]
    cameras.pop("basenames")

    subs = load_subs(args.path, args.subs, args.image)
    subs = [s for s in subs if s in cams_calib]
    if len(subs) < 2:
        raise RuntimeError("Need >=2 cameras in both images/ and intri/extri yml.")

    frame_sets = []
    for cam in subs:
        frames = list_frames(join(args.path, args.image, cam), args.ext)
        frame_sets.append(set(frames))
    common_frames = sorted(set.intersection(*frame_sets))
    common_frames = sample_list(common_frames, args.step)
    if args.start > 0:
        common_frames = common_frames[args.start:]
    if args.max_frames > 0:
        common_frames = common_frames[:args.max_frames]
    if len(common_frames) == 0:
        raise RuntimeError("No common frames found across selected cameras.")

    out_root = join(args.path, args.out)
    out_points = join(out_root, "points")
    os.makedirs(out_points, exist_ok=True)

    clouds_all = []
    stats_all = []
    for frame in tqdm(common_frames, desc="extract pointcloud"):
        stem = frame.replace(args.ext, "")
        out_npz = join(out_points, stem + ".npz")
        out_ply = join(out_points, stem + ".ply")
        if exists(out_npz) and (not args.save_ply or exists(out_ply)) and not args.force:
            data = np.load(out_npz)
            clouds_all.append(data["xyz"])
            stats_all.append({"frame": frame, "pairs": -1, "matches": -1, "points": int(data["xyz"].shape[0])})
            continue

        X, C, stats = build_frame_cloud(frame, {k: cameras[k] for k in subs}, args)
        is_keypoint = np.zeros((X.shape[0],), dtype=np.uint8)

        if args.k3d != "":
            k3d_root = args.k3d if args.k3d.startswith("/") else join(args.path, args.k3d)
            k3d_name = join(k3d_root, stem + ".json")
            X_k3d = read_keypoints3d_json(k3d_name, conf_thres=args.k3d_conf)
            if X_k3d.shape[0] > 0:
                color_k3d = np.asarray(args.k3d_color, dtype=np.uint8).reshape(1, 3)
                C_k3d = np.repeat(color_k3d, X_k3d.shape[0], axis=0)
                X = np.concatenate([X, X_k3d], axis=0)
                C = np.concatenate([C, C_k3d], axis=0)
                is_keypoint = np.concatenate([is_keypoint, np.ones((X_k3d.shape[0],), dtype=np.uint8)], axis=0)
                stats["k3d_points"] = int(X_k3d.shape[0])
            else:
                stats["k3d_points"] = 0
        else:
            stats["k3d_points"] = 0

        np.savez_compressed(
            out_npz,
            xyz=X.astype(np.float32),
            rgb=C.astype(np.uint8),
            is_keypoint=is_keypoint,
            frame=np.array([frame]),
        )
        if args.save_ply:
            write_ply_ascii(out_ply, X, C)
        clouds_all.append(X.astype(np.float32))
        stats["frame"] = frame
        stats_all.append(stats)

    with open(join(out_root, "pointclouds.pkl"), "wb") as f:
        pickle.dump(clouds_all, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "frames": common_frames,
        "cameras": subs,
        "stats": stats_all,
        "args": vars(args),
    }
    with open(join(out_root, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    npts = [s["points"] for s in stats_all]
    print(f"[PointCloud] frames={len(common_frames)} cams={len(subs)}")
    print(f"[PointCloud] points/frame mean={np.mean(npts):.1f} min={np.min(npts)} max={np.max(npts)}")
    print(f"[PointCloud] wrote: {out_root}")


if __name__ == "__main__":
    main()
