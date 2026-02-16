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

from easymocap.mytools import read_camera, write_intri, write_extri

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None
try:
    from scipy.sparse import lil_matrix
except Exception:
    lil_matrix = None


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


def read_keypoints3d_person(filename, pid=0, conf_thres=0.1):
    if not exists(filename):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    if isinstance(data, dict):
        data = data.get("annots", [])
    if not isinstance(data, list) or len(data) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    person = None
    for ann in data:
        ann_pid = ann.get("id", ann.get("personID", None))
        if ann_pid == pid:
            person = ann
            break
    if person is None:
        person = data[0]
    if "keypoints3d" not in person:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    k3d = np.asarray(person["keypoints3d"], dtype=np.float32)
    if k3d.ndim != 2 or k3d.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    ids = np.arange(k3d.shape[0], dtype=np.int32)
    if k3d.shape[1] >= 4:
        valid = k3d[:, 3] > conf_thres
        k3d = k3d[valid, :3]
        ids = ids[valid]
    else:
        k3d = k3d[:, :3]
    return k3d.astype(np.float32), ids.astype(np.int32)


def read_keypoints2d_person(filename, pid=0):
    if not exists(filename):
        return None
    try:
        with open(filename, "r") as f:
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
    person = None
    for ann in annots:
        ann_pid = ann.get("id", ann.get("personID", None))
        if ann_pid == pid:
            person = ann
            break
    if person is None:
        person = annots[0]
    k2d = person.get("keypoints", person.get("keypoints2d", None))
    if k2d is None:
        return None
    k2d = np.asarray(k2d, dtype=np.float32)
    if k2d.ndim != 2 or k2d.shape[1] < 2:
        return None
    if k2d.shape[1] == 2:
        k2d = np.hstack([k2d, np.ones((k2d.shape[0], 1), dtype=np.float32)])
    return k2d


def undistort_pixels(k2d, K, dist):
    pts = np.asarray(k2d[:, :2], dtype=np.float32).reshape(-1, 1, 2)
    pts_ud = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
    conf = k2d[:, 2:3] if k2d.shape[1] >= 3 else np.ones((k2d.shape[0], 1), dtype=np.float32)
    return np.hstack([pts_ud, conf]).astype(np.float32)


def init_roma_model(args):
    try:
        from romatch import roma_outdoor
    except Exception as e:
        raise RuntimeError(
            "RoMa is required for --matcher roma. "
            "Install dependency `romatch` in your environment."
        ) from e
    print(f"[Matcher] loading RoMa on device={args.roma_device}")
    return roma_outdoor(device=args.roma_device)


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


def triangulate_from_points(
    pts0,
    pts1,
    cam0,
    cam1,
    img0_ud,
    reproj_thres,
    use_ransac=False,
    ransac_method="essential",
    ransac_thres=1.0,
    ransac_prob=0.999,
    ransac_iters=2000,
):
    pts0 = np.asarray(pts0, dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    n_matches_raw = pts0.shape[0]
    if n_matches_raw < 8:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            n_matches_raw,
            n_matches_raw,
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

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
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    X = X[good].astype(np.float32)
    pts0 = pts0[good]
    pts1 = pts1[good]
    colors = []
    h, w = img0_ud.shape[:2]
    for p in pts0:
        x = int(np.clip(round(float(p[0])), 0, w - 1))
        y = int(np.clip(round(float(p[1])), 0, h - 1))
        b, g, r = img0_ud[y, x]
        colors.append([r, g, b])
    colors = np.asarray(colors, dtype=np.uint8)
    return X, colors, n_matches_raw, n_matches_used, pts0.astype(np.float32), pts1.astype(np.float32)


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
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=match_crosscheck)
    matches = matcher.match(des0, des1)
    matches = sorted(matches, key=lambda m: m.distance)
    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
    return triangulate_from_points(
        pts0,
        pts1,
        cam0,
        cam1,
        img0_ud,
        reproj_thres,
        use_ransac=use_ransac,
        ransac_method=ransac_method,
        ransac_thres=ransac_thres,
        ransac_prob=ransac_prob,
        ransac_iters=ransac_iters,
    )

def get_roma_matches(img0_ud, img1_ud, mask0, mask1, roma_model, args):
    from PIL import Image

    img0_rgb = cv2.cvtColor(img0_ud, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_ud, cv2.COLOR_BGR2RGB)
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    warp, certainty = roma_model.match(
        Image.fromarray(img0_rgb),
        Image.fromarray(img1_rgb),
        device=args.roma_device,
    )
    matches, certainty = roma_model.sample(warp, certainty)
    p0, p1 = roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)
    p0 = p0.detach().cpu().numpy().astype(np.float32)
    p1 = p1.detach().cpu().numpy().astype(np.float32)
    cert = certainty.detach().cpu().numpy().reshape(-1).astype(np.float32)

    valid = (
        (p0[:, 0] >= 0) & (p0[:, 0] < w0) & (p0[:, 1] >= 0) & (p0[:, 1] < h0) &
        (p1[:, 0] >= 0) & (p1[:, 0] < w1) & (p1[:, 1] >= 0) & (p1[:, 1] < h1)
    )
    if mask0 is not None:
        x0 = np.clip(np.round(p0[:, 0]).astype(np.int32), 0, w0 - 1)
        y0 = np.clip(np.round(p0[:, 1]).astype(np.int32), 0, h0 - 1)
        valid &= mask0[y0, x0] > 0
    if mask1 is not None:
        x1 = np.clip(np.round(p1[:, 0]).astype(np.int32), 0, w1 - 1)
        y1 = np.clip(np.round(p1[:, 1]).astype(np.int32), 0, h1 - 1)
        valid &= mask1[y1, x1] > 0

    p0 = p0[valid]
    p1 = p1[valid]
    cert = cert[valid]
    if p0.shape[0] == 0:
        return p0, p1

    if args.roma_max_matches > 0 and p0.shape[0] > args.roma_max_matches:
        idx = np.argsort(-cert)[: args.roma_max_matches]
        p0 = p0[idx]
        p1 = p1[idx]
    return p0, p1


def build_frame_cloud(frame, cameras, args, roma_model=None):
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

    feats = {}
    if args.matcher == "orb":
        orb = cv2.ORB_create(
            nfeatures=args.nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=args.fast_threshold,
        )
        for cam in cams_ok:
            item = data[cam]
            kp, des = orb.detectAndCompute(item["gray"], item["mask"])
            feats[cam] = {"kp": kp, "des": des}

    pair_cams = make_pairs(cams_ok, args.pair_mode)
    clouds = []
    colors = []
    obs_all = []
    offset = 0
    matches_total = 0
    ransac_inliers_total = 0
    for c0, c1 in pair_cams:
        if args.matcher == "orb":
            X, C, n_raw, n_used, uv0, uv1 = triangulate_pair(
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
        else:
            if roma_model is None:
                raise RuntimeError("matcher=roma requires initialized RoMa model")
            p0, p1 = get_roma_matches(
                data[c0]["img_ud"],
                data[c1]["img_ud"],
                data[c0]["mask"],
                data[c1]["mask"],
                roma_model,
                args,
            )
            X, C, n_raw, n_used, uv0, uv1 = triangulate_from_points(
                p0,
                p1,
                data[c0]["cam"],
                data[c1]["cam"],
                data[c0]["img_ud"],
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
        if args.ba:
            n = X.shape[0]
            idx = np.arange(offset, offset + n, dtype=np.int32)
            obs_all.append(
                {
                    "point_index": idx,
                    "cam0": c0,
                    "cam1": c1,
                    "uv0": uv0,
                    "uv1": uv1,
                }
            )
            offset += n
    if len(clouds) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), {
            "pairs": len(pair_cams),
            "matches": matches_total,
            "ransac_inliers": ransac_inliers_total,
            "points": 0,
        }, None

    X = np.concatenate(clouds, axis=0)
    C = np.concatenate(colors, axis=0)

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

    ba_pack = None
    if args.ba:
        selected = np.arange(offset, dtype=np.int32)
        if args.voxel_size > 0:
            # After voxel dedup, keep only points still present.
            selected = np.arange(X.shape[0], dtype=np.int32)
        if args.max_points > 0 and X.shape[0] > 0 and X.shape[0] <= len(selected):
            selected = np.arange(X.shape[0], dtype=np.int32)
        # Build map from old pre-filter indices to final point indices.
        # If voxel/max_points were applied, old-to-new cannot be fully recovered.
        # In this case, we conservatively disable BA cloud constraints for this frame.
        can_map = (args.voxel_size <= 0 and (args.max_points <= 0 or X.shape[0] == offset))
        if can_map and len(obs_all) > 0:
            obs_point = []
            obs_cam = []
            obs_uv = []
            for item in obs_all:
                pidx = item["point_index"]
                obs_point.append(pidx)
                obs_cam.append(np.array([item["cam0"]] * len(pidx)))
                obs_uv.append(item["uv0"])
                obs_point.append(pidx)
                obs_cam.append(np.array([item["cam1"]] * len(pidx)))
                obs_uv.append(item["uv1"])
            ba_pack = {
                "points3d": X.copy(),
                "obs_point": np.concatenate(obs_point, axis=0).astype(np.int32),
                "obs_cam": np.concatenate(obs_cam, axis=0),
                "obs_uv": np.concatenate(obs_uv, axis=0).astype(np.float32),
            }

    return X, C, {
        "pairs": len(pair_cams),
        "matches": matches_total,
        "ransac_inliers": ransac_inliers_total,
        "points": int(X.shape[0]),
    }, ba_pack


def pack_rt(rvec, tvec):
    return np.hstack([rvec.reshape(3), tvec.reshape(3)])


def unpack_rt(x):
    return x[:3].reshape(3, 1), x[3:6].reshape(3, 1)


def run_joint_ba(args, cameras, camnames, ref_cam, cloud_pack, k3d_pack, out_root):
    if least_squares is None:
        raise ImportError("scipy is required for --ba: pip install scipy")
    if len(cloud_pack["points3d"]) == 0 and len(k3d_pack["points3d"]) == 0:
        print("[BA] No cloud/k3d points available. Skip BA.")
        return None

    cam_order = [c for c in camnames if c != ref_cam]
    cam_init = []
    for cam in cam_order:
        rvec = cameras[cam]["Rvec"]
        tvec = cameras[cam]["T"]
        cam_init.append(pack_rt(rvec, tvec))
    cam_init = np.asarray(cam_init, dtype=np.float64).reshape(-1, 6)
    cam_prior = cam_init.copy()

    X_cloud = np.asarray(cloud_pack["points3d"], dtype=np.float64).reshape(-1, 3)
    X_k3d = np.asarray(k3d_pack["points3d"], dtype=np.float64).reshape(-1, 3)
    X_init = np.concatenate([X_cloud, X_k3d], axis=0) if X_k3d.size > 0 else X_cloud.copy()
    n_cloud = X_cloud.shape[0]
    n_k3d = X_k3d.shape[0]

    cam_idx = []
    pt_idx = []
    uv_obs = []
    w_obs = []
    for o in cloud_pack["obs"]:
        cam_idx.append(o[0])
        pt_idx.append(o[1])
        uv_obs.append([o[2], o[3]])
        w_obs.append(float(o[4]))
    for o in k3d_pack["obs"]:
        cam_idx.append(o[0])
        pt_idx.append(n_cloud + o[1])
        uv_obs.append([o[2], o[3]])
        w_obs.append(float(o[4]))
    cam_idx = np.asarray(cam_idx, dtype=np.int32)
    pt_idx = np.asarray(pt_idx, dtype=np.int32)
    uv_obs = np.asarray(uv_obs, dtype=np.float64)
    w_obs = np.asarray(w_obs, dtype=np.float64)

    camid_to_name = {i: c for i, c in enumerate(camnames)}
    opt_index = {cam: i for i, cam in enumerate(cam_order)}
    n_opt_cams = len(cam_order)
    n_points = X_init.shape[0]
    n_obs = len(cam_idx)

    def build_cam_state(cam_params):
        state = {}
        for cam in camnames:
            if cam == ref_cam:
                state[cam] = (cameras[cam]["R"], cameras[cam]["T"])
            else:
                i = opt_index[cam]
                rvec, tvec = unpack_rt(cam_params[i])
                R, _ = cv2.Rodrigues(rvec)
                state[cam] = (R, tvec)
        return state

    def compute_reproj_stats(cam_params, pts3d, obs_mask=None):
        cam_state = build_cam_state(cam_params)
        errs = []
        if obs_mask is None:
            obs_ids = range(n_obs)
        else:
            obs_ids = np.where(obs_mask)[0].tolist()
        for i in obs_ids:
            cam = camid_to_name[int(cam_idx[i])]
            R, T = cam_state[cam]
            K = cameras[cam]["K"]
            X = pts3d[int(pt_idx[i])].reshape(1, 3)
            rvec, _ = cv2.Rodrigues(R)
            uv_hat, _ = cv2.projectPoints(X, rvec, T, K, None)
            e = np.linalg.norm(uv_hat.reshape(2) - uv_obs[i])
            errs.append(float(e))
        if len(errs) == 0:
            return {"mean": float("nan"), "median": float("nan")}
        errs = np.asarray(errs, dtype=np.float64)
        return {"mean": float(np.mean(errs)), "median": float(np.median(errs))}

    def fun(x):
        ncam = len(cam_order) * 6
        cam_params = x[:ncam].reshape(-1, 6) if ncam > 0 else np.zeros((0, 6))
        pts3d = x[ncam:].reshape(-1, 3)
        cam_state = build_cam_state(cam_params)
        res = []
        for i in range(len(cam_idx)):
            cam = camid_to_name[int(cam_idx[i])]
            R, T = cam_state[cam]
            K = cameras[cam]["K"]
            X = pts3d[pt_idx[i]].reshape(1, 3)
            rvec, _ = cv2.Rodrigues(R)
            uv_hat, _ = cv2.projectPoints(X, rvec, T, K, None)
            duv = (uv_hat.reshape(2) - uv_obs[i]) * w_obs[i]
            res.extend(duv.tolist())
        if len(cam_order) > 0:
            sigma_r = max(float(args.ba_cam_sigma_r), 1e-8)
            sigma_t = max(float(args.ba_cam_sigma_t), 1e-8)
            for i in range(len(cam_order)):
                res.extend(((cam_params[i, :3] - cam_prior[i, :3]) / sigma_r).tolist())
                res.extend(((cam_params[i, 3:] - cam_prior[i, 3:]) / sigma_t).tolist())
        return np.asarray(res, dtype=np.float64)

    def build_jac_sparsity():
        if lil_matrix is None:
            return None
        ncam_params = n_opt_cams * 6
        npt_params = n_points * 3
        n_params = ncam_params + npt_params
        n_prior = ncam_params
        n_residuals = 2 * n_obs + n_prior
        A = lil_matrix((n_residuals, n_params), dtype=np.int8)

        off_cam = 0
        off_pt = ncam_params
        for i in range(n_obs):
            row0 = 2 * i
            row1 = row0 + 1
            cam_name = camid_to_name[int(cam_idx[i])]
            if cam_name != ref_cam:
                cidx = opt_index[cam_name]
                cbase = off_cam + cidx * 6
                A[row0, cbase : cbase + 6] = 1
                A[row1, cbase : cbase + 6] = 1
            pbase = off_pt + int(pt_idx[i]) * 3
            A[row0, pbase : pbase + 3] = 1
            A[row1, pbase : pbase + 3] = 1

        row_base = 2 * n_obs
        for i in range(ncam_params):
            A[row_base + i, off_cam + i] = 1
        return A

    x0 = np.concatenate([cam_init.reshape(-1), X_init.reshape(-1)], axis=0)
    obs_is_cloud = pt_idx < n_cloud
    obs_is_k3d = pt_idx >= n_cloud
    n_obs_cloud = int(obs_is_cloud.sum())
    n_obs_k3d = int(obs_is_k3d.sum())
    stats0 = compute_reproj_stats(cam_init, X_init)
    stats0_cloud = compute_reproj_stats(cam_init, X_init, obs_mask=obs_is_cloud)
    stats0_k3d = compute_reproj_stats(cam_init, X_init, obs_mask=obs_is_k3d)
    jac_sparsity = build_jac_sparsity()
    print(
        f"[BA] start: cams_opt={len(cam_order)} cloud_points={n_cloud} "
        f"k3d_points={n_k3d} obs={len(cam_idx)}"
    )
    print(f"[BA] observations: cloud={n_obs_cloud}, k3d={n_obs_k3d}")
    if n_k3d > 0 and n_obs_k3d == 0:
        print(
            "[BA][WARN] k3d points exist but no 2D keypoint observations loaded. "
            "Check --annot path and --pid/--ba_kpt_conf."
        )
    print(
        "[BA] reprojection BEFORE: "
        f"mean={stats0['mean']:.3f}px median={stats0['median']:.3f}px"
    )
    print(
        "[BA] reproj BEFORE (cloud): "
        f"mean={stats0_cloud['mean']:.3f}px median={stats0_cloud['median']:.3f}px"
    )
    print(
        "[BA] reproj BEFORE (k3d)  : "
        f"mean={stats0_k3d['mean']:.3f}px median={stats0_k3d['median']:.3f}px"
    )
    if jac_sparsity is not None:
        print("[BA] using sparse Jacobian structure")
    else:
        print("[BA] sparse Jacobian unavailable, fallback to dense finite-difference")
    result = least_squares(
        fun,
        x0,
        jac_sparsity=jac_sparsity,
        method="trf",
        x_scale="jac",
        ftol=1e-4,
        loss=args.ba_loss,
        f_scale=float(args.ba_f_scale),
        max_nfev=int(args.ba_max_nfev),
        verbose=2,
    )
    print(f"[BA] done: success={result.success}, cost={result.cost:.4f}")

    ncam = len(cam_order) * 6
    cam_opt = result.x[:ncam].reshape(-1, 6) if ncam > 0 else np.zeros((0, 6))
    pts_opt = result.x[ncam:].reshape(-1, 3)
    stats1 = compute_reproj_stats(cam_opt, pts_opt)
    stats1_cloud = compute_reproj_stats(cam_opt, pts_opt, obs_mask=obs_is_cloud)
    stats1_k3d = compute_reproj_stats(cam_opt, pts_opt, obs_mask=obs_is_k3d)
    print(
        "[BA] reprojection AFTER : "
        f"mean={stats1['mean']:.3f}px median={stats1['median']:.3f}px"
    )
    print(
        "[BA] reproj AFTER  (cloud): "
        f"mean={stats1_cloud['mean']:.3f}px median={stats1_cloud['median']:.3f}px"
    )
    print(
        "[BA] reproj AFTER  (k3d)  : "
        f"mean={stats1_k3d['mean']:.3f}px median={stats1_k3d['median']:.3f}px"
    )
    cam_state = build_cam_state(cam_opt)

    cams_out = {}
    for cam in camnames:
        R, T = cam_state[cam]
        cams_out[cam] = {
            "K": cameras[cam]["K"],
            "dist": cameras[cam]["dist"],
            "R": R,
            "Rvec": cv2.Rodrigues(R)[0],
            "T": T,
        }
        if "H" in cameras[cam]:
            cams_out[cam]["H"] = cameras[cam]["H"]
        if "W" in cameras[cam]:
            cams_out[cam]["W"] = cameras[cam]["W"]
    ba_root = join(out_root, "ba")
    os.makedirs(ba_root, exist_ok=True)
    write_intri(join(ba_root, "intri.yml"), cams_out)
    write_extri(join(ba_root, "extri.yml"), cams_out)

    return {
        "success": bool(result.success),
        "camnames": camnames,
        "ref_cam": ref_cam,
        "points_cloud_opt": pts_opt[:n_cloud].astype(np.float32),
        "points_k3d_opt": pts_opt[n_cloud:].astype(np.float32),
        "ba_root": ba_root,
    }


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
    parser.add_argument("--pair_mode", type=str, default="all", choices=["ring", "all"])
    parser.add_argument("--matcher", type=str, default="orb", choices=["orb", "roma"], help="feature matcher backend")
    parser.add_argument("--nfeatures", type=int, default=4000)
    parser.add_argument("--fast_threshold", type=int, default=10)
    parser.add_argument("--roma_device", type=str, default="cuda", help="device for RoMa matcher")
    parser.add_argument("--roma_max_matches", type=int, default=5000, help="max matches kept per pair for RoMa (-1 for all)")
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
    parser.add_argument("--pid", type=int, default=0, help="person id for k3d/2d association")
    parser.add_argument("--ba", action="store_true", help="experimental joint BA for cameras + cloud + keypoints3d")
    parser.add_argument("--ba_ref_cam", type=str, default="", help="fixed reference camera for BA (default first selected camera)")
    parser.add_argument("--ba_max_cloud_points", type=int, default=6000, help="max cloud points used in BA (-1 for all)")
    parser.add_argument("--ba_kpt_conf", type=float, default=0.1, help="2D keypoint confidence threshold for BA")
    parser.add_argument("--ba_loss", type=str, default="huber", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--ba_f_scale", type=float, default=3.0)
    parser.add_argument("--ba_max_nfev", type=int, default=40)
    parser.add_argument("--ba_cam_sigma_r", type=float, default=0.05, help="camera rotation prior sigma (rad)")
    parser.add_argument("--ba_cam_sigma_t", type=float, default=50.0, help="camera translation prior sigma (world unit)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.roma_max_matches == 0:
        args.roma_max_matches = -1

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
    roma_model = None
    if args.matcher == "roma":
        roma_model = init_roma_model(args)

    clouds_all = []
    stats_all = []
    cam_to_idx = {cam: i for i, cam in enumerate(subs)}
    cloud_points_global = []
    cloud_obs_global = []
    cloud_colors_global = []
    k3d_points_global = []
    k3d_obs_global = []
    k3d_frame_info = []
    k3d_root = args.k3d if args.k3d.startswith("/") else join(args.path, args.k3d)
    for frame in tqdm(common_frames, desc="extract pointcloud"):
        stem = frame.replace(args.ext, "")
        out_npz = join(out_points, stem + ".npz")
        out_ply = join(out_points, stem + ".ply")
        if exists(out_npz) and (not args.save_ply or exists(out_ply)) and not args.force:
            data = np.load(out_npz)
            clouds_all.append(data["xyz"])
            stats_all.append({"frame": frame, "pairs": -1, "matches": -1, "points": int(data["xyz"].shape[0])})
            continue

        X_cloud, C_cloud, stats, ba_pack = build_frame_cloud(
            frame,
            {k: cameras[k] for k in subs},
            args,
            roma_model=roma_model,
        )
        cloud_start = sum([p.shape[0] for p in cloud_points_global]) if len(cloud_points_global) > 0 else 0
        cloud_end = cloud_start + X_cloud.shape[0]
        cloud_points_global.append(X_cloud.astype(np.float32))
        cloud_colors_global.append(C_cloud.astype(np.uint8))
        if args.ba and ba_pack is not None:
            for i in range(ba_pack["obs_point"].shape[0]):
                cam = str(ba_pack["obs_cam"][i])
                if cam not in cam_to_idx:
                    continue
                pidx = int(ba_pack["obs_point"][i]) + cloud_start
                u, v = ba_pack["obs_uv"][i]
                cloud_obs_global.append((cam_to_idx[cam], pidx, float(u), float(v), 1.0))

        X = X_cloud.copy()
        C = C_cloud.copy()
        is_keypoint = np.zeros((X.shape[0],), dtype=np.uint8)

        if args.k3d != "":
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

        if args.ba and args.k3d != "":
            Xp, jids = read_keypoints3d_person(join(k3d_root, stem + ".json"), pid=args.pid, conf_thres=args.k3d_conf)
            k3d_start = sum([p.shape[0] for p in k3d_points_global]) if len(k3d_points_global) > 0 else 0
            k3d_points_global.append(Xp.astype(np.float32))
            k3d_frame_info.append((k3d_start, k3d_start + Xp.shape[0], stem, jids))
            if Xp.shape[0] > 0:
                for cam in subs:
                    annname = join(args.path, args.annot, cam, stem + ".json")
                    k2d = read_keypoints2d_person(annname, pid=args.pid)
                    if k2d is None:
                        continue
                    max_jid = int(jids.max()) if jids.size > 0 else -1
                    if max_jid >= k2d.shape[0]:
                        continue
                    k2d_sel = k2d[jids]
                    k2d_ud = undistort_pixels(k2d_sel, cameras[cam]["K"], cameras[cam]["dist"])
                    valid = k2d_ud[:, 2] > args.ba_kpt_conf
                    if valid.sum() == 0:
                        continue
                    vids = np.where(valid)[0]
                    for vi in vids:
                        u, v, conf = k2d_ud[vi]
                        pidx = k3d_start + int(vi)
                        w = float(np.clip(conf, 0.0, 1.0))
                        k3d_obs_global.append((cam_to_idx[cam], pidx, float(u), float(v), w))

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

    ba_result = None
    if args.ba:
        if args.ba_ref_cam != "":
            ref_cam = args.ba_ref_cam
        else:
            ref_cam = subs[0]
        cloud_points_all = np.concatenate(cloud_points_global, axis=0) if len(cloud_points_global) > 0 else np.zeros((0, 3), dtype=np.float32)
        cloud_colors_all = np.concatenate(cloud_colors_global, axis=0) if len(cloud_colors_global) > 0 else np.zeros((0, 3), dtype=np.uint8)
        cloud_obs = cloud_obs_global
        if args.ba_max_cloud_points > 0 and cloud_points_all.shape[0] > args.ba_max_cloud_points:
            rng = np.random.default_rng(args.seed)
            keep = rng.choice(cloud_points_all.shape[0], size=args.ba_max_cloud_points, replace=False)
            keep = np.sort(keep)
            remap = np.full((cloud_points_all.shape[0],), -1, dtype=np.int64)
            remap[keep] = np.arange(keep.shape[0], dtype=np.int64)
            cloud_points_all = cloud_points_all[keep]
            cloud_colors_all = cloud_colors_all[keep]
            cloud_obs = [o for o in cloud_obs if remap[o[1]] >= 0]
            cloud_obs = [(o[0], int(remap[o[1]]), o[2], o[3], o[4]) for o in cloud_obs]
            print(f"[BA] sampled cloud points: {keep.shape[0]}")
        k3d_points_all = np.concatenate(k3d_points_global, axis=0) if len(k3d_points_global) > 0 else np.zeros((0, 3), dtype=np.float32)
        ba_result = run_joint_ba(
            args,
            cameras,
            subs,
            ref_cam,
            {"points3d": cloud_points_all, "obs": cloud_obs, "colors": cloud_colors_all},
            {"points3d": k3d_points_all, "obs": k3d_obs_global},
            out_root,
        )
        if ba_result is not None:
            # Save refined cloud points used by BA.
            ba_points_dir = join(ba_result["ba_root"], "points")
            os.makedirs(ba_points_dir, exist_ok=True)
            np.savez_compressed(
                join(ba_points_dir, "cloud_points_ba.npz"),
                xyz=ba_result["points_cloud_opt"],
                rgb=cloud_colors_all[: ba_result["points_cloud_opt"].shape[0]],
            )
            # Save refined keypoints3d as per-frame files.
            if args.k3d != "" and ba_result["points_k3d_opt"].shape[0] > 0:
                out_k3d = join(ba_result["ba_root"], "keypoints3d")
                os.makedirs(out_k3d, exist_ok=True)
                cursor = 0
                for start, end, stem, jids in k3d_frame_info:
                    n = end - start
                    if n <= 0:
                        continue
                    pts = ba_result["points_k3d_opt"][cursor : cursor + n]
                    cursor += n
                    arr = np.zeros((int(jids.max()) + 1, 4), dtype=np.float32)
                    arr[:, 3] = 0.0
                    arr[jids, :3] = pts
                    arr[jids, 3] = 1.0
                    with open(join(out_k3d, stem + ".json"), "w") as f:
                        json.dump([{"id": args.pid, "keypoints3d": arr.tolist()}], f, indent=2)
                print(f"[BA] wrote refined keypoints3d: {out_k3d}")

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
