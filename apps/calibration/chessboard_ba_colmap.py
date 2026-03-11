"""
Use COLMAP's bundle_adjuster on chessboard corner observations.

Instead of running COLMAP's SIFT feature extraction and matching, this script
injects chessboard corners directly into a COLMAP sparse reconstruction, then
calls `colmap bundle_adjuster` to jointly refine camera intrinsics, extrinsics,
and 3D point positions.

Pipeline:
  1) Read camera parameters from intri.yml / extri.yml
  2) Read chessboard/<cam>/*.json corner detections
  3) Build multi-view tracks: each (frame, corner_id) becomes one 3D point
  4) Triangulate initial 3D points from camera pairs
  5) Write a COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
  6) Run `colmap bundle_adjuster`
  7) Read back the optimized model and write refined yml files

Inputs:
  - intri.yml / extri.yml
  - chessboard/<cam>/*.json

Outputs:
  - refined intri / extri yaml (intri_colmap_ba.yml, extri_colmap_ba.yml)
  - optimized 3D points npz (output/points_chess_colmap_ba.npz, key 'xyz')
    compatible with: python apps/calibration/vis_chess_sfm_ba.py <path>
"""

import os
import shutil
import tempfile
from collections import defaultdict
from glob import glob
from os.path import basename, join

import cv2
import numpy as np

from easymocap.mytools import read_json
from easymocap.mytools.camera_utils import read_camera, write_extri, write_intri
from easymocap.mytools.colmap_structure import (
    CAMERA_MODEL_NAMES,
    Camera,
    Image,
    Point3D,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    rotmat2qvec,
    write_cameras_binary,
    write_images_binary,
    write_points3d_binary,
)
from easymocap.mytools.debug_utils import run_cmd


def sample_list(lst, step):
    if step <= 1:
        return lst
    return lst[::step]


def resolve_path(root, path_or_name):
    if os.path.isabs(path_or_name):
        return path_or_name
    return join(root, path_or_name)


def parse_keypoints2d(data):
    """
    Robust parser for chessboard json formats.
    Supports:
      - {"keypoints2d": ...}
      - [{"keypoints2d": ...}, ...] (use first valid item)
      - raw list/array-like keypoints
    """
    if isinstance(data, dict):
        if "keypoints2d" not in data:
            return None
        arr = np.array(data["keypoints2d"], dtype=np.float64)
    elif isinstance(data, list):
        if len(data) == 0:
            return None
        if isinstance(data[0], dict):
            for item in data:
                if isinstance(item, dict) and "keypoints2d" in item:
                    arr = np.array(item["keypoints2d"], dtype=np.float64)
                    break
            else:
                return None
        else:
            arr = np.array(data, dtype=np.float64)
    else:
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        arr = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=np.float64)])
    return arr


def sample_frames_balanced(candidate_frames, frame_valid_cams, camnames, max_frames):
    """
    Select frames to be temporally spread while balancing camera coverage.
    """
    if max_frames <= 0 or len(candidate_frames) <= max_frames:
        return list(candidate_frames)

    n = len(candidate_frames)
    target = int(max_frames)
    selected = []
    selected_set = set()
    cam_counts = {cam: 0 for cam in camnames}
    frame_to_idx = {fr: i for i, fr in enumerate(candidate_frames)}

    # First pass: one frame per temporal bin, prioritizing under-represented cams.
    edges = np.linspace(0, n, target + 1)
    for bi in range(target):
        s = int(np.floor(edges[bi]))
        e = int(np.floor(edges[bi + 1]))
        if e <= s:
            e = min(s + 1, n)
        if s >= n:
            break
        center = 0.5 * (s + e - 1)
        best = None
        best_key = None
        for idx in range(s, e):
            fr = candidate_frames[idx]
            if fr in selected_set:
                continue
            valid = frame_valid_cams[fr]
            # Prefer frames that contain currently under-represented cameras.
            balance = sum(1.0 / (1.0 + cam_counts[c]) for c in valid)
            temporal = -abs(idx - center)
            key = (balance, len(valid), temporal)
            if best_key is None or key > best_key:
                best = fr
                best_key = key
        if best is None:
            continue
        selected.append(best)
        selected_set.add(best)
        for c in frame_valid_cams[best]:
            cam_counts[c] += 1

    # Fill any remaining slots using balance + temporal diversity.
    while len(selected) < target:
        chosen = None
        chosen_key = None
        selected_indices = [frame_to_idx[f] for f in selected] if selected else []
        for idx, fr in enumerate(candidate_frames):
            if fr in selected_set:
                continue
            valid = frame_valid_cams[fr]
            balance = sum(1.0 / (1.0 + cam_counts[c]) for c in valid)
            if selected_indices:
                min_dist = min(abs(idx - j) for j in selected_indices)
                spread = min_dist / max(1, n - 1)
            else:
                spread = 1.0
            key = (balance + 0.5 * spread, len(valid), spread)
            if chosen_key is None or key > chosen_key:
                chosen = fr
                chosen_key = key
        if chosen is None:
            break
        selected.append(chosen)
        selected_set.add(chosen)
        for c in frame_valid_cams[chosen]:
            cam_counts[c] += 1

    return sorted(selected, key=lambda f: frame_to_idx[f])


def undistort_uv(uv, K, dist):
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    return out


def triangulate_pair(c0, c1, uv0, uv1, cams):
    cam0, cam1 = cams[c0], cams[c1]
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


def triangulate_track(track_obs, cams, cam_centers):
    n = len(track_obs)
    if n < 2:
        return None
    best, best_score = None, 1e18
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
                uv_hat, _ = cv2.projectPoints(
                    X.reshape(1, 3), cams[c]["Rvec"], cams[c]["T"], cams[c]["K"], cams[c]["dist"]
                )
                errs.append(np.linalg.norm(uv_hat.reshape(2) - np.array([u, v])))
            score = float(np.mean(errs)) / baseline
            if score < best_score:
                best_score = score
                best = X
    return best


def compute_reprojection_error(cams, camnames, points3d, observations):
    """Compute per-observation reprojection errors."""
    errs = []
    for cam_idx, pt_idx, u, v, _w in observations:
        cam = camnames[cam_idx]
        uv_hat, _ = cv2.projectPoints(
            points3d[pt_idx].reshape(1, 3),
            cams[cam]["Rvec"],
            cams[cam]["T"],
            cams[cam]["K"],
            cams[cam]["dist"],
        )
        errs.append(np.linalg.norm(uv_hat.reshape(2) - np.array([u, v])))
    return np.array(errs)


def visualize_reprojection(
    root,
    cam,
    frame,
    cams,
    cams_opt,
    points_init,
    points3d_opt,
    observations,
    tracks,
    camnames,
    out_path=None,
):
    """
    Visualize detected corners, reprojected before BA, and reprojected after BA
    on a single image. Print per-point reprojection errors.
    """
    cam_to_idx = {c: i for i, c in enumerate(camnames)}
    if cam not in cam_to_idx:
        raise ValueError(f"Camera '{cam}' not in {camnames}")
    cam_idx = cam_to_idx[cam]

    # Filter observations for this (cam, frame)
    obs_here = []
    for k, (ci, pt_idx, u, v, conf) in enumerate(observations):
        if ci != cam_idx:
            continue
        if pt_idx >= len(tracks):
            continue
        if tracks[pt_idx]["frame"] != frame:
            continue
        obs_here.append((k, pt_idx, u, v))

    if not obs_here:
        print(
            f"[vis] No observations for cam={cam} frame={frame}. "
            f"Use a frame from the BA run (e.g. from used_frames)."
        )
        return

    # Load image (try images/cam/frame.ext, then cam/frame.ext)
    img_path = None
    for prefix in (join(root, "images", cam), join(root, cam)):
        for ext in (".jpg", ".png", ".jpeg"):
            p = join(prefix, frame + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is not None:
            break
    if img_path is None:
        print(f"[vis] Image not found: tried {root}/images/{cam}/{frame}.jpg|.png and {root}/{cam}/{frame}.jpg|.png")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[vis] Failed to load {img_path}")
        return

    vis = img.copy()
    h, w = vis.shape[:2]

    # Collect data per point
    pts_detected = []
    pts_reproj_before = []
    pts_reproj_after = []
    errs_before = []
    errs_after = []

    for k, pt_idx, u, v in obs_here:
        uv_before, _ = cv2.projectPoints(
            points_init[pt_idx].reshape(1, 3),
            cams[cam]["Rvec"],
            cams[cam]["T"],
            cams[cam]["K"],
            cams[cam]["dist"],
        )
        uv_after, _ = cv2.projectPoints(
            points3d_opt[pt_idx].reshape(1, 3),
            cams_opt[cam]["Rvec"],
            cams_opt[cam]["T"],
            cams_opt[cam]["K"],
            cams_opt[cam]["dist"],
        )
        uv_b = uv_before.reshape(2)
        uv_a = uv_after.reshape(2)
        eb = float(np.linalg.norm(uv_b - np.array([u, v])))
        ea = float(np.linalg.norm(uv_a - np.array([u, v])))

        pts_detected.append((u, v))
        pts_reproj_before.append(uv_b)
        pts_reproj_after.append(uv_a)
        errs_before.append(eb)
        errs_after.append(ea)

    pts_detected = np.array(pts_detected)
    pts_reproj_before = np.array(pts_reproj_before)
    pts_reproj_after = np.array(pts_reproj_after)
    errs_before = np.array(errs_before)
    errs_after = np.array(errs_after)

    # Draw: detected=green, reproj_before=red, reproj_after=blue
    radius = max(2, min(w, h) // 400)
    for i in range(len(pts_detected)):
        xd, yd = int(round(pts_detected[i, 0])), int(round(pts_detected[i, 1]))
        xb, yb = int(round(pts_reproj_before[i, 0])), int(round(pts_reproj_before[i, 1]))
        xa, ya = int(round(pts_reproj_after[i, 0])), int(round(pts_reproj_after[i, 1]))

        cv2.circle(vis, (xd, yd), radius, (0, 255, 0), -1)  # green: detected
        cv2.circle(vis, (xb, yb), radius, (0, 0, 255), -1)   # red: before BA
        cv2.circle(vis, (xa, ya), radius, (255, 0, 0), -1)   # blue: after BA

        # Line from detected to reproj_after (main error)
        cv2.line(vis, (xd, yd), (xa, ya), (255, 255, 0), 1)

    # Legend
    cv2.putText(
        vis, "green: detected", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
    )
    cv2.putText(
        vis, "red: reproj before BA", (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1
    )
    cv2.putText(
        vis, "blue: reproj after BA", (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1
    )
    cv2.putText(
        vis, "yellow line: detected -> after", (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1
    )

    # Print per-point errors
    print(f"\n[vis] Reprojection errors for cam={cam} frame={frame} (n={len(errs_before)} points)")
    print("  pt   err_before(px)   err_after(px)")
    print("  " + "-" * 36)
    for i in range(len(errs_before)):
        print(f"  {i:3d}   {errs_before[i]:12.3f}   {errs_after[i]:12.3f}")
    print("  " + "-" * 36)
    print(f"  mean {errs_before.mean():12.3f}   {errs_after.mean():12.3f}")
    print(f"  rms  {np.sqrt((errs_before**2).mean()):12.3f}   {np.sqrt((errs_after**2).mean()):12.3f}")

    # Save
    if out_path is None:
        out_path = join(root, "output", f"reproj_vis_{cam}_{frame}.jpg")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"[vis] Saved: {out_path}")


# ---------------------------------------------------------------------------
# COLMAP sparse model I/O helpers
# ---------------------------------------------------------------------------

def build_colmap_cameras(cams, camnames, img_hw):
    """One COLMAP camera per view, OPENCV model: fx fy cx cy k1 k2 p1 p2."""
    colmap_cams = {}
    for idx, cam in enumerate(camnames):
        K = cams[cam]["K"]
        dist = np.asarray(cams[cam]["dist"], dtype=np.float64).reshape(-1)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        k1 = float(dist[0]) if dist.size > 0 else 0.0
        k2 = float(dist[1]) if dist.size > 1 else 0.0
        p1 = float(dist[2]) if dist.size > 2 else 0.0
        p2 = float(dist[3]) if dist.size > 3 else 0.0
        h, w = img_hw.get(cam, (-1, -1))
        cam_id = idx + 1
        colmap_cams[cam_id] = Camera(
            id=cam_id,
            model="OPENCV",
            width=int(w),
            height=int(h),
            params=np.array([fx, fy, cx, cy, k1, k2, p1, p2], dtype=np.float64),
        )
    return colmap_cams


def build_colmap_images(cams, camnames, cam_id_map, per_image_xys, per_image_p3d_ids):
    """One COLMAP Image per camera view."""
    colmap_images = {}
    for idx, cam in enumerate(camnames):
        img_id = idx + 1
        cam_id = cam_id_map[cam]
        R = cams[cam]["R"].astype(np.float64)
        T = cams[cam]["T"].astype(np.float64).reshape(3)
        qvec = rotmat2qvec(R)
        xys = per_image_xys.get(cam, np.zeros((0, 2), dtype=np.float64))
        p3d_ids = per_image_p3d_ids.get(cam, np.array([], dtype=np.int64))
        colmap_images[img_id] = Image(
            id=img_id,
            qvec=qvec,
            tvec=T,
            camera_id=cam_id,
            name=f"{cam}.jpg",
            xys=xys,
            point3D_ids=p3d_ids,
        )
    return colmap_images


def build_colmap_points3d(points3d, point_tracks):
    """
    points3d: (N, 3)
    point_tracks: list of list[(image_id, point2d_idx)]
    """
    colmap_pts = {}
    for i in range(points3d.shape[0]):
        pid = i + 1
        track = point_tracks[i]
        image_ids = np.array([t[0] for t in track], dtype=np.int32)
        point2d_idxs = np.array([t[1] for t in track], dtype=np.int32)
        colmap_pts[pid] = Point3D(
            id=pid,
            xyz=points3d[i].astype(np.float64),
            rgb=np.array([128, 128, 128], dtype=np.uint8),
            error=0.0,
            image_ids=image_ids,
            point2D_idxs=point2d_idxs,
        )
    return colmap_pts


def read_back_colmap_model(sparse_path, camnames):
    """Read the COLMAP-optimized model and convert back to our camera dict."""
    colmap_cams = read_cameras_binary(join(sparse_path, "cameras.bin"))
    colmap_imgs = read_images_binary(join(sparse_path, "images.bin"))
    colmap_pts = read_points3d_binary(join(sparse_path, "points3D.bin"))

    name_to_cam = {}
    for cam in camnames:
        name_to_cam[f"{cam}.jpg"] = cam

    cams_out = {}
    for _img_id, img in colmap_imgs.items():
        cam_name = name_to_cam.get(img.name)
        if cam_name is None:
            continue
        colmap_cam = colmap_cams[img.camera_id]
        params = colmap_cam.params
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        k1 = params[4] if len(params) > 4 else 0.0
        k2 = params[5] if len(params) > 5 else 0.0
        p1 = params[6] if len(params) > 6 else 0.0
        p2 = params[7] if len(params) > 7 else 0.0
        dist = np.array([[k1, k2, p1, p2, 0.0]], dtype=np.float64)

        from easymocap.mytools.colmap_structure import qvec2rotmat

        R = qvec2rotmat(img.qvec).astype(np.float64)
        T = img.tvec.reshape(3, 1).astype(np.float64)
        Rvec = cv2.Rodrigues(R)[0].astype(np.float64)

        cams_out[cam_name] = {
            "K": K,
            "dist": dist,
            "R": R,
            "Rvec": Rvec,
            "T": T,
        }

    pts_list = []
    for _pid in sorted(colmap_pts.keys()):
        pts_list.append(colmap_pts[_pid].xyz)
    points3d_opt = np.array(pts_list, dtype=np.float64) if pts_list else np.zeros((0, 3))

    return cams_out, points3d_opt


def align_world_to_camera(cams, points3d, origin_cam):
    """
    Transform world so that origin_cam is at origin (identity R, zero T).
    Camera model: X_cam = R @ X_world + T. Camera center: C = -R^T @ T.
    """
    if origin_cam not in cams:
        raise RuntimeError(f"origin_cam '{origin_cam}' not in cameras {list(cams.keys())}")
    R0 = cams[origin_cam]["R"].astype(np.float64)
    T0 = cams[origin_cam]["T"].astype(np.float64).reshape(3)
    C0 = (-R0.T @ T0).reshape(3)

    cams_new = {}
    for cam, cd in cams.items():
        R = cd["R"].astype(np.float64)
        T = cd["T"].astype(np.float64).reshape(3)
        R_new = R @ R0.T
        T_new = (R @ C0 + T).reshape(3, 1)
        Rvec_new, _ = cv2.Rodrigues(R_new)
        cams_new[cam] = {**cd, "R": R_new, "T": T_new, "Rvec": Rvec_new}
    pts_new = (R0 @ (points3d - C0).T).T
    return cams_new, pts_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
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
        raise RuntimeError("Need >= 2 cameras with chessboard detections.")

    print(f"[colmap-BA] cameras: {camnames}")

    # Gather image sizes (needed for COLMAP camera model)
    img_hw = {}
    for cam in camnames:
        h, w = cams[cam].get("H", -1), cams[cam].get("W", -1)
        if h == -1 or w == -1:
            img_dir = join(root, "images", cam)
            if os.path.isdir(img_dir):
                sample = sorted(glob(join(img_dir, "*.jpg")))
                if not sample:
                    sample = sorted(glob(join(img_dir, "*.png")))
                if sample:
                    im = cv2.imread(sample[0])
                    if im is not None:
                        h, w = im.shape[:2]
        img_hw[cam] = (h, w)

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

    # Parse detections once and keep only frames with valid chessboard detections.
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
            k2d = parse_keypoints2d(data)
            if k2d is None:
                bad_json += 1
                continue
            if float(np.max(k2d[:, 2])) < args.conf:
                continue
            frame_k2d_cache[fr][cam] = k2d
            valid_cams.add(cam)
        frame_valid_cams[fr] = valid_cams

    candidate_frames = [fr for fr in all_frames if len(frame_valid_cams[fr]) >= args.min_views]
    if len(candidate_frames) == 0:
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
        f"[colmap-BA] frame selection: total_json_frames={len(all_frames)} "
        f"valid_frames={len(candidate_frames)} used_frames={len(used_frames)} "
        f"(max_frames={args.max_frames}) bad_or_invalid_json={bad_json}"
    )
    print(f"[colmap-BA] frame coverage per camera: {cam_frame_counts}")

    tracks = []
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
        rng = np.random.default_rng(args.seed)
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

    cam_to_idx = {c: i for i, c in enumerate(camnames)}

    points_init = []
    observations = []  # (cam_idx, pt_idx, u, v, conf)
    dropped = 0
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
        f"[colmap-BA] frames_used={len(used_frames)} kept={kept_frames} "
        f"tracks={len(tracks)} triangulated={points_init.shape[0]} dropped={dropped} "
        f"observations={len(observations)}"
    )
    print(
        f"[colmap-BA] corner points used: "
        f"2D_observations={len(observations)} 3D_tracks={points_init.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Compute reprojection error BEFORE
    # ------------------------------------------------------------------
    errs_before = compute_reprojection_error(cams, camnames, points_init, observations)
    print(
        f"[colmap-BA] reprojection BEFORE: "
        f"mean={errs_before.mean():.3f}px rms={np.sqrt((errs_before**2).mean()):.3f}px "
        f"n={len(errs_before)}"
    )

    # ------------------------------------------------------------------
    # Build COLMAP sparse model
    # ------------------------------------------------------------------
    # Per-image keypoint lists and point3D_id associations.
    # Each camera gets a flat list of 2D keypoints; the index into that list
    # is the point2D_idx used in the Point3D track.
    per_image_xys = {cam: [] for cam in camnames}
    per_image_p3d_ids = {cam: [] for cam in camnames}
    point_tracks = [[] for _ in range(points_init.shape[0])]

    for cam_idx, pt_idx, u, v, _conf in observations:
        cam = camnames[cam_idx]
        kp_idx = len(per_image_xys[cam])
        per_image_xys[cam].append([u, v])
        per_image_p3d_ids[cam].append(pt_idx + 1)  # COLMAP point IDs are 1-based
        img_id = cam_idx + 1  # COLMAP image IDs are 1-based
        point_tracks[pt_idx].append((img_id, kp_idx))

    for cam in camnames:
        if per_image_xys[cam]:
            per_image_xys[cam] = np.array(per_image_xys[cam], dtype=np.float64)
            per_image_p3d_ids[cam] = np.array(per_image_p3d_ids[cam], dtype=np.int64)
        else:
            per_image_xys[cam] = np.zeros((0, 2), dtype=np.float64)
            per_image_p3d_ids[cam] = np.array([], dtype=np.int64)

    cam_id_map = {cam: i + 1 for i, cam in enumerate(camnames)}

    colmap_cams = build_colmap_cameras(cams, camnames, img_hw)
    colmap_images = build_colmap_images(cams, camnames, cam_id_map, per_image_xys, per_image_p3d_ids)
    colmap_points = build_colmap_points3d(points_init, point_tracks)

    # Write to a temp workspace
    work_dir = args.work_dir
    if not work_dir:
        work_dir = tempfile.mkdtemp(prefix="colmap_chess_ba_")
    sparse_in = join(work_dir, "sparse_in")
    sparse_out = join(work_dir, "sparse_out")
    os.makedirs(sparse_in, exist_ok=True)
    os.makedirs(sparse_out, exist_ok=True)

    write_cameras_binary(colmap_cams, join(sparse_in, "cameras.bin"))
    write_images_binary(colmap_images, join(sparse_in, "images.bin"))
    write_points3d_binary(colmap_points, join(sparse_in, "points3D.bin"))
    print(f"[colmap-BA] wrote COLMAP sparse model to {sparse_in}")
    print(
        f"[colmap-BA]   cameras={len(colmap_cams)} images={len(colmap_images)} "
        f"points3D={len(colmap_points)}"
    )

    # Also create a dummy images/ folder so COLMAP doesn't complain
    dummy_img_dir = join(work_dir, "images")
    os.makedirs(dummy_img_dir, exist_ok=True)
    for cam in camnames:
        open(join(dummy_img_dir, f"{cam}.jpg"), "w").close()

    # ------------------------------------------------------------------
    # Run COLMAP bundle_adjuster
    # ------------------------------------------------------------------
    ba_flags = []
    if args.refine_intri:
        ba_flags.append("--BundleAdjustment.refine_focal_length 1")
        ba_flags.append("--BundleAdjustment.refine_principal_point 1")
        ba_flags.append("--BundleAdjustment.refine_extra_params 1")
    else:
        ba_flags.append("--BundleAdjustment.refine_focal_length 0")
        ba_flags.append("--BundleAdjustment.refine_principal_point 0")
        ba_flags.append("--BundleAdjustment.refine_extra_params 0")

    ba_flags.append(f"--BundleAdjustment.max_num_iterations {args.max_iter}")
    ba_flags.append(f"--BundleAdjustment.function_tolerance {args.func_tol}")

    ba_cmd = (
        f"{args.colmap} bundle_adjuster"
        f" --input_path {sparse_in}"
        f" --output_path {sparse_out}"
        f" {' '.join(ba_flags)}"
    )
    print(f"[colmap-BA] running: {ba_cmd}")
    run_cmd(ba_cmd)

    # ------------------------------------------------------------------
    # Read back results
    # ------------------------------------------------------------------
    cams_opt, points3d_opt = read_back_colmap_model(sparse_out, camnames)

    # Merge H/W and any keys the optimizer doesn't touch
    for cam in camnames:
        if cam in cams_opt:
            for key in ("H", "W"):
                if key in cams[cam]:
                    cams_opt[cam][key] = cams[cam][key]

    errs_after = compute_reprojection_error(cams_opt, camnames, points3d_opt, observations)
    print(
        f"[colmap-BA] reprojection AFTER : "
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
        print(f"[colmap-BA] aligned world to camera '{args.origin_cam}' (origin)")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    write_intri(out_intri, cams_opt)
    write_extri(out_extri, cams_opt)
    os.makedirs(os.path.dirname(out_points) or ".", exist_ok=True)
    np.savez_compressed(out_points, xyz=points3d_opt)
    print(f"[colmap-BA] wrote intrinsics : {out_intri}")
    print(f"[colmap-BA] wrote extrinsics : {out_extri}")
    print(f"[colmap-BA] wrote points3d   : {out_points} (xyz, N={points3d_opt.shape[0]})")
    print(f"[colmap-BA] visualize: python apps/calibration/vis_chess_sfm_ba.py {root}")

    # ------------------------------------------------------------------
    # Optional: visualize reprojection on a selected image
    # ------------------------------------------------------------------
    if args.vis_image:
        parts = args.vis_image.replace(",", " ").split()
        if len(parts) != 2:
            print(
                f"[vis] --vis_image expects 'cam,frame' e.g. '01,000000'. Got: {args.vis_image}"
            )
        else:
            vis_cam, vis_frame = parts[0].strip(), parts[1].strip()
            if vis_frame.endswith(".json"):
                vis_frame = vis_frame[:-5]
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

    if not args.keep_work_dir and not args.work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[colmap-BA] cleaned up temp dir: {work_dir}")
    else:
        print(f"[colmap-BA] work dir kept at: {work_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run COLMAP bundle_adjuster on chessboard corner observations."
    )
    parser.add_argument("path", type=str, help="dataset root")
    parser.add_argument("--intri", type=str, default="intri.yml")
    parser.add_argument("--extri", type=str, default="extri.yml")
    parser.add_argument("--chess", type=str, default="chessboard")
    parser.add_argument("--out_intri", type=str, default="intri_colmap_ba.yml")
    parser.add_argument("--out_extri", type=str, default="extri_colmap_ba.yml")
    parser.add_argument("--out_points", type=str, default="output/points_chess_colmap_ba.npz")

    parser.add_argument(
        "--origin_cam",
        type=str,
        default="01",
        help="camera to use as world origin (identity R, zero T); '' to disable",
    )
    parser.add_argument("--colmap", type=str, default="colmap", help="path to colmap binary")
    parser.add_argument("--work_dir", type=str, default="", help="temp workspace (auto if empty)")
    parser.add_argument("--keep_work_dir", action="store_true", help="keep the COLMAP workspace")

    parser.add_argument("--conf", type=float, default=0.1, help="min keypoint confidence")
    parser.add_argument("--min_views", type=int, default=2, help="min cameras per track")
    parser.add_argument("--step", type=int, default=1, help="sample chessboard frames by step")
    parser.add_argument("--max_frames", type=int, default=300, help="limit number of frames")
    parser.add_argument("--max_points", type=int, default=-1, help="cap tracks (-1 for all)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--refine_intri", dest="refine_intri", action="store_true")
    parser.add_argument("--no-refine_intri", dest="refine_intri", action="store_false")
    parser.add_argument("--max_iter", type=int, default=500, help="BA max iterations")
    parser.add_argument("--func_tol", type=float, default=1e-6, help="BA function tolerance")

    parser.add_argument(
        "--vis_image",
        type=str,
        default="",
        help="Visualize reprojection on image: 'cam,frame' e.g. '01,000000'",
    )
    parser.add_argument(
        "--vis_out",
        type=str,
        default="",
        help="Output path for reprojection vis (default: output/reproj_vis_<cam>_<frame>.jpg)",
    )

    parser.set_defaults(refine_intri=True)
    main(parser.parse_args())
