"""
Visualize chessboard SfM-BA / COLMAP-BA outputs:
  - points_chess_colmap_ba.npz (or points_chess_sfm_ba.npz)
  - intri_colmap_ba.yml / extri_colmap_ba.yml

Shows:
  - point cloud
  - camera centers + frustum wireframes
"""

import os
from os.path import join

import cv2
import numpy as np
import open3d as o3d

try:
    from easymocap.mytools.camera_utils import read_camera as _read_camera
except Exception:
    _read_camera = None


def _read_mat(fs, key):
    node = fs.getNode(key)
    if node.empty():
        return None
    return node.mat()


def read_camera_fallback(intri_path, extri_path):
    fs_i = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    fs_e = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    if not fs_i.isOpened():
        raise FileNotFoundError(intri_path)
    if not fs_e.isOpened():
        raise FileNotFoundError(extri_path)

    names_node = fs_i.getNode("names")
    camnames = []
    for i in range(names_node.size()):
        camnames.append(names_node.at(i).string())

    cameras = {}
    for cam in camnames:
        K = _read_mat(fs_i, f"K_{cam}")
        dist = _read_mat(fs_i, f"dist_{cam}")
        rvec = _read_mat(fs_e, f"R_{cam}")
        R = _read_mat(fs_e, f"Rot_{cam}")
        T = _read_mat(fs_e, f"T_{cam}")
        if R is None and rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
        if rvec is None and R is not None:
            rvec, _ = cv2.Rodrigues(R)
        if K is None or dist is None or R is None or rvec is None or T is None:
            raise RuntimeError(f"Missing camera params for {cam}")
        cameras[cam] = {"K": K, "dist": dist, "R": R, "Rvec": rvec, "T": T}

    fs_i.release()
    fs_e.release()
    cameras["basenames"] = camnames
    return cameras


def read_camera(intri_path, extri_path):
    if _read_camera is not None:
        return _read_camera(intri_path, extri_path)
    return read_camera_fallback(intri_path, extri_path)


def resolve_path(root, path_or_name):
    if os.path.isabs(path_or_name):
        return path_or_name
    return join(root, path_or_name)


def world_from_camera(camera, Xc):
    # Camera model: Xc = R * Xw + T  =>  Xw = R^T * (Xc - T)
    R = camera["R"]
    T = camera["T"]
    return (R.T @ (Xc.reshape(3, 1) - T)).reshape(3)


def get_camera_center(camera):
    return (-camera["R"].T @ camera["T"]).reshape(3)


def estimate_wh_from_k(camera):
    # If explicit image size is unavailable, infer rough size from principal point.
    K = camera["K"]
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    w = int(max(2 * cx, 64))
    h = int(max(2 * cy, 64))
    return w, h


def make_frustum_lines(camera, scale=0.35):
    K = camera["K"].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = estimate_wh_from_k(camera)

    # 5 points in world: camera center + 4 image-plane corners at depth=scale.
    c_world = get_camera_center(camera)
    z = float(scale)
    corners_uv = np.array(
        [
            [0.0, 0.0],
            [float(w), 0.0],
            [float(w), float(h)],
            [0.0, float(h)],
        ],
        dtype=np.float64,
    )
    corners_cam = np.zeros((4, 3), dtype=np.float64)
    corners_cam[:, 0] = (corners_uv[:, 0] - cx) / fx * z
    corners_cam[:, 1] = (corners_uv[:, 1] - cy) / fy * z
    corners_cam[:, 2] = z
    corners_world = np.stack([world_from_camera(camera, p) for p in corners_cam], axis=0)

    pts = np.vstack([c_world[None], corners_world])
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    return pts, lines


def make_line_set(points, lines, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def to_point_cloud(points, color=(0.85, 0.85, 0.85)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main(args):
    root = args.path
    intri_path = resolve_path(root, args.intri)
    extri_path = resolve_path(root, args.extri)
    points_path = resolve_path(root, args.points)

    if not os.path.exists(points_path):
        raise FileNotFoundError(points_path)
    cameras = read_camera(intri_path, extri_path)
    camnames = cameras.pop("basenames")

    data = np.load(points_path)
    if "xyz" not in data:
        raise KeyError(f"{points_path} must contain key 'xyz'")
    xyz = data["xyz"].astype(np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz shape must be (N,3), got {xyz.shape}")

    if args.max_points > 0 and xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz = xyz[keep]

    geometries = []
    geometries.append(to_point_cloud(xyz, color=(0.8, 0.8, 0.8)))
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size, origin=[0, 0, 0]))

    if len(args.subs) > 0:
        camnames = [c for c in camnames if c in args.subs]

    print(f"[VIS] points={xyz.shape[0]} cameras={len(camnames)}")
    for cam in camnames:
        pts, lines = make_frustum_lines(cameras[cam], scale=args.frustum_size)
        geometries.append(make_line_set(pts, lines, color=(0.15, 0.65, 1.0)))
        center = get_camera_center(cameras[cam]).reshape(3)
        print(f"[VIS] {cam} center: {center}")

    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset root")
    parser.add_argument("--intri", type=str, default="intri_colmap_ba.yml")
    parser.add_argument("--extri", type=str, default="extri_colmap_ba.yml")
    parser.add_argument("--points", type=str, default="points_chess_colmap_ba.npz")
    parser.add_argument("--subs", type=str, nargs="+", default=[], help="camera subset to visualize")
    parser.add_argument("--max_points", type=int, default=-1, help="random subsample for visualization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frustum_size", type=float, default=0.35)
    parser.add_argument("--axis_size", type=float, default=0.4)
    args = parser.parse_args()
    main(args)
