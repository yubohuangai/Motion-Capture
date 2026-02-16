"""
Visualize extracted point clouds from EasyMocap-style pointcloud output.

Expected folder layout:
    pointcloud/
      meta.pkl
      pointclouds.pkl
      points/
        000000.npz   # contains xyz, rgb
        000001.npz
        ...
        000000.ply   # optional
"""
import os
import pickle
import importlib
from os.path import join, exists

import numpy as np


def list_point_files(points_dir, ext):
    files = [f for f in os.listdir(points_dir) if f.endswith(ext)]
    files.sort()
    return files


def get_ordered_frames(pointcloud_root, points_dir):
    meta_name = join(pointcloud_root, "meta.pkl")
    if exists(meta_name):
        try:
            with open(meta_name, "rb") as f:
                meta = pickle.load(f)
            frames = meta.get("frames", [])
            if len(frames) > 0:
                stems = [os.path.splitext(fr)[0] for fr in frames]
                return stems
        except Exception:
            pass
    npz_names = list_point_files(points_dir, ".npz")
    if len(npz_names) > 0:
        return [os.path.splitext(n)[0] for n in npz_names]
    ply_names = list_point_files(points_dir, ".ply")
    return [os.path.splitext(n)[0] for n in ply_names]


def load_npz(npz_name):
    data = np.load(npz_name)
    xyz = data["xyz"].astype(np.float64)
    if "rgb" in data:
        rgb = data["rgb"].astype(np.float64) / 255.0
    else:
        rgb = np.full((xyz.shape[0], 3), 0.8, dtype=np.float64)
    return xyz, rgb


def import_o3d():
    """
    Prefer lightweight Open3D import path to avoid slow ml-related imports.
    """
    try:
        return importlib.import_module("open3d.cpu.pybind")
    except Exception:
        pass
    try:
        return importlib.import_module("open3d")
    except Exception as e:
        raise RuntimeError("open3d is required. Install with: pip install open3d") from e


def load_cloud(points_dir, stem):
    npz_name = join(points_dir, stem + ".npz")
    if exists(npz_name):
        return load_npz(npz_name)
    ply_name = join(points_dir, stem + ".ply")
    if exists(ply_name):
        o3d = import_o3d()
        pcd = o3d.io.read_point_cloud(ply_name)
        xyz = np.asarray(pcd.points, dtype=np.float64)
        if len(pcd.colors) > 0:
            rgb = np.asarray(pcd.colors, dtype=np.float64)
        else:
            rgb = np.full((xyz.shape[0], 3), 0.8, dtype=np.float64)
        return xyz, rgb
    raise FileNotFoundError(f"No point file for frame: {stem}")


def subsample_points(xyz, rgb, max_points):
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    rng = np.random.default_rng(0)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx], rgb[idx]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="/Users/yubo/data/s2/seq1/gt/check/pointcloud/",
        help="path to pointcloud folder",
    )
    parser.add_argument("--frame", type=int, default=-1, help="show one frame by index")
    parser.add_argument("--play", action="store_true", help="play sequence")
    parser.add_argument("--fps", type=float, default=20.0, help="playback fps")
    parser.add_argument("--every", type=int, default=1, help="frame stride")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--max_points", type=int, default=50000, help="per-frame point cap for rendering")
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--reset_view_each_frame", action="store_true")
    args = parser.parse_args()

    o3d = import_o3d()

    pointcloud_root = args.path
    points_dir = join(pointcloud_root, "points")
    if not exists(points_dir):
        raise FileNotFoundError(f"points directory not found: {points_dir}")

    stems = get_ordered_frames(pointcloud_root, points_dir)
    if len(stems) == 0:
        raise RuntimeError(f"No npz/ply files found in {points_dir}")
    stems = stems[::max(1, args.every)]
    if args.max_frames > 0:
        stems = stems[:args.max_frames]

    if args.frame >= 0:
        if args.frame >= len(stems):
            raise IndexError(f"--frame {args.frame} out of range [0, {len(stems)-1}]")
        stems = [stems[args.frame]]
        args.play = False

    if not args.play:
        xyz, rgb = load_cloud(points_dir, stems[0])
        xyz, rgb = subsample_points(xyz, rgb, args.max_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"PointCloud {stems[0]}")
        opt = vis.get_render_option()
        opt.point_size = float(args.point_size)
        opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        vis.add_geometry(pcd)
        vis.reset_view_point(True)
        vis.run()
        vis.destroy_window()
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PointCloud Sequence")
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)
    opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Initialize with first valid frame so view bbox is correct.
    xyz0, rgb0 = load_cloud(points_dir, stems[0])
    xyz0, rgb0 = subsample_points(xyz0, rgb0, args.max_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz0)
    pcd.colors = o3d.utility.Vector3dVector(rgb0)
    vis.add_geometry(pcd)
    vis.reset_view_point(True)

    delay = 1.0 / max(args.fps, 1e-6)
    for i, stem in enumerate(stems):
        xyz, rgb = load_cloud(points_dir, stem)
        xyz, rgb = subsample_points(xyz, rgb, args.max_points)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis.update_geometry(pcd)
        if args.reset_view_each_frame:
            vis.reset_view_point(True)
        vis.poll_events()
        vis.update_renderer()
        print(f"[{i+1:04d}/{len(stems):04d}] frame={stem} points={xyz.shape[0]}")
        import time
        time.sleep(delay)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
