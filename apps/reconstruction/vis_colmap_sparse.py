"""
Visualize a COLMAP sparse model: triangulated 3D points + camera frustums.

Usage:
    python apps/reconstruction/vis_colmap_sparse.py /path/to/sparse/0
    python apps/reconstruction/vis_colmap_sparse.py /path/to/sparse/0 --images_dir /path/to/images
"""

import argparse
import os
import sys
from os.path import join

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d is required: pip install open3d")
    sys.exit(1)

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..'))

from easymocap.mytools.colmap_structure import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3d_binary,
    read_points3D_text,
    qvec2rotmat,
)


def read_model_auto(sparse_dir):
    """Read COLMAP model, preferring binary over text."""
    if os.path.exists(join(sparse_dir, 'cameras.bin')):
        cameras = read_cameras_binary(join(sparse_dir, 'cameras.bin'))
        images = read_images_binary(join(sparse_dir, 'images.bin'))
        points3D = read_points3d_binary(join(sparse_dir, 'points3D.bin'))
    else:
        cameras = read_cameras_text(join(sparse_dir, 'cameras.txt'))
        images = read_images_text(join(sparse_dir, 'images.txt'))
        points3D = read_points3D_text(join(sparse_dir, 'points3D.txt'))
    return cameras, images, points3D


def camera_center(image):
    """Compute camera center in world coordinates: C = -R^T @ t."""
    R = qvec2rotmat(image.qvec)
    t = image.tvec
    return -R.T @ t


def create_camera_frustum(image, cam, scale=0.3):
    """Create a wireframe camera frustum as an Open3D LineSet."""
    R = qvec2rotmat(image.qvec)
    t = image.tvec
    C = -R.T @ t

    params = cam.params
    if cam.model == 'PINHOLE':
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif cam.model == 'SIMPLE_PINHOLE':
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]

    W, H = cam.width, cam.height
    half_w = W / (2 * fx) * scale
    half_h = H / (2 * fy) * scale
    d = scale

    # frustum corners in camera space
    corners_cam = np.array([
        [0, 0, 0],
        [-half_w, -half_h, d],
        [half_w, -half_h, d],
        [half_w, half_h, d],
        [-half_w, half_h, d],
    ])

    # transform to world space
    corners_world = (R.T @ (corners_cam.T - t.reshape(3, 1))).T

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set, C


def main():
    parser = argparse.ArgumentParser(
        description='Visualize COLMAP sparse model (points + cameras)',
    )
    parser.add_argument('sparse_dir', help='Path to sparse/0/ directory')
    parser.add_argument('--images_dir', default=None,
                        help='Path to images/ directory (for reading colors)')
    parser.add_argument('--frustum_scale', type=float, default=0.3,
                        help='Camera frustum display scale (default: 0.3)')
    parser.add_argument('--point_size', type=float, default=2.0,
                        help='Point rendering size (default: 2.0)')
    args = parser.parse_args()

    print(f'[vis_colmap] Reading model from {args.sparse_dir} ...')
    cameras, images, points3D = read_model_auto(args.sparse_dir)

    print(f'  Cameras: {len(cameras)}')
    print(f'  Images:  {len(images)}')
    print(f'  Points:  {len(points3D)}')

    geometries = []

    # 3D points
    if len(points3D) > 0:
        pts_xyz = np.array([p.xyz for p in points3D.values()])
        pts_rgb = np.array([p.rgb for p in points3D.values()]) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pts_rgb)
        geometries.append(pcd)
        print(f'  Point cloud bounds: min={pts_xyz.min(axis=0)}, max={pts_xyz.max(axis=0)}')
    else:
        print('  WARNING: No 3D points in the model')

    # Camera frustums
    cam_colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 0.5, 0], [0.5, 0, 1], [0, 0.5, 1],
        [1, 0, 0.5], [0.5, 1, 0],
    ]
    cam_centers = []

    for idx, (img_id, image) in enumerate(sorted(images.items())):
        cam = cameras[image.camera_id]
        frustum, center = create_camera_frustum(image, cam, args.frustum_scale)
        color = cam_colors[idx % len(cam_colors)]
        frustum.paint_uniform_color(color)
        geometries.append(frustum)
        cam_centers.append(center)
        print(f'  Camera {image.name}: center={center.round(3)}')

    # World origin axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(axes)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='COLMAP Sparse Model', width=1280, height=720)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
