"""
Visualize a 3D Gaussian Splatting .ply point cloud using Open3D.

Usage:
    python apps/reconstruction_classical/viz/vis_gaussians.py /path/to/point_cloud.ply
"""

import argparse
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("open3d is required: pip install open3d")
    sys.exit(1)

from plyfile import PlyData


def load_gaussian_ply(path):
    """Load a 3DGS .ply and extract positions + colors."""
    ply = PlyData.read(path)
    v = ply.elements[0]

    xyz = np.stack([v['x'], v['y'], v['z']], axis=1)

    # 3DGS stores colors as spherical harmonics coefficients (DC band)
    # f_dc_0/1/2 are the first SH coefficients; convert to RGB via sigmoid-like transform
    if 'f_dc_0' in v.data.dtype.names:
        C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
        r = v['f_dc_0'] * C0 + 0.5
        g = v['f_dc_1'] * C0 + 0.5
        b = v['f_dc_2'] * C0 + 0.5
        rgb = np.stack([r, g, b], axis=1).clip(0, 1)
    elif 'red' in v.data.dtype.names:
        rgb = np.stack([v['red'], v['green'], v['blue']], axis=1) / 255.0
    else:
        rgb = np.ones((len(xyz), 3)) * 0.5

    # Opacity for filtering low-confidence Gaussians
    if 'opacity' in v.data.dtype.names:
        opacity = 1.0 / (1.0 + np.exp(-v['opacity']))  # stored as logit
    else:
        opacity = np.ones(len(xyz))

    return xyz, rgb, opacity


def main():
    parser = argparse.ArgumentParser(description='Visualize 3DGS point cloud')
    parser.add_argument('ply', help='Path to point_cloud.ply')
    parser.add_argument('--opacity', type=float, default=0.5,
                        help='Filter Gaussians below this opacity (default: 0.5)')
    parser.add_argument('--point_size', type=float, default=1.0,
                        help='Point rendering size (default: 1.0)')
    parser.add_argument('--save', default=None,
                        help='Save filtered cloud to this .ply path')
    args = parser.parse_args()

    print(f'Loading {args.ply} ...')
    xyz, rgb, opacity = load_gaussian_ply(args.ply)
    print(f'  Total Gaussians: {len(xyz):,}')
    print(f'  Opacity: >0.5 solid={int((opacity>0.5).sum()):,}, '
          f'0.1-0.5={int(((opacity>=0.1)&(opacity<=0.5)).sum()):,}, '
          f'<0.1 ghost={int((opacity<0.1).sum()):,}')

    mask = opacity >= args.opacity
    xyz = xyz[mask]
    rgb = rgb[mask]
    opacity = opacity[mask]
    print(f'  After opacity filter (>={args.opacity}): {len(xyz):,}')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    if args.save:
        o3d.io.write_point_cloud(args.save, pcd)
        print(f'  Saved to {args.save}')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Gaussian Splatting Viewer', width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
