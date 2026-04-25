"""
Clean a dense point cloud: remove outliers, keep largest cluster, estimate normals.

Reads a .ply point cloud (e.g. COLMAP dense/fused.ply) and writes a cleaned
version ready for surface reconstruction.

Usage:
    python apps/reconstruction/tools/clean_pointcloud.py /path/to/data
    python apps/reconstruction/tools/clean_pointcloud.py /path/to/fused.ply
    # If given a directory, looks for colmap_ws/dense/fused.ply.
    # Writes cleaned.ply next to the input by default.

Cleaning steps (each prints before/after counts):
    1. Statistical outlier removal
    2. DBSCAN clustering — keep the largest cluster
    3. Radius outlier removal
    4. (Optional) Voxel downsampling
    5. Normal estimation
"""

import argparse
import os
import sys
import time
from os.path import join

import numpy as np
import open3d as o3d

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..', '..'))

from easymocap.mytools.colmap_structure import (
    read_images_binary,
    read_images_text,
    qvec2rotmat,
)


def resolve_input(path):
    """Accept a .ply file, a colmap_ws directory, or a data root."""
    path = os.path.abspath(path)
    if path.endswith('.ply') and os.path.isfile(path):
        return path
    candidates = [
        join(path, 'dense', 'fused.ply'),
        join(path, 'colmap_ws', 'dense', 'fused.ply'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return path


def find_sparse_dir(ply_path):
    """Walk up from the .ply to find the COLMAP sparse/0 directory."""
    d = os.path.dirname(ply_path)
    for _ in range(5):
        candidate = join(d, 'sparse', '0')
        if os.path.isdir(candidate):
            return candidate
        d = os.path.dirname(d)
    return None


def load_camera_centers(sparse_dir):
    """Read camera world positions from COLMAP sparse model."""
    images_bin = join(sparse_dir, 'images.bin')
    images_txt = join(sparse_dir, 'images.txt')
    if os.path.exists(images_bin):
        images = read_images_binary(images_bin)
    elif os.path.exists(images_txt):
        images = read_images_text(images_txt)
    else:
        return None
    centers = []
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        C = -R.T @ img.tvec
        centers.append(C)
    return np.array(centers)


def print_stats(pcd, label=""):
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n == 0:
        print(f'  {label}: 0 points')
        return
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    print(f'  {label}: {n:,} points')
    print(f'    bbox: X[{mn[0]:.4f}, {mx[0]:.4f}]  '
          f'Y[{mn[1]:.4f}, {mx[1]:.4f}]  '
          f'Z[{mn[2]:.4f}, {mx[2]:.4f}]')
    extent = mx - mn
    print(f'    extent: {extent[0]:.4f} x {extent[1]:.4f} x {extent[2]:.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='Clean a dense point cloud for surface reconstruction',
    )
    parser.add_argument(
        'input',
        help='.ply file, colmap_ws directory, or data root '
             '(auto-resolves to colmap_ws/dense/fused.ply)',
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Output .ply path (default: cleaned.ply next to input)',
    )

    g = parser.add_argument_group('statistical outlier removal')
    g.add_argument('--stat_neighbors', type=int, default=20,
                   help='K neighbors for statistical outlier test (default: 20)')
    g.add_argument('--stat_std', type=float, default=2.0,
                   help='Std ratio threshold (default: 2.0)')

    g = parser.add_argument_group('DBSCAN clustering')
    g.add_argument(
        '--dbscan', action=argparse.BooleanOptionalAction, default=True,
        help='Keep only the largest DBSCAN cluster (default: on)',
    )
    g.add_argument('--dbscan_eps', type=float, default=None,
                   help='DBSCAN epsilon (default: auto from point spacing)')
    g.add_argument('--dbscan_min', type=int, default=100,
                   help='DBSCAN min_points (default: 100)')

    g = parser.add_argument_group('radius outlier removal')
    g.add_argument('--radius_nb', type=int, default=16,
                   help='Min neighbors within radius (default: 16)')
    g.add_argument('--radius', type=float, default=None,
                   help='Search radius (default: auto from point spacing)')

    g = parser.add_argument_group('voxel downsampling')
    g.add_argument('--voxel', type=float, default=None,
                   help='Voxel size for downsampling (default: off)')

    g = parser.add_argument_group('normals')
    g.add_argument(
        '--normals', action=argparse.BooleanOptionalAction, default=True,
        help='Estimate normals (needed for Poisson reconstruction) (default: on)',
    )
    g.add_argument('--normal_radius', type=float, default=None,
                   help='Normal estimation search radius (default: auto)')

    args = parser.parse_args()

    # --- Resolve input ---
    ply_path = resolve_input(args.input)
    if not os.path.isfile(ply_path):
        print(f'[clean] ERROR: cannot find point cloud at {ply_path}', file=sys.stderr)
        if os.path.isdir(args.input):
            print(f'  Tried: {args.input}/dense/fused.ply and '
                  f'{args.input}/colmap_ws/dense/fused.ply', file=sys.stderr)
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        out_path = join(os.path.dirname(ply_path), 'cleaned.ply')

    print(f'[clean] Input:  {ply_path}')
    print(f'[clean] Output: {out_path}')

    t0 = time.time()
    pcd = o3d.io.read_point_cloud(ply_path)
    print_stats(pcd, 'input')

    if len(pcd.points) == 0:
        print('[clean] ERROR: point cloud is empty', file=sys.stderr)
        sys.exit(1)

    # Auto-compute point spacing for default parameters
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    sample_idx = np.random.default_rng(42).choice(len(pts), min(500, len(pts)), replace=False)
    nn_dists = []
    for i in sample_idx:
        _, idx, dist2 = tree.search_knn_vector_3d(pts[i], 6)
        nn_dists.extend(np.sqrt(dist2[1:]).tolist())
    median_spacing = float(np.median(nn_dists))
    print(f'  median nearest-neighbor spacing: {median_spacing:.6f}')

    # --- Step 1: Statistical outlier removal ---
    print(f'\n[clean] Step 1: Statistical outlier removal '
          f'(K={args.stat_neighbors}, std={args.stat_std})')
    n_before = len(pcd.points)
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=args.stat_neighbors, std_ratio=args.stat_std,
    )
    print(f'  removed {n_before - len(pcd.points):,} outliers')
    print_stats(pcd, 'after statistical')

    # --- Step 2: DBSCAN clustering ---
    if args.dbscan and len(pcd.points) > args.dbscan_min:
        eps = args.dbscan_eps if args.dbscan_eps is not None else median_spacing * 3
        print(f'\n[clean] Step 2: DBSCAN clustering '
              f'(eps={eps:.6f}, min_points={args.dbscan_min})')
        labels = np.asarray(pcd.cluster_dbscan(
            eps=eps, min_points=args.dbscan_min, print_progress=False,
        ))
        n_clusters = labels.max() + 1
        n_noise = (labels == -1).sum()
        print(f'  found {n_clusters} cluster(s), {n_noise:,} noise points')

        if n_clusters > 0:
            counts = np.bincount(labels[labels >= 0])
            largest = int(np.argmax(counts))
            for i, c in enumerate(counts):
                marker = ' <-- kept' if i == largest else ''
                print(f'    cluster {i}: {c:,} points{marker}')
            pcd = pcd.select_by_index(np.where(labels == largest)[0])
        print_stats(pcd, 'after DBSCAN')
    else:
        print('\n[clean] Step 2: DBSCAN skipped')

    # --- Step 3: Radius outlier removal ---
    r = args.radius if args.radius is not None else median_spacing * 3
    print(f'\n[clean] Step 3: Radius outlier removal '
          f'(nb={args.radius_nb}, radius={r:.6f})')
    n_before = len(pcd.points)
    pcd, ind = pcd.remove_radius_outlier(nb_points=args.radius_nb, radius=r)
    print(f'  removed {n_before - len(pcd.points):,} outliers')
    print_stats(pcd, 'after radius')

    # --- Step 4: Voxel downsampling ---
    if args.voxel is not None and args.voxel > 0:
        print(f'\n[clean] Step 4: Voxel downsampling (size={args.voxel})')
        n_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f'  {n_before:,} -> {len(pcd.points):,}')
        print_stats(pcd, 'after voxel')
    else:
        print('\n[clean] Step 4: Voxel downsampling skipped (use --voxel to enable)')

    # --- Step 5: Normal estimation ---
    if args.normals:
        nr = args.normal_radius if args.normal_radius is not None else median_spacing * 4
        print(f'\n[clean] Step 5: Normal estimation (radius={nr:.6f}, max_nn=30)')
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=30)
        )

        sparse_dir = find_sparse_dir(ply_path)
        cam_centers = load_camera_centers(sparse_dir) if sparse_dir else None

        if cam_centers is not None:
            print(f'  orienting normals toward {len(cam_centers)} camera centers')
            cam_centroid = cam_centers.mean(axis=0)
            pcd.orient_normals_towards_camera_location(cam_centroid)
        else:
            print('  WARNING: no COLMAP sparse model found — '
                  'falling back to tangent-plane orientation (may flip)')
            pcd.orient_normals_consistent_tangent_plane(k=15)

        print(f'  normals computed for {len(pcd.points):,} points')
    else:
        print('\n[clean] Step 5: Normal estimation skipped')

    # --- Save ---
    o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'[clean] Done in {elapsed:.1f}s')
    print_stats(pcd, 'final')
    print(f'  Saved: {out_path}')
    print(f'  Has normals: {pcd.has_normals()}')
    print(f'  Has colors:  {pcd.has_colors()}')
    print(f'{"="*60}')
    print(f'\nNext step:')
    print(f'  python apps/reconstruction/tools/pointcloud_to_mesh.py {out_path}')


if __name__ == '__main__':
    main()
