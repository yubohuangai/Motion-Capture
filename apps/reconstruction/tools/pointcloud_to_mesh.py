"""
Surface reconstruction from a cleaned point cloud via Poisson reconstruction.

Reads a .ply point cloud (with normals) and produces a triangle mesh with
vertex colors. If normals are missing, they are estimated automatically.

Usage:
    python apps/reconstruction/tools/pointcloud_to_mesh.py /path/to/cleaned.ply
    python apps/reconstruction/tools/pointcloud_to_mesh.py /path/to/data
    # If given a directory, looks for colmap_ws/dense/cleaned.ply.

Output: mesh.ply (triangle mesh with vertex colors) next to the input.
"""

import argparse
import os
import sys
import time
from os.path import join

import numpy as np
import open3d as o3d


def resolve_input(path):
    """Accept a .ply file, a colmap_ws directory, or a data root."""
    path = os.path.abspath(path)
    if path.endswith('.ply') and os.path.isfile(path):
        return path
    candidates = [
        join(path, 'dense', 'cleaned.ply'),
        join(path, 'colmap_ws', 'dense', 'cleaned.ply'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return path


def print_pcd_stats(pcd, label=""):
    pts = np.asarray(pcd.points)
    n = len(pts)
    if n == 0:
        print(f'  {label}: 0 points')
        return
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    print(f'  {label}: {n:,} points  '
          f'bbox=[{mn[0]:.4f}..{mx[0]:.4f}, {mn[1]:.4f}..{mx[1]:.4f}, {mn[2]:.4f}..{mx[2]:.4f}]')


def print_mesh_stats(mesh, label=""):
    verts = np.asarray(mesh.vertices)
    n_v, n_f = len(verts), len(mesh.triangles)
    if n_v == 0:
        print(f'  {label}: empty mesh')
        return
    mn, mx = verts.min(axis=0), verts.max(axis=0)
    print(f'  {label}: {n_v:,} vertices, {n_f:,} faces')
    print(f'    bbox: [{mn[0]:.4f}..{mx[0]:.4f}, {mn[1]:.4f}..{mx[1]:.4f}, {mn[2]:.4f}..{mx[2]:.4f}]')
    if mesh.is_watertight():
        print(f'    watertight: yes')
    else:
        print(f'    watertight: no')


def crop_mesh_to_pointcloud(mesh, pcd, margin_factor=0.1):
    """Remove Poisson vertices that extend far beyond the input point cloud."""
    pts = np.asarray(pcd.points)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    extent = mx - mn
    margin = extent * margin_factor
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=mn - margin, max_bound=mx + margin,
    )
    return mesh.crop(bbox)


def transfer_vertex_colors(mesh, pcd):
    """Transfer colors from the point cloud to mesh vertices via nearest-neighbor."""
    if not pcd.has_colors():
        print('  [color] point cloud has no colors, skipping')
        return mesh

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcd_colors = np.asarray(pcd.colors)
    verts = np.asarray(mesh.vertices)
    colors = np.zeros((len(verts), 3))

    for i in range(len(verts)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(verts[i], 1)
        colors[i] = pcd_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description='Surface reconstruction from point cloud (Poisson)',
    )
    parser.add_argument(
        'input',
        help='.ply point cloud or directory (auto-resolves to colmap_ws/dense/cleaned.ply)',
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Output mesh path (default: mesh.ply next to input)',
    )
    parser.add_argument('--depth', type=int, default=9,
                        help='Poisson octree depth — higher = finer detail (default: 9)')
    parser.add_argument('--density_quantile', type=float, default=0.05,
                        help='Remove faces below this density quantile (default: 0.05)')
    parser.add_argument('--crop_margin', type=float, default=0.1,
                        help='Crop margin as fraction of point cloud extent (default: 0.1)')
    parser.add_argument(
        '--smooth', action=argparse.BooleanOptionalAction, default=True,
        help='Apply Laplacian smoothing after reconstruction (default: on)',
    )
    parser.add_argument('--smooth_iterations', type=int, default=5,
                        help='Laplacian smoothing iterations (default: 5)')
    parser.add_argument(
        '--decimate', type=int, default=None,
        help='Target face count for mesh simplification (default: off)',
    )
    parser.add_argument(
        '--ball_pivot', action='store_true',
        help='Use ball-pivoting instead of Poisson (alternative method)',
    )

    args = parser.parse_args()

    # --- Resolve input ---
    ply_path = resolve_input(args.input)
    if not os.path.isfile(ply_path):
        print(f'[mesh] ERROR: cannot find point cloud at {ply_path}', file=sys.stderr)
        if os.path.isdir(args.input):
            print(f'  Tried: colmap_ws/dense/cleaned.ply under {args.input}',
                  file=sys.stderr)
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        out_path = join(os.path.dirname(ply_path), 'mesh.ply')

    print(f'[mesh] Input:  {ply_path}')
    print(f'[mesh] Output: {out_path}')

    t0 = time.time()
    pcd = o3d.io.read_point_cloud(ply_path)
    print_pcd_stats(pcd, 'input')

    if len(pcd.points) == 0:
        print('[mesh] ERROR: point cloud is empty', file=sys.stderr)
        sys.exit(1)

    # --- Ensure normals ---
    if not pcd.has_normals():
        print('[mesh] No normals found — estimating normals')
        pts = np.asarray(pcd.points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        sample_idx = np.random.default_rng(42).choice(
            len(pts), min(500, len(pts)), replace=False)
        nn_dists = []
        for i in sample_idx:
            _, idx, dist2 = tree.search_knn_vector_3d(pts[i], 6)
            nn_dists.extend(np.sqrt(dist2[1:]).tolist())
        nr = float(np.median(nn_dists)) * 4
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
    print(f'  normals: OK ({len(pcd.normals)} normals)')

    # --- Surface reconstruction ---
    if args.ball_pivot:
        print(f'\n[mesh] Ball-pivoting reconstruction')
        pts = np.asarray(pcd.points)
        tree = o3d.geometry.KDTreeFlann(pcd)
        sample_idx = np.random.default_rng(42).choice(
            len(pts), min(500, len(pts)), replace=False)
        nn_dists = []
        for i in sample_idx:
            _, idx, dist2 = tree.search_knn_vector_3d(pts[i], 6)
            nn_dists.extend(np.sqrt(dist2[1:]).tolist())
        avg_dist = float(np.mean(nn_dists))
        radii = [avg_dist * f for f in (1.0, 2.0, 4.0)]
        print(f'  radii: {[f"{r:.6f}" for r in radii]}')
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        print_mesh_stats(mesh, 'ball-pivot result')
    else:
        print(f'\n[mesh] Poisson reconstruction (depth={args.depth})')
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.depth)
        print_mesh_stats(mesh, 'raw Poisson')

        # Trim low-density faces (boundary artifacts)
        densities = np.asarray(densities)
        thresh = np.quantile(densities, args.density_quantile)
        print(f'\n[mesh] Trimming faces below density quantile '
              f'{args.density_quantile} (threshold={thresh:.4f})')
        face_verts = np.asarray(mesh.triangles)
        face_density = densities[face_verts].mean(axis=1)
        mesh.remove_triangles_by_mask(face_density <= thresh)
        mesh.remove_unreferenced_vertices()
        print_mesh_stats(mesh, 'after density trim')

    # --- Crop to point cloud bounds ---
    print(f'\n[mesh] Cropping to point cloud bounds (margin={args.crop_margin})')
    mesh = crop_mesh_to_pointcloud(mesh, pcd, margin_factor=args.crop_margin)
    mesh.remove_unreferenced_vertices()
    print_mesh_stats(mesh, 'after crop')

    # --- Transfer vertex colors ---
    print(f'\n[mesh] Transferring vertex colors from point cloud')
    mesh = transfer_vertex_colors(mesh, pcd)

    # --- Smoothing ---
    if args.smooth:
        print(f'\n[mesh] Laplacian smoothing ({args.smooth_iterations} iterations)')
        mesh = mesh.filter_smooth_laplacian(
            number_of_iterations=args.smooth_iterations)
        mesh.compute_vertex_normals()
        print_mesh_stats(mesh, 'after smoothing')

    # --- Decimation ---
    if args.decimate is not None:
        n_faces = len(mesh.triangles)
        if args.decimate < n_faces:
            print(f'\n[mesh] Decimating {n_faces:,} -> {args.decimate:,} faces')
            mesh = mesh.simplify_quadric_decimation(args.decimate)
            mesh.compute_vertex_normals()
            print_mesh_stats(mesh, 'after decimation')

    # --- Compute final normals and save ---
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False)

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'[mesh] Done in {elapsed:.1f}s')
    print_mesh_stats(mesh, 'final mesh')
    has_color = len(mesh.vertex_colors) > 0
    print(f'  vertex colors: {"yes" if has_color else "no"}')
    print(f'  saved: {out_path}')
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  file size: {size_mb:.1f} MB')
    print(f'{"="*60}')
    print(f'\nVisualize:')
    print(f'  python -c "import open3d as o3d; '
          f"mesh=o3d.io.read_triangle_mesh('{out_path}'); "
          f"mesh.compute_vertex_normals(); "
          f'o3d.visualization.draw_geometries([mesh])"')
    print(f'\nNext step:')
    print(f'  python apps/reconstruction/tools/texture_mesh.py {out_path} '
          f'--data /path/to/data')


if __name__ == '__main__':
    main()
