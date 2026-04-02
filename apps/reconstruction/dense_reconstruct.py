"""
COLMAP dense reconstruction with spatial-aware camera pairing.

Reads camera positions from the COLMAP sparse model and pairs each camera
with its K nearest spatial neighbors for stereo matching.  This prevents
false matches from repeated textures seen by distant/opposite cameras.

Pipeline:
    1. image_undistorter  (prepare dense workspace)
    2. Apply masks        (optional, blacks out background)
    3. Generate neighbor-based patch-match.cfg from camera positions
    4. patch_match_stereo (estimate depth maps)
    5. stereo_fusion      (fuse into dense point cloud)

Usage:
    python apps/reconstruction/dense_reconstruct.py \\
        /mnt/yubo/obj/cube/output/colmap \\
        --neighbor 3 --mask --min_num_pixels 2
"""

import argparse
import os
import subprocess
import shutil
import sys
from glob import glob
from os.path import join

import numpy as np

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..'))

from easymocap.mytools.colmap_structure import (
    read_images_binary,
    qvec2rotmat,
)


def get_camera_positions(sparse_dir):
    """Extract camera world positions from COLMAP sparse model.

    Returns dict[image_name] -> np.array([x, y, z])
    """
    images = read_images_binary(join(sparse_dir, 'images.bin'))
    positions = {}
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        center = -R.T @ t
        positions[img.name] = center
    return positions


def generate_neighbor_cfg(image_names, positions, neighbor_k, output_path):
    """Generate patch-match.cfg pairing each camera with its K nearest neighbors.

    Neighbors are determined by Euclidean distance between camera world positions,
    so this works for any camera layout (circular, linear, dome, etc.).
    """
    n = len(image_names)
    centers = np.array([positions[name] for name in image_names])

    dists = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)

    lines = []
    print(f'  Camera neighbor graph (K={neighbor_k}):')
    for i in range(n):
        sorted_idx = np.argsort(dists[i])
        # skip self (index 0), take K nearest
        nearest_idx = sorted_idx[1:neighbor_k + 1]
        sources = [image_names[j] for j in nearest_idx]
        source_dists = [dists[i, j] for j in nearest_idx]

        lines.append(image_names[i])
        lines.append(', '.join(sources))

        dist_str = ', '.join(f'{d:.3f}' for d in source_dists)
        print(f'    {image_names[i]} -> [{", ".join(sources)}]  (d={dist_str})')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Wrote {output_path}  ({n} cameras, K={neighbor_k} neighbors)')


def apply_masks(dense_img_dir, mask_dir, ext='.jpg'):
    """Apply foreground masks to dense images (set background to black)."""
    try:
        import cv2
    except ImportError:
        print('ERROR: OpenCV required for mask application')
        return False

    images = sorted(glob(join(dense_img_dir, f'*{ext}')))
    if not images:
        print(f'[dense] WARNING: no {ext} images found in {dense_img_dir}')
        return False

    count = 0
    for img_path in images:
        name = os.path.basename(img_path)
        mask_path = join(mask_dir, name + '.png')
        if not os.path.exists(mask_path):
            print(f'  {name}: no mask found at {mask_path}, skipping')
            continue
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        img[mask == 0] = 0
        cv2.imwrite(img_path, img)
        count += 1
        print(f'  Masked: {name}')

    print(f'[dense] Applied masks to {count}/{len(images)} images')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='COLMAP dense reconstruction with spatial camera pairing',
    )
    parser.add_argument('workspace',
                        help='COLMAP workspace (containing images/ and sparse/0/)')
    parser.add_argument('--neighbor', '-k', type=int, default=6,
                        help='Number of nearest camera neighbors for stereo matching '
                             '(default: 6)')
    parser.add_argument('--mask', action='store_true',
                        help='Apply masks from workspace/masks/ to dense images')
    parser.add_argument('--ext', default='.jpg',
                        help='Image extension (default: .jpg)')
    parser.add_argument('--colmap', default='colmap',
                        help='Path to COLMAP binary')
    parser.add_argument('--min_num_pixels', type=int, default=2,
                        help='stereo_fusion min_num_pixels (default: 2)')
    parser.add_argument('--max_reproj_error', type=float, default=2.0,
                        help='stereo_fusion max_reproj_error (default: 2.0)')
    parser.add_argument('--window_radius', type=int, default=5,
                        help='PatchMatchStereo window_radius (default: 5)')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='PatchMatchStereo num_iterations (default: 5)')
    parser.add_argument('--skip_undistort', action='store_true',
                        help='Skip image_undistorter (dense/ already exists)')
    args = parser.parse_args()

    ws = args.workspace
    dense_dir = join(ws, 'dense')
    img_dir = join(ws, 'images')
    sparse_dir = join(ws, 'sparse', '0')
    mask_dir = join(ws, 'masks')
    colmap = args.colmap

    # --- Step 0: Read camera positions from sparse model ---
    print(f'\n{"="*60}')
    print('[dense] Reading camera positions from sparse model')
    print(f'{"="*60}')
    positions = get_camera_positions(sparse_dir)
    for name, pos in sorted(positions.items()):
        print(f'  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]')

    # --- Step 1: Prepare dense workspace ---
    if not args.skip_undistort:
        print(f'\n{"="*60}')
        print('[dense] Step 1: image_undistorter')
        print(f'{"="*60}')
        if os.path.isdir(dense_dir):
            shutil.rmtree(dense_dir)
        cmd = (
            f'{colmap} image_undistorter'
            f' --image_path {img_dir}'
            f' --input_path {sparse_dir}'
            f' --output_path {dense_dir}'
            f' --output_type COLMAP'
        )
        print(f'  Running: {cmd}')
        subprocess.check_call(cmd, shell=True)
    else:
        print('[dense] Skipping image_undistorter (--skip_undistort)')

    dense_img_dir = join(dense_dir, 'images')

    # --- Step 2: Apply masks ---
    if args.mask and os.path.isdir(mask_dir):
        print(f'\n{"="*60}')
        print('[dense] Step 2: Applying masks to dense images')
        print(f'{"="*60}')
        apply_masks(dense_img_dir, mask_dir, args.ext)
    elif args.mask:
        print(f'[dense] WARNING: --mask set but {mask_dir} not found, skipping')

    # --- Step 3: Generate neighbor-based patch-match.cfg ---
    print(f'\n{"="*60}')
    print('[dense] Step 3: Generating spatial-neighbor patch-match.cfg')
    print(f'{"="*60}')
    image_names = sorted(
        f for f in os.listdir(dense_img_dir) if f.endswith(args.ext)
    )
    print(f'  Found {len(image_names)} images')

    cfg_path = join(dense_dir, 'stereo', 'patch-match.cfg')
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    generate_neighbor_cfg(image_names, positions, args.neighbor, cfg_path)

    # --- Step 4: Patch match stereo ---
    print(f'\n{"="*60}')
    print('[dense] Step 4: patch_match_stereo')
    print(f'{"="*60}')

    stereo_dir = join(dense_dir, 'stereo')
    for subdir in ['depth_maps', 'normal_maps']:
        d = join(stereo_dir, subdir)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    cmd = (
        f'{colmap} patch_match_stereo'
        f' --workspace_path {dense_dir}'
        f' --workspace_format COLMAP'
        f' --PatchMatchStereo.geom_consistency true'
        f' --PatchMatchStereo.window_radius {args.window_radius}'
        f' --PatchMatchStereo.num_iterations {args.num_iterations}'
    )
    print(f'  Running: {cmd}')
    subprocess.check_call(cmd, shell=True)

    # --- Step 5: Stereo fusion ---
    print(f'\n{"="*60}')
    print('[dense] Step 5: stereo_fusion')
    print(f'{"="*60}')
    fused_path = join(dense_dir, 'fused.ply')
    cmd = (
        f'{colmap} stereo_fusion'
        f' --workspace_path {dense_dir}'
        f' --workspace_format COLMAP'
        f' --output_path {fused_path}'
        f' --StereoFusion.min_num_pixels {args.min_num_pixels}'
        f' --StereoFusion.max_reproj_error {args.max_reproj_error}'
    )
    print(f'  Running: {cmd}')
    subprocess.check_call(cmd, shell=True)

    # --- Report ---
    print(f'\n{"="*60}')
    print(f'[dense] Done! Fused point cloud: {fused_path}')
    if os.path.exists(fused_path):
        size_mb = os.path.getsize(fused_path) / 1e6
        print(f'  File size: {size_mb:.1f} MB')
    print(f'{"="*60}')
    print(f'\nVisualize:')
    print(f'  python -c "import open3d as o3d; '
          f'pcd=o3d.io.read_point_cloud(\'{fused_path}\'); '
          f'print(len(pcd.points)); '
          f'o3d.visualization.draw_geometries([pcd])"')


if __name__ == '__main__':
    main()
