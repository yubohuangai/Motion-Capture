"""
COLMAP dense reconstruction with circular camera layout support.

For camera rigs where cameras are arranged in a circle (01, 02, ..., N),
this script generates a custom patch-match config that only pairs each
camera with its circular neighbors, preventing false matches from
repeated textures on opposite sides of the object.

Pipeline:
    1. image_undistorter  (prepare dense workspace)
    2. Apply masks        (optional, blacks out background)
    3. Generate circular patch-match.cfg
    4. patch_match_stereo (estimate depth maps)
    5. stereo_fusion      (fuse into dense point cloud)

Usage:
    python apps/reconstruction/dense_reconstruct.py \\
        /mnt/yubo/obj/cube/output/colmap \\
        --neighbor 3 --mask --min_num_pixels 2

    The input is the COLMAP workspace produced by export_colmap.py,
    which must contain images/ and sparse/0/.
"""

import argparse
import os
import subprocess
import shutil
from glob import glob
from os.path import join


def generate_circular_cfg(image_names, neighbor_k, output_path):
    """Generate patch-match.cfg for a circular camera arrangement.

    Each camera uses its ±neighbor_k nearest neighbors as source images,
    wrapping around the circle. Sources are ordered by proximity.
    """
    n = len(image_names)
    lines = []
    for i in range(n):
        ref = image_names[i]
        sources = []
        for delta in range(1, neighbor_k + 1):
            sources.append(image_names[(i + delta) % n])
            sources.append(image_names[(i - delta) % n])
        lines.append(ref)
        lines.append(', '.join(sources))

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[dense] Wrote {output_path}  ({n} cameras, ±{neighbor_k} neighbors)')


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
        description='COLMAP dense reconstruction with circular camera support',
    )
    parser.add_argument('workspace',
                        help='COLMAP workspace (containing images/ and sparse/0/)')
    parser.add_argument('--neighbor', '-k', type=int, default=3,
                        help='Number of circular neighbors per side (default: 3, '
                             'so each camera uses 2*k=6 source images)')
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

    # --- Step 3: Generate circular patch-match.cfg ---
    print(f'\n{"="*60}')
    print('[dense] Step 3: Generating circular patch-match.cfg')
    print(f'{"="*60}')
    image_names = sorted(
        f for f in os.listdir(dense_img_dir) if f.endswith(args.ext)
    )
    print(f'  Found {len(image_names)} images: {image_names}')

    cfg_path = join(dense_dir, 'stereo', 'patch-match.cfg')
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    generate_circular_cfg(image_names, args.neighbor, cfg_path)

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
