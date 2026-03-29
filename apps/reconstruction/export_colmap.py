"""
Export EasyMocap calibration (intri.yml / extri.yml) to a COLMAP workspace.

Produces a directory that 3D Gaussian Splatting, Nerfstudio, NeuS, and other
multi-view reconstruction frameworks can ingest directly:

    <output>/
    ├── images/          # undistorted, flat-named (01.jpg, 02.jpg, ...)
    ├── masks/           # optional (01.jpg.png, 02.jpg.png, ...)
    └── sparse/0/
        ├── cameras.bin  (+ cameras.txt)
        ├── images.bin   (+ images.txt)
        └── points3D.bin (+ points3D.txt)   [empty]

Usage:
    python apps/reconstruction/export_colmap.py /path/to/data \\
        --frame 0 --output /path/to/colmap_ws --undistort
"""

import argparse
import os
import shutil
import sys
from glob import glob
from os.path import join

import cv2
import numpy as np

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..'))

from easymocap.mytools.camera_utils import read_camera
from easymocap.mytools.colmap_structure import (
    Camera,
    Image,
    Point3D,
    rotmat2qvec,
    write_cameras_binary,
    write_cameras_text,
    write_images_binary,
    write_images_text,
    write_points3d_binary,
    write_points3D_text,
)


def detect_image_size(data_root, cam_names, frame, ext):
    """Read one image per camera to get (H, W). Returns dict[cam] -> (H, W)."""
    sizes = {}
    for cam in cam_names:
        pattern = join(data_root, 'images', cam, f'{frame:06d}{ext}')
        if not os.path.exists(pattern):
            candidates = sorted(glob(join(data_root, 'images', cam, f'*{ext}')))
            if len(candidates) == 0:
                raise FileNotFoundError(
                    f'No images found for camera {cam} in '
                    f'{join(data_root, "images", cam)}'
                )
            pattern = candidates[0]
        img = cv2.imread(pattern)
        if img is None:
            raise RuntimeError(f'Failed to read {pattern}')
        h, w = img.shape[:2]
        sizes[cam] = (h, w)
    return sizes


def build_colmap_cameras(cameras, cam_names, sizes, undistort):
    """
    Build COLMAP Camera entries (one per physical camera).

    When *undistort* is True the camera model is PINHOLE (4 params) with the
    optimal new-K obtained from cv2.getOptimalNewCameraMatrix.  Otherwise the
    original OPENCV model (8 params) is kept so that downstream tools handle
    distortion themselves.

    Returns
    -------
    colmap_cameras : dict[int, Camera]
    new_K_map : dict[str, np.ndarray]   (cam_name -> 3x3 new K, or original K)
    cam_id_map : dict[str, int]         (cam_name -> colmap camera_id)
    """
    colmap_cameras = {}
    new_K_map = {}
    cam_id_map = {}

    for idx, cam in enumerate(cam_names):
        cam_id = idx + 1
        cam_id_map[cam] = cam_id

        K = cameras[cam]['K']
        dist = cameras[cam]['dist']
        h, w = sizes[cam]

        if undistort:
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
            fx, fy, cx, cy = new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2]
            params = np.array([fx, fy, cx, cy], dtype=np.float64)
            model = 'PINHOLE'
            new_K_map[cam] = new_K
        else:
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            d = dist.flatten()
            k1 = d[0] if len(d) > 0 else 0.0
            k2 = d[1] if len(d) > 1 else 0.0
            p1 = d[2] if len(d) > 2 else 0.0
            p2 = d[3] if len(d) > 3 else 0.0
            params = np.array([fx, fy, cx, cy, k1, k2, p1, p2], dtype=np.float64)
            model = 'OPENCV'
            new_K_map[cam] = K

        colmap_cameras[cam_id] = Camera(
            id=cam_id, model=model, width=w, height=h, params=params,
        )

    return colmap_cameras, new_K_map, cam_id_map


def build_colmap_images(cameras, cam_names, cam_id_map, ext):
    """Build COLMAP Image entries with world-to-camera quaternion + translation."""
    colmap_images = {}
    for idx, cam in enumerate(cam_names):
        img_id = idx + 1
        R = cameras[cam]['R']
        T = cameras[cam]['T']
        qvec = rotmat2qvec(R)
        tvec = T.flatten()
        image_name = f'{cam}{ext}'
        colmap_images[img_id] = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=cam_id_map[cam],
            name=image_name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.array([], dtype=np.int64),
        )
    return colmap_images


def process_images(data_root, output_dir, cam_names, cameras,
                   new_K_map, frame, ext, undistort, mask_dir):
    """Copy (and optionally undistort) images + masks to the output workspace."""
    out_img_dir = join(output_dir, 'images')
    os.makedirs(out_img_dir, exist_ok=True)

    for cam in cam_names:
        src = join(data_root, 'images', cam, f'{frame:06d}{ext}')
        if not os.path.exists(src):
            candidates = sorted(glob(join(data_root, 'images', cam, f'*{ext}')))
            if len(candidates) == 0:
                raise FileNotFoundError(f'No image for camera {cam}')
            src = candidates[0]

        dst = join(out_img_dir, f'{cam}{ext}')

        if undistort:
            img = cv2.imread(src)
            K = cameras[cam]['K']
            dist = cameras[cam]['dist']
            new_K = new_K_map[cam]
            img_undist = cv2.undistort(img, K, dist, None, new_K)
            cv2.imwrite(dst, img_undist)
        else:
            shutil.copy2(src, dst)

        if mask_dir is not None:
            mask_src = join(data_root, mask_dir, cam, f'{frame:06d}.png')
            if os.path.exists(mask_src):
                out_mask_dir = join(output_dir, 'masks')
                os.makedirs(out_mask_dir, exist_ok=True)
                mask_dst = join(out_mask_dir, f'{cam}{ext}.png')
                if undistort:
                    mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
                    K = cameras[cam]['K']
                    dist = cameras[cam]['dist']
                    new_K = new_K_map[cam]
                    mask_undist = cv2.undistort(mask, K, dist, None, new_K)
                    cv2.imwrite(mask_dst, mask_undist)
                else:
                    shutil.copy2(mask_src, mask_dst)


def write_colmap_model(output_dir, colmap_cameras, colmap_images):
    """Write COLMAP sparse model in both binary and text format."""
    sparse_dir = join(output_dir, 'sparse', '0')
    os.makedirs(sparse_dir, exist_ok=True)

    empty_points = {}

    write_cameras_binary(colmap_cameras, join(sparse_dir, 'cameras.bin'))
    write_images_binary(colmap_images, join(sparse_dir, 'images.bin'))
    write_points3d_binary(empty_points, join(sparse_dir, 'points3D.bin'))

    write_cameras_text(colmap_cameras, join(sparse_dir, 'cameras.txt'))
    write_images_text(colmap_images, join(sparse_dir, 'images.txt'))
    write_points3D_text(empty_points, join(sparse_dir, 'points3D.txt'))


def main():
    parser = argparse.ArgumentParser(
        description='Export EasyMocap calibration to COLMAP workspace',
    )
    parser.add_argument('data', help='Root data path containing images/ and intri.yml/extri.yml')
    parser.add_argument('--output', '-o', required=True,
                        help='Output COLMAP workspace directory')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to export (default: 0)')
    parser.add_argument('--intri', default='intri.yml',
                        help='Intrinsics file name (default: intri.yml)')
    parser.add_argument('--extri', default='extri.yml',
                        help='Extrinsics file name (default: extri.yml)')
    parser.add_argument('--ext', default='.jpg',
                        help='Image extension (default: .jpg)')
    parser.add_argument('--undistort', action='store_true',
                        help='Undistort images and export PINHOLE cameras')
    parser.add_argument('--mask', default=None,
                        help='Mask sub-directory name (e.g. "mask" or "masks"). '
                             'If set, copies masks alongside images.')
    args = parser.parse_args()

    intri_path = join(args.data, args.intri)
    extri_path = join(args.data, args.extri)

    print(f'[export_colmap] Reading cameras from {intri_path}, {extri_path}')
    cams = read_camera(intri_path, extri_path)
    cam_names = cams.pop('basenames')
    print(f'[export_colmap] Found {len(cam_names)} cameras: {cam_names}')

    print('[export_colmap] Detecting image sizes ...')
    sizes = detect_image_size(args.data, cam_names, args.frame, args.ext)
    for cam in cam_names:
        h, w = sizes[cam]
        print(f'  {cam}: {w}x{h}')

    print('[export_colmap] Building COLMAP cameras ...')
    colmap_cameras, new_K_map, cam_id_map = build_colmap_cameras(
        cams, cam_names, sizes, args.undistort,
    )

    print('[export_colmap] Building COLMAP images ...')
    colmap_images = build_colmap_images(cams, cam_names, cam_id_map, args.ext)

    print(f'[export_colmap] Processing images (undistort={args.undistort}) ...')
    process_images(
        args.data, args.output, cam_names, cams,
        new_K_map, args.frame, args.ext, args.undistort, args.mask,
    )

    print('[export_colmap] Writing COLMAP sparse model ...')
    write_colmap_model(args.output, colmap_cameras, colmap_images)

    print(f'[export_colmap] Done. Output at: {args.output}')
    print(f'  images/        — {len(cam_names)} images')
    print(f'  sparse/0/      — cameras.{{bin,txt}}, images.{{bin,txt}}, points3D.{{bin,txt}}')
    print()
    print('Next steps:')
    print(f'  3DGS:       python train.py -s {args.output} --iterations 7000')
    print(f'  Nerfstudio: ns-train neus-facto --data {args.output} colmap')


if __name__ == '__main__':
    main()
