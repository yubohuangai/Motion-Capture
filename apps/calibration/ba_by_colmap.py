"""
Bundle adjust camera extrinsics with COLMAP using existing intri/extri as priors.

Pipeline:
  1) Copy one frame per camera to a COLMAP workspace.
  2) Create COLMAP database + initial model from intri/extri.
  3) Run COLMAP feature extraction + matching.
  4) Run point triangulation + bundle adjustment.

Notes:
  - Requires COLMAP installed and in PATH (or use --colmap).
  - Uses OPENCV camera model (fx, fy, cx, cy, k1, k2, p1, p2).
  - Intrinsics are taken from intri.yml; extrinsics from extri.yml.
"""

import os
from os.path import join
import argparse
import cv2
import numpy as np

from easymocap.mytools.camera_utils import read_camera, write_camera
from easymocap.mytools.colmap_wrapper import (
    COLMAPDatabase,
    create_empty_db,
    colmap_feature_extract,
    colmap_feature_match,
    colmap_ba,
    copy_images,
)
from easymocap.mytools.colmap_structure import (
    Camera,
    Image,
    rotmat2qvec,
    write_cameras_binary,
    write_images_binary,
    write_points3d_binary,
)
from easymocap.mytools.debug_utils import log, mywarn


def _opencv_params_from_k_dist(K, dist):
    dist = dist.reshape(-1)
    k1 = float(dist[0]) if dist.size > 0 else 0.0
    k2 = float(dist[1]) if dist.size > 1 else 0.0
    p1 = float(dist[2]) if dist.size > 2 else 0.0
    p2 = float(dist[3]) if dist.size > 3 else 0.0
    if dist.size > 4 and abs(dist[4]) > 1e-9:
        mywarn("COLMAP OPENCV model ignores k3; dropping k3 from intrinsics.")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return [fx, fy, cx, cy, k1, k2, p1, p2]


def create_initial_model(out_dir, cameras, image_names):
    sparse_dir = join(out_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # One camera per view (no shared intrinsics).
    cameras_colmap = {}
    images_colmap = {}
    cam_ids = {}

    for cam_id, cam in enumerate(sorted(image_names.keys()), start=1):
        img_path = image_names[cam]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        height, width = img.shape[:2]

        K = cameras[cam]["K"]
        dist = cameras[cam]["dist"]
        params = _opencv_params_from_k_dist(K, dist)

        cam_obj = Camera(
            id=cam_id,
            model="OPENCV",
            width=width,
            height=height,
            params=params,
        )
        cameras_colmap[cam_id] = cam_obj
        cam_ids[cam] = cam_id

    for img_id, cam in enumerate(sorted(image_names.keys()), start=1):
        R = cameras[cam]["R"]
        T = cameras[cam]["T"]
        qvec = rotmat2qvec(R)
        tvec = T.reshape(3)
        name = os.path.basename(image_names[cam])
        img_obj = Image(
            id=img_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=cam_ids[cam],
            name=name,
            xys=[],
            point3D_ids=[],
        )
        images_colmap[img_id] = img_obj

    write_cameras_binary(cameras_colmap, join(sparse_dir, "cameras.bin"))
    write_images_binary(images_colmap, join(sparse_dir, "images.bin"))
    write_points3d_binary({}, join(sparse_dir, "points3D.bin"))

    return sparse_dir


def main():
    parser = argparse.ArgumentParser(
        description="Bundle adjust with COLMAP using intri/extri priors."
    )
    parser.add_argument("path", type=str, help="Dataset root (contains images/)")
    parser.add_argument("out", type=str, help="Output COLMAP workspace")
    parser.add_argument("--intri", type=str, default=None, help="Path to intri.yml")
    parser.add_argument("--extri", type=str, default=None, help="Path to extri.yml")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to copy")
    parser.add_argument("--colmap", type=str, default="colmap", help="COLMAP binary")
    parser.add_argument("--add_mask", action="store_true", help="Use masks if present")
    parser.add_argument("--mask", type=str, default="mask", help="Mask folder name")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for SIFT/matching")
    parser.add_argument("--export_opencv", action="store_true", help="Write intri.yml/extri.yml from COLMAP result")
    args = parser.parse_args()

    intri_path = args.intri or join(args.path, "intri.yml")
    extri_path = args.extri or join(args.path, "extri.yml")

    if not os.path.exists(intri_path) or not os.path.exists(extri_path):
        raise FileNotFoundError(f"Missing intri/extri: {intri_path}, {extri_path}")

    # Prepare workspace and copy images
    os.makedirs(args.out, exist_ok=True)
    ok, image_names = copy_images(args.path, args.out, nf=args.frame, mask=args.mask, add_mask=args.add_mask)
    if not ok:
        raise RuntimeError("Failed to copy images for COLMAP workspace.")

    # Load intri/extri as priors
    cameras = read_camera(intri_path, extri_path)
    cameras.pop("basenames", None)

    # Initialize COLMAP database + model
    db_path = join(args.out, "database.db")
    create_empty_db(db_path)
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    # Add cameras and images with priors
    for cam in sorted(image_names.keys()):
        img = cv2.imread(image_names[cam])
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_names[cam]}")
        height, width = img.shape[:2]
        params = _opencv_params_from_k_dist(cameras[cam]["K"], cameras[cam]["dist"])
        camera_id = db.add_camera(
            model=4,  # OPENCV
            width=width,
            height=height,
            params=params,
            prior_focal_length=False,
        )
        qvec = rotmat2qvec(cameras[cam]["R"])
        tvec = cameras[cam]["T"].reshape(3)
        db.add_image(
            name=os.path.basename(image_names[cam]),
            camera_id=camera_id,
            prior_q=qvec,
            prior_t=tvec,
        )

    db.commit()
    db.close()

    # Write initial model for point_triangulator + BA
    create_initial_model(args.out, cameras, image_names)

    # Feature extraction / matching
    colmap_feature_extract(args.colmap, args.out, share_camera=False, add_mask=args.add_mask, gpu=args.gpu)
    colmap_feature_match(args.colmap, args.out, gpu=args.gpu)

    # Triangulate + bundle adjust
    colmap_ba(args.colmap, args.out, with_init=True)

    if args.export_opencv:
        # Export refined cameras to OpenCV YAML
        from easymocap.mytools.colmap_structure import read_model, qvec2rotmat
        sparse_dir = join(args.out, "sparse", "0")
        cams, imgs, _ = read_model(sparse_dir, ".bin")
        cameras_out = {}
        for img in imgs.values():
            cam = cams[img.camera_id]
            p = cam.params
            if cam.model == "OPENCV":
                fx, fy, cx, cy, k1, k2, p1, p2 = p
                K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float64).reshape(3, 3)
                dist = np.array([[k1, k2, p1, p2, 0.0]], dtype=np.float64)
            else:
                raise RuntimeError(f"Unsupported camera model for export: {cam.model}")
            R = qvec2rotmat(img.qvec)
            T = img.tvec.reshape(3, 1)
            cameras_out[os.path.splitext(img.name)[0]] = {
                "K": K,
                "dist": dist,
                "R": R,
                "T": T,
            }
        write_camera(cameras_out, sparse_dir)
        log(f"[COLMAP] Exported intri.yml/extri.yml to {sparse_dir}")


if __name__ == "__main__":
    main()
