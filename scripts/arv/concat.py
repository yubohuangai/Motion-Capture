# coding: utf-8
import os
import cv2
from glob import glob
import argparse

pose_2d_dir = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/01_view40/pose2D"
pose_3d_dir = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/01_view40/output_3D/images"
concat_dir = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/01_view40/output_3D/concat"

parser = argparse.ArgumentParser(description="Concatenate images")
parser.add_argument(
    "--mode",
    choices=["filename", "index"],
    default="index",
    help="filename: match by name | index: index from pose_3d filename"
)
args = parser.parse_args()

os.makedirs(concat_dir, exist_ok=True)


def list_images(dir_path, exts=("jpg", "png")):
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(dir_path, f"*.{ext}")))
    return sorted(paths)


pose_3d_paths = list_images(pose_3d_dir)
pose_2d_paths = list_images(pose_2d_dir)

num_2d = len(pose_2d_paths)

for pose_3d_path in pose_3d_paths:
    filename = os.path.basename(pose_3d_path)
    stem, _ = os.path.splitext(filename)

    if args.mode == "filename":
        pose_2d_path = os.path.join(pose_2d_dir, filename)
        if not os.path.exists(pose_2d_path):
            print(f"[WARN] 2D image not found for {filename}")
            continue

    else:  # index mode
        try:
            idx = int(stem)   # "000006" → 6
        except ValueError:
            print(f"[WARN] Cannot parse index from {filename}")
            continue

        if idx >= num_2d:
            print(f"[WARN] Index {idx} out of range for 2D images")
            continue

        pose_2d_path = pose_2d_paths[idx]

    pose_3d_img = cv2.imread(pose_3d_path)
    pose_2d_img = cv2.imread(pose_2d_path)

    if pose_3d_img is None or pose_2d_img is None:
        print(f"[WARN] Failed to read {filename}")
        continue

    # Resize 2D to match 3D height
    if pose_3d_img.shape[0] != pose_2d_img.shape[0]:
        h = pose_3d_img.shape[0]
        w = int(pose_2d_img.shape[1] * h / pose_2d_img.shape[0])
        pose_2d_img = cv2.resize(pose_2d_img, (w, h))

    concat_img = cv2.hconcat([pose_2d_img, pose_3d_img])

    out_path = os.path.join(concat_dir, filename)
    cv2.imwrite(out_path, concat_img)

print(f"[DONE] Concatenation finished using mode '{args.mode}'")