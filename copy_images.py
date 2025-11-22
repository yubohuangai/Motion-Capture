#!/usr/bin/env python3
import os
import shutil
from tqdm import tqdm
import argparse

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def copy_images(src_dir, dst_dir, start, end):
    """Copy images from src_dir to dst_dir, continue numbering from existing images."""
    os.makedirs(dst_dir, exist_ok=True)

    # Load source images
    src_files = sorted([f for f in os.listdir(src_dir)
                        if f.lower().endswith(ALLOWED_EXTENSIONS)])
    selected_files = src_files[start-1:end]  # 1-based indexing

    # Determine next index in destination
    existing = sorted([f for f in os.listdir(dst_dir)
                       if f.lower().endswith(ALLOWED_EXTENSIONS)])
    if existing:
        last_file = existing[-1]
        try:
            last_idx = int(os.path.splitext(last_file)[0])
        except ValueError:
            last_idx = len(existing) - 1
        next_idx = last_idx + 1
    else:
        next_idx = 0

    print(f"Copying {len(selected_files)} images from {src_dir} → {dst_dir}, starting at index {next_idx:06d}")

    for f in tqdm(selected_files, desc=f"Copying {os.path.basename(src_dir)}"):
        ext = os.path.splitext(f)[1]
        new_name = f"{next_idx:06d}{ext}"
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, new_name))
        next_idx += 1


def process_all_cameras(src_root, dst_root, start, end):
    """Find all camera folders under src_root and copy images to dst_root."""
    cams = sorted([d for d in os.listdir(src_root)
                   if os.path.isdir(os.path.join(src_root, d))])
    print(f"Found {len(cams)} camera folders: {cams}")

    for cam in cams:
        src_dir = os.path.join(src_root, cam, "images")  # expects 'images' subfolder
        dst_dir = os.path.join(dst_root, cam)            # destination per camera
        if not os.path.exists(src_dir):
            print(f"Warning: source folder not found: {src_dir}")
            continue
        copy_images(src_dir, dst_dir, start, end)


def parse_args():
    parser = argparse.ArgumentParser(description="Copy images from multiple cameras with continuous numbering.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing camera folders (01, 02, ...).")
    parser.add_argument("--dst", type=str, required=True,
                        help="Destination root folder to copy images into.")
    parser.add_argument("--start", type=int, default=1,
                        help="Start index of images to copy (1-based).")
    parser.add_argument("--end", type=int, default=None,
                        help="End index of images to copy (inclusive).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_cameras(args.root, args.dst, args.start, args.end)

