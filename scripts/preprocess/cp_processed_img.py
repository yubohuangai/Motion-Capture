#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_camera_dirs(src_root: Path):
    return sorted([d for d in src_root.iterdir() if d.is_dir()])


def list_images(cam_dir: Path):
    return sorted([f for f in cam_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS])


def copy_index_range(src_root: Path, dst_root: Path, start_idx: int, end_idx: int):
    camera_dirs = list_camera_dirs(src_root)
    if not camera_dirs:
        raise FileNotFoundError(f"No camera folders found under {src_root}")

    for cam_dir in camera_dirs:
        images = list_images(cam_dir)
        selected = images[start_idx:end_idx + 1]
        if not selected:
            print(f"Warning: no images selected for {cam_dir}")
            continue

        dst_cam_dir = dst_root / cam_dir.name
        dst_cam_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(selected, desc=f"Copying {cam_dir.name}", leave=False):
            shutil.copy2(img_path, dst_cam_dir / img_path.name)

        print(f"{cam_dir.name}: copied {len(selected)} images to {dst_cam_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy a fixed image index range for each camera folder."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/mnt/yubo/emily/highknees/images",
        help="Source root containing camera folders (e.g. 01, 02, ...).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/mnt/yubo/emily/highknees100/images",
        help="Destination root where camera folders will be created.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (0-based, inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=99,
        help="End index (0-based, inclusive).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.start < 0 or args.end < args.start:
        raise ValueError("Invalid range: require 0 <= start <= end")

    copy_index_range(Path(args.src), Path(args.dst), args.start, args.end)
