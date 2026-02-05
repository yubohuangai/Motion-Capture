#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

CONFIG = {
    "base_root": "/mnt/yubo/lab/raw",
    "tasks": [
        {
            "name": "ground",
            "dst": "/mnt/yubo/lab/ground",
            "mode": "first",
        },
        {
            "name": "board",
            "dst": "/mnt/yubo/lab/board",
            "start": "/mnt/yubo/lab/raw/01/images/1770250495383885349.jpg",
            "end": "/mnt/yubo/lab/raw/01/images/1770250521086395757.jpg",
        },
        {
            "name": "box",
            "dst": "/mnt/yubo/lab/box",
            "start": "/mnt/yubo/lab/raw/01/images/1770250528687138135.jpg",
            "end": "/mnt/yubo/lab/raw/01/images/1770250539521529682.jpg",
        },
        {
            "name": "cube",
            "dst": "/mnt/yubo/lab/cube",
            "start": "/mnt/yubo/lab/raw/01/images/1770250546488828235.jpg",
            "end": "/mnt/yubo/lab/raw/01/images/1770250570124518728.jpg",
        },
        {
            "name": "chair",
            "dst": "/mnt/yubo/lab/chair",
            "start": "/mnt/yubo/lab/raw/01/images/1770250580758890739.jpg",
            "end": "/mnt/yubo/lab/raw/01/images/1770250584792618053.jpg",
        },
    ],
}


def get_camera_ids(root):
    root_path = Path(root)
    if not root_path.exists():
        return []
    cam_ids = []
    for entry in root_path.iterdir():
        if entry.is_dir() and entry.name.isdigit():
            cam_ids.append(int(entry.name))
    return sorted(cam_ids)


def list_images(img_dir):
    return sorted([f for f in os.listdir(img_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)])


def index_from_filename(img_dir, filename):
    files = list_images(img_dir)
    if filename not in files:
        raise FileNotFoundError(f"{filename} not found in {img_dir}")
    return files.index(filename) + 1  # 1-based


def copy_range(src_dir, dst_dir, start_idx, end_idx):
    os.makedirs(dst_dir, exist_ok=True)
    files = list_images(src_dir)
    selected = files[start_idx - 1:end_idx]
    for f in tqdm(selected, desc=f"Copying {os.path.basename(src_dir)}"):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))


def copy_first(src_dir, dst_dir):
    files = list_images(src_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    first = files[0]
    shutil.copy2(os.path.join(src_dir, first), os.path.join(dst_dir, first))


def run_task(base_root, task):
    cam_ids = get_camera_ids(base_root)
    if not cam_ids:
        raise RuntimeError(f"No camera folders found under {base_root}")

    cam01_dir = Path(base_root) / "01" / "images"
    if not cam01_dir.exists():
        raise FileNotFoundError(f"Camera 01 images not found: {cam01_dir}")

    if task.get("mode") == "first":
        for cam_id in cam_ids:
            src_dir = Path(base_root) / f"{cam_id:02d}" / "images"
            dst_dir = Path(task["dst"]) / f"{cam_id:02d}"
            copy_first(str(src_dir), str(dst_dir))
        return

    start_name = Path(task["start"]).name
    end_name = Path(task["end"]).name
    start_idx = index_from_filename(str(cam01_dir), start_name)
    end_idx = index_from_filename(str(cam01_dir), end_name)
    if end_idx < start_idx:
        raise ValueError(f"End index before start index for task {task['name']}")

    for cam_id in cam_ids:
        src_dir = Path(base_root) / f"{cam_id:02d}" / "images"
        dst_dir = Path(task["dst"]) / f"{cam_id:02d}"
        copy_range(str(src_dir), str(dst_dir), start_idx, end_idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Copy specific ranges from multi-camera image folders.")
    parser.add_argument(
        "base_root",
        nargs="?",
        default=CONFIG["base_root"],
        help="Root directory containing 01/, 02/, ... camera folders",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for task in CONFIG["tasks"]:
        print(f"=== Task: {task['name']} → {task['dst']} ===")
        run_task(args.base_root, task)
