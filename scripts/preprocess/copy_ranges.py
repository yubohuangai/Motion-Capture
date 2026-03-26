#!/usr/bin/env python3
"""
Copy frames from multi-camera layout: ``<base_root>/<cam_id>/images/*`` → ``<dst>/images/<cam_id>/000000.*``.

**Task modes** (set on each task dict):

- ``"mode": "first"`` — copy only the first image per camera (e.g. ground reference).
- ``"mode": "all"`` — copy every image per camera, sorted, renumbered from ``000000``.
- *(no mode)* — copy a **range**: set ``"start"`` and ``"end"`` to **full paths** to two
  filenames under ``.../01/images/``. Indices are taken from camera ``01`` and applied
  to every camera (same frame indices for all views).

**CLI:** ``python copy_ranges.py [base_root] [--task NAME]``

See ``EXAMPLE_TASKS_LAB`` below for a multi-segment lab workflow you can paste into ``CONFIG["tasks"]``.
"""
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Active jobs (edit ``base_root`` / ``tasks`` for your dataset).
CONFIG = {
    "base_root": "/mnt/yubo/charuco/raw",
    "tasks": [
        {
            "name": "board",
            "dst": "/mnt/yubo/charuco/board",
            "mode": "all",
        },
    ],
}

# Reference only — not executed. Copy entries into CONFIG["tasks"] when you need the same patterns.
EXAMPLE_TASKS_LAB = [
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
]
EXAMPLE_BASE_ROOT_LAB = "/mnt/yubo/lab/raw"


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
    next_idx = 0
    for f in tqdm(selected, desc=f"Copying {os.path.basename(src_dir)}"):
        ext = os.path.splitext(f)[1]
        new_name = f"{next_idx:06d}{ext}"
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, new_name))
        next_idx += 1


def copy_first(src_dir, dst_dir):
    files = list_images(src_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)
    first = files[0]
    ext = os.path.splitext(first)[1]
    shutil.copy2(os.path.join(src_dir, first), os.path.join(dst_dir, f"{0:06d}{ext}"))


def copy_all(src_dir, dst_dir):
    """Copy all images in sorted order, renumbered from 000000."""
    files = list_images(src_dir)
    if not files:
        return
    copy_range(src_dir, dst_dir, 1, len(files))


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
            dst_dir = Path(task["dst"]) / "images" / f"{cam_id:02d}"
            copy_first(str(src_dir), str(dst_dir))
        return

    if task.get("mode") == "all":
        for cam_id in cam_ids:
            src_dir = Path(base_root) / f"{cam_id:02d}" / "images"
            dst_dir = Path(task["dst"]) / "images" / f"{cam_id:02d}"
            copy_all(str(src_dir), str(dst_dir))
        return

    start_name = Path(task["start"]).name
    end_name = Path(task["end"]).name
    start_idx = index_from_filename(str(cam01_dir), start_name)
    end_idx = index_from_filename(str(cam01_dir), end_name)
    if end_idx < start_idx:
        raise ValueError(f"End index before start index for task {task['name']}")

    for cam_id in cam_ids:
        src_dir = Path(base_root) / f"{cam_id:02d}" / "images"
        dst_dir = Path(task["dst"]) / "images" / f"{cam_id:02d}"
        copy_range(str(src_dir), str(dst_dir), start_idx, end_idx)


def parse_args():
    parser = argparse.ArgumentParser(description="Copy specific ranges from multi-camera image folders.")
    parser.add_argument(
        "base_root",
        nargs="?",
        default=CONFIG["base_root"],
        help="Root directory containing 01/, 02/, ... camera folders",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        metavar="NAME",
        help="Run only the task with this name (default: all tasks in CONFIG)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks = CONFIG["tasks"]
    if args.task is not None:
        tasks = [t for t in tasks if t["name"] == args.task]
        if not tasks:
            raise SystemExit(f"No task named {args.task!r}. Known: {[t['name'] for t in CONFIG['tasks']]}")
    for task in tasks:
        print(f"=== Task: {task['name']} → {task['dst']} ===")
        run_task(args.base_root, task)
