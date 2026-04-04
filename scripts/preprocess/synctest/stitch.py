from __future__ import annotations

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import math

from session_paths import load_config

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _list_frame_files(folder: Path) -> list[str]:
    out = []
    for f in os.listdir(folder):
        suf = Path(f).suffix.lower()
        if suf in _IMAGE_SUFFIXES:
            out.append(f)
    return sorted(out)


def _frame_dirs_for_cameras(camera_dirs: list[Path], images_subdir: str | None) -> list[Path]:
    """Per-camera folder that contains frame files (e.g. .../01 or .../01/images)."""
    if not images_subdir or not str(images_subdir).strip():
        return camera_dirs
    sub = Path(str(images_subdir).strip())
    return [d / sub for d in camera_dirs]


def stitch_images_grid(config, columns=None):
    stitch_root = config.get("stitch_root")
    if not stitch_root or not os.path.exists(stitch_root):
        raise FileNotFoundError(f"Stitch root not found or not defined: {stitch_root}")
    stitch_root = Path(stitch_root)

    stitch_start = config.get("stitch_start", None)
    stitch_end = config.get("stitch_end", None)
    stitch_last_n = config.get("stitch_last_n", None)
    stitch_images_subdir = config.get("stitch_images_subdir", None)
    stitch_columns_cfg = config.get("stitch_columns", None)

    # Only immediate subdirs with all-digit names (e.g. 01, 02).
    subdirs = sorted([d for d in stitch_root.iterdir() if d.is_dir() and d.name.isdigit()],
                     key=lambda x: int(x.name))
    num_videos = len(subdirs)
    if num_videos == 0:
        raise ValueError("No valid numbered subdirectories found in stitch_root.")

    # Layout: stitch_columns in config (e.g. 1 = single column), else sqrt grid; override via columns= argument.
    if columns is None:
        if stitch_columns_cfg is not None:
            columns = int(stitch_columns_cfg)
            if columns < 1:
                raise ValueError("stitch_columns must be >= 1")
        else:
            columns = math.ceil(math.sqrt(num_videos))
    rows = math.ceil(num_videos / columns)

    # Optional: font for drawing video index
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    frame_dirs = _frame_dirs_for_cameras(subdirs, stitch_images_subdir)
    for cam, fd in zip(subdirs, frame_dirs):
        if not fd.is_dir():
            raise FileNotFoundError(
                f"Missing frame folder {fd} (camera {cam.name}). "
                "Set stitch_images_subdir to '' if frames live directly under each camera folder."
            )

    # Get sorted list of images per camera (.png / .jpg / .jpeg)
    subdir_files = [_list_frame_files(d) for d in frame_dirs]
    num_frames = min(len(files) for files in subdir_files)
    if num_frames == 0:
        hint = (
            f" under <camera>/{stitch_images_subdir}/"
            if stitch_images_subdir and str(stitch_images_subdir).strip()
            else " under each <camera>/ folder "
        )
        raise ValueError(
            f"No images (.png/.jpg/.jpeg) found{hint}below {stitch_root}. "
            "Use stitch_images_subdir (e.g. images) when frames are in …/01/images/*.jpg."
        )

    # Determine start and end indices
    if stitch_last_n is not None:
        n = int(stitch_last_n)
        if n <= 0:
            raise ValueError("stitch_last_n must be positive")
        start_idx = max(0, num_frames - n)
        end_idx = num_frames
    else:
        start_idx = stitch_start if stitch_start is not None else 0
        end_idx = stitch_end if stitch_end is not None else num_frames
        start_idx = max(0, start_idx)
        end_idx = min(num_frames, end_idx)

    # Output directory with range in name
    output_dir = stitch_root / f"stitched_{start_idx}_{end_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in tqdm(range(start_idx, end_idx), desc="Stitching"):
        imgs = []
        for i, files in enumerate(subdir_files):
            img_path = frame_dirs[i] / files[frame_idx]
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), f"V{i}", fill=(255, 0, 0), font=font)
            imgs.append(img)

        # Determine size for uniform resizing
        min_width = min(img.width for img in imgs)
        min_height = min(img.height for img in imgs)
        resized_imgs = [img.resize((min_width, min_height)) for img in imgs]

        stitched_width = columns * min_width
        stitched_height = rows * min_height
        stitched_img = Image.new('RGB', (stitched_width, stitched_height))

        for idx, img in enumerate(resized_imgs):
            row_idx = idx // columns
            col_idx = idx % columns
            x = col_idx * min_width
            y = row_idx * min_height
            stitched_img.paste(img, (x, y))

        output_filename = f"{frame_idx:06d}.png"
        stitched_img.save(output_dir / output_filename)

if __name__ == "__main__":
    config = load_config("config.yaml")
    stitch_images_grid(config)
