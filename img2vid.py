#!/usr/bin/env python3
import os
from glob import glob
import cv2
import argparse


def collect_images(image_dir):
    """Return sorted list of images in image_dir; if empty, search subdirectories."""
    # Direct images first
    imgs = sorted(
        [p for p in glob(os.path.join(image_dir, '*.*'))
         if p.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: x.lower()
    )
    if imgs:
        return imgs

    # Otherwise search immediate subdirectories
    subdirs = [d for d in glob(os.path.join(image_dir, '*')) if os.path.isdir(d)]
    combined = []
    for sd in sorted(subdirs):
        sd_imgs = sorted(
            [p for p in glob(os.path.join(sd, '*.*'))
             if p.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: x.lower()
        )
        combined.extend(sd_imgs)
    return combined


import imageio

def format_range_suffix(start_idx, end_idx):
    if start_idx is None and end_idx is None:
        return "full"
    start = start_idx if start_idx is not None else 0
    end = end_idx if end_idx is not None else "end"
    return f"{start}-{end}"


def create_gif_from_images(image_paths, fps, output_path):
    duration = 1.0 / fps  # seconds per frame

    frames = []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    if not frames:
        print("No valid frames for GIF.")
        return

    imageio.mimsave(output_path, frames, duration=duration)
    print(f"GIF written to: {output_path} (FPS={fps}, Frames={len(frames)})")


def create_video_from_images_cv2(
    image_dir,
    start_idx=None,
    end_idx=None,
    fps=30,
    output_path=None,
    to_gif=False
):
    image_dir = os.path.abspath(image_dir)
    folder_name = os.path.basename(image_dir.rstrip('/'))

    range_suffix = format_range_suffix(start_idx, end_idx)

    if output_path is None:
        output_path = os.path.join(
            image_dir,
            f"{folder_name}_{range_suffix}.mp4"
        )
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect images
    image_paths = collect_images(image_dir)

    if not image_paths:
        print(f"No images found in {image_dir} or its subdirectories.")
        return

    # Frame selection
    if start_idx is not None or end_idx is not None:
        start_idx = start_idx or 0
        end_idx = end_idx or len(image_paths)
        image_paths = image_paths[start_idx:end_idx]

    if to_gif:
        if output_path is None:
            output_path = os.path.join(image_dir, f"{folder_name}.gif")
        else:
            output_path = os.path.splitext(output_path)[0] + ".gif"

        create_gif_from_images(image_paths, fps, output_path)
        return
    # Load first frame to get size
    frame = cv2.imread(image_paths[0])
    if frame is None:
        print("Error: Could not read first frame:", image_paths[0])
        return

    height, width, _ = frame.shape

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video written to: {output_path} (FPS={fps}, Frames={len(image_paths)})")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a video from a folder of images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images or image subfolders.")
    parser.add_argument("--output_path", type=str, help="Output MP4 file path.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for the output video.")
    parser.add_argument("--start_idx", type=int, default=None, help="Start frame index (0-based).")
    parser.add_argument("--end_idx", type=int, default=None, help="End frame index (exclusive).")
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Export GIF instead of MP4"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_video_from_images_cv2(
        args.image_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        fps=args.fps,
        output_path=args.output_path,
        to_gif=args.gif
    )