"""
File path: scripts/postprocess/img2vid.py
"""

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


def format_range_suffix(start_idx, end):
    if start_idx is None and end is None:
        return "full"
    start = start_idx if start_idx is not None else 0
    end = end if end is not None else "end"
    return f"{start}-{end}"


# Max dimension for MPEG-4 compatibility (encoder rejects very large frames)
MAX_DIM = 4096


def _compute_output_size(width, height, max_dim=MAX_DIM):
    """Return (out_w, out_h) downscaled if either dimension exceeds max_dim."""
    if width <= max_dim and height <= max_dim:
        return width, height
    scale = min(max_dim / width, max_dim / height)
    out_w = int(round(width * scale))
    out_h = int(round(height * scale))
    # Ensure even dimensions for some codecs
    out_w = out_w - (out_w % 2)
    out_h = out_h - (out_h % 2)
    return max(2, out_w), max(2, out_h)


def create_video_from_images_cv2(
    image_dir,
    start_idx=None,
    end=None,
    fps=30,
    output_path=None,
    max_dim=MAX_DIM,
):
    image_dir = os.path.abspath(image_dir)
    folder_name = os.path.basename(image_dir.rstrip('/'))

    range_suffix = format_range_suffix(start_idx, end)

    if output_path is None:
        output_path = os.path.join(
            image_dir,
            f"{folder_name}_{range_suffix}.mp4"
        )
    else:
        output_path = os.path.abspath(output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    # Collect images
    image_paths = collect_images(image_dir)

    if not image_paths:
        print(f"No images found in {image_dir} or its subdirectories.")
        return

    # Frame selection
    if start_idx is not None or end is not None:
        start_idx = start_idx or 0
        end = end or len(image_paths)
        image_paths = image_paths[start_idx:end]

    # Load first frame to get size
    frame = cv2.imread(image_paths[0])
    if frame is None:
        print("Error: Could not read first frame:", image_paths[0])
        return

    height, width, _ = frame.shape
    out_width, out_height = _compute_output_size(width, height, max_dim)

    if (out_width, out_height) != (width, height):
        print(f"Downscaling {width}x{height} -> {out_width}x{out_height} (max_dim={max_dim})")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame.shape[1] != out_width or frame.shape[0] != out_height:
            frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
        out.write(frame)

    out.release()
    print(f"Video written to: {output_path} (FPS={fps}, Frames={len(image_paths)})")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a video from a folder of images.")
    parser.add_argument("src", nargs="?", type=str, help="Directory containing images or image subfolders.")
    parser.add_argument("--src", dest="src_flag", type=str, help="Directory containing images or image subfolders.")
    parser.add_argument("--dst", type=str, default=None, help="Output MP4 file path (default: src folder + .mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for the output video.")
    parser.add_argument("--start", type=int, default=None, help="Start frame index (0-based).")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive).")
    parser.add_argument("--max_dim", type=int, default=MAX_DIM, help=f"Max width/height for MPEG-4 compatibility (default: {MAX_DIM}).")
    args = parser.parse_args()

    if args.src is None:
        args.src = args.src_flag
    if args.src is None:
        parser.error("the following arguments are required: src (positional) or --src")

    if args.dst is None:
        src_abs = os.path.abspath(args.src.rstrip('/'))
        args.dst = f"{src_abs}.mp4"

    return args

if __name__ == "__main__":
    args = parse_args()
    create_video_from_images_cv2(
        args.src,
        start_idx=args.start,
        end=args.end,
        fps=args.fps,
        output_path=args.dst,
        max_dim=args.max_dim,
    )