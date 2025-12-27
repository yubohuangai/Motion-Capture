#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile


def video_to_gif(video_path, output_path=None, scale=None, loop=0):
    video_path = os.path.abspath(video_path)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + ".gif"
    else:
        output_path = os.path.abspath(output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # Build filter string
    filters = []
    if scale is not None:
        filters.append(f"scale={scale}:-1:flags=lanczos")
    filters.append("palettegen=stats_mode=diff")

    filter_str = ",".join(filters)

    with tempfile.TemporaryDirectory() as tmpdir:
        palette_path = os.path.join(tmpdir, "palette.png")

        # --------------------------
        # 1) Generate palette
        # --------------------------
        palette_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", filter_str,
            palette_path
        ]

        print("Generating palette:")
        print(" ".join(palette_cmd))
        subprocess.run(palette_cmd, check=True)

        # --------------------------
        # 2) Apply palette
        # --------------------------
        gif_filters = []
        if scale is not None:
            gif_filters.append(f"scale={scale}:-1:flags=lanczos")
        gif_filters.append(f"paletteuse=dither=bayer:bayer_scale=5")

        gif_filter_str = ",".join(gif_filters)

        gif_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", palette_path,
            "-lavfi", gif_filter_str,
            "-loop", str(loop),
            output_path
        ]

        print("Creating GIF:")
        print(" ".join(gif_cmd))
        subprocess.run(gif_cmd, check=True)

    print(f"GIF written to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert video to GIF keeping original FPS.")
    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument("--output", type=str, help="Output GIF path")
    parser.add_argument("--scale", type=int, help="Resize width (keeps aspect ratio)")
    parser.add_argument("--loop", type=int, default=0, help="0 = infinite loop")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    video_to_gif(
        video_path=args.video,
        output_path=args.output,
        scale=args.scale,
        loop=args.loop
    )