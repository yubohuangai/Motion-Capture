#!/usr/bin/env python3
import argparse
import os
import subprocess

def video_to_gif(video_path, output_path=None, scale=None, loop=0):
    video_path = os.path.abspath(video_path)

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + ".gif"
    else:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vf_parts = []
    if scale is not None:
        vf_parts.append(f"scale={scale}:-1:flags=lanczos")

    vf = ",".join(vf_parts) if vf_parts else None

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-loop", str(loop), output_path]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
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