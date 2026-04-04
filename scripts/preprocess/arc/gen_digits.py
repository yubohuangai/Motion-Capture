#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

def generate_images(output_dir, total_frames=99):
    """
    Images are cropped to minimal size around the text.

    Args:
        output_dir (str): Folder to save images.
        total_frames (int): Number of images to generate.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Use a readable font
    try:
        # font = ImageFont.truetype("arial.ttf", 48) # Windows
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 300) # Mac
    except IOError:
        font = ImageFont.load_default()
    print("Font used:", font)

    for i in range(total_frames + 1):
        # Temporary large image
        temp_img = Image.new("RGB", (1000, 200), "white")
        draw = ImageDraw.Draw(temp_img)

        text = f"{i:02d}"

        # Get tight bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Create final cropped image
        img = Image.new("RGB", (text_w, text_h), "white")
        draw_final = ImageDraw.Draw(img)
        draw_final.text((-bbox[0], -bbox[1]), text, fill="black", font=font)

        # Save image
        filename = os.path.join(output_dir, f"{i:02d}.jpg")
        img.save(filename)

    print(f"{total_frames+1} tight images saved in '{output_dir}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate number images with tight cropping.")
    parser.add_argument("--outdir", type=str, default="output", required=False, help="Output directory for generated images")
    parser.add_argument("--total_frames", type=int, default=99, help="Number of frames to generate (default 99)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_images(output_dir=args.outdir, total_frames=args.total_frames)