"""
scripts/preprocess/crop_omni.py

Crop ERP (equirectangular) 360° images into specific views
(e.g. left view, back view) for omnidirectional cameras.
"""

import os
import cv2
import argparse
from glob import glob


def make_left_view(img):
    h, w = img.shape[:2]
    return img[:, :w // 2]


def make_back_view(img):
    """
    Back view from ERP:
    - right 1/4 + left 1/4
    - mirror both
    - stitch
    """
    h, w = img.shape[:2]
    q = w // 4

    left_q = img[:, :q]
    right_q = img[:, -q:]

    left_q = cv2.flip(left_q, 1)
    right_q = cv2.flip(right_q, 1)

    return cv2.hconcat([right_q, left_q])


def main():
    parser = argparse.ArgumentParser(
        description="Generate left or back view from equirectangular images"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source image directory"
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination directory"
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["left", "back"],
        required=True,
        help="Output view type"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="*",
        help="Image extension (jpg, png, etc.)"
    )

    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    pattern = os.path.join(args.src, f"*.{args.ext}") if args.ext != "*" \
              else os.path.join(args.src, "*")
    img_paths = sorted(glob(pattern))

    if not img_paths:
        print(f"[ERROR] No images found in {args.src}")
        return

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read {img_path}")
            continue

        if args.view == "left":
            out = make_left_view(img)
        elif args.view == "back":
            out = make_back_view(img)
        else:
            raise ValueError("Unknown view type")

        out_path = os.path.join(args.dst, os.path.basename(img_path))
        cv2.imwrite(out_path, out)

    print(f"[DONE] Generated '{args.view}' view")
    print(f"[SRC] {args.src}")
    print(f"[DST] {args.dst}")


if __name__ == "__main__":
    main()