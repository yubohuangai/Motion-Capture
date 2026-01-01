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


def make_front_view(img):
    h, w = img.shape[:2]
    q = w // 4
    return img[:, q:3 * q]


def make_back_view(img):
    h, w = img.shape[:2]
    q = w // 4
    left_q = img[:, :q]
    right_q = img[:, -q:]
    return cv2.hconcat([right_q, left_q])


def make_right_view(img):
    h, w = img.shape[:2]
    return img[:, w // 2:]


def make_custom_view(img, center_ratio=0.32, width_ratio=0.5):
    """
    Crop a horizontal slice of the image centered at center_ratio * width
    width_ratio: fraction of the total width to crop
    """
    h, w = img.shape[:2]
    cw = int(center_ratio * w)           # center pixel
    half_w = int(width_ratio * w / 2)    # half width of crop

    left = max(cw - half_w, 0)
    right = min(cw + half_w, w)

    return img[:, left:right]


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
        choices=["left", "right", "front", "back", "custom"],
        required=True,
        help="Output view type"
    )
    parser.add_argument(
        "--center_ratio",
        type=float,
        default=0.31,
        help="Center of custom view as fraction of width"
    )
    parser.add_argument(
        "--width_ratio",
        type=float,
        default=0.5,
        help="Width of custom view as fraction of width"
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

        if args.view == "custom":
            out = make_custom_view(img, center_ratio=args.center_ratio, width_ratio=args.width_ratio)
        elif args.view == "left":
            out = make_left_view(img)
        elif args.view == "right":
            out = make_right_view(img)
        elif args.view == "front":
            out = make_front_view(img)
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
