import os
import cv2
import argparse
from glob import glob

def main():
    parser = argparse.ArgumentParser(
        description="Crop the left half of each image in a directory"
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
        help="Destination directory for cropped images"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="*",
        help="Image extension filter, e.g. jpg, png (default: all)"
    )

    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    pattern = os.path.join(args.src, f"*.{args.ext}") if args.ext != "*" \
              else os.path.join(args.src, "*")
    img_paths = sorted(glob(pattern))

    if len(img_paths) == 0:
        print(f"[ERROR] No images found in {args.src}")
        return

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read {img_path}")
            continue

        h, w = img.shape[:2]
        left_half = img[:, :w // 2]

        out_path = os.path.join(args.dst, os.path.basename(img_path))
        cv2.imwrite(out_path, left_half)

    print(f"[DONE] Cropped {len(img_paths)} images")
    print(f"[SRC] {args.src}")
    print(f"[DST] {args.dst}")

if __name__ == "__main__":
    main()