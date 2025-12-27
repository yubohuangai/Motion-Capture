from pathlib import Path
import sys
import re

ROOT = Path("/Users/yubo/data/s2/seq1/images")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

DIGIT_DIR = re.compile(r"^\d{2}$")


def rename_images_06d(folder: Path):
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)

    if not images:
        print(f"[WARN] No images found in {folder}")
        return

    for idx, img in enumerate(images, start=0):
        new_name = f"{idx:06d}{img.suffix.lower()}"
        new_path = folder / new_name

        if img.name != new_name:
            img.rename(new_path)

    print(f"[OK] Renamed {len(images)} images in {folder.name}")


def main():
    if not ROOT.exists():
        print(f"[ERROR] Root directory does not exist: {ROOT}")
        sys.exit(1)

    subdirs = sorted(
        p for p in ROOT.iterdir()
        if p.is_dir() and DIGIT_DIR.match(p.name)
    )

    if not subdirs:
        print(
            "[ERROR] No valid subdirectories found.\n"
            "Expected subdirectories named like: 01, 02, 03, ...\n"
            f"Checked under: {ROOT}"
        )
        sys.exit(1)

    print(f"[INFO] Found {len(subdirs)} subdirectories:")
    for sd in subdirs:
        print(f"  - {sd.name}")

    for subdir in subdirs:
        rename_images_06d(subdir)


if __name__ == "__main__":
    main()