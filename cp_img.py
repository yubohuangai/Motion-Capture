import os
import shutil
from tqdm import tqdm

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")

# =======================
# Control parameters
# =======================
SRC_ROOT = "/mnt/yubo/emily/raw_motion"     # Source root directory containing camera folders
DST_ROOT = "/mnt/yubo/emily/motion/images"  # Destination root directory for output images
START_IDX = 1                              # Start index (1-based)
END_IDX = None                               # End index (inclusive)
RENAME_AFTER_COPY = True                    # Rename files after copying
# =======================

def copy_images(src_dir, dst_dir, start, end):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(ALLOWED_EXTENSIONS)])
    selected_files = files[start-1:end]  # convert to 1-based indexing

    print(f"Copying {len(selected_files)} images from {src_dir} to {dst_dir}")
    for f in tqdm(selected_files, desc=f"Copying {os.path.basename(src_dir)}"):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

def rename_images(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(ALLOWED_EXTENSIONS)])
    for i, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]  # preserve original extension
        old_path = os.path.join(folder, filename)
        new_name = f"{i:06d}{ext}"
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
    print(f"Renamed {len(files)} images in {folder}")

def process_all_cameras(src_root, dst_root, start, end, rename=False):
    cams = sorted([d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))])
    print(f"Found {len(cams)} camera folders: {cams}")

    for cam in cams:
        src_dir = os.path.join(src_root, cam, "images")  # include 'images' subfolder
        dst_dir = os.path.join(dst_root, cam)            # destination already includes 'images'
        copy_images(src_dir, dst_dir, start, end)
        if rename:
            rename_images(dst_dir)

if __name__ == "__main__":
    process_all_cameras(SRC_ROOT, DST_ROOT, START_IDX, END_IDX, rename=RENAME_AFTER_COPY)
