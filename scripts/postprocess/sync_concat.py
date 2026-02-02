import os
import re
import glob
import cv2
import numpy as np

# ---- Paths ----
orig_dir = "/Users/yubo/data/kickoff"      # 30 fps frames: frame_XXXXXX.jpg
pred_dir = "/Users/yubo/data/frames"      # 60 fps frames: frame_XXXXXX.png
out_dir  = "/Users/yubo/data/kickoff_concat"  # output concatenated frames

os.makedirs(out_dir, exist_ok=True)

# ---- FPS + anchor frames ----
fps_orig = 30.0
fps_pred = 60.0
ratio = fps_pred / fps_orig  # 2.0

i0 = 400   # original anchor index (frame_000400.jpg)
j0 = 382   # predict anchor index (frame_000382.png)


def get_frame_index(path):
    """Extract integer index from 'frame_000400.jpg' or 'frame_000382.png'."""
    name = os.path.basename(path)
    m = re.search(r'(\d+)(?=\.[^.]+$)', name)
    if not m:
        return None
    return int(m.group(1))


# All original (30 fps) frames
orig_files = sorted(glob.glob(os.path.join(orig_dir, "frame_*.jpg")))

print(f"Found {len(orig_files)} original frames.")

for orig_path in orig_files:
    i = get_frame_index(orig_path)
    if i is None:
        print(f"Skip (no index): {orig_path}")
        continue

    # Compute corresponding 60 fps frame index using anchor + fps ratio
    j_float = ratio * (i - i0) + j0
    j = int(round(j_float))

    pred_path = os.path.join(pred_dir, f"frame_{j:06d}.png")

    if not os.path.exists(pred_path):
        print(f"Missing predict frame for original {i}: {pred_path}")
        continue

    # Read images
    img_orig = cv2.imread(orig_path)
    img_pred = cv2.imread(pred_path)

    if img_orig is None or img_pred is None:
        print(f"Failed to read one of: {orig_path}, {pred_path}")
        continue

    # Make heights match (resize predict to original height, keep aspect ratio)
    h_o, w_o = img_orig.shape[:2]
    h_p, w_p = img_pred.shape[:2]

    if h_o != h_p:
        new_h = h_o
        new_w = int(w_p * (new_h / h_p))
        img_pred = cv2.resize(img_pred, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Concatenate horizontally: [original | predict]
    concat = np.hstack([img_orig, img_pred])

    out_name = os.path.basename(orig_path)  # keep original frame name
    out_path = os.path.join(out_dir, out_name)

    cv2.imwrite(out_path, concat)
    # print(f"Saved: {out_path}")

print("Done. Concatenated frames written to:", out_dir)