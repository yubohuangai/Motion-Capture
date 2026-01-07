"""
Filepath: scripts/postprocess/concat_per_frame.py

Concatenate per-frame keypoint JSON files into a single keypoints.json.
Supports single-person 2D or 3D keypoints.
Output shape: (1, T, J, 2) or (1, T, J, 3)
"""

import os
import json
import argparse
import numpy as np
from glob import glob


# ---------- args ----------
parser = argparse.ArgumentParser(
    description="Concatenate per-frame keypoints JSON files"
)
parser.add_argument(
    "--input_dir",
    type=str,
    default="/Users/yubo/data/s2/seq1/360/output/poseformerv2/view32_fov150/input_2D/keypoints",
    help="Directory containing per-frame JSON files (e.g. 000000.json)"
)
parser.add_argument(
    "--dim",
    choices=["2d", "3d"],
    default="2d",
    help="Keypoint dimension type"
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output keypoints.json path (default: input_dir + '.json')"
)
args = parser.parse_args()

input_dir = args.input_dir
output_json = args.output or (input_dir.rstrip("/") + ".json")

# ---------- collect frames ----------
json_paths = sorted(glob(os.path.join(input_dir, "*.json")))
assert len(json_paths) > 0, f"No JSON files found in {input_dir}"

all_frames = []

for json_path in json_paths:
    with open(json_path, "r") as f:
        frame_data = json.load(f)

    # single person assumption
    person = frame_data[0]

    if args.dim == "2d":
        kpts = np.asarray(person["keypoints"], dtype=np.float32)
        kpts = kpts[:, :2]  # drop confidence
    else:
        kpts = np.asarray(person["keypoints3d"], dtype=np.float32)
        kpts = kpts[:, :3]  # drop confidence

    all_frames.append(kpts)

# ---------- stack ----------
reconstruction = np.stack(all_frames, axis=0)  # (T, J, 2) or (T, J, 3)
reconstruction = np.expand_dims(reconstruction, axis=0)  # (1, T, J, 2) or (1, T, J, 3)

print(f"Reconstructed shape: {reconstruction.shape}")

# ---------- save ----------
out_data = {
    "reconstruction": reconstruction.tolist()
}

with open(output_json, "w") as f:
    json.dump(out_data, f)

print(f"[DONE] Saved concatenated file to:\n{output_json}")