"""
Filepath: scripts/postprocess/split_per_frame.py

Split keypoints.json into per-frame JSON files.
Supports single-person 2D or 3D keypoints.
"""

import json
import os
import numpy as np
import argparse


# ---------- args ----------
parser = argparse.ArgumentParser(
    description="Split single-person keypoints.json into per-frame files"
)
parser.add_argument(
    "--input",
    type=str,
    default="/Users/yubo/data/s2/seq1/360/output/poseformerv2/01_view40/output_3D/output_keypoints_3d.json",
    help="Path to input keypoints.json"
)
parser.add_argument(
    "--dim",
    choices=["2d", "3d"],
    default="2d",
    help="Keypoint dimension type"
)
parser.add_argument(
    "--confidence",
    type=float,
    default=1.0,
    help="Default confidence value"
)
args = parser.parse_args()

input_json = args.input
DEFAULT_CONFIDENCE = args.confidence
PERSON_ID = 0

# ---------- output dir ----------
output_dir = os.path.splitext(input_json)[0]
os.makedirs(output_dir, exist_ok=True)

# ---------- load ----------
with open(input_json, "r") as f:
    data = json.load(f)

reconstruction = np.asarray(data["reconstruction"], dtype=np.float32)
print(f"Loaded reconstruction shape: {reconstruction.shape}")

# ---------- normalize shapes ----------
if args.dim == "2d":
    # Accept (T, J, 2) or (1, T, J, 2)
    if reconstruction.ndim == 4:
        assert reconstruction.shape[0] == 1, "Only single-person input is supported"
        reconstruction = reconstruction[0]
    elif reconstruction.ndim != 3:
        raise ValueError("Invalid 2D reconstruction shape")

elif args.dim == "3d":
    # Expect (T, J, 3)
    if reconstruction.ndim != 3:
        raise ValueError("Invalid 3D reconstruction shape")

num_frames, num_joints, coord_dim = reconstruction.shape
print(f"Frames: {num_frames}, Joints: {num_joints}, Dim: {coord_dim}")

# ---------- process ----------
for frame_idx in range(num_frames):
    frame_kpts = reconstruction[frame_idx]

    # append confidence
    conf = np.full((num_joints, 1), DEFAULT_CONFIDENCE, dtype=np.float32)
    frame_kpts_out = np.concatenate([frame_kpts, conf], axis=1)

    if args.dim == "3d":
        frame_json = [{
            "id": PERSON_ID,
            "keypoints3d": frame_kpts_out.tolist()
        }]
    else:  # 2d
        frame_json = [{
            "id": PERSON_ID,
            "keypoints": frame_kpts_out.tolist()
        }]

    out_path = os.path.join(output_dir, f"{frame_idx:06d}.json")
    with open(out_path, "w") as f:
        json.dump(frame_json, f, indent=2)

print(f"[DONE] Saved {num_frames} frames to:\n{output_dir}")