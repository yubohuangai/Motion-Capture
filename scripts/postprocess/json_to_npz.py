"""
Filepath: scripts/postprocess/json_to_npz.py

Convert JSON back to NPZ (reverse of npz_to_json.py)
"""

import numpy as np
import json
import os

json_path = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/view32_fov150/input_2D/keypoints.json"
npz_path = json_path.replace(".json", ".npz")

with open(json_path, "r") as f:
    data = json.load(f)

npz_data = {}

for key, value in data.items():
    arr = np.array(value)
    npz_data[key] = arr

np.savez(npz_path, **npz_data)

print(f"[DONE] Saved NPZ to: {npz_path}")