"""
Filepath: scripts/postprocess/npz_to_json.py
"""

import numpy as np
import json
import os

npz_path = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/view32_fov150/input_2D/keypoints.npz"
json_path = npz_path.replace(".npz", ".json")

# Load NPZ
data = np.load(npz_path, allow_pickle=True)

# print("Keys in NPZ:", data.files)

json_data = {}

for key in data.files:
    arr = data[key]
    # Convert numpy arrays to lists (JSON serializable)
    if isinstance(arr, np.ndarray):
        json_data[key] = arr.tolist()
    else:
        json_data[key] = arr

# Save to JSON
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)

# print(f"Saved JSON to: {json_path}")