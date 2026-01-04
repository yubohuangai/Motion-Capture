import json
import os
import numpy as np

# ---------- paths ----------
input_json = (
    "/Users/yubo/data/s2/seq1/output/mhformer/"
    "view32_fisheye_fov150/output_3D/output_keypoints_3d.json"
)

output_dir = (
    "/Users/yubo/data/s2/seq1/output/mhformer/"
    "view32_fisheye_fov150/output_3D/keypoints3d"
)

os.makedirs(output_dir, exist_ok=True)

# ---------- config ----------
DEFAULT_CONFIDENCE = 1.0   # change if you want (e.g. 8.0 or 10.0)
PERSON_ID = 0

# ---------- load ----------
with open(input_json, "r") as f:
    data = json.load(f)

# shape: (num_frames, num_joints, 3)
reconstruction = np.array(data["reconstruction"], dtype=np.float32)

num_frames, num_joints, _ = reconstruction.shape
print(f"Loaded {num_frames} frames, {num_joints} joints")

# ---------- process ----------
for frame_idx in range(num_frames):
    frame_kpts = reconstruction[frame_idx]

    # append confidence column
    conf = np.full((num_joints, 1), DEFAULT_CONFIDENCE, dtype=np.float32)
    frame_kpts_4d = np.concatenate([frame_kpts, conf], axis=1)

    frame_json = [
        {
            "id": PERSON_ID,
            "keypoints3d": frame_kpts_4d.tolist()
        }
    ]

    out_path = os.path.join(output_dir, f"{frame_idx:06d}.json")

    with open(out_path, "w") as f:
        json.dump(frame_json, f, indent=2)

print(f"Saved {num_frames} frame files to:\n{output_dir}")