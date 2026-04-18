"""
File: proj_annots.py
Project 2D keypoints from fisheye image space to perspective image space.
Compatible with the fisheye_to_perspective() image warp.
"""

import os
import json
import numpy as np
from glob import glob

# -----------------------
# CONFIG
# -----------------------
FOV_DEG = 150
K1 = 0.05
K2 = 0.005

# -----------------------
# CORE PROJECTION
# -----------------------
def fisheye_kpts_to_perspective(
    keypoints,
    Wf, Hf,
    Wp, Hp,
    fov_deg=150
):
    """
    keypoints: (N, 3) [x_fish, y_fish, conf]
    returns:   (N, 3) [x_persp, y_persp, conf]
    """

    keypoints = np.asarray(keypoints, dtype=np.float32)
    out = keypoints.copy()

    fov = np.deg2rad(fov_deg)
    f = (Wp / 2) / np.tan(fov / 2)
    f_fish = (min(Wf, Hf) / 2) / (np.pi / 2)

    for i, (xf, yf, c) in enumerate(keypoints):
        if c <= 0:
            continue

        # fisheye image center
        dx = xf - Wf / 2
        dy = Hf / 2 - yf

        r = np.sqrt(dx * dx + dy * dy)
        if r < 1e-6:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arctan2(dy, dx)

            # invert fisheye radial distortion (Newton iterations)
            phi = r / f_fish
            for _ in range(5):
                phi = (r / f_fish - K1 * phi**3 - K2 * phi**5)

        if phi > np.pi / 2:
            out[i, 2] = 0.0
            continue

        # ray direction
        Z = np.cos(phi)
        X = np.sin(phi) * np.cos(theta)
        Y = np.sin(phi) * np.sin(theta)

        # perspective projection
        xp = f * (X / Z) + Wp / 2
        yp = Hp / 2 - f * (Y / Z)

        if xp < 0 or xp >= Wp or yp < 0 or yp >= Hp:
            out[i, 2] = 0.0
            continue

        out[i, 0] = xp
        out[i, 1] = yp

    return out


# -----------------------
# PROCESS JSON SEQUENCE
# -----------------------
def project_sequence(json_files, output_dir, image_shape):
    os.makedirs(output_dir, exist_ok=True)

    Hf, Wf = image_shape
    Hp, Wp = Hf, Wf  # same output size as your image warp

    for jp in json_files:
        with open(jp, "r") as f:
            data = json.load(f)

        for annot in data["annots"]:
            kpts = np.asarray(annot["keypoints"], dtype=np.float32)
            kpts_proj = fisheye_kpts_to_perspective(
                kpts, Wf, Hf, Wp, Hp, fov_deg=FOV_DEG
            )
            annot["keypoints"] = kpts_proj.tolist()

        out_path = os.path.join(output_dir, os.path.basename(jp))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Projected keypoints saved to: {output_dir}")


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    root = "/Users/yubo/data/s2/seq1/360/view32_fisheye"
    json_dir = os.path.join(root, "annots-rtm-refined")

    subdirs = [
        d for d in os.listdir(json_dir)
        if os.path.isdir(os.path.join(json_dir, d))
    ]
    assert len(subdirs) == 1

    subdir = subdirs[0]
    json_files = sorted(glob(os.path.join(json_dir, subdir, "*.json")))

    # IMPORTANT: use original fisheye image size
    # Example: 2880x2880
    image_shape = (2880, 2880)

    output_dir = os.path.join(root, "annots-rtm-perspective", subdir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    project_sequence(json_files, output_dir, image_shape)