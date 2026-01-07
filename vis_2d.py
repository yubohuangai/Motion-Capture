'''
File: vis_2d.py
'''

import os
import json
import cv2
import numpy as np
from glob import glob

# -----------------------
# CONFIG
# -----------------------
root = "/Users/yubo/data/s2/seq1/360/view32_fisheye"
json_dir = os.path.join(root, "annots-refined")  # folder containing JSONs
output_dir = os.path.join(root, "output2d-refined")
os.makedirs(output_dir, exist_ok=True)

FIRST_PERSON_ONLY = True  # <-- set True to visualize only first person

# -----------------------
# Haple-26 skeleton with body-part grouping
# -----------------------
HAPLE_PARTS = {
    "face": {"pairs": [[0, 1], [0, 2], [1, 3], [2, 4]], "color": (0, 255, 255)},
    "head": {"pairs": [[17, 18]], "color": (0, 165, 255)},
    "torso": {"pairs": [[18, 5], [18, 6], [18, 19], [11, 19], [12, 19]], "color": (0, 255, 0)},
    "left_arm": {"pairs": [[5, 7], [7, 9]], "color": (255, 0, 0)},
    "right_arm": {"pairs": [[6, 8], [8, 10]], "color": (0, 0, 255)},
    "left_leg": {"pairs": [[11, 13], [13, 15]], "color": (255, 128, 0)},
    "right_leg": {"pairs": [[12, 14], [14, 16]], "color": (128, 0, 255)},
    "left_foot": {"pairs": [[15, 24], [24, 20], [24, 22]], "color": (255, 255, 0)},
    "right_foot": {"pairs": [[16, 25], [25, 23], [25, 21]], "color": (255, 0, 255)},
}

# drawing parameters
JOINT_RADIUS = 6
BONE_THICKNESS = 4

# -----------------------
# PROCESS ALL JSON FILES
# -----------------------
json_files = sorted(glob(os.path.join(json_dir, "**/*.json")))

for json_path in json_files:
    # load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    img_path = os.path.join(root, data["filename"])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}, skipping.")
        continue

    # draw annotations
    annots_to_draw = data["annots"]
    if FIRST_PERSON_ONLY and len(annots_to_draw) > 0:
        annots_to_draw = [annots_to_draw[0]]  # only first person

    for annot in annots_to_draw:
        pid = annot["personID"]
        kpts = np.array(annot["keypoints"])  # (26, 3)

        # ---- draw joints (white outline + red center) ----
        for x, y, c in kpts:
            if c > 0.2:
                cv2.circle(img, (int(x), int(y)), JOINT_RADIUS + 2, (255, 255, 255), -1)
                cv2.circle(img, (int(x), int(y)), JOINT_RADIUS, (0, 0, 255), -1)

        # ---- draw bones by body part ----
        for part in HAPLE_PARTS.values():
            color = part["color"]
            for i, j in part["pairs"]:
                if i < len(kpts) and j < len(kpts):
                    xi, yi, ci = kpts[i]
                    xj, yj, cj = kpts[j]
                    if ci > 0.2 and cj > 0.2:
                        cv2.line(
                            img,
                            (int(xi), int(yi)),
                            (int(xj), int(yj)),
                            color,
                            BONE_THICKNESS,
                            lineType=cv2.LINE_AA
                        )

        # ---- person ID near head ----
        hx, hy, _ = kpts[17]
        cv2.putText(
            img,
            f"ID {pid}",
            (int(hx), int(hy) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # save output
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)