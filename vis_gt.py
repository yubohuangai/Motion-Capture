'''
File: vis_gt.py
'''

import json
import numpy as np
import matplotlib.pyplot as plt

# -------- config --------
ground_truth = "/Users/yubo/data/s2/seq1/rtm/keypoints3d/000000.json"

gt_pairs = [
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
    [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
    [14, 19], [19, 20], [14, 21],
    [11, 22], [22, 23], [11, 24]
]
# ------------------------

# Load JSON
with open(ground_truth, "r") as f:
    infos = json.load(f)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for info in infos:
    keypoints3d = np.asarray(info["keypoints3d"])  # (25, 4)

    x0, y0, z0 = keypoints3d[:, 0], keypoints3d[:, 1], keypoints3d[:, 2]

    # visualization-only remap
    x = y0
    y = z0
    z = -x0   # X is visually up and upright

    ax.scatter(x, y, z, s=20)

    for i, j in gt_pairs:
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [z[i], z[j]],
            linewidth=2
        )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Skeleton")
ax.view_init(elev=20, azim=225)

# ---- equal axis scaling ----
def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d()
    ])
    center = limits.mean(axis=1)
    radius = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

set_axes_equal(ax)
plt.show()