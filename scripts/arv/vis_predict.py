'''
File: vis_predict.py
'''
import json
import numpy as np
import matplotlib.pyplot as plt

# -------- config --------
predict = "/Users/yubo/data/s2/seq1/output/mhformer/view32_fisheye_fov150/output_3D/keypoints3d/000000.json"

predict_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                 [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                 [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
# ------------------------

# Load JSON
with open(predict, "r") as f:
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

    for i, j in predict_pairs:
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