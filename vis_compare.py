import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- config ----------
gt_path = "/Users/yubo/data/s2/seq1/rtm/keypoints3d/000000.json"
pred_path = "/Users/yubo/data/s2/seq1/output/mhformer/view32_fisheye_fov150/output_3D/keypoints3d/000000.json"

gt_pairs = [
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
    [1, 0], [0, 15], [15, 17], [0, 16], [16, 18],
    [14, 19], [19, 20], [14, 21],
    [11, 22], [22, 23], [11, 24]
]

pred_pairs = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16]
]

# ---------- rigid alignment ----------
def rigid_align_3d(P, G):
    Pc = P.mean(axis=0)
    Gc = G.mean(axis=0)

    P0 = P - Pc
    G0 = G - Gc

    H = P0.T @ G0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = Gc - R @ Pc
    return R, t

# ---------- load data ----------
with open(gt_path) as f:
    gt = np.asarray(json.load(f)[0]["keypoints3d"])[:, :3]

with open(pred_path) as f:
    pred = np.asarray(json.load(f)[0]["keypoints3d"])[:, :3]

# ---------- anchor correspondences ----------
P_idx = [0, 14, 11]   # predict
G_idx = [8, 2, 5]     # gt

R, t = rigid_align_3d(pred[P_idx], gt[G_idx])
pred_aligned = (R @ pred.T).T + t

# ---------- visualization remap ----------
def vis_coords(kpts):
    x0, y0, z0 = kpts[:, 0], kpts[:, 1], kpts[:, 2]
    x = y0
    y = z0
    z = -x0
    return x, y, z

# ---------- plot ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# GT
xg, yg, zg = vis_coords(gt)
ax.scatter(xg, yg, zg, c="tab:blue", s=25, label="GT")
for i, j in gt_pairs:
    ax.plot([xg[i], xg[j]], [yg[i], yg[j]], [zg[i], zg[j]],
            c="tab:blue", linewidth=2)

# Predict (aligned)
xp, yp, zp = vis_coords(pred_aligned)
ax.scatter(xp, yp, zp, c="tab:red", s=25, label="Predict (Aligned)")
for i, j in pred_pairs:
    ax.plot([xp[i], xp[j]], [yp[i], yp[j]], [zp[i], zp[j]],
            c="tab:red", linewidth=2)

ax.set_title("GT (blue) vs Predict aligned to GT (red)")
ax.legend()
ax.view_init(elev=20, azim=225)

# ---------- equal axis ----------
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