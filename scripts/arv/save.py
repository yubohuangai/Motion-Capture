import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- config ----------
gt_dir = "/Users/yubo/data/s2/seq1/gt/rtm/keypoints3d"
pred_dir = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/01_view40/output_3D/output_keypoints_3d"
save_path = "/Users/yubo/data/s2/seq1/360/output/poseformerv2/01_view40/output_3D/images"

FPS_RATIO = 1  # 30fps / 5fps

# ---------- skeleton pairs ----------
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

# ---------- visualization remap ----------
def vis_coords(kpts):
    x0, y0, z0 = kpts[:, 0], kpts[:, 1], kpts[:, 2]
    x = -y0  # horizontal flip
    y = z0
    z = -x0
    return x, y, z

# ---------- collect files ----------
gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".json"))[1:]  # skip first
pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".json"))[::FPS_RATIO]

assert len(gt_files) <= len(pred_files), "Not enough pred frames!"

os.makedirs(save_path, exist_ok=True)

# ---------- compute global bounding box ----------
all_coords = []
for gt_file, pred_file in zip(gt_files, pred_files):
    # load GT
    with open(os.path.join(gt_dir, gt_file)) as f:
        gt = np.asarray(json.load(f)[0]["keypoints3d"])[:, :3]
    # load prediction
    with open(os.path.join(pred_dir, pred_file)) as f:
        pred = np.asarray(json.load(f)[0]["keypoints"])[:, :3]

    # align prediction to GT
    P_idx = [0, 14, 11]
    G_idx = [8, 2, 5]
    R, t = rigid_align_3d(pred[P_idx], gt[G_idx])
    pred_aligned = (R @ pred.T).T + t

    # remap coordinates
    xg, yg, zg = vis_coords(gt)
    xp, yp, zp = vis_coords(pred_aligned)

    all_coords.append(np.stack([xg, yg, zg], axis=1))
    all_coords.append(np.stack([xp, yp, zp], axis=1))

all_coords = np.concatenate(all_coords, axis=0)
x_min, y_min, z_min = all_coords.min(axis=0)
x_max, y_max, z_max = all_coords.max(axis=0)
x_center, y_center, z_center = (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
radius = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2

# ---------- main loop ----------
for gt_file, pred_file in zip(gt_files, pred_files):
    gt_path = os.path.join(gt_dir, gt_file)
    pred_path = os.path.join(pred_dir, pred_file)

    with open(gt_path) as f:
        gt = np.asarray(json.load(f)[0]["keypoints3d"])[:, :3]
    with open(pred_path) as f:
        pred = np.asarray(json.load(f)[0]["keypoints"])[:, :3]

    # align prediction
    P_idx = [0, 14, 11]
    G_idx = [8, 2, 5]
    R, t = rigid_align_3d(pred[P_idx], gt[G_idx])
    pred_aligned = (R @ pred.T).T + t

    # ---------- plot ----------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # GT
    xg, yg, zg = vis_coords(gt)
    ax.scatter(xg, yg, zg, c="tab:blue", s=25)
    for i, j in gt_pairs:
        ax.plot([xg[i], xg[j]], [yg[i], yg[j]], [zg[i], zg[j]],
                c="tab:blue", linewidth=2)

    # Pred
    xp, yp, zp = vis_coords(pred_aligned)
    ax.scatter(xp, yp, zp, c="tab:red", s=25)
    for i, j in pred_pairs:
        ax.plot([xp[i], xp[j]], [yp[i], yp[j]], [zp[i], zp[j]],
                c="tab:red", linewidth=2)

    ax.view_init(elev=7, azim=-30)

    # ---------- fixed axes with correct proportions ----------
    scale = 0.9  # zoom in (<1) or out (>1)

    # Compute global center
    x_center, y_center, z_center = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

    # Maximum range for equal scaling
    x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
    max_range = max(x_range, y_range, z_range) * scale

    # Set limits for all axes with the same half-length
    half = max_range / 2
    ax.set_xlim([x_center - half, x_center + half])
    ax.set_ylim([y_center - half, y_center + half])
    ax.set_zlim([z_center - half, z_center + half])
    ax.set_box_aspect((1, 1, 1))
    plt.subplots_adjust(0, 0, 1, 1)

    # ---------- save ----------
    base_name = os.path.splitext(pred_file)[0]
    save_file = os.path.join(save_path, f"{base_name}.jpg")
    plt.savefig(save_file, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

print(f"All frames saved to: {save_path}")