"""
File: refine_predictive.py

Temporal refinement + smoothing + predictive filtering for 2D pose keypoints.
Designed for HAPLE-26 format: (26, 3) [x, y, confidence]
"""

import os
import json
import numpy as np
from glob import glob

# -----------------------
# CONFIG
# -----------------------
CONF_THRESH = 0.3          # confidence below this is considered bad
TEMPORAL_RADIUS = 3        # +-n frames for temporal refinement
MIN_VALID_NEIGHBORS = 2

# temporal weights
TEMPORAL_WEIGHTS = {0: 0.5, 1: 0.8, 2: 0.7, 3: 0.6}

# smoothing
SMOOTHING_ALPHA_HIGH = 0.7
SMOOTHING_ALPHA_LOW = 0.05
SMOOTHING_CONF_REF = 0.6

# predictive filter
PREDICTIVE_ALPHA = 0.7   # blend predicted position with current
VELOCITY_DECAY = 0.8     # smooth velocity estimate
MAX_PRED_FRAMES = 5      # max consecutive frames to predict


# -----------------------
# UTILITIES
# -----------------------
def confidence_to_alpha(c):
    """Map confidence to smoothing alpha."""
    if c <= CONF_THRESH:
        return SMOOTHING_ALPHA_LOW
    return SMOOTHING_ALPHA_LOW + (
        (SMOOTHING_ALPHA_HIGH - SMOOTHING_ALPHA_LOW)
        * min(c / SMOOTHING_CONF_REF, 1.0)
    )


def temporal_refine_joint(joint_idx, frame_idx, person_id, all_frames):
    """Refine a joint using temporal neighbors."""
    samples, weights = [], []

    for dt in range(-TEMPORAL_RADIUS, TEMPORAL_RADIUS + 1):
        t = frame_idx + dt
        if t not in all_frames: continue
        if person_id not in all_frames[t]: continue

        kpts = all_frames[t][person_id]
        x, y, c = kpts[joint_idx]

        if c <= 0: continue
        w = c * TEMPORAL_WEIGHTS.get(abs(dt), 0.0)
        if w <= 0: continue

        samples.append([x, y])
        weights.append(w)

    if len(samples) < MIN_VALID_NEIGHBORS:
        return None

    samples = np.asarray(samples)
    weights = np.asarray(weights)
    refined_xy = np.average(samples, axis=0, weights=weights)
    refined_conf = float(np.clip(np.mean(weights), 0.0, 1.0))
    return refined_xy[0], refined_xy[1], refined_conf


def temporal_smooth_sequence(all_frames):
    """Confidence-aware EMA smoothing."""
    smoothed_frames, prev_state = {}, {}

    for t in sorted(all_frames.keys()):
        smoothed_frames[t] = {}

        for pid, kpts in all_frames[t].items():
            kpts = kpts.copy()
            if pid not in prev_state:
                smoothed_frames[t][pid] = kpts
                prev_state[pid] = kpts
                continue

            prev_kpts = prev_state[pid]
            out = kpts.copy()
            for j in range(kpts.shape[0]):
                x, y, c = kpts[j]
                px, py, _ = prev_kpts[j]
                alpha = confidence_to_alpha(c)
                out[j, 0] = alpha * x + (1 - alpha) * px
                out[j, 1] = alpha * y + (1 - alpha) * py
                out[j, 2] = c
            smoothed_frames[t][pid] = out
            prev_state[pid] = out

    return smoothed_frames


# -----------------------
# PREDICTIVE FILTER
# -----------------------
def predictive_filter(all_frames):
    """Predict missing/low-confidence joints using velocity."""
    pred_frames = {}
    prev_state = {}
    prev_velocity = {}

    for t in sorted(all_frames.keys()):
        pred_frames[t] = {}
        for pid, kpts in all_frames[t].items():
            kpts = kpts.copy()
            if pid not in prev_state:
                # first frame: init state and velocity
                pred_frames[t][pid] = kpts
                prev_state[pid] = kpts
                prev_velocity[pid] = np.zeros((kpts.shape[0], 2), dtype=np.float32)
                continue

            out = kpts.copy()
            for j in range(kpts.shape[0]):
                x, y, c = kpts[j]
                px, py, pc = prev_state[pid][j]
                vx, vy = prev_velocity[pid][j]

                # if confidence low, predict
                if c <= CONF_THRESH:
                    # predicted position
                    pred_x = px + vx
                    pred_y = py + vy
                    # blend predicted with current (if any)
                    out[j, 0] = PREDICTIVE_ALPHA * pred_x + (1 - PREDICTIVE_ALPHA) * x
                    out[j, 1] = PREDICTIVE_ALPHA * pred_y + (1 - PREDICTIVE_ALPHA) * y
                # update velocity
                vx_new = VELOCITY_DECAY * vx + (x - px)
                vy_new = VELOCITY_DECAY * vy + (y - py)
                prev_velocity[pid][j] = np.array([vx_new, vy_new], dtype=np.float32)

            pred_frames[t][pid] = out
            prev_state[pid] = out

    return pred_frames


# -----------------------
# MAIN PIPELINE
# -----------------------
def temporal_refine_sequence(json_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_frames = {}

    # -------- load all frames --------
    for idx, jp in enumerate(json_files):
        with open(jp, "r") as f:
            data = json.load(f)
        frame_people = {}
        for annot in data["annots"]:
            frame_people[annot["personID"]] = np.array(
                annot["keypoints"], dtype=np.float32
            )
        all_frames[idx] = frame_people

    # -------- temporal refinement (bad joints only) --------
    for idx in all_frames:
        for pid in all_frames[idx]:
            kpts = all_frames[idx][pid]
            for j in range(kpts.shape[0]):
                if kpts[j, 2] >= CONF_THRESH:
                    continue
                refined = temporal_refine_joint(j, idx, pid, all_frames)
                if refined is not None:
                    kpts[j] = refined

    # -------- temporal smoothing --------
    all_frames = temporal_smooth_sequence(all_frames)

    # -------- predictive filtering --------
    all_frames = predictive_filter(all_frames)

    # -------- save results --------
    for idx, jp in enumerate(json_files):
        with open(jp, "r") as f:
            data = json.load(f)
        for annot in data["annots"]:
            pid = annot["personID"]
            annot["keypoints"] = all_frames[idx][pid].tolist()
        out_path = os.path.join(output_dir, os.path.basename(jp))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Temporal refinement + smoothing + predictive filtering done. Saved to: {output_dir}")


# -----------------------
# ENTRY POINT
# -----------------------
if __name__ == "__main__":
    root = "/Users/yubo/data/s2/seq1/360/view32_fisheye"
    json_dir = os.path.join(root, "annots")
    subdirs = [d for d in os.listdir(json_dir) if os.path.isdir(os.path.join(json_dir, d))]
    assert len(subdirs) == 1, "Expected exactly one subdir under annots"
    subdir = subdirs[0]
    json_files = sorted(glob(os.path.join(json_dir, subdir, "*.json")))
    output_dir = os.path.join(root, "annots-refined", subdir)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    temporal_refine_sequence(json_files, output_dir)