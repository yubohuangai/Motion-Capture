"""
Generate foreground masks for multi-view object reconstruction.

Modes:
  1. **background subtraction** (default): requires a background-only frame.
     Computes per-pixel difference, thresholds, and cleans with morphology.
  2. **SAM**: full-image automatic masks—SAM1 (`segment_anything`) or SAM2
     (`SAM2AutomaticMaskGenerator`).  Choose with ``--sam_backend``.
  3. **hybrid**: background subtraction mask → bounding box → SAM on a **crop**
     of that box (default) or the full frame (`--hybrid_sam_space full`).
     SAM can output multiple masks; the one with best **IoU** vs. the diff mask
     is kept when a prior is available.  Use `--hybrid_combine intersect` to
     AND with bg_sub.  By default, **post-SAM center-blob** refinement runs
     (same spatial idea as center_only on the diff mask) to drop extra SAM
     blobs; disable with ``--no_post_sam_center_only``.

**SAM 2** ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)) is
often stronger than SAM 1, but the upstream project requires **Python ≥3.10**
and **torch ≥2.5.1**—use a separate conda env, not an older (e.g. 3.9) stack.
Pass ``--sam_backend sam2``, ``--sam2_checkpoint`` (``.pt``), and optionally
``--sam2_config`` (Hydra config name, default matches SAM 2.1 Hiera Large).

Output layout:
    <data>/masks/<cam>/NNNNNN.png   (255 = foreground, 0 = background)
    By default also writes mask overlays to ``<data>/mask_vis/`` and
    foreground-only RGB to ``<data>/foreground_images/`` (disable with
    ``--no_vis`` / ``--no_save_fg_images``).  Background-subtraction modes
    use **center-only** blob filtering by default (disable with ``--no_center_only``).

Usage:
    # Minimal hybrid (defaults: mode=hybrid, frame=0, bg_frame=0, threshold=30,
    # SAM checkpoint at <Motion-Capture>/data/sam/sam_vit_h_4b8939.pth)
    python apps/reconstruction/generate_masks.py /path/to/data --bg_data /path/to/background

    # Background subtraction only
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --mode bg_sub --frame 0 --bg_frame 100 --threshold 30

    # SAM automatic masks (needs segment_anything + model checkpoint)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --mode sam --sam_checkpoint /path/to/sam_vit_h.pth

    # Hybrid with SAM 1 (ViT, .pth)
    python apps/reconstruction/generate_masks.py /mnt/yubo/obj/cube \\
        --bg_data /mnt/yubo/obj/background --frame 0 --bg_frame 0 \\
        --mode hybrid --sam_backend sam1 --sam_checkpoint /path/to/sam_vit_h.pth

    # Hybrid with SAM 2 (.pt + yaml config name; see sam2 repo checkpoints)
    python apps/reconstruction/generate_masks.py /mnt/yubo/obj/cube \\
        --bg_data /mnt/yubo/obj/background --frame 0 --bg_frame 0 \\
        --mode hybrid --sam_backend sam2 \\
        --sam2_checkpoint /path/to/sam2.1_hiera_large.pt

    # Shrink a slightly bloated diff mask (no SAM)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --bg_frame 100 --erode 3
"""

import argparse
import contextlib
import os
import sys
from glob import glob
from os.path import join

import cv2
import numpy as np

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..'))

# Repo root (Motion-Capture): default SAM path works regardless of cwd.
_REPO_ROOT = os.path.normpath(join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_SAM_CHECKPOINT = join(_REPO_ROOT, 'data', 'sam', 'sam_vit_h_4b8939.pth')


def get_cam_names(data_root):
    img_dir = join(data_root, 'images')
    cams = sorted([
        d for d in os.listdir(img_dir)
        if os.path.isdir(join(img_dir, d))
    ])
    return cams


def find_image(data_root, cam, frame, ext):
    path = join(data_root, 'images', cam, f'{frame:06d}{ext}')
    if os.path.exists(path):
        return path
    candidates = sorted(glob(join(data_root, 'images', cam, f'*{ext}')))
    if len(candidates) == 0:
        raise FileNotFoundError(f'No images for camera {cam}')
    if frame < len(candidates):
        return candidates[frame]
    return candidates[0]


def save_mask_overlay(img_bgr, mask, vis_dir, cam, frame):
    """Save mask overlay only: green tint on foreground + red contour (same size as input)."""
    os.makedirs(vis_dir, exist_ok=True)
    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr)
    green[:, :, 1] = 255
    overlay[mask > 0] = cv2.addWeighted(overlay, 0.5, green, 0.5, 0)[mask > 0]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    out_path = join(vis_dir, f'{cam}_{frame:06d}.jpg')
    cv2.imwrite(out_path, overlay)
    return out_path


def save_foreground_image(img_bgr, mask, fg_dir, cam, frame, ext):
    """Black out background; save as ``<cam>_<frame>.<ext>`` in one folder (like mask_vis)."""
    os.makedirs(fg_dir, exist_ok=True)
    fg_img = img_bgr.copy()
    fg_img[mask == 0] = 0
    out_path = join(fg_dir, f'{cam}_{frame:06d}{ext}')
    cv2.imwrite(out_path, fg_img)
    return out_path


def locate_object_bbox(mask, max_area_ratio=0.10, min_area_ratio=0.0005,
                       bottom_ratio=0.75):
    """
    Find the bounding box of the object blob on the table.

    Works on a lightly-cleaned mask (opening only, no closing) so the person
    and object stay as separate contours.  Returns (x, y, w, h) of the best
    candidate or None.
    """
    h, w = mask.shape[:2]
    img_area = h * w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio
    y_cutoff = h * (1.0 - bottom_ratio)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        blob_cy = by + bh / 2.0
        if blob_cy < y_cutoff:
            continue
        candidates.append((area, (bx, by, bw, bh)))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def compute_background_subtraction_mask(fg_path, bg_path, threshold, morph_kernel,
                                        center_only, max_area_ratio):
    """Return uint8 mask (0 / 255) from frame differencing; no dilate/erode."""
    fg = cv2.imread(fg_path)
    bg = cv2.imread(bg_path)
    if fg is None:
        raise FileNotFoundError(f'Cannot read foreground: {fg_path}')
    if bg is None:
        raise FileNotFoundError(f'Cannot read background: {bg_path}')
    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)

    diff = np.abs(fg_f - bg_f).max(axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel),
    )

    if center_only:
        # --- Pass 1: locate the object with conservative settings ---
        coarse = (diff > threshold).astype(np.uint8) * 255
        small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, small_k,
                                  iterations=1)
        bbox = locate_object_bbox(coarse, max_area_ratio=max_area_ratio)

        if bbox is not None:
            bx, by, bw, bh = bbox
            h, w = diff.shape[:2]
            # tight bbox: 10% padding (dilate adds margin later)
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)

            # --- Pass 2: same threshold but restricted to bbox ---
            mask = np.zeros((h, w), dtype=np.uint8)
            roi = diff[y1:y2, x1:x2]
            mask[y1:y2, x1:x2] = (roi > threshold).astype(np.uint8) * 255

            # close to fill gaps within the box, open to remove speckles
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                    iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_k,
                                    iterations=1)

            # keep only the largest contour, use convex hull for clean shape
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                biggest = max(cnts, key=cv2.contourArea)
                hull = cv2.convexHull(biggest)
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [hull], -1, 255, cv2.FILLED)
        else:
            mask = np.zeros(diff.shape[:2], dtype=np.uint8)
    else:
        mask = (diff > threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
            mask = filled

    return mask


def refine_hybrid_mask_single_blob(mask_uint8, morph_kernel, max_area_ratio):
    """
    After SAM, keep one primary blob using the same spatial prior as
    ``center_only`` on the diff mask: light opening → ``locate_object_bbox`` →
    restrict to padded ROI → close/open → largest contour → convex hull.

    If no bbox is found on the opened mask, falls back to the largest contour
    on the opened mask (convex hull).
    """
    if mask_uint8 is None or mask_uint8.sum() == 0:
        return mask_uint8

    h, w = mask_uint8.shape[:2]
    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel),
    )
    coarse = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, small_k, iterations=1)
    bbox = locate_object_bbox(coarse, max_area_ratio=max_area_ratio)

    if bbox is None:
        cnts, _ = cv2.findContours(coarse, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask_uint8
        biggest = max(cnts, key=cv2.contourArea)
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [cv2.convexHull(biggest)], -1, 255, cv2.FILLED)
        return out

    bx, by, bw, bh = bbox
    pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)

    mask = np.zeros((h, w), dtype=np.uint8)
    roi = mask_uint8[y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = np.where(roi > 127, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        cnts, _ = cv2.findContours(coarse, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask_uint8
        biggest = max(cnts, key=cv2.contourArea)
        out = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(out, [cv2.convexHull(biggest)], -1, 255, cv2.FILLED)
        return out

    biggest = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(biggest)
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(out, [hull], -1, 255, cv2.FILLED)
    return out


def apply_mask_postprocess(mask, dilate=0, erode=0):
    """Optional dilate then erode on binary uint8 mask."""
    out = mask
    if dilate > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        out = cv2.dilate(out, dk, iterations=1)
    if erode > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
        out = cv2.erode(out, ek, iterations=1)
    return out


def bbox_from_mask(mask, pad_ratio=0.0):
    """
    Tight axis-aligned box around the largest foreground contour.
    pad_ratio expands each side by a fraction of (w, h).  Returns xyxy numpy
    array or None if empty.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    h, w = mask.shape[:2]
    if bw <= 0 or bh <= 0:
        return None
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(w, bx + bw + pad_x)
    y2 = min(h, by + bh + pad_y)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _mask_from_predict_output(masks_arr):
    """Normalize SAM1/SAM2 ``predict()`` mask output to HxW uint8 {0, 255}."""
    m = np.asarray(masks_arr)
    if m.ndim == 4:
        m = m[0]
    if m.ndim == 3:
        m = m[0]
    return (m.astype(np.uint8) > 0).astype(np.uint8) * 255


def _binary_iou(mask_a, mask_b):
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / union) if union > 0 else 0.0


def hybrid_sam_predict_mask(
    predictor,
    image_rgb,
    box_xyxy,
    *,
    use_sam2,
    use_cuda,
    mask_prior=None,
):
    """
    Box prompt in the same pixel frame as ``image_rgb``.
    If ``mask_prior`` is set (HxW, same size as image), requests multiple
    candidate masks and returns the one with highest IoU to the prior.
    """
    import torch

    box = np.asarray(box_xyxy, dtype=np.float32)
    multimask = mask_prior is not None
    if use_sam2:
        with torch.inference_mode():
            ctx = (
                torch.autocast('cuda', dtype=torch.bfloat16)
                if use_cuda else contextlib.nullcontext()
            )
            with ctx:
                predictor.set_image(image_rgb)
                masks, _, _ = predictor.predict(
                    box=box.astype(np.float32),
                    multimask_output=multimask,
                )
    else:
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=multimask,
        )
    if not multimask:
        return _mask_from_predict_output(masks)
    ms = np.asarray(masks)
    if ms.ndim == 4:
        ms = ms[0]
    best_m, best_s = None, -1.0
    for i in range(ms.shape[0]):
        mi = _mask_from_predict_output(ms[i : i + 1])
        s = _binary_iou(mi, mask_prior)
        if s > best_s:
            best_s = s
            best_m = mi
    return best_m if best_m is not None else _mask_from_predict_output(masks)


def background_subtraction(data_root, cam_names, frame, bg_frame, ext,
                           threshold, morph_kernel, output_dir, vis_dir=None,
                           bg_data=None, center_only=False, dilate=0, erode=0,
                           max_area_ratio=0.10, save_fg_dir=None):
    """Frame-difference masking with optional smart blob selection.

    When --center_only is set, uses a two-pass strategy:
      Pass 1 (locate): light opening on high-threshold diff → find object bbox
      Pass 2 (refine): same threshold inside tight bbox → largest blob only
    """
    bg_root = bg_data or data_root
    for cam in cam_names:
        fg_path = find_image(data_root, cam, frame, ext)
        bg_path = find_image(bg_root, cam, bg_frame, ext)

        mask = compute_background_subtraction_mask(
            fg_path, bg_path, threshold, morph_kernel, center_only,
            max_area_ratio,
        )
        if center_only and mask.sum() == 0:
            print(f'  {cam}: WARNING - no object blob found')

        mask = apply_mask_postprocess(mask, dilate=dilate, erode=erode)

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, mask)
        print(f'  {cam}: {out_path}  (fg pixels: {mask.sum() // 255})')

        if save_fg_dir is not None:
            img_bgr = cv2.imread(fg_path)
            out_fg_path = save_foreground_image(
                img_bgr, mask, save_fg_dir, cam, frame, ext,
            )
            print(f'    fg: {out_fg_path}')

        if vis_dir is not None:
            img_bgr = cv2.imread(fg_path)
            vis_path = save_mask_overlay(img_bgr, mask, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def sam_segmentation(data_root, cam_names, frame, ext, output_dir,
                     sam_checkpoint, sam_model_type, vis_dir=None,
                     save_fg_dir=None):
    """Use Segment Anything to produce masks from auto-detected boxes."""
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        print('ERROR: segment_anything is not installed.')
        print('  pip install segment-anything')
        print('  Download checkpoint from https://github.com/facebookresearch/segment-anything')
        sys.exit(1)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[SAM] Loading model {sam_model_type} from {sam_checkpoint} ...')
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=1000,
    )

    for cam in cam_names:
        img_path = find_image(data_root, cam, frame, ext)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = mask_generator.generate(img_rgb)

        combined = np.zeros(img.shape[:2], dtype=np.uint8)
        for m in masks:
            combined[m['segmentation']] = 255

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, combined)
        print(f'  {cam}: {out_path}  ({len(masks)} segments)')

        if save_fg_dir is not None:
            out_fg_path = save_foreground_image(
                img, combined, save_fg_dir, cam, frame, ext,
            )
            print(f'    fg: {out_fg_path}')

        if vis_dir is not None:
            vis_path = save_mask_overlay(img, combined, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def sam2_segmentation(data_root, cam_names, frame, ext, output_dir,
                      sam2_checkpoint, sam2_config, vis_dir=None,
                      save_fg_dir=None):
    """Full-image automatic masks via SAM2 ``SAM2AutomaticMaskGenerator``."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print('ERROR: sam2 is not installed.')
        print('  See https://github.com/facebookresearch/sam2 (Python>=3.10, torch>=2.5.1)')
        sys.exit(1)

    import torch

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    cfg = sam2_config or 'configs/sam2.1/sam2.1_hiera_l.yaml'
    print(f'[SAM2] Loading config={cfg!r} checkpoint={sam2_checkpoint!r} ...')
    model = build_sam2(cfg, sam2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=1000,
    )

    for cam in cam_names:
        img_path = find_image(data_root, cam, frame, ext)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            ctx = (
                torch.autocast('cuda', dtype=torch.bfloat16)
                if use_cuda else contextlib.nullcontext()
            )
            with ctx:
                masks = mask_generator.generate(img_rgb)

        combined = np.zeros(img.shape[:2], dtype=np.uint8)
        for m in masks:
            combined[m['segmentation']] = 255

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, combined)
        print(f'  {cam}: {out_path}  ({len(masks)} segments)')

        if save_fg_dir is not None:
            out_fg_path = save_foreground_image(
                img, combined, save_fg_dir, cam, frame, ext,
            )
            print(f'    fg: {out_fg_path}')

        if vis_dir is not None:
            vis_path = save_mask_overlay(img, combined, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def hybrid_bg_sam(data_root, cam_names, frame, bg_frame, ext,
                  threshold, morph_kernel, output_dir, vis_dir=None,
                  bg_data=None, center_only=False, dilate=0, erode=0,
                  max_area_ratio=0.10, save_fg_dir=None,
                  sam_checkpoint=None, sam_model_type='vit_h',
                  hybrid_combine='sam', sam_box_pad_ratio=0.12,
                  sam_backend='sam1', sam2_checkpoint=None,
                  sam2_config=None, hybrid_sam_space='crop',
                  post_sam_center_only=True):
    """
    Background subtraction mask → bbox → SAM1/SAM2 box prompt.

    ``hybrid_sam_space``:
      * ``crop`` — crop the padded bbox, run SAM inside (less global confusion).
      * ``full`` — ``set_image`` on the full frame (previous behavior).

    ``hybrid_combine``:
      * ``sam`` — use SAM mask only (after optional IoU pick vs. diff mask).
      * ``intersect`` — bitwise AND of diff mask and SAM.

    ``post_sam_center_only``:
      If True, run :func:`refine_hybrid_mask_single_blob` after SAM (and after
      ``intersect`` if used) to remove stray SAM regions using the same table /
      center spatial prior as traditional ``center_only``.
    """
    import torch

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    use_sam2 = sam_backend == 'sam2'

    if use_sam2:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            print('ERROR: sam2 is not installed.')
            print('  See https://github.com/facebookresearch/sam2 (Python>=3.10, torch>=2.5.1)')
            sys.exit(1)
        cfg = sam2_config or 'configs/sam2.1/sam2.1_hiera_l.yaml'
        print(f'[hybrid] Loading SAM2 config={cfg!r} checkpoint={sam2_checkpoint!r} ...')
        sam2_model = build_sam2(cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
    else:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            print('ERROR: segment_anything is not installed.')
            print('  pip install segment-anything')
            print('  Download checkpoint from https://github.com/facebookresearch/segment-anything')
            sys.exit(1)

        print(f'[hybrid] Loading SAM1 {sam_model_type} from {sam_checkpoint} ...')
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        predictor = SamPredictor(sam)

    print(f'[hybrid] SAM image space: {hybrid_sam_space}, '
          f'post_sam_center_only: {post_sam_center_only}')

    bg_root = bg_data or data_root
    for cam in cam_names:
        fg_path = find_image(data_root, cam, frame, ext)
        bg_path = find_image(bg_root, cam, bg_frame, ext)

        mask_bg = compute_background_subtraction_mask(
            fg_path, bg_path, threshold, morph_kernel, center_only,
            max_area_ratio,
        )
        if center_only and mask_bg.sum() == 0:
            print(f'  {cam}: WARNING - no object blob found (bg_sub empty)')

        img = cv2.imread(fg_path)
        if img is None:
            raise FileNotFoundError(f'Cannot read foreground: {fg_path}')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        box = bbox_from_mask(mask_bg, pad_ratio=sam_box_pad_ratio)
        if box is None or mask_bg.sum() == 0:
            mask = mask_bg
            print(f'  {cam}: hybrid fallback — no bbox, using bg_sub only')
        else:
            if hybrid_sam_space == 'crop':
                xi1, yi1, xi2, yi2 = [int(round(float(v))) for v in box]
                xi1 = max(0, xi1)
                yi1 = max(0, yi1)
                xi2 = min(img_rgb.shape[1], max(xi1 + 1, xi2))
                yi2 = min(img_rgb.shape[0], max(yi1 + 1, yi2))
                min_side = 32
                if (xi2 - xi1) < min_side or (yi2 - yi1) < min_side:
                    mask_sam = hybrid_sam_predict_mask(
                        predictor, img_rgb, box,
                        use_sam2=use_sam2, use_cuda=use_cuda,
                        mask_prior=mask_bg,
                    )
                else:
                    crop_rgb = img_rgb[yi1:yi2, xi1:xi2]
                    mask_c = mask_bg[yi1:yi2, xi1:xi2]
                    box_l = bbox_from_mask(mask_c, pad_ratio=0.05)
                    if box_l is None or mask_c.sum() == 0:
                        mask_sam = hybrid_sam_predict_mask(
                            predictor, img_rgb, box,
                            use_sam2=use_sam2, use_cuda=use_cuda,
                            mask_prior=mask_bg,
                        )
                    else:
                        mask_sam_c = hybrid_sam_predict_mask(
                            predictor, crop_rgb, box_l,
                            use_sam2=use_sam2, use_cuda=use_cuda,
                            mask_prior=mask_c,
                        )
                        mask_sam = np.zeros(mask_bg.shape[:2], dtype=np.uint8)
                        mask_sam[yi1:yi2, xi1:xi2] = mask_sam_c
            else:
                mask_sam = hybrid_sam_predict_mask(
                    predictor, img_rgb, box,
                    use_sam2=use_sam2, use_cuda=use_cuda,
                    mask_prior=mask_bg,
                )
            if hybrid_combine == 'intersect':
                mask = cv2.bitwise_and(mask_bg, mask_sam)
            else:
                mask = mask_sam

        if post_sam_center_only and mask is not None and mask.sum() > 0:
            mask = refine_hybrid_mask_single_blob(
                mask, morph_kernel, max_area_ratio,
            )

        mask = apply_mask_postprocess(mask, dilate=dilate, erode=erode)

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, mask)
        print(f'  {cam}: {out_path}  (fg pixels: {mask.sum() // 255})')

        if save_fg_dir is not None:
            out_fg_path = save_foreground_image(
                img, mask, save_fg_dir, cam, frame, ext,
            )
            print(f'    fg: {out_fg_path}')

        if vis_dir is not None:
            vis_path = save_mask_overlay(img, mask, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def _ensure_sam_checkpoint(path, parser):
    """Exit with a clear message if the checkpoint path is missing (not a placeholder)."""
    if path is None:
        return
    if not os.path.isfile(path):
        parser.error(
            f'SAM checkpoint file not found: {path}\n'
            'Download a .pth from '
            'https://github.com/facebookresearch/segment-anything#model-checkpoints '
            'and pass the real path, e.g.\n'
            '  --sam_checkpoint /path/where/you/saved/sam_vit_h_4b8939.pth\n'
            '(Replace /path/to/sam_vit_h.pth in examples with your actual file.)'
        )


def _ensure_sam2_checkpoint(path, parser):
    if path is None:
        return
    if not os.path.isfile(path):
        parser.error(
            f'SAM2 checkpoint file not found: {path}\n'
            'Download a .pt from https://github.com/facebookresearch/sam2 '
            '(see README / checkpoints) and pass e.g.\n'
            '  --sam2_checkpoint /path/to/sam2.1_hiera_large.pt\n'
            'Match --sam2_config to the checkpoint (default is SAM 2.1 Hiera Large).'
        )


def main():
    parser = argparse.ArgumentParser(
        description='Generate foreground masks for multi-view reconstruction',
    )
    parser.add_argument('data', help='Root data path (with images/<cam>/)')
    parser.add_argument('--frame', type=int, default=0, help='Target frame index')
    parser.add_argument('--ext', default='.jpg', help='Image extension')
    parser.add_argument('--output', default=None,
                        help='Output mask directory (default: <data>/masks)')
    parser.add_argument('--mode', choices=['bg_sub', 'sam', 'hybrid'], default='hybrid',
                        help='Mask generation mode (default: hybrid = bg_sub bbox + SAM)')

    bg_group = parser.add_argument_group('background subtraction')
    bg_group.add_argument('--bg_frame', type=int, default=0,
                          help='Background-only frame index (default: 0; used by bg_sub / hybrid)')
    bg_group.add_argument('--bg_data', default=None,
                          help='Separate data root for background frames '
                               '(if background is in a different directory)')
    bg_group.add_argument('--threshold', type=float, default=30.0,
                          help='Pixel difference threshold (0-255)')
    bg_group.add_argument('--morph_kernel', type=int, default=7,
                          help='Morphology kernel size')
    bg_group.add_argument('--no_center_only', action='store_true',
                          help='Disable center-only blob filtering on the '
                               'background-subtraction mask (bg_sub and hybrid; '
                               'hybrid still uses this mask for bbox + optional IoU prior)')
    bg_group.add_argument('--max_area_ratio', type=float, default=0.15,
                          help='Max blob area as fraction of image (blobs '
                               'larger than this are discarded, default 0.15)')
    bg_group.add_argument('--dilate', type=int, default=0,
                          help='Dilate final mask by N pixels (adds margin around object)')
    bg_group.add_argument('--erode', type=int, default=0,
                          help='Erode final mask by N pixels (shrinks mask; reduces '
                               'included background/table halo)')

    sam_group = parser.add_argument_group('SAM / hybrid')
    sam_group.add_argument('--sam_backend', choices=['sam1', 'sam2'], default='sam1',
                           help='segment_anything (sam1) vs SAM 2 (sam2); sam2 needs '
                                'Python>=3.10 and torch>=2.5.1 per upstream')
    sam_group.add_argument(
        '--sam_checkpoint',
        default=DEFAULT_SAM_CHECKPOINT,
        help=f'SAM1 .pth checkpoint (default: {DEFAULT_SAM_CHECKPOINT})',
    )
    sam_group.add_argument('--sam_model_type', default='vit_h',
                           help='SAM1 model type (vit_h, vit_l, vit_b)')
    sam_group.add_argument('--sam2_checkpoint', default=None,
                           help='SAM2 .pt checkpoint (sam_backend sam2, modes sam/hybrid)')
    sam_group.add_argument('--sam2_config', default=None,
                           help='SAM2 Hydra config name (default: configs/sam2.1/'
                                'sam2.1_hiera_l.yaml — pair with matching .pt)')
    sam_group.add_argument('--hybrid_combine', choices=['sam', 'intersect'],
                           default='sam',
                           help='hybrid only: use SAM mask only, or AND with bg_sub')
    sam_group.add_argument('--sam_box_pad_ratio', type=float, default=0.12,
                           help='hybrid only: expand bg_sub bbox by this fraction '
                                'before SAM box prompt (each side)')
    sam_group.add_argument('--hybrid_sam_space', choices=['crop', 'full'],
                           default='crop',
                           help='hybrid only: run SAM on cropped ROI around bg_sub '
                                'bbox (crop) or on the full frame (full)')
    sam_group.add_argument('--no_post_sam_center_only', action='store_true',
                           help='hybrid only: skip post-SAM single-blob refinement '
                                '(refine_hybrid_mask_single_blob; on by default)')

    parser.add_argument('--no_vis', action='store_true',
                        help='Do not save per-camera overlay JPEGs to <data>/mask_vis/')
    parser.add_argument('--no_save_fg_images', action='store_true',
                        help='Do not save foreground-only RGB to <data>/foreground_images/')

    args = parser.parse_args()

    center_only = not args.no_center_only
    post_sam_center_only = not args.no_post_sam_center_only
    use_vis = not args.no_vis
    save_fg = not args.no_save_fg_images

    output_dir = args.output or join(args.data, 'masks')
    vis_dir = join(args.data, 'mask_vis') if use_vis else None
    fg_out_dir = join(args.data, 'foreground_images') if save_fg else None
    cam_names = get_cam_names(args.data)
    print(f'[generate_masks] Cameras: {cam_names}')
    print(f'[generate_masks] Mode: {args.mode}, frame: {args.frame}, '
          f'sam_backend: {args.sam_backend}'
          + (f', hybrid_sam_space: {args.hybrid_sam_space}' if args.mode == 'hybrid' else '')
          + (f', center_only: {center_only}' if args.mode in ('bg_sub', 'hybrid') else '')
          + (f', post_sam_center_only: {post_sam_center_only}' if args.mode == 'hybrid' else ''))
    print(f'[generate_masks] mask_vis: {use_vis}, foreground_images: {save_fg}')
    print(f'[generate_masks] Output: {output_dir}')
    if vis_dir:
        print(f'[generate_masks] Vis overlays: {vis_dir}')
    if fg_out_dir:
        print(f'[generate_masks] Foreground images: {fg_out_dir}')

    if args.mode == 'bg_sub':
        background_subtraction(
            args.data, cam_names, args.frame, args.bg_frame, args.ext,
            args.threshold, args.morph_kernel, output_dir, vis_dir,
            bg_data=args.bg_data, center_only=center_only,
            dilate=args.dilate, erode=args.erode,
            max_area_ratio=args.max_area_ratio,
            save_fg_dir=fg_out_dir,
        )
    elif args.mode == 'sam':
        if args.sam_backend == 'sam1':
            if args.sam_checkpoint is None:
                parser.error('--sam_checkpoint is required for sam mode with sam_backend sam1')
            _ensure_sam_checkpoint(args.sam_checkpoint, parser)
            sam_segmentation(
                args.data, cam_names, args.frame, args.ext, output_dir,
                args.sam_checkpoint, args.sam_model_type, vis_dir,
                save_fg_dir=fg_out_dir,
            )
        else:
            if args.sam2_checkpoint is None:
                parser.error('--sam2_checkpoint is required for sam mode with sam_backend sam2')
            _ensure_sam2_checkpoint(args.sam2_checkpoint, parser)
            sam2_segmentation(
                args.data, cam_names, args.frame, args.ext, output_dir,
                args.sam2_checkpoint, args.sam2_config, vis_dir,
                save_fg_dir=fg_out_dir,
            )
    elif args.mode == 'hybrid':
        if args.sam_backend == 'sam1':
            if args.sam_checkpoint is None:
                parser.error('--sam_checkpoint is required for hybrid mode with sam_backend sam1')
            _ensure_sam_checkpoint(args.sam_checkpoint, parser)
        else:
            if args.sam2_checkpoint is None:
                parser.error('--sam2_checkpoint is required for hybrid mode with sam_backend sam2')
            _ensure_sam2_checkpoint(args.sam2_checkpoint, parser)
        hybrid_bg_sam(
            args.data, cam_names, args.frame, args.bg_frame, args.ext,
            args.threshold, args.morph_kernel, output_dir, vis_dir,
            bg_data=args.bg_data, center_only=center_only,
            dilate=args.dilate, erode=args.erode,
            max_area_ratio=args.max_area_ratio,
            save_fg_dir=fg_out_dir,
            sam_checkpoint=args.sam_checkpoint,
            sam_model_type=args.sam_model_type,
            hybrid_combine=args.hybrid_combine,
            sam_box_pad_ratio=args.sam_box_pad_ratio,
            sam_backend=args.sam_backend,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            hybrid_sam_space=args.hybrid_sam_space,
            post_sam_center_only=post_sam_center_only,
        )

    print('[generate_masks] Done.')


if __name__ == '__main__':
    main()
