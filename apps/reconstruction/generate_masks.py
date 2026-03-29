"""
Generate foreground masks for multi-view object reconstruction.

Two modes:
  1. **background subtraction** (default): requires a background-only frame.
     Computes per-pixel difference, thresholds, and cleans with morphology.
  2. **SAM** (Segment Anything Model): uses a bounding-box or point prompt to
     segment the foreground in each view.  Requires `segment_anything` to be
     installed.

Output layout mirrors the image directory:
    <data>/masks/<cam>/NNNNNN.png   (255 = foreground, 0 = background)

Usage:
    # Background subtraction (fast, works well in controlled studios)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --bg_frame 100 --threshold 30

    # SAM-based segmentation (needs segment_anything + model checkpoint)
    python apps/reconstruction/generate_masks.py /path/to/data \\
        --frame 0 --mode sam --sam_checkpoint /path/to/sam_vit_h.pth
"""

import argparse
import os
import sys
from glob import glob
from os.path import join

import cv2
import numpy as np

sys.path.insert(0, join(os.path.dirname(__file__), '..', '..'))


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
    """Save a side-by-side image: original | mask overlay (green tint on foreground)."""
    os.makedirs(vis_dir, exist_ok=True)
    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr)
    green[:, :, 1] = 255
    overlay[mask > 0] = cv2.addWeighted(overlay, 0.5, green, 0.5, 0)[mask > 0]

    # draw contour outline in red
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    combined = np.hstack([img_bgr, overlay])
    out_path = join(vis_dir, f'{cam}_{frame:06d}.jpg')
    cv2.imwrite(out_path, combined)
    return out_path


def _com_ref_masked(mask, roi_top_frac):
    """Center of mass using only foreground below roi_top_frac * height (table region)."""
    h, w = mask.shape[:2]
    roi = mask.copy()
    cut = int(h * roi_top_frac)
    if cut > 0:
        roi[:cut, :] = 0
    m = cv2.moments(roi, binaryImage=True)
    if m['m00'] > 1e-3:
        return m['m10'] / m['m00'], m['m01'] / m['m00']
    m2 = cv2.moments(mask, binaryImage=True)
    if m2['m00'] > 1e-3:
        return m2['m10'] / m2['m00'], m2['m01'] / m2['m00']
    return w / 2.0, h * 0.65


def _nearest_mask_point(mask, x, y):
    """Integer (x,y) on mask closest to (x,y); fallback to image center."""
    h, w = mask.shape[:2]
    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    if mask[yi, xi] > 0:
        return xi, yi
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return w // 2, h // 2
    d = (xs.astype(np.float64) - x) ** 2 + (ys.astype(np.float64) - y) ** 2
    j = int(np.argmin(d))
    return int(xs[j]), int(ys[j])


def select_object_blob(mask, erode_size=41, max_area_ratio=0.05,
                       max_aspect=3.0, roi_top_frac=0.38,
                       min_centroid_y_frac=0.36):
    """
    Isolate the object (compact blob on the table) from people/clutter.

    Uses center-of-mass of the difference mask in the *lower* image (table
    ROI) as the reference -- not fixed bottom-center -- so side cameras
    (e.g. box on the right) are not biased toward wrong blobs.  Rejects
    components whose centroid lies above min_centroid_y_frac * H to drop
    upper-field false positives (chairs, frame glare).
    """
    h, w = mask.shape[:2]
    img_area = h * w
    ref_x, ref_y = _com_ref_masked(mask, roi_top_frac)
    min_cy = h * min_centroid_y_frac

    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    eroded = cv2.erode(mask, ek, iterations=3)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded, connectivity=8,
    )
    if num_labels <= 1:
        return mask

    def pick_label(require_lower_centroid):
        best_lbl = -1
        best_d = float('inf')
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 300:
                continue
            if area > img_area * max_area_ratio:
                continue
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect > max_aspect:
                continue
            cx, cy = centroids[lbl]
            if require_lower_centroid and cy < min_cy:
                continue
            d = ((cx - ref_x) ** 2 + (cy - ref_y) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_lbl = lbl
        return best_lbl

    best_label = pick_label(require_lower_centroid=True)
    if best_label < 0:
        best_label = pick_label(require_lower_centroid=False)

    if best_label < 0:
        # Erosion removed all candidates: keep component of original mask that
        # contains the COM anchor (handles thin objects).
        sx, sy = _nearest_mask_point(mask, ref_x, ref_y)
        lbl_mask, _, stats2, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        seed_l = int(lbl_mask[sy, sx])
        if seed_l <= 0:
            return mask
        selected = (lbl_mask == seed_l).astype(np.uint8) * 255
        return cv2.bitwise_and(mask, selected)

    selected_eroded = ((labels == best_label) * 255).astype(np.uint8)
    grow_size = erode_size * 4
    grow_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (grow_size, grow_size),
    )
    region = cv2.dilate(selected_eroded, grow_k, iterations=1)
    result = cv2.bitwise_and(mask, region)

    # If we killed almost everything, the wrong blob was chosen; anchor to COM.
    if mask.sum() > 0 and result.sum() < 0.12 * mask.sum():
        sx, sy = _nearest_mask_point(mask, ref_x, ref_y)
        lbl_mask, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        seed_l = int(lbl_mask[sy, sx])
        if seed_l > 0:
            result = ((lbl_mask == seed_l).astype(np.uint8) * 255)
            result = cv2.bitwise_and(mask, result)
    return result


def background_subtraction(data_root, cam_names, frame, bg_frame, ext,
                           threshold, morph_kernel, output_dir, vis_dir=None,
                           bg_data=None, center_only=False, dilate=0,
                           roi_top_frac=0.38, min_centroid_y_frac=0.36):
    """Simple frame-difference masking."""
    bg_root = bg_data or data_root
    for cam in cam_names:
        fg_path = find_image(data_root, cam, frame, ext)
        bg_path = find_image(bg_root, cam, bg_frame, ext)

        fg = cv2.imread(fg_path).astype(np.float32)
        bg = cv2.imread(bg_path).astype(np.float32)

        diff = np.abs(fg - bg).max(axis=2)
        mask = (diff > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
            mask = filled

        if center_only:
            mask = select_object_blob(
                mask, roi_top_frac=roi_top_frac,
                min_centroid_y_frac=min_centroid_y_frac,
            )

        if dilate > 0:
            dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
            mask = cv2.dilate(mask, dk, iterations=1)

        out_dir = join(output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f'{frame:06d}.png')
        cv2.imwrite(out_path, mask)
        print(f'  {cam}: {out_path}  (fg pixels: {mask.sum() // 255})')

        if vis_dir is not None:
            img_bgr = cv2.imread(fg_path)
            vis_path = save_mask_overlay(img_bgr, mask, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def sam_segmentation(data_root, cam_names, frame, ext, output_dir,
                     sam_checkpoint, sam_model_type, vis_dir=None):
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

        if vis_dir is not None:
            vis_path = save_mask_overlay(img, combined, vis_dir, cam, frame)
            print(f'    vis: {vis_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate foreground masks for multi-view reconstruction',
    )
    parser.add_argument('data', help='Root data path (with images/<cam>/)')
    parser.add_argument('--frame', type=int, default=0, help='Target frame index')
    parser.add_argument('--ext', default='.jpg', help='Image extension')
    parser.add_argument('--output', default=None,
                        help='Output mask directory (default: <data>/masks)')
    parser.add_argument('--mode', choices=['bg_sub', 'sam'], default='bg_sub',
                        help='Mask generation mode')

    bg_group = parser.add_argument_group('background subtraction')
    bg_group.add_argument('--bg_frame', type=int, default=None,
                          help='Background-only frame index (required for bg_sub)')
    bg_group.add_argument('--bg_data', default=None,
                          help='Separate data root for background frames '
                               '(if background is in a different directory)')
    bg_group.add_argument('--threshold', type=float, default=30.0,
                          help='Pixel difference threshold (0-255)')
    bg_group.add_argument('--morph_kernel', type=int, default=7,
                          help='Morphology kernel size')
    bg_group.add_argument('--center_only', action='store_true',
                          help='Isolate object on table: COM in lower image, reject '
                               'upper-field blobs, erosion + grow-back (see --roi_top)')
    bg_group.add_argument('--roi_top', type=float, default=0.38,
                          help='With --center_only: ignore top fraction of image when '
                               'computing center-of-mass anchor (0-1)')
    bg_group.add_argument('--min_blob_y', type=float, default=0.36,
                          help='With --center_only: min normalized centroid y for '
                               'eroded blobs (0-1); rejects upper false positives')
    bg_group.add_argument('--dilate', type=int, default=0,
                          help='Dilate final mask by N pixels (adds margin around object)')

    sam_group = parser.add_argument_group('SAM segmentation')
    sam_group.add_argument('--sam_checkpoint', default=None,
                           help='Path to SAM model checkpoint')
    sam_group.add_argument('--sam_model_type', default='vit_h',
                           help='SAM model type (vit_h, vit_l, vit_b)')

    parser.add_argument('--vis', action='store_true',
                        help='Save mask overlay visualizations to mask_vis/')

    args = parser.parse_args()

    output_dir = args.output or join(args.data, 'masks')
    vis_dir = join(args.data, 'mask_vis') if args.vis else None
    cam_names = get_cam_names(args.data)
    print(f'[generate_masks] Cameras: {cam_names}')
    print(f'[generate_masks] Mode: {args.mode}, frame: {args.frame}')
    print(f'[generate_masks] Output: {output_dir}')
    if vis_dir:
        print(f'[generate_masks] Vis overlays: {vis_dir}')

    if args.mode == 'bg_sub':
        if args.bg_frame is None:
            parser.error('--bg_frame is required for background subtraction mode')
        background_subtraction(
            args.data, cam_names, args.frame, args.bg_frame, args.ext,
            args.threshold, args.morph_kernel, output_dir, vis_dir,
            bg_data=args.bg_data, center_only=args.center_only,
            dilate=args.dilate, roi_top_frac=args.roi_top,
            min_centroid_y_frac=args.min_blob_y,
        )
    elif args.mode == 'sam':
        if args.sam_checkpoint is None:
            parser.error('--sam_checkpoint is required for SAM mode')
        sam_segmentation(
            args.data, cam_names, args.frame, args.ext, output_dir,
            args.sam_checkpoint, args.sam_model_type, vis_dir,
        )

    print('[generate_masks] Done.')


if __name__ == '__main__':
    main()
