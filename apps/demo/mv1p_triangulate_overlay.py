#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  Multi-view triangulation + reprojection visualization on same image.
  Based on EasyMocap/apps/demo/mv1p.py
  Author: Yubo Huang, adapted version 2025-10-29
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from tqdm import tqdm
import numpy as np
import cv2
from os.path import join
from easymocap.mytools import simple_recon_person, projectN3
from easymocap.smplmodel import check_keypoints
from easymocap.dataset import CONFIG, MV1PMF


# --- fallback draw_points2d if easymocap.visualize.draw is missing ---
def draw_points2d(img, keypoints, kintree=None, color=(0, 255, 0), radius=4, thickness=2):
    """Draw 2D skeleton with optional kinematic tree connections."""
    img_drawn = img.copy()
    keypoints = np.asarray(keypoints)

    # Draw limbs if kintree is provided
    if kintree is not None:
        for (i, j) in kintree:
            if keypoints[i, 2] > 0 and keypoints[j, 2] > 0:
                p1 = tuple(map(int, keypoints[i, :2]))
                p2 = tuple(map(int, keypoints[j, :2]))
                cv2.line(img_drawn, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    # Draw joints
    for kpt in keypoints:
        if kpt[2] > 0:
            p = tuple(map(int, kpt[:2]))
            cv2.circle(img_drawn, p, radius, color, -1, lineType=cv2.LINE_AA)

    return img_drawn

def compute_reprojection_error(keypoints2d, kpts_repro):
    """Compute mean per-joint reprojection error (pixels)."""
    conf = (keypoints2d[..., -1] > 0) & (kpts_repro[..., -1] > 0)
    diff = (keypoints2d[..., :2] - kpts_repro[..., :2]) * conf[..., None]
    err = np.sqrt((diff ** 2).sum(axis=-1))
    if conf.sum() == 0:
        return np.nan
    return err[conf].mean()


def draw_overlay(images, annots, kpts_repro, outdir, nf, kintree=None):
    """Draw detections (2D) and reprojections (2D) together on each camera view."""
    os.makedirs(outdir, exist_ok=True)
    keypoints2d = annots['keypoints']
    for vid, img in enumerate(images):
        img_disp = img.copy()
        kpt_det = keypoints2d[vid]
        kpt_rep = kpts_repro[vid]

        # OpenCV uses BGR (not RGB):
        # - Orange detections: (0, 128, 255)
        # - Red reprojections: (64, 64, 255)
        img_disp = draw_points2d(img_disp, kpt_det, kintree=kintree, color=(0, 128, 255))
        img_disp = draw_points2d(img_disp, kpt_rep, kintree=kintree, color=(64, 64, 255))

        out_path = join(outdir, f"{nf:06d}_{vid:02d}.jpg")
        cv2.imwrite(out_path, img_disp)


def mv1pmf_triangulate_overlay(dataset, args):
    start, end = args.start, min(args.end, len(dataset))
    MIN_CONF_THRES = args.thres2d
    dataset.no_img = False
    vis_dir = join(args.out, 'vis_overlay')
    os.makedirs(vis_dir, exist_ok=True)
    config = getattr(dataset, 'config', None)
    if isinstance(config, dict):
        kintree = config.get('kintree', None)
    else:
        kintree = getattr(config, 'kintree', None)

    print(f"[INFO] Triangulating frames {start}-{end-1}...")
    errors = []
    frame_iter = range(start, end)
    if not args.no_bar:
        frame_iter = tqdm(frame_iter, desc="Triangulate")

    for nf in frame_iter:
        images, annots = dataset[nf]

        keypoints2d = annots['keypoints']
        check_keypoints(keypoints2d, WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)

        # Triangulate to 3D
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, dataset.Pall)
        err = compute_reprojection_error(keypoints2d, kpts_repro)
        errors.append(err)
        print(f"[Frame {nf:06d}] mean reprojection error = {err:.2f}px")

        # Save 3D keypoints
        dataset.write_keypoints3d(keypoints3d, nf)

        # Draw overlay (detections + reprojections)
        draw_overlay(images, annots, kpts_repro, outdir=vis_dir, nf=nf, kintree=kintree)

    avg_error = float(np.nanmean(np.array(errors))) if len(errors) > 0 else float('nan')
    print(f"[DONE] All overlays saved to {vis_dir}")
    print(f"[DONE] Average reprojection error over sequence = {avg_error:.2f}px")


if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    parser = load_parser()
    parser.add_argument('--no_vis', action='store_true', help='disable visualization')
    parser.add_argument('--no_bar', action='store_true', help='disable tqdm progress bar')
    args = parse_parser(parser)

    print("""
  Demo: multi-view triangulation + overlay visualization
    Input : {} => {}
    Output: {}
""".format(args.path, ', '.join(args.sub), args.out))

    dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
                     config=CONFIG[args.body], kpts_type=args.body,
                     undis=args.undis, no_img=False, verbose=args.verbose)
    dataset.writer.save_origin = args.save_origin

    mv1pmf_triangulate_overlay(dataset, args)