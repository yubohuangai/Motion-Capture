"""
Extract person masks using Segment Anything (SAM) for multiview image folders.

This script follows the same dataset layout/CLI style as extract_keypoints.py:
    {path}/images/{sub}/%06d.jpg
    {path}/{annot}/{sub}/%06d.json
and writes:
    {path}/{mask}/{sub}/%06d.png
"""
import os
from os.path import join
import argparse
from tqdm import tqdm
import numpy as np
import cv2


def load_subs(path, subs):
    if len(subs) == 0:
        subs = sorted(os.listdir(join(path, "images")))
    subs = [sub for sub in subs if os.path.isdir(join(path, "images", sub))]
    if len(subs) == 0:
        subs = [""]
    return subs


def load_annots(annotname):
    if not os.path.exists(annotname):
        return []
    import json
    with open(annotname, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("annots", [])
    if isinstance(data, list):
        return data
    return []


def clip_bbox_xyxy(bbox, width, height):
    x1, y1, x2, y2 = bbox[:4]
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def mask_from_annot(predictor, annot, image_shape, args):
    h, w = image_shape[:2]
    mask = None
    # Prefer bbox prompts for stability.
    if "bbox" in annot and len(annot["bbox"]) >= 4:
        bbox = clip_bbox_xyxy(annot["bbox"], w, h)
        if bbox is not None and (len(annot["bbox"]) < 5 or annot["bbox"][4] >= args.bbox_thres):
            masks, scores, _ = predictor.predict(
                box=bbox,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = masks[best]
    # Optional keypoint prompt fallback.
    if mask is None and args.prompt == "keypoints" and "keypoints" in annot:
        kpts = np.asarray(annot["keypoints"], dtype=np.float32)
        if kpts.ndim == 2 and kpts.shape[1] >= 3:
            valid = kpts[:, 2] > args.kpt_thres
            if np.any(valid):
                points = kpts[valid, :2]
                labels = np.ones((points.shape[0],), dtype=np.int32)
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                mask = masks[best]
    return mask


def extract_mask_one_sub(image_root, annot_root, mask_root, args):
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as e:
        raise RuntimeError(
            "segment_anything is required. Install with "
            "`pip install git+https://github.com/facebookresearch/segment-anything.git`"
        ) from e

    os.makedirs(mask_root, exist_ok=True)
    device = args.device
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    imgnames = sorted([n for n in os.listdir(image_root) if n.endswith(args.ext)])
    for imgname in tqdm(imgnames, desc=f"mask {os.path.basename(mask_root) or 'root'}"):
        base = imgname.replace(args.ext, "")
        outname = join(mask_root, base + ".png")
        if os.path.exists(outname) and not args.force:
            continue

        image = cv2.imread(join(image_root, imgname))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        annots = load_annots(join(annot_root, base + ".json"))
        mask_all = np.zeros(image.shape[:2], dtype=np.uint8)
        for annot in annots:
            pred = mask_from_annot(predictor, annot, image.shape, args)
            if pred is None:
                continue
            mask_all[pred] = 255

        if args.erosion > 0:
            kernel = np.ones((args.erosion, args.erosion), np.uint8)
            mask_all = cv2.erode(mask_all, kernel, iterations=1)
        if args.dilation > 0:
            kernel = np.ones((args.dilation, args.dilation), np.uint8)
            mask_all = cv2.dilate(mask_all, kernel, iterations=1)

        cv2.imwrite(outname, mask_all)
        if args.vis:
            vis = image.copy()
            vis[mask_all == 0] = 0
            cv2.imshow("mask", cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2)))
            cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--subs", type=str, nargs="+", default=[])
    parser.add_argument("--annot", type=str, default="annots")
    parser.add_argument("--mask", type=str, default="masks")
    parser.add_argument("--ext", type=str, default=".jpg")
    parser.add_argument("--prompt", type=str, default="keypoints", choices=["bbox", "keypoints"])
    parser.add_argument("--bbox_thres", type=float, default=0.1)
    parser.add_argument("--kpt_thres", type=float, default=0.2)
    parser.add_argument("--sam_checkpoint", type=str, default="data/models/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_model_type", type=str, default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--erosion", type=int, default=0)
    parser.add_argument("--dilation", type=int, default=0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(join(args.path, "images")) and os.path.exists(join(args.path, "videos")):
        cmd = f"python3 apps/preprocess/extract_image.py {args.path}"
        os.system(cmd)

    subs = load_subs(args.path, args.subs)
    if len(args.gpus) != 0:
        from easymocap.mytools.debug_utils import run_cmd
        nproc = len(args.gpus)
        plist = []
        for i in range(nproc):
            sublist = subs[i::nproc]
            if len(sublist) == 0:
                continue
            cmd = f"export CUDA_VISIBLE_DEVICES={args.gpus[i]} && python3 apps/preprocess/extract_mask.py {args.path}"
            cmd += f" --subs {' '.join(sublist)} --annot {args.annot} --mask {args.mask} --ext {args.ext}"
            cmd += f" --prompt {args.prompt} --bbox_thres {args.bbox_thres} --kpt_thres {args.kpt_thres}"
            cmd += f" --sam_checkpoint {args.sam_checkpoint} --sam_model_type {args.sam_model_type} --device cuda"
            cmd += f" --erosion {args.erosion} --dilation {args.dilation}"
            if args.force:
                cmd += " --force"
            if args.vis:
                cmd += " --vis"
            cmd += " &"
            print(cmd)
            p = run_cmd(cmd, bg=False)
            plist.extend(p)
        for p in plist:
            p.join()
        return

    for sub in subs:
        image_root = join(args.path, "images", sub)
        annot_root = join(args.path, args.annot, sub)
        mask_root = join(args.path, args.mask, sub)
        if not os.path.exists(image_root):
            print(f"[WARN] image root not found: {image_root}")
            continue
        if not os.path.exists(annot_root):
            print(f"[WARN] annot root not found: {annot_root}")
            continue
        extract_mask_one_sub(image_root, annot_root, mask_root, args)


if __name__ == "__main__":
    main()
