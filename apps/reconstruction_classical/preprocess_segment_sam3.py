"""Run SAM 3 over every camera view to produce binary foreground masks.

Usage
-----
    python -m apps.reconstruction_classical.preprocess_segment_sam3 \
        /Users/yubo/data/cow_1/10465 \
        --prompt "cattle or cow" \
        --frame 0 \
        --score_thr 0.5

Outputs
-------
    <data_root>/masks/<cam>/<frame:06d>.png  (uint8, 255 = foreground)
    <data_root>/masks/<cam>/<frame:06d>_overlay.jpg  (visual sanity check)

Notes
-----
* SAM 3 requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+. It does NOT run on
  macOS / MPS. Run this on the cluster (or any CUDA box) before running
  ``run_stage_a.py`` if you want masked sparse reconstruction.
* The ``masks/`` directory is consumed by ``stage_a_classical/sparse.py`` (and
  later by MVS) only if the mask file for a given (cam, frame) actually
  exists; otherwise that step falls back to the full image.
* Multiple detected instances for the prompt are unioned into one binary mask.
* Optional erosion (``--erode_px``) is applied before saving so silhouette
  edges (where SAM 3 is least reliable) are not used downstream.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


def _list_camera_dirs(images_root: Path) -> list[Path]:
    return sorted(p for p in images_root.iterdir() if p.is_dir())


def _pick_frame(cam_dir: Path, frame: int) -> Optional[Path]:
    cand = cam_dir / f"{frame:06d}.jpg"
    if cand.exists():
        return cand
    files = sorted(cam_dir.glob("*.jpg"))
    if not files:
        return None
    return files[frame] if frame < len(files) else files[0]


def _to_binary_union(masks: "np.ndarray | list", H: int, W: int,
                     scores: Optional[np.ndarray] = None,
                     score_thr: float = 0.5) -> np.ndarray:
    """Union of all per-instance binary masks above ``score_thr``.

    SAM 3 returns ``state['masks']`` of shape (N, 1, H, W) (bool/0-1).
    """
    if hasattr(masks, "detach"):
        masks = masks.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    if scores is not None and hasattr(scores, "detach"):
        scores = scores.detach().float().cpu().numpy()
    if masks.ndim == 4:                      # (N,1,H,W)
        masks = masks[:, 0]
    if masks.size == 0:
        return np.zeros((H, W), dtype=np.uint8)
    keep = np.ones(masks.shape[0], dtype=bool)
    if scores is not None:
        keep = scores >= score_thr
    if not keep.any():
        return np.zeros((H, W), dtype=np.uint8)
    union = masks[keep].any(axis=0).astype(np.uint8) * 255
    return union


def _save_overlay(image_bgr: np.ndarray, mask_u8: np.ndarray,
                  out_path: Path) -> None:
    overlay = image_bgr.copy()
    color = np.zeros_like(image_bgr)
    color[..., 1] = 255                       # green
    alpha = (mask_u8 > 0).astype(np.float32)[..., None] * 0.4
    overlay = (overlay * (1 - alpha) + color * alpha).astype(np.uint8)
    cv2.imwrite(str(out_path), overlay,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def main() -> None:
    p = argparse.ArgumentParser(description="SAM 3 foreground masks")
    p.add_argument("data_root", type=str,
                   help="root containing images/<cam>/<frame>.jpg")
    p.add_argument("--prompt", type=str, action="append", default=None,
                   help="text concept to segment. Pass multiple times to "
                        "ensemble prompts; per-prompt masks are unioned. "
                        "Default: ['cow', 'cow tail'] — the tail prompt is "
                        "needed because SAM 3 text queries routinely miss "
                        "thin low-contrast appendages.")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--score_thr", type=float, default=0.3,
                   help="discard SAM 3 instances with score below this. "
                        "Low default (0.3) because thin parts often score "
                        "below 0.5 even when correct, and a loose mask is "
                        "preferred for reconstruction.")
    p.add_argument("--erode_px", type=int, default=0,
                   help="erode mask by N px (tight silhouette; default off).")
    p.add_argument("--dilate_px", type=int, default=8,
                   help="dilate mask by N px after any erode (loose silhouette; "
                        "preferred for reconstruction — losing a foreground "
                        "pixel costs a 3D point, gaining a background one is "
                        "filtered by RANSAC/MVS).")
    p.add_argument("--max_long_side", type=int, default=2048,
                   help="downsize image before inference if its long side "
                        "exceeds this. The mask is then upsampled to native "
                        "resolution. Set 0 to disable.")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda | cpu (cpu is intended for debugging only)")
    p.add_argument("--no_overlay", action="store_true",
                   help="skip writing the green-overlay JPEG previews")
    args = p.parse_args()
    prompts = args.prompt if args.prompt else ["cow", "cow tail"]

    data_root = Path(args.data_root)
    images_root = data_root / "images"
    if not images_root.is_dir():
        raise SystemExit(f"no images/ directory under {data_root}")
    masks_root = data_root / "masks"
    masks_root.mkdir(exist_ok=True)

    # ------- Lazy SAM 3 import so the script is at least importable on Mac --
    try:
        import torch
        from PIL import Image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        raise SystemExit(
            f"SAM 3 import failed: {e}\n"
            "Install on a CUDA host per https://github.com/facebookresearch/sam3 :\n"
            "  conda create -n sam3 python=3.12 && conda activate sam3\n"
            "  pip install torch==2.10.0 torchvision \\\n"
            "      --index-url https://download.pytorch.org/whl/cu128\n"
            "  pip install -e /path/to/sam3\n"
            "Also run `hf auth login` and accept the SAM 3 HF gated repo."
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[sam3] CUDA not available; falling back to CPU "
              "(this will be very slow)")
        device = "cpu"

    print(f"[sam3] building model on {device} ...")
    model = build_sam3_image_model().to(device).eval()
    processor = Sam3Processor(model)

    # SAM 3 ships fp32 weights but its forward pass expects bf16 activations;
    # every example notebook enters a global autocast before calling set_image.
    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    cam_dirs = _list_camera_dirs(images_root)
    print(f"[sam3] found {len(cam_dirs)} cameras under {images_root}")
    print(f"[sam3] prompts: {prompts}")

    for cam_dir in cam_dirs:
        cam = cam_dir.name
        img_path = _pick_frame(cam_dir, args.frame)
        if img_path is None:
            print(f"[sam3] {cam}: no image, skipping")
            continue

        bgr_full = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr_full is None:
            print(f"[sam3] {cam}: failed to decode {img_path}, skipping")
            continue
        H_full, W_full = bgr_full.shape[:2]

        # --- optional pre-resize for inference -----------------------------
        if args.max_long_side > 0 and max(H_full, W_full) > args.max_long_side:
            s = args.max_long_side / max(H_full, W_full)
            W_inf = max(1, int(round(W_full * s)))
            H_inf = max(1, int(round(H_full * s)))
            bgr_inf = cv2.resize(bgr_full, (W_inf, H_inf),
                                 interpolation=cv2.INTER_AREA)
        else:
            bgr_inf = bgr_full
            H_inf, W_inf = H_full, W_full

        rgb_inf = cv2.cvtColor(bgr_inf, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb_inf)

        # --- run SAM 3 (one call per prompt, then union) -------------------
        state = processor.set_image(pil)
        union_inf = np.zeros((H_inf, W_inf), dtype=np.uint8)
        n_inst = 0
        for prompt in prompts:
            out = processor.set_text_prompt(state=state, prompt=prompt)
            masks = out.get("masks")
            scores = out.get("scores")
            n_inst += 0 if masks is None else int(getattr(masks, "shape", [0])[0])
            part = _to_binary_union(masks, H_inf, W_inf,
                                    scores=scores,
                                    score_thr=args.score_thr)
            union_inf = np.maximum(union_inf, part)

        # --- upscale back to native resolution -----------------------------
        if (H_inf, W_inf) != (H_full, W_full):
            mask_full = cv2.resize(union_inf, (W_full, H_full),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            mask_full = union_inf

        # --- erode silhouette ----------------------------------------------
        if args.erode_px > 0 and mask_full.any():
            k = max(1, int(args.erode_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)
            mask_full = cv2.erode(mask_full, kernel, iterations=1)

        # --- dilate silhouette ---------------------------------------------
        if args.dilate_px > 0 and mask_full.any():
            k = max(1, int(args.dilate_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)
            mask_full = cv2.dilate(mask_full, kernel, iterations=1)

        # --- write ---------------------------------------------------------
        out_dir = masks_root / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{args.frame:06d}.png"
        cv2.imwrite(str(out_png), mask_full)
        coverage = float((mask_full > 0).mean())
        print(f"[sam3] {cam}: {n_inst} instance(s) "
              f"-> coverage {coverage * 100:5.1f}% -> {out_png}")

        if not args.no_overlay:
            overlay_path = out_dir / f"{args.frame:06d}_overlay.jpg"
            _save_overlay(bgr_full, mask_full, overlay_path)

    print(f"[sam3] done. Masks under {masks_root}")


if __name__ == "__main__":
    main()
