"""NeuS training loop — photometric + Eikonal + (optional) mask losses.

No data priors, no pretraining. Everything is randomly initialised (except for
the geometric-initialisation sphere) and optimised purely on the 11 observed
images of the scene.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import NeuSDataset
from .models import ColorNetwork, SDFNetwork, SingleVariance
from .renderer import NeuSRenderer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    n_iters: int = 100_000
    batch_rays: int = 512
    lr: float = 5e-4
    lr_warmup_iters: int = 5_000
    lr_final_factor: float = 0.1          # cosine decays from lr -> lr * factor
    weight_eikonal: float = 0.1
    weight_mask: float = 0.0              # only used if masks are provided
    n_samples: int = 64
    n_importance: int = 64
    up_sample_steps: int = 4
    anneal_end_iters: int = 50_000        # linear ramp of s (variance) scale
    log_every: int = 200
    val_every: int = 5_000
    ckpt_every: int = 10_000
    device: str = "cpu"

    def to_json(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_lr(iter_i: int, cfg: TrainConfig) -> float:
    """Warmup + cosine decay learning rate schedule."""
    if iter_i < cfg.lr_warmup_iters:
        return cfg.lr * (iter_i + 1) / cfg.lr_warmup_iters
    progress = (iter_i - cfg.lr_warmup_iters) / max(1, cfg.n_iters - cfg.lr_warmup_iters)
    progress = min(max(progress, 0.0), 1.0)
    factor = cfg.lr_final_factor + 0.5 * (1.0 - cfg.lr_final_factor) * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * factor


def _anneal_ratio(iter_i: int, cfg: TrainConfig) -> float:
    """0 -> 1 over the first ``anneal_end_iters`` steps (NeuS s-schedule)."""
    return min(1.0, iter_i / max(1, cfg.anneal_end_iters))


def _psnr(mse: torch.Tensor) -> float:
    eps = 1e-12
    return float(-10.0 * torch.log10(mse + eps))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(path: str | Path, iter_i: int, sdf_net, color_net, variance,
                    optimizer, cfg: TrainConfig) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iter": iter_i,
        "sdf": sdf_net.state_dict(),
        "color": color_net.state_dict(),
        "variance": variance.state_dict(),
        "optim": optimizer.state_dict(),
        "cfg": cfg.to_json(),
    }, path)


def load_checkpoint(path: str | Path, sdf_net, color_net, variance,
                    optimizer=None) -> int:
    ck = torch.load(path, map_location="cpu")
    sdf_net.load_state_dict(ck["sdf"])
    color_net.load_state_dict(ck["color"])
    variance.load_state_dict(ck["variance"])
    if optimizer is not None and "optim" in ck:
        optimizer.load_state_dict(ck["optim"])
    return int(ck.get("iter", 0))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(dataset: NeuSDataset,
          out_dir: str | Path,
          cfg: Optional[TrainConfig] = None,
          resume_from: Optional[str | Path] = None,
          ) -> Dict[str, object]:
    """Optimise the SDF/color networks against ``dataset``.

    Returns a dict with the final models and the rendered validation images.
    """
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(exist_ok=True)
    (out_dir / "val").mkdir(exist_ok=True)
    device = torch.device(cfg.device)

    sdf_net = SDFNetwork().to(device)
    color_net = ColorNetwork().to(device)
    variance = SingleVariance(init_val=0.3).to(device)
    renderer = NeuSRenderer(sdf_net, color_net, variance,
                            n_samples=cfg.n_samples,
                            n_importance=cfg.n_importance,
                            up_sample_steps=cfg.up_sample_steps,
                            perturb=True,
                            bbox_radius=1.0)

    params = [
        {"params": sdf_net.parameters()},
        {"params": color_net.parameters()},
        {"params": variance.parameters()},
    ]
    optim = torch.optim.Adam(params, lr=cfg.lr)

    start_iter = 0
    if resume_from is not None and Path(resume_from).exists():
        start_iter = load_checkpoint(resume_from, sdf_net, color_net, variance, optim) + 1

    # Save config for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.to_json(), f, indent=2)

    log_file = open(out_dir / "train.log", "a", buffering=1)
    print(f"[train] starting at iter {start_iter}, device={device}", file=log_file)

    t0 = time.perf_counter()
    running_mse = 0.0
    running_eik = 0.0
    running_cnt = 0
    for it in range(start_iter, cfg.n_iters):
        lr_now = _cosine_lr(it, cfg)
        for g in optim.param_groups:
            g["lr"] = lr_now

        rays_o, rays_d, rgb_gt = dataset.sample_rays(cfg.batch_rays)
        out = renderer.render(rays_o, rays_d,
                              background_rgb=torch.zeros(3, device=device),
                              compute_normals=True, compute_eikonal=True)
        rgb_pred = out["rgb"]
        mse = F.mse_loss(rgb_pred, rgb_gt)

        loss = mse
        if cfg.weight_eikonal > 0:
            grads = out["gradients"]
            gnorm = grads.norm(dim=-1)
            eik = ((gnorm - 1.0) ** 2).mean()
            loss = loss + cfg.weight_eikonal * eik
        else:
            eik = torch.tensor(0.0, device=device)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        running_mse += float(mse.detach())
        running_eik += float(eik.detach())
        running_cnt += 1

        if (it + 1) % cfg.log_every == 0:
            mean_mse = running_mse / running_cnt
            mean_eik = running_eik / running_cnt
            inv_s = float(variance().detach())
            dt = time.perf_counter() - t0
            msg = (f"iter {it+1:>7d}/{cfg.n_iters} "
                   f"lr={lr_now:.2e} mse={mean_mse:.4f} "
                   f"psnr={_psnr(torch.tensor(mean_mse)):.2f} "
                   f"eik={mean_eik:.4f} inv_s={inv_s:.1f} "
                   f"{dt:.1f}s")
            print(msg, file=log_file); print(msg, flush=True)
            running_mse = running_eik = 0.0; running_cnt = 0
            t0 = time.perf_counter()

        if (it + 1) % cfg.val_every == 0:
            _render_validation(renderer, dataset, out_dir / "val", it + 1, device)

        if (it + 1) % cfg.ckpt_every == 0:
            save_checkpoint(out_dir / "ckpt" / f"iter_{it+1:07d}.pt",
                            it, sdf_net, color_net, variance, optim, cfg)

    save_checkpoint(out_dir / "ckpt" / "final.pt", cfg.n_iters - 1,
                    sdf_net, color_net, variance, optim, cfg)
    log_file.close()
    return {"sdf_net": sdf_net, "color_net": color_net, "variance": variance,
            "renderer": renderer}


# ---------------------------------------------------------------------------
# Validation rendering
# ---------------------------------------------------------------------------


def _render_validation(renderer: NeuSRenderer,
                       dataset: NeuSDataset,
                       out_dir: Path,
                       iter_i: int,
                       device: torch.device,
                       chunk: int = 2048,
                       max_views: int = 3,
                       downsample: int = 4) -> None:
    """Render a handful of training views at reduced resolution for QA."""
    import cv2
    n = min(max_views, dataset.n_views)
    for vi in range(n):
        rays_o, rays_d, rgb_gt, (H, W) = dataset.rays_for_view(vi)
        # stride-sampled pixels so the image fits in one MPS/CUDA pass
        Hs, Ws = H // downsample, W // downsample
        ii = torch.arange(Hs, device=device) * downsample
        jj = torch.arange(Ws, device=device) * downsample
        iu, ju = torch.meshgrid(ii, jj, indexing="ij")
        idx = (iu * W + ju).reshape(-1)
        ro = rays_o[idx]; rd = rays_d[idx]
        rgb_out = []
        for s in range(0, ro.shape[0], chunk):
            sub = renderer.render(ro[s:s + chunk], rd[s:s + chunk],
                                  background_rgb=torch.zeros(3, device=device),
                                  compute_normals=False, compute_eikonal=False)
            rgb_out.append(sub["rgb"].detach().cpu().numpy())
        rgb = np.concatenate(rgb_out, axis=0).reshape(Hs, Ws, 3)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        path = out_dir / f"{dataset.view_name(vi)}_{iter_i:07d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
