"""NeuS volume renderer — turns SDF + color networks into rendered colors.

We implement Wang et al.'s "unbiased, occlusion-aware" SDF-to-alpha formula:

    alpha_i = clip((Phi_s(f_i) - Phi_s(f_{i+1})) / Phi_s(f_i), 0, 1)

where ``f_i`` is the SDF at sample i and ``Phi_s`` is the sigmoid with
learnable scale. Colors are composed via the usual back-to-front alpha
compositing weights ``w_i = alpha_i * prod_{j<i} (1 - alpha_j)``.

Hierarchical sampling: a coarse uniform pass along the ray produces weights
which we invert to importance-sample a finer set of points near the surface.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ray / sphere intersection (objects are normalized to the unit sphere)
# ---------------------------------------------------------------------------


def ray_sphere_intersection(rays_o: torch.Tensor,
                            rays_d: torch.Tensor,
                            radius: float = 1.0,
                            eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect rays with a sphere centred at the origin.

    Returns (near, far, hit_mask). Near/far are scalar t-values; where the
    ray misses the sphere, near=far=0 and hit_mask=False.
    """
    b = 2.0 * (rays_o * rays_d).sum(dim=-1)
    c = (rays_o * rays_o).sum(dim=-1) - radius * radius
    disc = b * b - 4.0 * c
    hit = disc > 0
    sq = torch.sqrt(torch.clamp(disc, min=eps))
    t1 = 0.5 * (-b - sq)
    t2 = 0.5 * (-b + sq)
    near = torch.clamp(t1, min=eps)
    far = torch.clamp(t2, min=near + eps)
    near = torch.where(hit, near, torch.zeros_like(near))
    far = torch.where(hit, far, torch.zeros_like(far))
    return near, far, hit


# ---------------------------------------------------------------------------
# NeuS SDF-to-alpha compositing
# ---------------------------------------------------------------------------


def sdf_to_alpha(sdf: torch.Tensor,
                 dists: torch.Tensor,
                 inv_s: torch.Tensor,
                 cos_val: torch.Tensor) -> torch.Tensor:
    """Compute NeuS alpha values along a set of rays.

    Parameters
    ----------
    sdf : (R, S) SDF at each sample.
    dists : (R, S-1) distances between consecutive samples.
    inv_s : scalar, output of :class:`SingleVariance.forward`.
    cos_val : (R, S-1) estimated cos(theta) at each interval (expected
        to be <= 0 when the ray is moving inward).

    Returns
    -------
    alpha : (R, S-1) opacity per interval.
    """
    # mid-point SDFs (avoid bias at interval edges)
    prev_sdf = sdf[:, :-1]
    next_sdf = sdf[:, 1:]
    mid_sdf = 0.5 * (prev_sdf + next_sdf)
    # Estimated SDF at interval start / end assuming linear interpolation
    estimated_prev = mid_sdf - cos_val * dists * 0.5
    estimated_next = mid_sdf + cos_val * dists * 0.5
    prev_cdf = torch.sigmoid(estimated_prev * inv_s)
    next_cdf = torch.sigmoid(estimated_next * inv_s)
    alpha = torch.clamp((prev_cdf - next_cdf) / (prev_cdf + 1e-5), 0.0, 1.0)
    return alpha


def weights_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Back-to-front compositing weights w_i = alpha_i * prod_{j<i} (1-alpha_j)."""
    ones = torch.ones_like(alpha[:, :1])
    trans = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
    return alpha * trans


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_uniform(near: torch.Tensor, far: torch.Tensor, n_samples: int,
                   perturb: bool) -> torch.Tensor:
    """Uniformly sample ``n_samples`` t-values in each [near, far] interval."""
    t = torch.linspace(0.0, 1.0, n_samples, device=near.device)
    z = near[:, None] + (far - near)[:, None] * t[None, :]
    if perturb:
        mids = 0.5 * (z[:, 1:] + z[:, :-1])
        upper = torch.cat([mids, z[:, -1:]], dim=-1)
        lower = torch.cat([z[:, :1], mids], dim=-1)
        t_rand = torch.rand_like(z)
        z = lower + (upper - lower) * t_rand
    return z


def sample_pdf(bins: torch.Tensor,
               weights: torch.Tensor,
               n_samples: int,
               perturb: bool) -> torch.Tensor:
    """Inverse-CDF sampling from piecewise-constant weights (NeRF style)."""
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
    if perturb:
        u = torch.rand(*cdf.shape[:-1], n_samples, device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
        u = u.expand(*cdf.shape[:-1], n_samples)
    u = u.contiguous()
    idx = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(idx - 1, 0, cdf.shape[-1] - 1)
    above = torch.clamp(idx, 0, cdf.shape[-1] - 1)
    inds = torch.stack([below, above], dim=-1)
    cdf_g = torch.gather(cdf[:, None].expand(-1, n_samples, -1), 2, inds)
    bins_g = torch.gather(bins[:, None].expand(-1, n_samples, -1), 2, inds)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------


class NeuSRenderer:
    """Render pixel colors (and optionally densities / normals) for batches of rays.

    The renderer holds references to the SDF / color / variance networks; calling
    ``render(rays_o, rays_d)`` returns a dict of compositied outputs.
    """

    def __init__(self,
                 sdf_net,
                 color_net,
                 variance,
                 n_samples: int = 64,
                 n_importance: int = 64,
                 up_sample_steps: int = 4,
                 perturb: bool = True,
                 bbox_radius: float = 1.0) -> None:
        self.sdf_net = sdf_net
        self.color_net = color_net
        self.variance = variance
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.bbox_radius = bbox_radius

    # ------------------------------------------------------------------

    def _upsample(self, rays_o, rays_d, z_vals, sdf, n_importance):
        """One iteration of NeuS-style importance resampling."""
        batch_size, n_samples = z_vals.shape
        prev_z = z_vals[:, :-1]; next_z = z_vals[:, 1:]
        prev_sdf = sdf[:, :-1]; next_sdf = sdf[:, 1:]
        mid_sdf = 0.5 * (prev_sdf + next_sdf)
        cos_val = (next_sdf - prev_sdf) / (next_z - prev_z + 1e-5)
        cos_val = torch.clamp(cos_val, max=0.0)
        dists = next_z - prev_z
        with torch.no_grad():
            # Use a smaller inv_s for the resampling pass so the CDF is smoother
            for_s = 32.0
            prev_cdf = torch.sigmoid((mid_sdf - cos_val * dists * 0.5) * for_s)
            next_cdf = torch.sigmoid((mid_sdf + cos_val * dists * 0.5) * for_s)
            alpha = torch.clamp((prev_cdf - next_cdf) / (prev_cdf + 1e-5), 0.0, 1.0)
            weights = weights_from_alpha(alpha)
            # bins are the S edges (z_vals); weights cover the S-1 intervals.
            new_z = sample_pdf(z_vals, weights, n_importance, perturb=self.perturb).detach()
        return new_z

    def _cat_z(self, rays_o, rays_d, z_vals, new_z):
        z_all, _ = torch.sort(torch.cat([z_vals, new_z], dim=-1), dim=-1)
        pts = rays_o[:, None] + rays_d[:, None] * z_all[..., None]
        return z_all, pts

    # ------------------------------------------------------------------

    def render(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
               background_rgb: torch.Tensor | None = None,
               compute_normals: bool = True,
               compute_eikonal: bool = True,
               ) -> Dict[str, torch.Tensor]:
        """Render a batch of rays.

        ``rays_o``, ``rays_d`` are expected to be in the *normalised* object
        frame (the caller is responsible for the world->object transform and
        for normalising ``rays_d``).
        """
        R = rays_o.shape[0]
        near, far, hit = ray_sphere_intersection(rays_o, rays_d, self.bbox_radius)
        # for rays that miss the sphere, fall back to a trivial range so the
        # code path stays fully vectorised; their color is masked to background.
        near = torch.where(hit, near, torch.full_like(near, 0.9))
        far = torch.where(hit, far, torch.full_like(far, 1.1))

        z_vals = sample_uniform(near, far, self.n_samples, perturb=self.perturb)

        # Iterative importance resampling (NeuS style)
        if self.n_importance > 0 and self.up_sample_steps > 0:
            per_step = max(self.n_importance // self.up_sample_steps, 1)
            with torch.no_grad():
                for _ in range(self.up_sample_steps):
                    pts = rays_o[:, None] + rays_d[:, None] * z_vals[..., None]
                    flat = pts.reshape(-1, 3)
                    sdf = self.sdf_net.sdf(flat).reshape(*z_vals.shape)
                    new_z = self._upsample(rays_o, rays_d, z_vals, sdf, per_step)
                    z_vals, _ = torch.sort(torch.cat([z_vals, new_z], dim=-1), dim=-1)

        S = z_vals.shape[1]
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        pts = rays_o[:, None] + rays_d[:, None] * z_vals[..., None]
        flat_pts = pts.reshape(-1, 3)

        # SDF + feature + gradient (need grad for normals and Eikonal)
        need_grad = compute_normals or compute_eikonal
        if need_grad:
            flat_pts = flat_pts.detach().clone().requires_grad_(True)
        sdf_feat = self.sdf_net(flat_pts)
        sdf = sdf_feat[:, :1] / self.sdf_net.scale
        feat = sdf_feat[:, 1:]
        if need_grad:
            d_output = torch.ones_like(sdf, requires_grad=False)
            gradients = torch.autograd.grad(
                outputs=sdf, inputs=flat_pts,
                grad_outputs=d_output, create_graph=True, retain_graph=True)[0]
        else:
            gradients = torch.zeros_like(flat_pts)

        sdf = sdf.reshape(R, S)
        feat = feat.reshape(R, S, -1)
        gradients = gradients.reshape(R, S, 3)

        # cos value at each interval (for unbiased alpha)
        # gradients here are unit-ish if Eikonal is enforced; otherwise we
        # normalize before taking the dot product to stay well-defined.
        grad_norm = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-6)
        true_cos = (rays_d[:, None] * grad_norm).sum(dim=-1)          # (R, S)
        # NeuS: iter_cos = -max(-true_cos * 0.5 + 0.5 * ... )
        # We follow the reference implementation's "iter_cos" smoothing:
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * 0.0 + F.relu(-true_cos))
        # Use interval-wise average cos
        cos_mid = 0.5 * (iter_cos[:, :-1] + iter_cos[:, 1:])
        inv_s = self.variance().clamp(max=1.0e4)
        alpha = sdf_to_alpha(sdf, dists, inv_s, cos_mid)

        # Color at midpoints
        mid_pts = 0.5 * (pts[:, :-1] + pts[:, 1:])
        mid_feat = 0.5 * (feat[:, :-1] + feat[:, 1:])
        mid_grad = 0.5 * (gradients[:, :-1] + gradients[:, 1:])
        mid_grad = mid_grad / (mid_grad.norm(dim=-1, keepdim=True) + 1e-6)
        view_dirs = rays_d[:, None].expand(R, S - 1, 3)
        color_flat = self.color_net(
            mid_pts.reshape(-1, 3),
            mid_grad.reshape(-1, 3),
            view_dirs.reshape(-1, 3),
            mid_feat.reshape(-1, mid_feat.shape[-1]),
        ).reshape(R, S - 1, 3)

        weights = weights_from_alpha(alpha)                          # (R, S-1)
        rgb = (weights[..., None] * color_flat).sum(dim=1)            # (R, 3)
        acc = weights.sum(dim=-1, keepdim=True)                       # (R, 1)
        if background_rgb is not None:
            rgb = rgb + (1.0 - acc) * background_rgb

        # Only include samples that lie inside the sphere (rays missing it
        # contribute no surface loss)
        inside = hit.float().unsqueeze(-1)
        rgb = rgb * inside + (1.0 - inside) * (background_rgb if background_rgb is not None else torch.zeros_like(rgb))

        out = {
            "rgb": rgb,
            "weights": weights,
            "z_vals": z_vals,
            "sdf": sdf,
            "alpha": alpha,
            "acc": acc,
            "hit": hit,
        }
        if compute_eikonal:
            out["gradients"] = gradients
        return out
