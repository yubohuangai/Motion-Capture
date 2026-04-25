"""Extract a triangle mesh from the trained NeuS SDF.

We sample the zero-level set with marching cubes on a regular grid inside the
unit sphere (object coordinates), then:

1. Colour the vertices by querying the ColorNetwork along the outward SDF
   gradient as the fake "view direction" (yields a view-independent albedo).
2. Transform vertices back into *world* coordinates using the dataset's
   ``object_to_world`` map so the mesh aligns with Stage A outputs.

Marching cubes comes from scikit-image if installed, otherwise from a PyMCubes
fallback. Output PLY format mirrors Stage A's ``write_ply_mesh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from ..common.io_utils import write_ply_mesh
from .dataset import NeuSDataset
from .models import ColorNetwork, SDFNetwork


# ---------------------------------------------------------------------------
# Marching cubes backend
# ---------------------------------------------------------------------------


def _marching_cubes(volume: np.ndarray, iso: float,
                    spacing: Tuple[float, float, float],
                    origin: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) in the same coordinate frame as the volume.

    ``volume[i, j, k]`` corresponds to world point ``origin + (i, j, k) * spacing``.
    """
    try:
        from skimage.measure import marching_cubes           # type: ignore
        v, f, _, _ = marching_cubes(volume, level=iso, spacing=spacing)
        v = v + np.asarray(origin, dtype=np.float32)
        return v.astype(np.float32), f.astype(np.int32)
    except Exception:
        import mcubes                                        # type: ignore  # PyMCubes
        v, f = mcubes.marching_cubes(volume, iso)
        v = v * np.asarray(spacing, dtype=np.float32) + np.asarray(origin, dtype=np.float32)
        return v.astype(np.float32), f.astype(np.int32)


# ---------------------------------------------------------------------------
# SDF grid evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def sdf_grid(sdf_net: SDFNetwork,
             resolution: int,
             bbox_min: np.ndarray,
             bbox_max: np.ndarray,
             device: torch.device,
             chunk: int = 65536) -> np.ndarray:
    """Evaluate the SDF on a regular grid of shape (resolution,)*3."""
    xs = torch.linspace(bbox_min[0], bbox_max[0], resolution, device=device)
    ys = torch.linspace(bbox_min[1], bbox_max[1], resolution, device=device)
    zs = torch.linspace(bbox_min[2], bbox_max[2], resolution, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    out = torch.empty(pts.shape[0], device=device)
    for s in range(0, pts.shape[0], chunk):
        out[s:s + chunk] = sdf_net.sdf(pts[s:s + chunk]).squeeze(-1)
    vol = out.reshape(resolution, resolution, resolution).cpu().numpy()
    return vol


# ---------------------------------------------------------------------------
# Vertex colouring
# ---------------------------------------------------------------------------


@torch.no_grad()
def color_vertices(verts_obj: np.ndarray,
                   sdf_net: SDFNetwork,
                   color_net: ColorNetwork,
                   device: torch.device,
                   chunk: int = 32768) -> np.ndarray:
    """Colour vertices by evaluating the radiance field along the SDF normal.

    View direction is taken as -normal (camera looking straight at the surface)
    which gives a roughly Lambertian, view-independent tint.
    """
    v = torch.from_numpy(verts_obj.astype(np.float32)).to(device)
    out = np.empty((v.shape[0], 3), dtype=np.float32)
    for s in range(0, v.shape[0], chunk):
        sub = v[s:s + chunk].clone().requires_grad_(True)
        with torch.enable_grad():
            sdf_feat = sdf_net(sub)
            sdf = sdf_feat[:, :1] / sdf_net.scale
            feat = sdf_feat[:, 1:]
            g = torch.autograd.grad(sdf.sum(), sub, create_graph=False)[0]
        n = g / (g.norm(dim=-1, keepdim=True) + 1e-6)
        view = -n
        rgb = color_net(sub.detach(), n.detach(), view.detach(), feat.detach())
        out[s:s + chunk] = rgb.detach().cpu().numpy()
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_mesh(sdf_net: SDFNetwork,
                 color_net: ColorNetwork,
                 dataset: NeuSDataset,
                 out_path: str | Path,
                 resolution: int = 256,
                 bbox_radius: float = 1.0,
                 iso: float = 0.0,
                 device: Optional[torch.device] = None,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Run marching cubes + vertex colouring and write a PLY.

    Returns (vertices_world, faces).
    """
    device = device or next(sdf_net.parameters()).device
    bbox_min = np.array([-bbox_radius] * 3, dtype=np.float32)
    bbox_max = np.array([+bbox_radius] * 3, dtype=np.float32)

    volume = sdf_grid(sdf_net, resolution, bbox_min, bbox_max, device)
    spacing = tuple((bbox_max - bbox_min) / (resolution - 1))
    origin = tuple(bbox_min)
    verts_obj, faces = _marching_cubes(volume, iso=iso, spacing=spacing, origin=origin)

    if verts_obj.size == 0:
        print("[extract_mesh] marching cubes produced no vertices (check training / iso level)")
        # still write an empty placeholder so downstream paths don't break
        write_ply_mesh(out_path, verts_obj.reshape(-1, 3), faces.reshape(-1, 3))
        return verts_obj, faces

    colors = color_vertices(verts_obj, sdf_net, color_net, device)
    verts_world = dataset.object_to_world(verts_obj).astype(np.float32)
    write_ply_mesh(out_path, verts_world, faces, colors=colors)
    print(f"[extract_mesh] wrote {verts_world.shape[0]} verts / {faces.shape[0]} faces -> {out_path}")
    return verts_world, faces
