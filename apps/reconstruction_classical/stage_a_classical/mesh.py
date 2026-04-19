"""Screened Poisson surface reconstruction from an oriented point cloud.

Open3D is required for this step. If the package is missing we fall back
to returning only the point cloud output; callers can still run Stage B
(NeuS) downstream and meshing can be re-done later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..common.cameras import Camera
from ..common.io_utils import write_ply_mesh, write_ply_points
from .fuse import FusedCloud


def _safe_import_o3d():
    try:
        import open3d as o3d        # type: ignore
        return o3d
    except Exception as exc:        # pragma: no cover
        print(f"[mesh] open3d unavailable ({exc}); skipping Poisson meshing.")
        return None


def to_open3d(cloud: FusedCloud):
    o3d = _safe_import_o3d()
    if o3d is None:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.points.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(cloud.normals.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cloud.colors.astype(np.float64) / 255.0)
    return pcd


def poisson_mesh(cloud: FusedCloud,
                 depth: int = 9,
                 density_percentile: float = 5.0,
                 ):
    """Run screened Poisson with density-based pruning.

    Parameters
    ----------
    depth : octree depth passed to Open3D's ``create_from_point_cloud_poisson``.
    density_percentile : vertices in the bottom percentile of point density
        are trimmed to remove the characteristic "balloon" artifact.
    """
    o3d = _safe_import_o3d()
    if o3d is None or cloud.points.shape[0] == 0:
        return None, None
    pcd = to_open3d(cloud)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False)
    densities = np.asarray(densities)
    cutoff = np.percentile(densities, density_percentile)
    keep_mask = densities >= cutoff
    mesh.remove_vertices_by_mask(~keep_mask)
    mesh.compute_vertex_normals()
    return mesh, densities


def save_mesh(mesh, path: str | Path, export_with_color: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mesh is None:
        return
    import open3d as o3d
    o3d.io.write_triangle_mesh(str(path), mesh,
                               write_ascii=False,
                               write_vertex_colors=export_with_color)


def save_fused_cloud(cloud: FusedCloud, path: str | Path) -> None:
    write_ply_points(path, cloud.points, colors=cloud.colors, normals=cloud.normals)


# ---------------------------------------------------------------------------
# Bounding-box cropping (useful for isolating the cow)
# ---------------------------------------------------------------------------


def crop_to_bbox(cloud: FusedCloud,
                 center: np.ndarray,
                 extent: np.ndarray) -> FusedCloud:
    """Axis-aligned crop: keep points within ``center +- 0.5 * extent``."""
    c = np.asarray(center, dtype=np.float32).reshape(3)
    e = np.asarray(extent, dtype=np.float32).reshape(3)
    lo, hi = c - 0.5 * e, c + 0.5 * e
    keep = np.all((cloud.points >= lo) & (cloud.points <= hi), axis=1)
    return FusedCloud(cloud.points[keep], cloud.normals[keep],
                      cloud.colors[keep], cloud.confidence[keep])
