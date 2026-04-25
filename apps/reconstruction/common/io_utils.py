"""Point cloud / mesh I/O and small logging helpers."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# PLY writers — kept dependency-free so users without open3d still get output
# ---------------------------------------------------------------------------


def write_ply_points(path: str | Path,
                     points: np.ndarray,
                     colors: Optional[np.ndarray] = None,
                     normals: Optional[np.ndarray] = None) -> None:
    """Write a binary-little-endian PLY point cloud.

    Parameters
    ----------
    points  : (N,3) float.
    colors  : (N,3) uint8 in [0,255], optional.
    normals : (N,3) float, optional.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = points.shape[0]
    props = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    arrays = [points]
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        assert normals.shape[0] == n
        props += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
        arrays.append(normals)
    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        assert colors.shape[0] == n
        props += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        arrays.append(colors)
    dtype = np.dtype(props)
    out = np.empty(n, dtype=dtype)
    cur = 0
    for fname, ftype in props:
        # figure out which array this belongs to
        pass
    # Simpler approach: rebuild structured array column-by-column.
    out["x"] = points[:, 0]; out["y"] = points[:, 1]; out["z"] = points[:, 2]
    if normals is not None:
        out["nx"] = normals[:, 0]; out["ny"] = normals[:, 1]; out["nz"] = normals[:, 2]
    if colors is not None:
        out["red"] = colors[:, 0]; out["green"] = colors[:, 1]; out["blue"] = colors[:, 2]

    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for fname, ftype in props:
        t = {"f4": "float", "u1": "uchar"}[ftype]
        header_lines.append(f"property {t} {fname}")
    header_lines.append("end_header\n")
    header = ("\n".join(header_lines)).encode("ascii")

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(out.tobytes())


def write_ply_mesh(path: str | Path,
                   vertices: np.ndarray,
                   faces: np.ndarray,
                   colors: Optional[np.ndarray] = None,
                   normals: Optional[np.ndarray] = None) -> None:
    """Write a binary-little-endian PLY triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    n = vertices.shape[0]; m = faces.shape[0]

    v_props = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        v_props += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        v_props += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    v_arr = np.empty(n, dtype=np.dtype(v_props))
    v_arr["x"] = vertices[:, 0]; v_arr["y"] = vertices[:, 1]; v_arr["z"] = vertices[:, 2]
    if normals is not None:
        v_arr["nx"] = normals[:, 0]; v_arr["ny"] = normals[:, 1]; v_arr["nz"] = normals[:, 2]
    if colors is not None:
        v_arr["red"] = colors[:, 0]; v_arr["green"] = colors[:, 1]; v_arr["blue"] = colors[:, 2]

    # face block: count (uchar=3) then 3 int32 indices
    f_dtype = np.dtype([("n", "u1"), ("a", "i4"), ("b", "i4"), ("c", "i4")])
    f_arr = np.empty(m, dtype=f_dtype)
    f_arr["n"] = 3
    f_arr["a"] = faces[:, 0]; f_arr["b"] = faces[:, 1]; f_arr["c"] = faces[:, 2]

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for fname, ftype in v_props:
        t = {"f4": "float", "u1": "uchar"}[ftype]
        header.append(f"property {t} {fname}")
    header += [f"element face {m}", "property list uchar int vertex_indices", "end_header\n"]
    header_bytes = ("\n".join(header)).encode("ascii")

    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(v_arr.tobytes())
        f.write(f_arr.tobytes())


# ---------------------------------------------------------------------------
# Small timing context manager
# ---------------------------------------------------------------------------


@contextmanager
def timed(label: str, stream=sys.stdout):
    t0 = time.perf_counter()
    print(f"[{label}] start", file=stream, flush=True)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[{label}] done in {dt:.2f}s", file=stream, flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p
