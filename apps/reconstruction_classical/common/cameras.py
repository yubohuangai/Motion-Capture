"""Camera model and geometric primitives for optimization-based reconstruction.

Intrinsics/extrinsics follow OpenCV convention:
    x_cam = R @ X_world + T
    x_img = K @ x_cam  (then divide by z)
so R, T are world -> camera. The camera center in world frame is C = -R^T @ T.

Calibration is read from the OpenCV YAML files produced elsewhere in this
repository (intri.yml / extri.yml). Distortion coefficients may be all zero
(already-rectified inputs) or a standard OpenCV (k1,k2,p1,p2,k3) vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Camera:
    """A single calibrated pinhole camera with optional distortion."""

    name: str
    K: np.ndarray              # (3,3) intrinsic
    dist: np.ndarray           # (5,) distortion (k1,k2,p1,p2,k3); zeros if none
    R: np.ndarray              # (3,3) rotation, world->camera
    T: np.ndarray              # (3,1) translation, world->camera
    width: int                 # image width in pixels
    height: int                # image height in pixels

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        if self.dist.size == 4:
            self.dist = np.concatenate([self.dist, [0.0]])
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.T = np.asarray(self.T, dtype=np.float64).reshape(3, 1)

    # --- derived quantities --------------------------------------------------

    @property
    def Rt(self) -> np.ndarray:
        """3x4 [R|T] world-to-camera matrix."""
        return np.hstack([self.R, self.T])

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix K @ [R|T]."""
        return self.K @ self.Rt

    @property
    def center(self) -> np.ndarray:
        """3-vector: camera center in the world frame."""
        return (-self.R.T @ self.T).reshape(3)

    @property
    def forward(self) -> np.ndarray:
        """3-vector: camera's +Z (viewing) axis in the world frame."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    # --- projection ---------------------------------------------------------

    def project(self, pts_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project world points to pixel coords (no distortion).

        Parameters
        ----------
        pts_world : (N,3) or (3,N) world-space points.

        Returns
        -------
        uv : (N,2) pixel coordinates.
        z  : (N,)  camera-space depths (can be negative if behind).
        """
        X = np.asarray(pts_world, dtype=np.float64)
        if X.ndim == 1:
            X = X[None]
        if X.shape[0] == 3 and X.shape[1] != 3:
            X = X.T
        X_cam = X @ self.R.T + self.T.reshape(1, 3)           # (N,3)
        z = X_cam[:, 2].copy()
        z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
        uv_h = X_cam @ self.K.T                                # (N,3)
        uv = uv_h[:, :2] / z_safe[:, None]
        return uv, z

    def project_distorted(self, pts_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project with the stored distortion coefficients applied."""
        X = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(self.R)
        uv, _ = cv2.projectPoints(X[:, None, :], rvec, self.T, self.K, self.dist)
        _, z = self.project(X)
        return uv.reshape(-1, 2), z

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Back-project pixel coords at given camera-frame depths to world.

        Assumes undistorted pixels; call :meth:`undistort_points` first if
        the inputs come from a distorted image.
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        depth = np.asarray(depth, dtype=np.float64).reshape(-1)
        ones = np.ones((uv.shape[0], 1))
        uvh = np.concatenate([uv, ones], axis=1)              # (N,3)
        rays_cam = uvh @ np.linalg.inv(self.K).T               # (N,3), unit-depth dir in cam frame
        X_cam = rays_cam * depth[:, None]
        X_world = (X_cam - self.T.reshape(1, 3)) @ self.R      # world = R^T (X_cam - T)
        return X_world

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        """Undistort pixel coordinates (uses stored K, dist)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if np.allclose(self.dist, 0.0):
            return uv.reshape(-1, 2)
        und = cv2.undistortPoints(uv, self.K, self.dist, P=self.K)
        return und.reshape(-1, 2)

    # --- pairwise geometry --------------------------------------------------

    def relative_to(self, other: "Camera") -> Tuple[np.ndarray, np.ndarray]:
        """Return (R_rel, T_rel) such that x_self = R_rel @ x_other + T_rel."""
        R_rel = self.R @ other.R.T
        T_rel = self.T - R_rel @ other.T
        return R_rel, T_rel

    def fundamental_to(self, other: "Camera") -> np.ndarray:
        """Fundamental matrix F with x_other^T F x_self = 0 (both in pixels).

        Useful for epipolar-guided filtering of putative matches.
        """
        R_rel, T_rel = other.relative_to(self)     # other_coords = R_rel @ self_coords + T_rel
        Tx = np.array([[0, -T_rel[2, 0], T_rel[1, 0]],
                       [T_rel[2, 0], 0, -T_rel[0, 0]],
                       [-T_rel[1, 0], T_rel[0, 0], 0]])
        E = Tx @ R_rel
        F = np.linalg.inv(other.K).T @ E @ np.linalg.inv(self.K)
        return F

    def plane_induced_homography(self, other: "Camera", depth: float,
                                 normal: Optional[np.ndarray] = None) -> np.ndarray:
        """Homography mapping ``other`` image pixels onto ``self`` image pixels
        for a fronto-parallel plane at distance ``depth`` from ``self``.

        The plane passes through the 3D point at depth ``depth`` along the
        +Z axis of ``self`` and is normal to that axis (unless ``normal``
        overrides it, given in the world frame as a unit vector).

        Returns
        -------
        H : (3,3) so that p_self ~ H @ p_other (both in homogeneous pixels).
        """
        if normal is None:
            # plane normal in self's camera frame is (0,0,1); in world it is R_self^T @ (0,0,1)
            n_cam_self = np.array([0.0, 0.0, 1.0])
        else:
            n_cam_self = self.R @ np.asarray(normal, dtype=np.float64).reshape(3)
            n_cam_self = n_cam_self / (np.linalg.norm(n_cam_self) + 1e-12)
        # Express relative pose: x_self = R_rel @ x_other + T_rel
        R_rel, T_rel = self.relative_to(other)
        # Classic plane-induced homography (see Hartley & Zisserman Eq. 13.2)
        d = float(depth)
        H_cam = R_rel + (T_rel @ n_cam_self.reshape(1, 3)) / d
        H = self.K @ H_cam @ np.linalg.inv(other.K)
        return H


# ---------------------------------------------------------------------------
# Reading/writing from intri.yml + extri.yml via OpenCV's FileStorage
# ---------------------------------------------------------------------------


def _read_mat(fs: cv2.FileStorage, key: str) -> Optional[np.ndarray]:
    node = fs.getNode(key)
    if node.empty():
        return None
    return node.mat()


def _read_int(fs: cv2.FileStorage, key: str) -> Optional[int]:
    node = fs.getNode(key)
    if node.empty():
        return None
    return int(node.real())


def _read_name_list(fs: cv2.FileStorage, key: str = "names") -> List[str]:
    node = fs.getNode(key)
    out: List[str] = []
    for i in range(node.size()):
        s = node.at(i).string()
        if not s:
            s = str(int(node.at(i).real()))
        out.append(s)
    return out


def load_cameras(intri_path: str | Path,
                 extri_path: str | Path,
                 image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
                 ) -> Dict[str, Camera]:
    """Load all cameras from the two YAML files.

    Parameters
    ----------
    image_sizes : optional dict mapping camera name -> (H, W). If not provided,
        values of H_<cam>, W_<cam> in intri.yml are used, or (-1,-1) if missing.
    """
    intri_path = str(intri_path); extri_path = str(extri_path)
    assert Path(intri_path).exists(), intri_path
    assert Path(extri_path).exists(), extri_path
    intri = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    try:
        names = _read_name_list(intri, "names")
        cams: Dict[str, Camera] = {}
        for n in names:
            K = _read_mat(intri, f"K_{n}")
            dist = _read_mat(intri, f"dist_{n}")
            if dist is None:
                dist = _read_mat(intri, f"D_{n}")
            if dist is None:
                dist = np.zeros((1, 5))
            H = _read_int(intri, f"H_{n}")
            W = _read_int(intri, f"W_{n}")
            if image_sizes is not None and n in image_sizes:
                H, W = image_sizes[n]
            if H is None: H = -1
            if W is None: W = -1
            Rvec = _read_mat(extri, f"R_{n}")
            T = _read_mat(extri, f"T_{n}")
            if Rvec is None or T is None:
                raise ValueError(f"missing extrinsics for camera {n}")
            if Rvec.shape == (3, 1) or Rvec.shape == (1, 3):
                Rmat, _ = cv2.Rodrigues(Rvec)
            else:
                Rmat = Rvec
            cams[n] = Camera(name=n, K=K, dist=dist.reshape(-1),
                             R=Rmat, T=T.reshape(3, 1), width=int(W), height=int(H))
    finally:
        intri.release(); extri.release()
    return cams


def scale_camera(cam: Camera, sx: float, sy: Optional[float] = None) -> Camera:
    """Return a new Camera corresponding to an image resized by (sx, sy).

    Distortion coefficients are preserved (they act in normalized camera
    coordinates, which are unaffected by image scaling).
    """
    if sy is None:
        sy = sx
    K2 = cam.K.copy()
    K2[0, 0] *= sx; K2[0, 2] *= sx
    K2[1, 1] *= sy; K2[1, 2] *= sy
    new_w = max(1, int(round(cam.width * sx))) if cam.width > 0 else -1
    new_h = max(1, int(round(cam.height * sy))) if cam.height > 0 else -1
    return Camera(name=cam.name, K=K2, dist=cam.dist.copy(),
                  R=cam.R.copy(), T=cam.T.copy(), width=new_w, height=new_h)


def scene_bounds_from_cameras(cams: Sequence[Camera]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (center, extent, radius) derived from camera centers.

    This is a rough prior: the scene is assumed to lie near the centroid of
    the rig with a radius proportional to rig spread. Callers can refine it
    from a triangulated sparse cloud.
    """
    C = np.stack([c.center for c in cams], axis=0)
    mn, mx = C.min(0), C.max(0)
    center = 0.5 * (mn + mx)
    extent = (mx - mn)
    radius = 0.5 * float(np.linalg.norm(extent))
    return center, extent, radius
