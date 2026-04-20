"""
Visualize chessboard SfM-BA / COLMAP-BA outputs:
  - points_chess_colmap_ba.npz (or points_chess_sfm_ba.npz)
  - intri_colmap_ba.yml / extri_colmap_ba.yml

Shows:
  - point cloud
  - camera centers + frustum wireframes (default depth scales with median pairwise camera distance)
  - camera name labels below each camera
  - distance lines + labels for specified camera pairs (e.g. 01-02, 01-06)

Saves camera_info.json with centers and pairwise distances.

Applies OpenCV→Open3D coordinate transform (flip Y) so the scene displays
right-side-up (OpenCV uses Y-down, Open3D/OpenGL uses Y-up).
"""

import json
import os
from os.path import join

import cv2
import numpy as np
import open3d as o3d

try:
    from easymocap.mytools.camera_utils import read_camera as _read_camera
except Exception:
    _read_camera = None


def _read_mat(fs, key):
    node = fs.getNode(key)
    if node.empty():
        return None
    return node.mat()


def read_camera_fallback(intri_path, extri_path):
    fs_i = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    fs_e = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    if not fs_i.isOpened():
        raise FileNotFoundError(intri_path)
    if not fs_e.isOpened():
        raise FileNotFoundError(extri_path)

    names_node = fs_i.getNode("names")
    camnames = []
    for i in range(names_node.size()):
        camnames.append(names_node.at(i).string())

    cameras = {}
    for cam in camnames:
        K = _read_mat(fs_i, f"K_{cam}")
        dist = _read_mat(fs_i, f"dist_{cam}")
        rvec = _read_mat(fs_e, f"R_{cam}")
        R = _read_mat(fs_e, f"Rot_{cam}")
        T = _read_mat(fs_e, f"T_{cam}")
        if R is None and rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
        if rvec is None and R is not None:
            rvec, _ = cv2.Rodrigues(R)
        if K is None or dist is None or R is None or rvec is None or T is None:
            raise RuntimeError(f"Missing camera params for {cam}")
        cameras[cam] = {"K": K, "dist": dist, "R": R, "Rvec": rvec, "T": T}

    fs_i.release()
    fs_e.release()
    cameras["basenames"] = camnames
    return cameras


def read_camera(intri_path, extri_path):
    if _read_camera is not None:
        return _read_camera(intri_path, extri_path)
    return read_camera_fallback(intri_path, extri_path)


def resolve_path(root, path_or_name):
    if os.path.isabs(path_or_name):
        return path_or_name
    return join(root, path_or_name)


def opencv_to_opengl(pts):
    """Transform OpenCV (X right, Y down, Z forward) to Open3D/OpenGL
    (X right, Y up, Z backward). This is a 180 deg rotation about X:
    flip BOTH Y and Z. Flipping only Y would invert handedness and
    produce a mirrored scene (rotations cannot undo that)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    out = pts.copy()
    out[:, 1] = -out[:, 1]
    out[:, 2] = -out[:, 2]
    return out.squeeze()


def world_from_camera(camera, Xc):
    # Camera model: Xc = R * Xw + T  =>  Xw = R^T * (Xc - T)
    R = camera["R"]
    T = camera["T"]
    return (R.T @ (Xc.reshape(3, 1) - T)).reshape(3)


def get_camera_center(camera):
    return (-camera["R"].T @ camera["T"]).reshape(3)


def estimate_wh_from_k(camera):
    # If explicit image size is unavailable, infer rough size from principal point.
    K = camera["K"]
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    w = int(max(2 * cx, 64))
    h = int(max(2 * cy, 64))
    return w, h


def make_frustum_lines(camera, scale=0.35):
    K = camera["K"].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = estimate_wh_from_k(camera)

    # 5 points in world: camera center + 4 image-plane corners at depth=scale.
    c_world = get_camera_center(camera)
    z = float(scale)
    corners_uv = np.array(
        [
            [0.0, 0.0],
            [float(w), 0.0],
            [float(w), float(h)],
            [0.0, float(h)],
        ],
        dtype=np.float64,
    )
    corners_cam = np.zeros((4, 3), dtype=np.float64)
    corners_cam[:, 0] = (corners_uv[:, 0] - cx) / fx * z
    corners_cam[:, 1] = (corners_uv[:, 1] - cy) / fy * z
    corners_cam[:, 2] = z
    corners_world = np.stack([world_from_camera(camera, p) for p in corners_cam], axis=0)

    pts = np.vstack([c_world[None], corners_world])
    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    return pts, lines


def make_line_set(points, lines, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    colors = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (lines.shape[0], 1))
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def to_point_cloud(points, color=(0.85, 0.85, 0.85)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color, dtype=np.float64).reshape(1, 3), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def median_pairwise_camera_distance(cam_centers):
    """
    Median distance over all unordered camera pairs (robust scene scale).
    Returns None if fewer than two centers.
    """
    names = list(cam_centers.keys())
    if len(names) < 2:
        return None
    P = np.array([cam_centers[n] for n in names], dtype=np.float64)
    dists = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dists.append(float(np.linalg.norm(P[i] - P[j])))
    return float(np.median(dists))


def parse_dist_pairs(s):
    """Parse '01-02,01-06' into [('01','02'), ('01','06')]."""
    if not s:
        return []
    pairs = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pairs.append((a.strip(), b.strip()))
    return pairs


def get_default_mat(color=(0.15, 0.65, 1.0), shader="defaultUnlit"):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = shader
    mat.base_color = [color[0], color[1], color[2], 1.0]
    return mat


def run_gui_visualization(
    geometries,
    labels,
    distance_labels=None,
    bbox_expand=0.8,
    label_scale=1.2,
):
    """Run Open3D GUI with 3D labels. labels: camera (position, text). distance_labels: [(pair, dist_str)] for side panel."""
    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering

    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("Chessboard SfM-BA", 1600, 900)
    em = window.theme.font_size

    # Side panel for distances (aside the main graph)
    panel = gui.Vert(em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
    panel.add_child(gui.Label("Distances"))
    if distance_labels:
        for pair, dist_str in distance_labels:
            panel.add_child(gui.Label(f"  {pair}: {dist_str}"))
    else:
        panel.add_child(gui.Label("  (no pairs)"))

    # Scene widget
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    widget.scene.set_background([0.2, 0.2, 0.2, 1.0])
    widget.scene.scene.set_sun_light([-1, -1, -1], [1, 1, 1], 100000)
    widget.scene.scene.enable_sun_light(True)

    def _on_layout(ctx):
        rect = window.content_rect
        panel_width = max(200, int(rect.width * 0.21))
        widget.frame = gui.Rect(rect.x, rect.y, rect.width - panel_width, rect.height)
        panel.frame = gui.Rect(rect.x + rect.width - panel_width, rect.y, panel_width, rect.height)

    window.set_on_layout(_on_layout)
    window.add_child(widget)
    window.add_child(panel)

    # Add geometries
    for i, geom in enumerate(geometries):
        if isinstance(geom, o3d.geometry.PointCloud):
            mat = get_default_mat((0.8, 0.8, 0.8), "defaultUnlit")
        elif isinstance(geom, o3d.geometry.LineSet):
            mat = get_default_mat((0.15, 0.65, 1.0), "defaultUnlit")
        elif isinstance(geom, o3d.geometry.TriangleMesh):
            mat = get_default_mat((1, 1, 1), "defaultLit")
        else:
            mat = get_default_mat()
        widget.scene.add_geometry(f"geom_{i}", geom, mat)

    # Add 3D labels (camera names only; distances go in side panel)
    for pos, text in labels:
        pos = np.asarray(pos, dtype=np.float64)
        lbl = widget.add_3d_label(pos, str(text))
        lbl.color = gui.Color(1.0, 1.0, 1.0)
        lbl.scale = float(label_scale)

    # Setup camera to fit bbox
    all_pts = []
    for g in geometries:
        if hasattr(g, "points"):
            all_pts.append(np.asarray(g.points))
        elif hasattr(g, "vertices"):
            all_pts.append(np.asarray(g.vertices))
    if all_pts:
        pts = np.vstack(all_pts)
        center = pts.mean(axis=0)
        r = float(np.max(np.linalg.norm(pts - center, axis=1)) * bbox_expand) + 1e-6
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            center - r, center + r
        )
    else:
        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        center = np.zeros(3)
    widget.setup_camera(60, bbox, center)

    gui.Application.instance.run()


def main(args):
    root = args.path
    intri_path = resolve_path(root, args.intri)
    extri_path = resolve_path(root, args.extri)
    points_path = resolve_path(root, args.points)

    if not os.path.exists(points_path):
        raise FileNotFoundError(points_path)
    cameras = read_camera(intri_path, extri_path)
    camnames = cameras.pop("basenames")

    data = np.load(points_path)
    if "xyz" not in data:
        raise KeyError(f"{points_path} must contain key 'xyz'")
    xyz = data["xyz"].astype(np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz shape must be (N,3), got {xyz.shape}")

    if args.max_points > 0 and xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz = xyz[keep]

    if len(args.subs) > 0:
        camnames = [c for c in camnames if c in args.subs]

    # Compute camera centers
    cam_centers = {}
    for cam in camnames:
        c = get_camera_center(cameras[cam]).reshape(3)
        cam_centers[cam] = c.tolist()

    # Compute pairwise distances and save camera_info.json
    dist_pairs = parse_dist_pairs(args.dist_pairs)
    distances = {}
    for a, b in dist_pairs:
        if a in cam_centers and b in cam_centers:
            d = float(np.linalg.norm(np.array(cam_centers[a]) - np.array(cam_centers[b])))
            key = f"{a}-{b}"
            distances[key] = d

    camera_info = {
        "centers": cam_centers,
        "distances": distances,
    }
    info_path = resolve_path(root, args.camera_info)
    os.makedirs(os.path.dirname(info_path) or ".", exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(camera_info, f, indent=2)
    print(f"[VIS] saved camera_info: {info_path}")

    med_pair = median_pairwise_camera_distance(cam_centers)
    if med_pair is not None:
        print(f"[VIS] median pairwise camera distance: {med_pair:.3f}m (used for default symbol scale)")
    # Frustum depth / axis / labels: explicit CLI overrides, else scale with rig size
    if args.frustum_size is not None:
        frustum_depth = float(args.frustum_size)
    elif med_pair is not None and med_pair > 1e-6:
        frustum_depth = float(np.clip(med_pair * args.frustum_rel, 0.02, 2.0))
    else:
        frustum_depth = 0.35
    if args.axis_size is not None:
        axis_size = float(args.axis_size)
    elif med_pair is not None and med_pair > 1e-6:
        axis_size = float(np.clip(med_pair * args.axis_rel, 0.05, 0.6))
    else:
        axis_size = 0.4
    if args.label_offset is not None:
        label_off = float(args.label_offset)
    elif med_pair is not None and med_pair > 1e-6:
        label_off = float(np.clip(med_pair * args.label_rel, 0.025, 0.25))
    else:
        label_off = 0.08
    gui_label_scale = 1.2
    if med_pair is not None and med_pair > 1e-6:
        gui_label_scale = float(np.clip(0.35 + 0.55 * med_pair, 0.45, 2.0))

    geometries = []
    xyz_vis = opencv_to_opengl(xyz)
    geometries.append(to_point_cloud(xyz_vis, color=(0.8, 0.8, 0.8)))
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    )

    # Build geometries and labels
    label_offset = np.array([0, -label_off, 0], dtype=np.float64)  # below camera

    print(f"[VIS] points={xyz.shape[0]} cameras={len(camnames)}")
    labels = []
    for cam in camnames:
        pts, lines = make_frustum_lines(cameras[cam], scale=frustum_depth)
        pts_vis = opencv_to_opengl(pts)
        geometries.append(make_line_set(pts_vis, lines, color=(0.15, 0.65, 1.0)))
        center = np.array(cam_centers[cam], dtype=np.float64)
        label_pos = opencv_to_opengl(center + label_offset)
        labels.append((label_pos, cam))
        print(f"[VIS] {cam} center: {center}")

    # Add distance lines (labels shown in side panel)
    distance_labels = []
    for a, b in dist_pairs:
        if a in cam_centers and b in cam_centers:
            ca = np.array(cam_centers[a], dtype=np.float64)
            cb = np.array(cam_centers[b], dtype=np.float64)
            line_pts = np.stack([ca, cb], axis=0)
            line_pts_vis = opencv_to_opengl(line_pts)
            line_lines = np.array([[0, 1]], dtype=np.int32)
            geometries.append(make_line_set(line_pts_vis, line_lines, color=(1.0, 0.5, 0.0)))
            d = distances.get(f"{a}-{b}", np.linalg.norm(cb - ca))
            distance_labels.append((f"{a}-{b}", f"{d:.3f}m"))
            print(f"[VIS] {a}-{b} distance: {d:.3f}m")

    if args.legacy:
        o3d.visualization.draw_geometries(geometries)
        print("[VIS] (legacy mode: no labels; use default viz for labels)")
    else:
        run_gui_visualization(
            geometries,
            labels,
            distance_labels=distance_labels,
            label_scale=gui_label_scale,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="dataset root")
    parser.add_argument("--intri", type=str, default="intri_colmap_ba.yml")
    parser.add_argument("--extri", type=str, default="extri_colmap_ba.yml")
    parser.add_argument("--points", type=str, default="output/points_chess_colmap_ba.npz")
    parser.add_argument("--camera_info", type=str, default="output/camera_info.json")
    parser.add_argument(
        "--dist_pairs",
        type=str,
        default="01-02,01-06",
        help="camera pairs to show distance (e.g. 01-02,01-06)",
    )
    parser.add_argument(
        "--label_offset",
        type=float,
        default=None,
        help="label offset below camera (m); default scales with median camera spacing",
    )
    parser.add_argument("--subs", type=str, nargs="+", default=[], help="camera subset to visualize")
    parser.add_argument("--max_points", type=int, default=-1, help="random subsample for visualization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frustum_size",
        type=float,
        default=None,
        help="frustum depth along optical axis (m); default = median_pairwise * --frustum_rel",
    )
    parser.add_argument(
        "--frustum_rel",
        type=float,
        default=0.12,
        help="when --frustum_size omitted: depth = median pairwise cam distance * this",
    )
    parser.add_argument(
        "--axis_size",
        type=float,
        default=None,
        help="world axis frame size (m); default scales with median camera spacing",
    )
    parser.add_argument(
        "--axis_rel",
        type=float,
        default=0.065,
        help="when --axis_size omitted: axis length = median pairwise * this",
    )
    parser.add_argument(
        "--label_rel",
        type=float,
        default=0.035,
        help="when --label_offset omitted: offset = median pairwise * this",
    )
    parser.add_argument("--legacy", action="store_true", help="use draw_geometries (no labels)")
    args = parser.parse_args()
    main(args)
