"""
Render a turntable MP4 of a triangle mesh (e.g. Poisson output mesh.ply).

On macOS, Filament OffscreenRenderer (EGL headless) is unavailable; this script
tries CPU offscreen first, then falls back to Visualizer + screen capture.

Usage:
    python apps/reconstruction/render_mesh_turntable.py \\
        /path/to/mesh.ply -o /path/to/turntable.mp4

Requires: open3d, numpy; for MP4: imageio + imageio-ffmpeg (or save PNGs + ffmpeg).
"""

from __future__ import annotations

import argparse
import os
import sys
from os.path import join

import numpy as np

# Open3D is imported in main() *after* optional env vars (macOS CPU / headless).


def _mesh_center_and_radius(mesh):
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return np.zeros(3), 1.0
    c = v.mean(axis=0)
    r = float(np.max(np.linalg.norm(v - c, axis=1)))
    return c, max(r, 1e-6)


def _intrinsic_from_vertical_fov(o3d, width, height, fov_deg_vertical):
    """Intrinsics matching ViewControl::ConvertToPinholeCameraParameters (vertical FOV, fx == fy)."""
    vfov = np.radians(float(fov_deg_vertical))
    tan_half = np.tan(vfov * 0.5)
    f = float(height) / tan_half / 2.0
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(int(width), int(height), f, f, cx, cy)
    return intrinsic


def _open3d_viewcontrol_extrinsic(eye, center, world_up=(0.0, 1.0, 0.0)):
    """4x4 extrinsic exactly as ViewControl::ConvertToPinholeCameraParameters (Open3D legacy Visualizer).

    *front_dir* is Open3D's view ``front_``: from *center* (lookat) toward *eye*, not the ray into the scene.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    wup = np.asarray(world_up, dtype=np.float64).reshape(3)
    front_dir = eye - center
    fn = np.linalg.norm(front_dir)
    if fn < 1e-12:
        front_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        front_dir = front_dir / fn
    if abs(np.dot(front_dir, wup)) > 0.99:
        wup = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # Same as ViewControl::SetProjectionParameters: right = up × front
    right_dir = np.cross(wup, front_dir)
    rn = np.linalg.norm(right_dir)
    if rn < 1e-12:
        wup = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right_dir = np.cross(wup, front_dir)
        rn = np.linalg.norm(right_dir) + 1e-12
    right_dir = right_dir / rn
    up_dir = np.cross(front_dir, right_dir)
    up_dir = up_dir / (np.linalg.norm(up_dir) + 1e-12)
    T = np.eye(4, dtype=np.float64)
    T[0, :3] = right_dir
    T[1, :3] = -up_dir
    T[2, :3] = -front_dir
    T[0, 3] = -float(np.dot(right_dir, eye))
    T[1, 3] = float(np.dot(up_dir, eye))
    T[2, 3] = float(np.dot(front_dir, eye))
    T[3, 3] = 1.0
    return T


def orbit_eye(center, radius, theta, axis="y", elevation=0.15):
    """Camera position on a circle around *center* (horizontal orbit for axis=y)."""
    c = np.asarray(center, dtype=np.float64)
    if axis == "y":
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        off = np.array([x, elevation * radius, z], dtype=np.float64)
    elif axis == "z":
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        off = np.array([x, y, elevation * radius], dtype=np.float64)
    else:
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        off = np.array([elevation * radius, y, z], dtype=np.float64)
    return c + off


def _save_video_or_pngs(rgb_frames, out_path, fps):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None

    if imageio is not None:
        try:
            imageio.mimsave(out_path, rgb_frames, fps=fps, codec="libx264", quality=8)
            print(f"[turntable] Wrote {out_path} ({len(rgb_frames)} frames, {fps} fps)")
            return
        except Exception as e:
            print(f"[turntable] imageio MP4 failed ({e}); saving PNG sequence",
                  file=sys.stderr)

    stem = os.path.splitext(out_path)[0]
    seq_dir = stem + "_frames"
    os.makedirs(seq_dir, exist_ok=True)
    import open3d as o3d
    for i, fr in enumerate(rgb_frames):
        o3d.io.write_image(join(seq_dir, f"{i:04d}.png"), o3d.geometry.Image(fr))
    print(f"[turntable] Saved {len(rgb_frames)} PNGs under {seq_dir}/")
    print("[turntable] Encode MP4:")
    print(
        f"  ffmpeg -y -framerate {fps} -i {seq_dir}/%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p {out_path}"
    )


def render_turntable_offscreen(
    mesh,
    *,
    width,
    height,
    frames,
    fov,
    axis,
    zoom,
    elevation,
    bg,
):
    import open3d as o3d

    center, r = _mesh_center_and_radius(mesh)
    cam_radius = r * zoom
    rendering = o3d.visualization.rendering
    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(bg) + [1.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    if not mesh.has_vertex_colors():
        mat.base_color = (0.85, 0.82, 0.78, 1.0)

    renderer.scene.add_geometry("mesh", mesh, mat)
    try:
        renderer.scene.scene.set_sun_light([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 100000)
        renderer.scene.scene.enable_sun_light(True)
    except Exception:
        pass

    thetas = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    rgb_frames = []
    for i, th in enumerate(thetas):
        eye = orbit_eye(center, cam_radius, th, axis=axis, elevation=elevation)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        renderer.setup_camera(fov, center, eye, up)
        img = renderer.render_to_image()
        rgb_frames.append(np.asarray(img))
        if (i + 1) % 30 == 0 or i == 0:
            print(f"[turntable] frame {i + 1}/{frames}")
    return rgb_frames


def render_turntable_visualizer(
    mesh,
    *,
    width,
    height,
    frames,
    fov,
    axis,
    zoom,
    elevation,
    bg,
):
    """Classic Visualizer + capture_screen_float_buffer (works on macOS).

    Uses pinhole extrinsics each frame so camera distance matches *orbit_eye* (same as
    offscreen). set_lookat/set_front/set_zoom alone does not use orbit radius for distance.
    """
    import open3d as o3d

    center, r = _mesh_center_and_radius(mesh)

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name="turntable", width=width, height=height, visible=False):
        raise RuntimeError("create_window failed (need a display / GUI session on this machine)")

    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color((0.82, 0.78, 0.74))
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg, dtype=np.float64)
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    cam_param = ctr.convert_to_pinhole_camera_parameters()
    cam_param.intrinsic = _intrinsic_from_vertical_fov(o3d, width, height, fov)

    cam_radius = r * zoom
    thetas = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rgb_frames = []
    for i, th in enumerate(thetas):
        eye = orbit_eye(center, cam_radius, th, axis=axis, elevation=elevation)
        cam_param.extrinsic = _open3d_viewcontrol_extrinsic(eye, center, world_up)
        # Without allow_arbitrary, ViewControl recomputes lookat/zoom from bbox and clamps zoom,
        # breaking a true orbit around *center*.
        try:
            ret = ctr.convert_from_pinhole_camera_parameters(
                cam_param, allow_arbitrary=True
            )
        except TypeError:
            ret = ctr.convert_from_pinhole_camera_parameters(cam_param)
        if ret is False:
            raise RuntimeError("convert_from_pinhole_camera_parameters failed")
        vis.poll_events()
        vis.update_renderer()
        buf = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        rgb = (np.clip(buf, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb_frames.append(rgb)
        if (i + 1) % 30 == 0 or i == 0:
            print(f"[turntable] frame {i + 1}/{frames} (Visualizer)")

    vis.destroy_window()
    return rgb_frames


def render_turntable(
    mesh_path,
    out_path,
    *,
    width=960,
    height=720,
    frames=120,
    fps=30,
    fov=50.0,
    axis="y",
    zoom=3.0,
    elevation=0.12,
    bg=(0.1, 0.1, 0.1),
    prefer_visualizer=False,
):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        print(f"[turntable] ERROR: empty mesh: {mesh_path}", file=sys.stderr)
        sys.exit(1)
    mesh.compute_vertex_normals()

    rgb_frames = None
    err_off = None

    if not prefer_visualizer:
        try:
            print("[turntable] Trying Filament OffscreenRenderer …")
            rgb_frames = render_turntable_offscreen(
                mesh,
                width=width, height=height, frames=frames, fov=fov,
                axis=axis, zoom=zoom, elevation=elevation, bg=bg,
            )
        except RuntimeError as e:
            err_off = e
            msg = str(e)
            if "EGL" in msg or "Headless" in msg or "headless" in msg.lower():
                print(f"[turntable] Offscreen not supported: {msg}", file=sys.stderr)
            else:
                print(f"[turntable] Offscreen failed: {msg}", file=sys.stderr)

    if rgb_frames is None:
        print("[turntable] Using Visualizer fallback (macOS-friendly) …")
        try:
            rgb_frames = render_turntable_visualizer(
                mesh,
                width=width, height=height, frames=frames, fov=fov,
                axis=axis, zoom=zoom, elevation=elevation, bg=bg,
            )
        except Exception as e2:
            if err_off:
                print(f"[turntable] Visualizer also failed: {e2}", file=sys.stderr)
                print(f"[turntable] Original offscreen error was: {err_off}", file=sys.stderr)
            raise

    _save_video_or_pngs(rgb_frames, out_path, fps)
    print(f"[turntable] Resolution {width}x{height}")


def main():
    p = argparse.ArgumentParser(description="Turntable MP4 from a mesh PLY/OBJ")
    p.add_argument("mesh", help="Path to mesh (.ply, .obj, ...)")
    p.add_argument("-o", "--output", default=None, help="Output .mp4 (default: <mesh>_turntable.mp4)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fov", type=float, default=50.0)
    p.add_argument("--axis", choices=("y", "z", "x"), default="y")
    p.add_argument(
        "--zoom",
        type=float,
        default=3.0,
        help="Orbit radius = mesh_radius * zoom (larger = camera farther; affects offscreen + visualizer)",
    )
    p.add_argument("--elevation", type=float, default=0.12)
    p.add_argument(
        "--visualizer",
        action="store_true",
        help="Skip OffscreenRenderer; use Visualizer only (faster to debug on Mac)",
    )
    args = p.parse_args()

    # Filament EGL headless is not on macOS; CPU path must be set before import.
    if sys.platform == "darwin":
        os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")

    try:
        import open3d as o3d  # noqa: F401
    except ImportError:
        print("open3d is required: pip install open3d", file=sys.stderr)
        sys.exit(1)

    out = args.output
    if out is None:
        out = os.path.splitext(args.mesh)[0] + "_turntable.mp4"

    render_turntable(
        args.mesh,
        out,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
        fov=args.fov,
        axis=args.axis,
        zoom=args.zoom,
        elevation=args.elevation,
        prefer_visualizer=args.visualizer,
    )


if __name__ == "__main__":
    main()
