'''
  @ Date: 2021-04-13 19:46:51
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-13 17:56:25
  @ FilePath: /EasyMocap/apps/demo/mv1p.py
'''
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

from tqdm import tqdm
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.mytools import simple_recon_person, Timer, projectN3
from easymocap.pipeline import smpl_from_keypoints3d2d
import os
from os.path import join
import numpy as np
import cv2
import json


def check_repro_error(keypoints3d, kpts_repro, keypoints2d, P, MAX_REPRO_ERROR):
    square_diff = (keypoints2d[:, :, :2] - kpts_repro[:, :, :2])**2 
    conf = keypoints3d[None, :, -1:]
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2])*conf)**2).sum(axis=-1))
    vv, jj = np.where(dist > MAX_REPRO_ERROR)
    if vv.shape[0] > 0:
        keypoints2d[vv, jj, -1] = 0.
        keypoints3d, kpts_repro = simple_recon_person(keypoints2d, P)
    mean_repro_error = (dist[conf[..., 0] > 0]).mean() if np.any(conf[..., 0] > 0) else 0
    return keypoints3d, kpts_repro, mean_repro_error


def mv1pmf_skel(dataset, check_repro=True, args=None):
    MIN_CONF_THRES = args.thres2d
    no_img = not (args.vis_det or args.vis_repro)
    dataset.no_img = no_img
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    kpts_repro = None

    repro_errors = []

    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
        if check_repro:
            keypoints3d, kpts_repro, repro_error = check_repro_error(
                keypoints3d, kpts_repro, annots['keypoints'],
                P=dataset.Pall, MAX_REPRO_ERROR=args.MAX_REPRO_ERROR
            )
            # log_time(f"[Frame {nf:04d}] mean reprojection error = {repro_error:.2f}px")
            tqdm.write(f"[Frame {nf:04d}] mean reprojection error = {repro_error:.2f}px")
            repro_errors.append(repro_error)
        else:
            repro_error = np.nan
        # keypoints3d, kpts_repro = robust_triangulate(annots['keypoints'], dataset.Pall, config=config, ret_repro=True)
        kp3ds.append(keypoints3d)
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis)

    np.save(join(args.out, "reprojection_error.npy"), np.array(repro_errors))
    log_time(f"Average reprojection error over sequence: {np.nanmean(repro_errors):.2f}px")    # smooth the skeleton
    if args.smooth3d > 0:
        kp3ds = smooth_skeleton(kp3ds, args.smooth3d)
    for nf in tqdm(range(len(kp3ds)), desc='dump'):
        dataset.write_keypoints3d(kp3ds[nf], nf+start)

def mv1pmf_smpl(dataset, args, weight_pose=None, weight_shape=None):
    dataset.skel_path = args.skel
    kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    keypoints2d, bboxes = [], []
    dataset.no_img = True
    for nf in tqdm(range(start, end), desc='loading'):
        images, annots = dataset[nf]
        keypoints2d.append(annots['keypoints'])
        bboxes.append(annots['bbox'])
    kp3ds = dataset.read_skeleton(start, end)
    keypoints2d = np.stack(keypoints2d)
    bboxes = np.stack(bboxes)
    kp3ds = check_keypoints(kp3ds, 1)
    silhouette_points = None
    need_sil = (args.shape_silhouette or args.vis_shape_silhouette
                or getattr(args, 'refine_shape', False))
    if need_sil:
        silhouette_points = load_silhouette_points(dataset, start, end, args)
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    params = smpl_from_keypoints3d2d(body_model, kp3ds, keypoints2d, bboxes, 
        dataset.Pall, config=dataset.config, args=args,
        weight_shape=weight_shape, weight_pose=weight_pose,
        silhouette_points=silhouette_points)

    # Extract pre-refinement shapes (stashed by smpl_from_keypoints3d2d) for
    # before/after visualization.
    pre_refine_shapes = params.pop('_pre_refine_shapes', None)

    # write out the results
    vis_sil = args.vis_shape_silhouette
    dataset.no_img = not (args.vis_smpl or args.vis_repro or vis_sil)
    if vis_sil and silhouette_points is None:
        print('[Shape silhouette] no silhouette points loaded, skip visualization.')
        vis_sil = False
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        param = select_nf(params, nf-start)
        dataset.write_smpl(param, nf)
        vertices = None
        if args.write_vertices or args.vis_smpl or vis_sil:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
        if args.write_smpl_full:
            param_full = param.copy()
            param_full['poses'] = body_model.full_poses(param['poses'])
            dataset.write_smpl(param_full, nf, mode='smpl_full')
        if args.write_vertices:
            write_data = [{'id': 0, 'vertices': vertices[0]}]
            dataset.write_vertices(write_data, nf)
        if args.vis_smpl:
            dataset.vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)
        if vis_sil:
            frame_pts = silhouette_points[nf-start]
            has_contour = any(
                (p.shape[0] > 0 if isinstance(p, np.ndarray) else len(p) > 0)
                for p in frame_pts)
            if has_contour:
                verts_before = None
                if pre_refine_shapes is not None:
                    param_before = param.copy()
                    param_before['shapes'] = pre_refine_shapes
                    verts_before = body_model(
                        return_verts=True, return_tensor=False,
                        **param_before)[0]
                vis_shape_silhouette_overlay(
                    dataset, images, vertices[0], frame_pts, nf, args,
                    vertices_before=verts_before,
                    faces=body_model.faces
                )
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, dataset.Pall)
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis, mode='repro_smpl')

def load_silhouette_points(dataset, start, end, args):
    mask_root = join(dataset.root, args.shape_mask_root)
    annot_root = join(dataset.root, args.annot)
    has_json_masks = os.path.exists(annot_root)
    has_image_masks = os.path.exists(mask_root)
    if not has_json_masks and not has_image_masks:
        print(f"[Shape silhouette] neither annot root ({annot_root}) nor mask root ({mask_root}) exists, skipping silhouette term.")
        return None
    rng = np.random.default_rng(0)
    silhouettes = []
    pid = getattr(args, 'pid', 0)
    if isinstance(pid, (list, tuple)):
        pid = pid[0]
    for nf in tqdm(range(start, end), desc='loading masks'):
        frame_points = []
        use_frame = ((nf - start) % max(args.shape_mask_frame_step, 1) == 0)
        for cam in dataset.cams:
            points = np.zeros((0, 2), dtype=np.float32)
            if use_frame:
                imgname = dataset.imagelist[cam][nf]
                base, ext = os.path.splitext(imgname)
                # Prefer json masks saved in annots[*].mask by extract_mask.py.
                annotname = join(annot_root, cam, base + '.json')
                if os.path.exists(annotname):
                    try:
                        with open(annotname, 'r') as f:
                            data = json.load(f)
                        annots = data.get('annots', []) if isinstance(data, dict) else data
                        if isinstance(annots, list) and len(annots) > 0:
                            target = None
                            for ann in annots:
                                if ann.get('id', 0) == pid and 'mask' in ann:
                                    target = ann
                                    break
                            if target is None:
                                for ann in annots:
                                    if 'mask' in ann:
                                        target = ann
                                        break
                            if target is not None:
                                pts = np.asarray(target.get('mask', []), dtype=np.float32)
                                if pts.ndim == 2 and pts.shape[1] >= 2 and pts.shape[0] > 0:
                                    points = pts[:, :2]
                    except Exception:
                        pass
                # Fallback to image masks for backward compatibility.
                if points.shape[0] == 0 and has_image_masks:
                    candidates = [
                        join(mask_root, cam, base + '.png'),
                        join(mask_root, cam, base + '.jpg'),
                        join(mask_root, cam, imgname),
                    ]
                    maskname = next((name for name in candidates if os.path.exists(name)), None)
                    if maskname is not None:
                        mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            ys, xs = np.where(mask > args.shape_mask_thr)
                            if xs.shape[0] > 0:
                                points = np.stack([xs, ys], axis=1).astype(np.float32)
                if points.shape[0] > args.shape_mask_max_points:
                    sel = rng.choice(points.shape[0], size=args.shape_mask_max_points, replace=False)
                    points = points[sel]
            frame_points.append(points)
        silhouettes.append(frame_points)
    return silhouettes


def _mesh_silhouette_contour(mesh_xy_all, faces, h, w):
    """Rasterise the projected mesh to a mask and return its contours."""
    if faces is not None and mesh_xy_all.shape[0] > 0:
        tris = mesh_xy_all[faces].astype(np.int32)  # (F, 3, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, tris, 255)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    if mesh_xy_all.shape[0] >= 3:
        hull = cv2.convexHull(np.round(mesh_xy_all).astype(np.int32))
        return [hull]
    return None


def _draw_silhouette_overlay(image, mesh_xy_all, mask_xy_all, label,
                             faces=None, draw_points=False,
                             mesh_sil_indices=None):
    """Draw mesh silhouette contour (red) and GT mask contour (green).

    When *faces* is provided the mesh is rasterised via its triangles so we
    can extract a true silhouette contour.  Otherwise falls back to the convex
    hull of the projected vertices.

    When draw_points=True, draws individual points instead of contours so you
    can see how many points are used. mesh_sil_indices: vertex indices for
    mesh silhouette boundary (from _silhouette_vertex_indices).
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    pt_radius = 2

    if draw_points:
        # --- mesh silhouette points (red) ---
        if mesh_sil_indices is not None and mesh_sil_indices.shape[0] > 0:
            mesh_pts = mesh_xy_all[mesh_sil_indices]
            for pt in mesh_pts:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(canvas, (x, y), pt_radius, (0, 0, 255), -1)
        # --- GT mask points (green) ---
        mask_xy = np.round(mask_xy_all).astype(np.int32)
        valid = ((mask_xy[:, 0] >= 0) & (mask_xy[:, 0] < w) &
                 (mask_xy[:, 1] >= 0) & (mask_xy[:, 1] < h))
        mask_xy = mask_xy[valid]
        for pt in mask_xy:
            cv2.circle(canvas, tuple(pt), pt_radius, (0, 255, 0), -1)
        n_mesh = mesh_sil_indices.shape[0] if mesh_sil_indices is not None else 0
        n_gt = mask_xy.shape[0]
        label = f'{label}  mesh:{n_mesh}  GT:{n_gt}'
    else:
        # --- mesh silhouette contour (red) ---
        mesh_contour = _mesh_silhouette_contour(mesh_xy_all, faces, h, w)
        if mesh_contour is not None and len(mesh_contour) > 0:
            cv2.drawContours(canvas, mesh_contour, -1, (0, 0, 255), 2)

        # --- GT mask contour (green) ---
        mask_xy = np.round(mask_xy_all).astype(np.int32)
        valid = ((mask_xy[:, 0] >= 0) & (mask_xy[:, 0] < w) &
                 (mask_xy[:, 1] >= 0) & (mask_xy[:, 1] < h))
        mask_xy = mask_xy[valid]
        if mask_xy.shape[0] > 0:
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            gt_mask[mask_xy[:, 1], mask_xy[:, 0]] = 255
            gt_mask = cv2.dilate(gt_mask, np.ones((3, 3), np.uint8), iterations=1)
            gt_contours, _ = cv2.findContours(
                gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, gt_contours, -1, (0, 255, 0), 2)

    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def vis_shape_silhouette_overlay(dataset, images, vertices, frame_points, nf, args,
                                 vertices_before=None, faces=None):
    """Visualize projected mesh silhouette contour vs GT mask contour.

    When vertices_before is provided (pre-refinement mesh), a side-by-side
    BEFORE | AFTER comparison is saved so the user can see the shape improvement.

    When args.vis_silhouette_points is True, draws points instead of contours
    and shows mesh/GT point counts in the label.
    """
    import torch
    from easymocap.pyfitting.optimize_simple import (
        _build_edge_face_adjacency,
        _silhouette_vertex_indices,
    )

    out_root = join(args.out, 'shape_silhouette')
    proj = projectN3(vertices, dataset.Pall)[..., :2]
    proj_before = (projectN3(vertices_before, dataset.Pall)[..., :2]
                   if vertices_before is not None else None)

    draw_points = getattr(args, 'vis_silhouette_points', False)
    edge_v_t = edge_f_t = faces_t = None
    if draw_points and faces is not None:
        edge_v, edge_f = _build_edge_face_adjacency(faces)
        edge_v_t = torch.tensor(edge_v, dtype=torch.long)
        edge_f_t = torch.tensor(edge_f, dtype=torch.long)
        faces_t = torch.tensor(faces, dtype=torch.long)

    def _mesh_sil_idx(proj_nv):
        if edge_v_t is None or proj_nv.shape[0] == 0:
            return None
        p = torch.tensor(proj_nv, dtype=torch.float32)
        idx = _silhouette_vertex_indices(p, faces_t, edge_v_t, edge_f_t)
        return idx.cpu().numpy()

    for nv, cam in enumerate(dataset.cams):
        mask_pts = (frame_points[nv] if frame_points is not None
                    else np.zeros((0, 2), dtype=np.float32))

        mesh_sil_idx = _mesh_sil_idx(proj[nv]) if draw_points else None
        after = _draw_silhouette_overlay(
            images[nv], proj[nv], mask_pts,
            'AFTER  GT(green) mesh(red)', faces=faces,
            draw_points=draw_points, mesh_sil_indices=mesh_sil_idx)

        if proj_before is not None:
            mesh_sil_idx_before = _mesh_sil_idx(proj_before[nv]) if draw_points else None
            before = _draw_silhouette_overlay(
                images[nv], proj_before[nv], mask_pts,
                'BEFORE  GT(green) mesh(red)', faces=faces,
                draw_points=draw_points, mesh_sil_indices=mesh_sil_idx_before)
            canvas = np.concatenate([before, after], axis=1)
        else:
            canvas = after

        outname = join(out_root, cam, f'{nf:06d}.jpg')
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, canvas)


if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    from easymocap.mytools.debug_utils import log, log_time, mywarn, myerror

    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    args = parse_parser(parser)

    log_time("Starting EasyMocap mv1p pipeline...")
    log(f"Input path: {args.path}")
    log(f"Output directory: {args.out}")
    log(f"Model: {args.model}, Gender: {args.gender}, Body type: {args.body}")

    try:
        skel_path = join(args.out, 'keypoints3d')
        dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
                         config=CONFIG[args.body], kpts_type=args.body,
                         undis=args.undis, no_img=False, verbose=args.verbose)
        dataset.writer.save_origin = args.save_origin

        if args.skel or not os.path.exists(skel_path):
            log_time("Running 3D keypoint triangulation...")
            mv1pmf_skel(dataset, check_repro=True, args=args)

        log_time("Fitting SMPL model...")
        mv1pmf_smpl(dataset, args)
        log_time("All processing complete!")

    except Exception as e:
        myerror(f"Pipeline failed: {e}")
        raise
    