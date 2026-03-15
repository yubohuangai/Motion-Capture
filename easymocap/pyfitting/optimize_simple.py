'''
  @ Date: 2020-11-19 10:49:26
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-05-25 19:51:12
  @ FilePath: /EasyMocap/easymocap/pyfitting/optimize_simple.py
'''
import cv2
import numpy as np
import torch
from .lbfgs import LBFGS 
from .optimize import FittingMonitor, grad_require, FittingLog
from .lossfactory import LossSmoothBodyMean, LossRegPoses
from .lossfactory import LossKeypoints3D, LossKeypointsMV2D, LossSmoothBody, LossRegPosesZero, LossInit, LossSmoothPoses


def _build_edge_face_adjacency(faces_np):
    """Build edge-to-face adjacency from triangle mesh topology (called once).

    Returns:
        edges:      (E, 2) int64 – vertex-index pair per edge
        edge_faces: (E, 2) int64 – face-index pair per edge (-1 for boundary)
    """
    from collections import defaultdict
    emap = defaultdict(list)
    for fi in range(faces_np.shape[0]):
        for j in range(3):
            v0, v1 = int(faces_np[fi, j]), int(faces_np[fi, (j + 1) % 3])
            key = (min(v0, v1), max(v0, v1))
            emap[key].append(fi)
    E = len(emap)
    edges = np.empty((E, 2), dtype=np.int64)
    edge_faces = np.full((E, 2), -1, dtype=np.int64)
    for i, (key, fids) in enumerate(emap.items()):
        edges[i] = key
        edge_faces[i, 0] = fids[0]
        if len(fids) > 1:
            edge_faces[i, 1] = fids[1]
    return edges, edge_faces


def _silhouette_vertex_indices(proj_2d, faces_t, edge_v_t, edge_f_t):
    """Return unique vertex indices on the projected-mesh silhouette boundary.

    A silhouette edge separates a front-facing triangle from a back-facing one
    (or is a boundary edge of a front-facing triangle).  The 2D face
    orientation is determined by the sign of the cross-product of two edge
    vectors in screen space.

    All inputs/outputs live on the same device (GPU-friendly, no CPU round-trips).
    """
    v0 = proj_2d[faces_t[:, 0]]
    v1 = proj_2d[faces_t[:, 1]]
    v2 = proj_2d[faces_t[:, 2]]
    cross_z = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
               (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
    front = cross_z > 0

    boundary = edge_f_t[:, 1] < 0
    f0_front = front[edge_f_t[:, 0]]
    safe_f1 = edge_f_t[:, 1].clamp(min=0)
    f1_front = front[safe_f1]

    interior_sil = (~boundary) & (f0_front != f1_front)
    boundary_sil = boundary & f0_front
    sil_mask = interior_sil | boundary_sil

    if not sil_mask.any():
        return torch.empty(0, dtype=torch.long, device=proj_2d.device)
    return torch.unique(edge_v_t[sil_mask].reshape(-1))


def _outside_gt_contour(gt_pts_np, query_pts_np):
    """Return boolean mask: True for query points outside the GT contour polygon.

    Renders the GT contour as a filled polygon and tests each query point.
    Points outside the image bounds are considered outside.
    """
    nq = query_pts_np.shape[0]
    if gt_pts_np.shape[0] < 3:
        return np.ones(nq, dtype=bool)
    all_pts = np.concatenate([gt_pts_np, query_pts_np], axis=0)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int) + 2
    h = int(max(min(max_xy[1], 4096), 1))
    w = int(max(min(max_xy[0], 4096), 1))
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = gt_pts_np.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(mask, [poly], 255)
    qi = query_pts_np.astype(np.int32)
    oob = (qi[:, 0] < 0) | (qi[:, 0] >= w) | (qi[:, 1] < 0) | (qi[:, 1] >= h)
    qi = np.clip(qi, 0, np.array([w - 1, h - 1]))
    result = mask[qi[:, 1], qi[:, 0]] == 0
    result[oob] = True
    return result


def _outer_silhouette_vertex_indices(proj_2d, faces_t, edge_v_t, edge_f_t,
                                     gt_pts_np, device):
    """Return vertex indices on the OUTER silhouette boundary only.

    Excludes internal boundaries (arm-hole, mouth opening, ear cavity, etc.)
    by rasterizing the mesh and keeping only vertices on the outer contour.
    Matches the contour visualization (RETR_EXTERNAL).
    """
    sil_idx = _silhouette_vertex_indices(proj_2d, faces_t, edge_v_t, edge_f_t)
    if sil_idx.numel() == 0:
        return sil_idx

    proj_np = proj_2d.detach().cpu().numpy()
    faces_np = faces_t.cpu().numpy()
    sil_idx_np = sil_idx.cpu().numpy()

    all_pts = np.concatenate([gt_pts_np, proj_np[sil_idx_np]], axis=0)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int) + 2
    h = int(max(min(max_xy[1], 4096), 1))
    w = int(max(min(max_xy[0], 4096), 1))

    tris = proj_np[faces_np].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, tris, 255)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return sil_idx

    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, 2)

    keep = []
    for i in range(sil_idx_np.shape[0]):
        idx = sil_idx_np[i]
        x = int(round(proj_np[idx, 0]))
        y = int(round(proj_np[idx, 1]))
        if 0 <= x < w and 0 <= y < h and contour_mask[y, x] > 0:
            keep.append(i)
    if len(keep) == 0:
        return sil_idx
    out_indices = sil_idx_np[keep]
    return torch.tensor(out_indices, dtype=torch.long, device=device)


def optimizeShape(body_model, body_params, keypoints3d,
    weight_loss, kintree, cfg=None):
    """ simple function for optimizing model shape given 3d keypoints

    Args:
        body_model (SMPL model)
        params_init (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints (nFrames, nJoints, 3): 3D keypoints
        weight (Dict): string:float
        kintree ([[src, dst]]): list of list:int
        cfg (Config): Config Node controling running mode
    """
    device = body_model.device
    # 计算不同的骨长
    kintree = np.array(kintree, dtype=int)
    # limb_length: nFrames, nLimbs, 1
    limb_length = np.linalg.norm(keypoints3d[:, kintree[:, 1], :3] - keypoints3d[:, kintree[:, 0], :3], axis=2, keepdims=True)
    # conf: nFrames, nLimbs, 1
    limb_conf = np.minimum(keypoints3d[:, kintree[:, 1], 3:], keypoints3d[:, kintree[:, 0], 3:])
    limb_length = torch.Tensor(limb_length).to(device)
    limb_conf = torch.Tensor(limb_conf).to(device)
    body_params = {key:torch.Tensor(val).to(device) for key, val in body_params.items()}
    body_params_init = {key:val.clone() for key, val in body_params.items()}
    opt_params = [body_params['shapes']]
    grad_require(opt_params, True)
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe', max_iter=10)
    nFrames = keypoints3d.shape[0]
    verbose = False
    def closure(debug=False):
        optimizer.zero_grad()
        keypoints3d = body_model(return_verts=False, return_tensor=True, only_shape=True, **body_params)
        src = keypoints3d[:, kintree[:, 0], :3] #.detach()
        dst = keypoints3d[:, kintree[:, 1], :3]
        direct_est = (dst - src).detach()
        direct_norm = torch.norm(direct_est, dim=2, keepdim=True)
        direct_normalized = direct_est/(direct_norm + 1e-4)
        err = dst - src - direct_normalized * limb_length
        loss_dict = {
            's3d': torch.sum(err**2*limb_conf)/nFrames, 
            'reg_shapes': torch.sum(body_params['shapes']**2)}
        if 'init_shape' in weight_loss.keys():
            loss_dict['init_shape'] = torch.sum((body_params['shapes'] - body_params_init['shapes'])**2)
        # fittingLog.step(loss_dict, weight_loss)
        if verbose:
            print(' '.join([key + ' %.3f'%(loss_dict[key].item()*weight_loss[key]) 
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        if not debug:
            loss.backward()
            return loss
        else:
            return loss_dict

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key].item()
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe')
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def optimizeShapeSilhouette(body_model, body_params, silhouette_points, Pall, weight_loss, max_iter=20, max_verts=1000, max_pairs=200):
    """Refine shape with contour-to-contour silhouette Chamfer loss.

    Args:
        body_model: SMPL body model.
        body_params: dict of fitted parameters (poses/Rh/Th/shapes).
        silhouette_points: nested list [nFrames][nViews], each item (N, 2) ndarray.
        Pall: (nViews, 3, 4) projection matrices.
        weight_loss: dict, uses keys {'chamfer', 'reg_shapes'}.
        max_iter: LBFGS max iterations.
        max_verts: unused (kept for API compat).
        max_pairs: max frame-view pairs for silhouette loss.
    """
    if silhouette_points is None:
        return body_params
    nFrames = len(silhouette_points)
    if nFrames == 0:
        return body_params
    nViews = len(silhouette_points[0]) if len(silhouette_points[0]) > 0 else 0
    if nViews == 0:
        return body_params
    if weight_loss.get('chamfer', 0.) <= 0.:
        return body_params

    device = body_model.device
    Pall_t = torch.tensor(Pall, dtype=torch.float32, device=device)
    body_params = {key: torch.tensor(val, dtype=torch.float32, device=device) for key, val in body_params.items()}
    body_params_init = {key: val.clone() for key, val in body_params.items()}
    opt_params = [body_params['shapes']]
    grad_require(opt_params, True)
    optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe', max_iter=max_iter)
    ones_cache = {}

    points_t = []
    for nf in range(nFrames):
        frame_points = []
        for nv in range(nViews):
            pts = silhouette_points[nf][nv]
            if pts is None or len(pts) == 0:
                frame_points.append(None)
                continue
            frame_points.append(torch.tensor(pts, dtype=torch.float32, device=device))
        points_t.append(frame_points)
    valid_pairs = [(nf, nv) for nf in range(nFrames) for nv in range(nViews) if points_t[nf][nv] is not None]
    if len(valid_pairs) == 0:
        return body_params
    if max_pairs is not None and max_pairs > 0 and len(valid_pairs) > max_pairs:
        sample_idx = np.linspace(0, len(valid_pairs) - 1, max_pairs, dtype=int)
        valid_pairs = [valid_pairs[i] for i in sample_idx]

    edge_v, edge_f = _build_edge_face_adjacency(body_model.faces)
    edge_v_t = torch.tensor(edge_v, dtype=torch.long, device=device)
    edge_f_t = torch.tensor(edge_f, dtype=torch.long, device=device)
    faces_t = body_model.faces_tensor.to(device)

    def closure(debug=False):
        optimizer.zero_grad()
        verts = body_model(return_verts=True, return_tensor=True, **body_params)
        nv_all = verts.shape[1]
        if nv_all not in ones_cache:
            ones_cache[nv_all] = torch.ones(
                (verts.shape[0], nv_all, 1), dtype=verts.dtype, device=device)
        verts_h = torch.cat([verts, ones_cache[nv_all]], dim=2)
        point_cam = torch.einsum('vab,fnb->vfna', Pall_t, verts_h)
        proj = point_cam[..., :2] / torch.clamp(point_cam[..., 2:3], min=1e-6)

        chamfer_loss = torch.tensor(0., device=device)
        count = 0
        for nf, nv in valid_pairs:
            gt_contour = points_t[nf][nv]
            pv = proj[nv, nf]
            sil_idx = _outer_silhouette_vertex_indices(
                pv, faces_t, edge_v_t, edge_f_t,
                gt_contour.detach().cpu().numpy(), device)
            if sil_idx.numel() == 0:
                continue
            mesh_contour = pv[sil_idx]
            outside = _outside_gt_contour(
                gt_contour.detach().cpu().numpy(),
                mesh_contour.detach().cpu().numpy())
            outside_t = torch.tensor(outside, dtype=torch.bool,
                                     device=device)
            if not outside_t.any():
                continue
            dists = torch.cdist(mesh_contour[outside_t], gt_contour, p=2)
            chamfer_loss = chamfer_loss + dists.min(dim=1)[0].mean()
            count += 1
        if count > 0:
            chamfer_loss = chamfer_loss / count

        loss_dict = {
            'chamfer': chamfer_loss,
            'reg_shapes': torch.sum(body_params['shapes']**2)
        }
        if 'init_shape' in weight_loss.keys():
            loss_dict['init_shape'] = torch.sum((body_params['shapes'] - body_params_init['shapes'])**2)

        loss = sum([loss_dict[key] * weight_loss.get(key, 0.) for key in loss_dict.keys()])
        if debug:
            return loss_dict
        loss.backward()
        return loss

    fitting = FittingMonitor(ftol=1e-4)
    fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    body_params = {key: val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def optimizeShapeWithPose(body_model, body_params, keypoints3d,
    keypoints2d, bboxes, Pall, weight_loss,
    silhouette_points=None, max_verts=1000, max_pairs=200):
    """Refine shape parameters using the current pose estimate.

    Inspired by smplfitter's vertex/joint-level shape fitting, but adapted for
    multi-view scenarios with 2D keypoints and camera matrices.

    Unlike optimizeShape (which only matches bone lengths in T-pose), this
    function uses the fitted pose to leverage:
      - Direct 3D joint position matching (many more constraints than bone lengths)
      - Multi-view 2D joint reprojection (uses image evidence from all cameras)
      - Optional silhouette Chamfer loss (captures body width/depth)

    Args:
        body_model: SMPL/SMPLX body model.
        body_params: dict of current fitted parameters (poses/Rh/Th/shapes/expression).
        keypoints3d: (nFrames, nJoints, 4) triangulated 3D keypoints with confidence.
        keypoints2d: (nFrames, nViews, nJoints, 3) multi-view 2D keypoints, or None.
        bboxes: (nFrames, nViews, 5) bounding boxes per view, or None.
        Pall: (nViews, 3, 4) projection matrices.
        weight_loss: dict of loss weights (k3d_shape, k2d_shape, chamfer,
                     reg_shapes, init_shape).
        silhouette_points: nested list [nFrames][nViews] of (N, 2) arrays, or None.
        max_verts: vertex subsample count for silhouette Chamfer.
        max_pairs: max frame-view pairs for silhouette loss.
    """
    device = body_model.device
    nFrames = keypoints3d.shape[0]

    body_params = {key: torch.Tensor(val).to(device)
                   for key, val in body_params.items()}
    body_params_init = {key: val.clone() for key, val in body_params.items()}

    opt_params = [body_params['shapes']]
    grad_require(opt_params, True)

    kp3d = torch.Tensor(keypoints3d).to(device)
    kp3d_conf = kp3d[..., 3:]
    kp3d_pos = kp3d[..., :3]

    has_2d = (keypoints2d is not None and bboxes is not None
              and Pall is not None
              and weight_loss.get('k2d_shape', 0.) > 0.)
    if has_2d:
        kp2d = torch.Tensor(keypoints2d).to(device).transpose(0, 1)
        kp2d_conf = kp2d[..., 2:3]
        kp2d_pos = kp2d[..., :2]
        nViews = kp2d.shape[0]
        bbox = torch.Tensor(bboxes).to(device).transpose(0, 1)
        bbox_size = torch.clamp(
            (bbox[..., 2:4] - bbox[..., :2]).max(dim=-1, keepdim=True)[0],
            min=1.)
        inv_bbox = 1. / bbox_size
        Pall_t = torch.Tensor(Pall).to(device)
    else:
        nViews = 0
        Pall_t = (torch.Tensor(Pall).to(device)
                  if Pall is not None else None)

    chamfer_weight = weight_loss.get('chamfer', 0.)
    has_sil = (silhouette_points is not None and chamfer_weight > 0.)
    valid_pairs = []
    sil_pts_t = []
    if has_sil:
        if Pall_t is None:
            Pall_t = torch.Tensor(Pall).to(device)
        for nf in range(len(silhouette_points)):
            frame_pts = []
            for nv_pts in silhouette_points[nf]:
                if nv_pts is None or len(nv_pts) == 0:
                    frame_pts.append(None)
                else:
                    frame_pts.append(
                        torch.tensor(nv_pts, dtype=torch.float32,
                                     device=device))
            sil_pts_t.append(frame_pts)
        nSilViews = len(silhouette_points[0]) if len(silhouette_points) > 0 else 0
        valid_pairs = [
            (nf, nv) for nf in range(len(silhouette_points))
            for nv in range(nSilViews) if sil_pts_t[nf][nv] is not None]
        if 0 < max_pairs < len(valid_pairs):
            idx = np.linspace(0, len(valid_pairs) - 1, max_pairs, dtype=int)
            valid_pairs = [valid_pairs[i] for i in idx]

    # Precompute mesh edge-face adjacency (topology only, done once)
    edge_v_t, edge_f_t, faces_t = None, None, None
    if has_sil and len(valid_pairs) > 0:
        edge_v, edge_f = _build_edge_face_adjacency(body_model.faces)
        edge_v_t = torch.tensor(edge_v, dtype=torch.long, device=device)
        edge_f_t = torch.tensor(edge_f, dtype=torch.long, device=device)
        faces_t = body_model.faces_tensor.to(device)

    optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe', max_iter=30)
    ones_cache = {}

    def closure(debug=False):
        optimizer.zero_grad()

        loss_dict = {}

        need_joints = (weight_loss.get('k3d_shape', 0.) > 0. or has_2d)
        if need_joints:
            joints = body_model(return_verts=False, return_tensor=True,
                                **body_params)
            nJ = min(joints.shape[1], kp3d_pos.shape[1])

            if weight_loss.get('k3d_shape', 0.) > 0.:
                diff = (joints[:, :nJ, :3] - kp3d_pos[:, :nJ]) * kp3d_conf[:, :nJ]
                loss_dict['k3d_shape'] = torch.sum(diff ** 2) / nFrames

            if has_2d:
                nJ_homo = joints.shape[1]
                key_homo = nJ_homo
                if key_homo not in ones_cache:
                    ones_cache[key_homo] = torch.ones(
                        nFrames, nJ_homo, 1, device=device)
                joints_h = torch.cat([joints[..., :3], ones_cache[key_homo]],
                                     dim=-1)
                point_cam = torch.einsum('vab,fnb->vfna', Pall_t, joints_h)
                proj_2d = point_cam[..., :2] / torch.clamp(
                    point_cam[..., 2:3], min=1e-6)
                nJ2d = min(proj_2d.shape[2], kp2d_pos.shape[2])
                diff_2d = ((proj_2d[:, :, :nJ2d] - kp2d_pos[:, :, :nJ2d])
                           * kp2d_conf[:, :, :nJ2d]
                           * inv_bbox.unsqueeze(2))
                loss_dict['k2d_shape'] = torch.sum(diff_2d ** 2) / nFrames / nViews

        if has_sil and len(valid_pairs) > 0:
            verts = body_model(return_verts=True, return_tensor=True,
                               **body_params)
            nv_all = verts.shape[1]
            if nv_all not in ones_cache:
                ones_cache[nv_all] = torch.ones(
                    nFrames, nv_all, 1, dtype=verts.dtype, device=device)
            verts_h = torch.cat([verts, ones_cache[nv_all]], dim=2)
            proj_v = torch.einsum('vab,fnb->vfna', Pall_t, verts_h)
            proj_v_2d = proj_v[..., :2] / torch.clamp(
                proj_v[..., 2:3], min=1e-6)
            chamfer_loss = torch.tensor(0., device=device)
            count = 0
            for nf, nv in valid_pairs:
                gt_contour = sil_pts_t[nf][nv]
                pv = proj_v_2d[nv, nf]           # (V, 2)
                sil_idx = _outer_silhouette_vertex_indices(
                    pv, faces_t, edge_v_t, edge_f_t,
                    gt_contour.detach().cpu().numpy(), device)
                if sil_idx.numel() == 0:
                    continue
                mesh_contour = pv[sil_idx]        # (S, 2)
                outside = _outside_gt_contour(
                    gt_contour.detach().cpu().numpy(),
                    mesh_contour.detach().cpu().numpy())
                outside_t = torch.tensor(outside, dtype=torch.bool,
                                         device=device)
                if not outside_t.any():
                    continue
                dists = torch.cdist(mesh_contour[outside_t], gt_contour, p=2)
                chamfer_loss = chamfer_loss + dists.min(dim=1)[0].mean()
                count += 1
            if count > 0:
                chamfer_loss = chamfer_loss / count
            loss_dict['chamfer'] = chamfer_loss

        loss_dict['reg_shapes'] = torch.sum(body_params['shapes'] ** 2)
        if weight_loss.get('init_shape', 0.) > 0.:
            loss_dict['init_shape'] = torch.sum(
                (body_params['shapes'] - body_params_init['shapes']) ** 2)

        loss = sum([loss_dict[key] * weight_loss.get(key, 0.)
                    for key in loss_dict.keys()])
        if debug:
            return loss_dict
        loss.backward()
        return loss

    # Log loss before optimization.
    with torch.no_grad():
        ld_before = closure(debug=True)
    print('[Shape refine] BEFORE: ' + '  '.join(
        '{} {:.4f}(w={})'.format(k, v.item(), weight_loss.get(k, 0.))
        for k, v in ld_before.items()))
    print('[Shape refine] betas: ' + ' '.join(
        '{:.3f}'.format(b) for b in body_params['shapes'].detach().cpu().view(-1).tolist()))

    fitting = FittingMonitor(ftol=1e-5)
    fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)

    with torch.no_grad():
        ld_after = closure(debug=True)
    print('[Shape refine] AFTER:  ' + '  '.join(
        '{} {:.4f}(w={})'.format(k, v.item(), weight_loss.get(k, 0.))
        for k, v in ld_after.items()))
    print('[Shape refine] betas: ' + ' '.join(
        '{:.3f}'.format(b) for b in body_params['shapes'].detach().cpu().view(-1).tolist()))

    body_params = {key: val.detach().cpu().numpy()
                   for key, val in body_params.items()}
    return body_params


N_BODY = 25
N_HAND = 21

def interp(left_value, right_value, weight, key='poses'):
    if key == 'Rh':
        return left_value * weight + right_value * (1 - weight)
    elif key == 'Th':
        return left_value * weight + right_value * (1 - weight)
    elif key == 'poses':
        return left_value * weight + right_value * (1 - weight)

def get_interp_by_keypoints(keypoints):
    if len(keypoints.shape) == 3: # (nFrames, nJoints, 3)
        conf = keypoints[..., -1]
    elif len(keypoints.shape) == 4: # (nViews, nFrames, nJoints)
        conf = keypoints[..., -1].sum(axis=0)
    else:
        raise NotImplementedError
    not_valid_frames = np.where(conf.sum(axis=1) < 0.01)[0].tolist()
    # 遍历空白帧，选择起点和终点
    ranges = []
    if len(not_valid_frames) > 0:
        start = not_valid_frames[0]
        for i in range(1, len(not_valid_frames)):
            if not_valid_frames[i] == not_valid_frames[i-1] + 1:
                pass
            else:# 改变位置了
                end = not_valid_frames[i-1]
                ranges.append((start, end))
                start = not_valid_frames[i]
        ranges.append((start, not_valid_frames[-1]))
    def interp_func(params):
        nFrames = params['poses'].shape[0]
        for start, end in ranges:
            # Handle boundary ranges safely (missing left/right valid frame).
            left = start - 1
            right = end + 1
            if left < 0 and right >= nFrames:
                # All frames invalid, keep as-is.
                continue
            if left < 0:
                for nf in range(start, end + 1):
                    for key in ['Rh', 'Th', 'poses']:
                        params[key][nf] = params[key][right]
                continue
            if right >= nFrames:
                for nf in range(start, end + 1):
                    for key in ['Rh', 'Th', 'poses']:
                        params[key][nf] = params[key][left]
                continue
            for nf in range(start, end+1):
                weight = (nf - left)/(right - left)
                for key in ['Rh', 'Th', 'poses']:
                    params[key][nf] = interp(params[key][left], params[key][right], 1-weight, key=key)
        return params
    return interp_func

def interp_by_k3d(conf, params):
    for key in ['Rh', 'Th', 'poses']:
        params[key] = params[key].clone()
    # Totally invalid frames
    not_valid_frames = torch.nonzero(conf.sum(dim=1).squeeze() < 0.01)[:, 0].detach().cpu().numpy().tolist()
    # 遍历空白帧，选择起点和终点
    ranges = []
    if len(not_valid_frames) > 0:
        start = not_valid_frames[0]
        for i in range(1, len(not_valid_frames)):
            if not_valid_frames[i] == not_valid_frames[i-1] + 1:
                pass
            else:# 改变位置了
                end = not_valid_frames[i-1]
                ranges.append((start, end))
                start = not_valid_frames[i]
        ranges.append((start, not_valid_frames[-1]))
    nFrames = params['poses'].shape[0]
    for start, end in ranges:
        # Handle boundary ranges safely (missing left/right valid frame).
        left = start - 1
        right = end + 1
        if left < 0 and right >= nFrames:
            # All frames invalid, keep as-is.
            continue
        if left < 0:
            for nf in range(start, end + 1):
                for key in ['Rh', 'Th', 'poses']:
                    params[key][nf] = params[key][right]
            continue
        if right >= nFrames:
            for nf in range(start, end + 1):
                for key in ['Rh', 'Th', 'poses']:
                    params[key][nf] = params[key][left]
            continue
        for nf in range(start, end+1):
            weight = (nf - left)/(right - left)
            for key in ['Rh', 'Th', 'poses']:
                params[key][nf] = interp(params[key][left], params[key][right], 1-weight, key=key)
    return params

def deepcopy_tensor(body_params):
    for key in body_params.keys():
        body_params[key] = body_params[key].clone()
    return body_params

def dict_of_tensor_to_numpy(body_params):
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def get_prepare_smplx(body_params, cfg, nFrames):
    zero_pose = torch.zeros((nFrames, 3), device=cfg.device)
    if not cfg.OPT_HAND and cfg.model in ['smplh', 'smplx']:
        zero_pose_hand = torch.zeros((nFrames, body_params['poses'].shape[1] - 66), device=cfg.device)
    elif cfg.OPT_HAND and not cfg.OPT_EXPR and cfg.model == 'smplx':
        zero_pose_face = torch.zeros((nFrames, body_params['poses'].shape[1] - 78), device=cfg.device)

    def pack(new_params):
        if not cfg.OPT_HAND and cfg.model in ['smplh', 'smplx']:
            new_params['poses'] = torch.cat([zero_pose, new_params['poses'][:, 3:66], zero_pose_hand], dim=1)
        else:
            new_params['poses'] = torch.cat([zero_pose, new_params['poses'][:, 3:]], dim=1)
        return new_params
    return pack

def get_optParams(body_params, cfg, extra_params):
    for key, val in body_params.items():
        body_params[key] = torch.Tensor(val).to(cfg.device)
    if cfg is None:
        opt_params = [body_params['Rh'], body_params['Th'], body_params['poses']]
    else:
        if extra_params is not None:
            opt_params = extra_params
        else:
            opt_params = []
        if cfg.OPT_R:
            opt_params.append(body_params['Rh'])
        if cfg.OPT_T:
            opt_params.append(body_params['Th'])
        if cfg.OPT_POSE:
            opt_params.append(body_params['poses'])
        if cfg.OPT_SHAPE:
            opt_params.append(body_params['shapes'])
        if cfg.OPT_EXPR and cfg.model == 'smplx':
            opt_params.append(body_params['expression'])
    return opt_params

def _optimizeSMPL(body_model, body_params, prepare_funcs, postprocess_funcs, 
    loss_funcs, extra_params=None,
    weight_loss={}, cfg=None):
    """ A common interface for different optimization.

    Args:
        body_model (SMPL model)
        body_params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        prepare_funcs (List): functions for prepare
        loss_funcs (Dict): functions for loss
        weight_loss (Dict): weight
        cfg (Config): Config Node controling running mode
    """
    loss_funcs = {key: val for key, val in loss_funcs.items() if key in weight_loss.keys() and weight_loss[key] > 0.}
    if cfg.verbose:
        print('Loss Functions: ')
        for key, func in loss_funcs.items():
            print('  -> {:15s}: {}'.format(key, func.__doc__))
    opt_params = get_optParams(body_params, cfg, extra_params)
    grad_require(opt_params, True)
    optimizer = LBFGS(opt_params, 
        line_search_fn='strong_wolfe')
    PRINT_STEP = 100
    records = []
    def closure(debug=False):
        # 0. Prepare body parameters => new_params
        optimizer.zero_grad()
        new_params = body_params.copy()
        for func in prepare_funcs:
            new_params = func(new_params)
        # 1. Compute keypoints => kpts_est
        kpts_est = body_model(return_verts=False, return_tensor=True, **new_params)
        # 2. Compute loss => loss_dict
        loss_dict = {key:func(kpts_est=kpts_est, **new_params) for key, func in loss_funcs.items()}
        # 3. Summary and log
        cnt = len(records)
        if cfg.verbose and cnt % PRINT_STEP == 0:
            print('{:-6d}: '.format(cnt) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        records.append(loss.item())
        if debug:
            return loss_dict
        loss.backward()
        return loss

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    if cfg.verbose:
        print('{:-6d}: '.format(len(records)) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key]) 
            for key in loss_dict.keys() if weight_loss[key]>0]))
    loss_dict = {key:val.item() for key, val in loss_dict.items()}
    # post-process the body_parameters
    for func in postprocess_funcs:
        body_params = func(body_params)
    return body_params

def optimizePose3D(body_model, params, keypoints3d, weight, cfg):
    """ 
        simple function for optimizing model pose given 3d keypoints

    Args:
        body_model (SMPL model)
        params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints3d (nFrames, nJoints, 4): 3D keypoints
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    nFrames = keypoints3d.shape[0]
    prepare_funcs = [
        deepcopy_tensor,
        get_prepare_smplx(params, cfg, nFrames),
        get_interp_by_keypoints(keypoints3d)
    ]
    loss_funcs = {
        'k3d': LossKeypoints3D(keypoints3d, cfg).body,
        'smooth_body': LossSmoothBodyMean(cfg).body,
        'smooth_poses': LossSmoothPoses(1, nFrames, cfg).poses,
        'reg_poses': LossRegPoses(cfg).reg_body,
        'init_poses': LossInit(params, cfg).init_poses,
    }
    if body_model.model_type != 'mano':
        loss_funcs['reg_poses_zero'] = LossRegPosesZero(keypoints3d, cfg).__call__
    if cfg.OPT_HAND:
        loss_funcs['k3d_hand'] = LossKeypoints3D(keypoints3d, cfg, norm='l1').hand
        loss_funcs['reg_hand'] = LossRegPoses(cfg).reg_hand
        # loss_funcs['smooth_hand'] = LossSmoothPoses(1, nFrames, cfg).hands
        loss_funcs['smooth_hand'] = LossSmoothBodyMean(cfg).hand

    if cfg.OPT_EXPR:
        loss_funcs['k3d_face'] = LossKeypoints3D(keypoints3d, cfg, norm='l1').face
        loss_funcs['reg_head'] = LossRegPoses(cfg).reg_head
        loss_funcs['reg_expr'] = LossRegPoses(cfg).reg_expr
        loss_funcs['smooth_head'] = LossSmoothPoses(1, nFrames, cfg).head

    postprocess_funcs = [
        get_interp_by_keypoints(keypoints3d),
        dict_of_tensor_to_numpy
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, weight_loss=weight, cfg=cfg)
    return params

def optimizePose2D(body_model, params, bboxes, keypoints2d, Pall, weight, cfg):
    """ 
        simple function for optimizing model pose given 3d keypoints

    Args:
        body_model (SMPL model)
        params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints2d (nFrames, nViews, nJoints, 4): 2D keypoints of each view
        bboxes: (nFrames, nViews, 5)
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    # transpose to (nViews, nFrames, 5)
    bboxes = bboxes.transpose(1, 0, 2)
    # transpose to => keypoints2d: (nViews, nFrames, nJoints, 3)
    keypoints2d = keypoints2d.transpose(1, 0, 2, 3)
    nViews, nFrames = keypoints2d.shape[:2]
    prepare_funcs = [
        deepcopy_tensor,
        get_prepare_smplx(params, cfg, nFrames),
        get_interp_by_keypoints(keypoints2d)
    ]
    loss_funcs = {
        'k2d': LossKeypointsMV2D(keypoints2d, bboxes, Pall, cfg).__call__,
        'smooth_body': LossSmoothBodyMean(cfg).body,
        'init_poses': LossInit(params, cfg).init_poses,
        'smooth_poses': LossSmoothPoses(nViews, nFrames, cfg).poses,
        'reg_poses': LossRegPoses(cfg).reg_body,
    }
    if body_model.model_type != 'mano':
        loss_funcs['reg_poses_zero'] = LossRegPosesZero(keypoints2d, cfg).__call__
    if cfg.OPT_SHAPE:
        loss_funcs['init_shapes'] = LossInit(params, cfg).init_shapes
    if cfg.OPT_HAND:
        loss_funcs['reg_hand'] = LossRegPoses(cfg).reg_hand
        # loss_funcs['smooth_hand'] = LossSmoothPoses(1, nFrames, cfg).hands
        loss_funcs['smooth_hand'] = LossSmoothBodyMean(cfg).hand

    if cfg.OPT_EXPR:
        loss_funcs['reg_head'] = LossRegPoses(cfg).reg_head
        loss_funcs['reg_expr'] = LossRegPoses(cfg).reg_expr
        loss_funcs['smooth_head'] = LossSmoothPoses(1, nFrames, cfg).head

    loss_funcs = {key:val for key, val in loss_funcs.items() if key in weight.keys()}
    
    postprocess_funcs = [
        get_interp_by_keypoints(keypoints2d),
        dict_of_tensor_to_numpy
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, weight_loss=weight, cfg=cfg)
    return params