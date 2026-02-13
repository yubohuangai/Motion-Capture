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

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
except Exception:
    least_squares = None
    lil_matrix = None


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


def _pack_rt(rvec, tvec):
    return np.hstack([rvec.reshape(3), tvec.reshape(3)])


def _unpack_rt(x):
    rvec = x[:3].reshape(3, 1)
    tvec = x[3:6].reshape(3, 1)
    return rvec, tvec


def _rebuild_dataset_projection(dataset):
    for cam in dataset.cams:
        cam_data = dataset.cameras[cam]
        cam_data['RT'] = np.hstack((cam_data['R'], cam_data['T']))
        cam_data['P'] = cam_data['K'] @ cam_data['RT']
    dataset.Pall = np.stack([dataset.cameras[cam]['P'] for cam in dataset.cams])


def _compute_repro_error(keypoints3d, keypoints2d, Pall):
    kpts_repro = projectN3(keypoints3d, Pall)
    conf = (keypoints3d[None, :, -1:] > 0) * (keypoints2d[:, :, -1:] > 0)
    dist = np.sqrt((((kpts_repro[..., :2] - keypoints2d[..., :2]) * conf) ** 2).sum(axis=-1))
    return (dist[conf[..., 0] > 0]).mean() if np.any(conf[..., 0] > 0) else 0.0


def joint_ba_refine(dataset, keypoints2d_all, kp3ds_all, args):
    if least_squares is None or lil_matrix is None:
        raise ImportError("scipy is required for --joint_ba: pip install scipy")
    if dataset.cameras is None:
        raise RuntimeError("Camera parameters are required for --joint_ba")

    n_frames = keypoints2d_all.shape[0]
    frame_indices = list(range(0, n_frames, max(1, args.joint_ba_step)))
    if args.joint_ba_max_frames > 0:
        frame_indices = frame_indices[:args.joint_ba_max_frames]
    if len(frame_indices) < 2:
        mywarn("Too few frames selected for joint BA. Skipping joint BA.")
        return kp3ds_all

    camnames = dataset.cams
    cam0 = camnames[0]
    cam_order = camnames[1:]
    K_dict = {cam: dataset.cameras[cam]['K'].astype(np.float64) for cam in camnames}
    R0_dict = {cam: dataset.cameras[cam]['R'].astype(np.float64) for cam in camnames}
    T0_dict = {cam: dataset.cameras[cam]['T'].astype(np.float64) for cam in camnames}

    # Keep only frame/joint pairs with enough 2D support.
    frame_infos = []
    total_obs = 0
    for nf in frame_indices:
        kpts2d = keypoints2d_all[nf]
        kp3d = kp3ds_all[nf]
        valid_joint = ((kpts2d[..., 2] > 0).sum(axis=0) >= 2) & (kp3d[:, 3] > 0)
        joint_ids = np.where(valid_joint)[0]
        if joint_ids.size == 0:
            continue
        obs_by_cam = []
        obs_count = 0
        for nv in range(len(camnames)):
            vis_local = np.where(kpts2d[nv, joint_ids, 2] > 0)[0].astype(np.int64)
            obs_by_cam.append(vis_local)
            obs_count += int(vis_local.size)
        total_obs += obs_count
        frame_infos.append({
            'nf': nf,
            'joint_ids': joint_ids.astype(np.int64),
            'kpts2d': kpts2d.astype(np.float64),
            'xyz0': kp3d[joint_ids, :3].astype(np.float64),
            'obs_by_cam': obs_by_cam,
        })
    if len(frame_infos) < 2:
        mywarn("Not enough valid frame/joint observations for joint BA. Skipping.")
        return kp3ds_all

    x0 = []
    cam_prior = {}
    for cam in cam_order:
        rvec0, _ = cv2.Rodrigues(R0_dict[cam])
        tvec0 = T0_dict[cam]
        cam_prior[cam] = (rvec0.copy(), tvec0.copy())
        x0.append(_pack_rt(rvec0, tvec0))
    for info in frame_infos:
        x0.append(info['xyz0'].reshape(-1))
    x0 = np.concatenate(x0, axis=0)

    # Variable block index mapping for sparse Jacobian.
    cam_col_start = {}
    for icam, cam in enumerate(cam_order):
        cam_col_start[cam] = 6 * icam
    offset = 6 * len(cam_order)
    for info in frame_infos:
        info['var_start'] = offset
        info['n_joint'] = int(info['joint_ids'].shape[0])
        offset += 3 * info['n_joint']

    # Build Jacobian sparsity: each (u,v) residual depends on one camera block (except cam0)
    # and one local 3D joint block.
    n_vars = x0.size
    n_rows = 2 * total_obs + (6 * len(cam_order) if args.joint_ba_lambda_cam > 0 else 0)
    jac_sparsity = lil_matrix((n_rows, n_vars), dtype=np.int8)
    row = 0
    for info in frame_infos:
        for nv, cam in enumerate(camnames):
            vis_local = info['obs_by_cam'][nv]
            for local_idx in vis_local:
                point_col = info['var_start'] + 3 * int(local_idx)
                jac_sparsity[row:row+2, point_col:point_col+3] = 1
                if cam != cam0:
                    c0 = cam_col_start[cam]
                    jac_sparsity[row:row+2, c0:c0+6] = 1
                row += 2
    if args.joint_ba_lambda_cam > 0:
        for cam in cam_order:
            c0 = cam_col_start[cam]
            jac_sparsity[row:row+6, c0:c0+6] = 1
            row += 6
    assert row == n_rows, (row, n_rows)

    log_time(
        f"[joint_ba] frames={len(frame_infos)} views={len(camnames)} "
        f"obs={total_obs} variables={x0.size} (sparse_jacobian)"
    )

    def residual_func(x):
        idx = 0
        cam_vars = {
            cam0: (
                np.zeros((3, 1), dtype=np.float64),
                R0_dict[cam0],
                T0_dict[cam0]
            )
        }
        for cam in cam_order:
            rvec, tvec = _unpack_rt(x[idx:idx+6])
            R, _ = cv2.Rodrigues(rvec)
            cam_vars[cam] = (rvec, R, tvec)
            idx += 6

        residuals = []
        for info in frame_infos:
            joint_ids = info['joint_ids']
            n_joint = info['n_joint']
            xyz = x[idx:idx + 3 * n_joint].reshape(n_joint, 3)
            idx += 3 * n_joint
            for nv, cam in enumerate(camnames):
                vis_local = info['obs_by_cam'][nv]
                if vis_local.size == 0:
                    continue
                xyz_vis = xyz[vis_local]
                uv_obs = info['kpts2d'][nv, joint_ids[vis_local], :2]
                conf = info['kpts2d'][nv, joint_ids[vis_local], 2:3]

                _, R, T = cam_vars[cam]
                Xc = (R @ xyz_vis.T + T).T
                z = np.maximum(Xc[:, 2:3], 1e-6)
                proj = (K_dict[cam] @ Xc.T).T
                uv = proj[:, :2] / z
                diff = (uv - uv_obs) * np.sqrt(np.clip(conf, 1e-8, None))
                residuals.append(diff.reshape(-1))

        if args.joint_ba_lambda_cam > 0:
            w = np.sqrt(args.joint_ba_lambda_cam)
            for cam in cam_order:
                rvec0, tvec0 = cam_prior[cam]
                rvec, _, tvec = cam_vars[cam]
                residuals.append((w * (rvec - rvec0)).reshape(-1))
                residuals.append((w * (tvec - tvec0)).reshape(-1))

        if len(residuals) == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(residuals, axis=0)

    res0 = residual_func(x0)
    before = np.sqrt((res0.reshape(-1, 2) ** 2).sum(axis=1)).mean() if res0.size > 0 else 0.0
    log_time(f"[joint_ba] weighted reprojection before: {before:.3f}")

    result = least_squares(
        residual_func,
        x0,
        jac_sparsity=jac_sparsity,
        x_scale='jac',
        method='trf',
        loss=args.joint_ba_loss,
        f_scale=args.joint_ba_f_scale,
        max_nfev=args.joint_ba_max_nfev,
        verbose=2 if args.verbose else 0,
    )
    log_time(f"[joint_ba] success={result.success}, cost={result.cost:.4f}, msg={result.message}")
    res1 = residual_func(result.x)
    after = np.sqrt((res1.reshape(-1, 2) ** 2).sum(axis=1)).mean() if res1.size > 0 else 0.0
    log_time(f"[joint_ba] weighted reprojection after : {after:.3f}")

    # Unpack optimized camera extrinsics.
    idx = 0
    for cam in cam_order:
        rvec, tvec = _unpack_rt(result.x[idx:idx+6])
        R, _ = cv2.Rodrigues(rvec)
        dataset.cameras[cam]['R'] = R.astype(np.float64)
        dataset.cameras[cam]['T'] = tvec.astype(np.float64)
        idx += 6
    _rebuild_dataset_projection(dataset)

    # Unpack optimized 3D points for selected frames.
    kp3ds_refined = kp3ds_all.copy()
    for info in frame_infos:
        joint_ids = info['joint_ids']
        n_joint = joint_ids.shape[0]
        xyz = result.x[idx:idx + 3 * n_joint].reshape(n_joint, 3)
        idx += 3 * n_joint
        kp3ds_refined[info['nf'], joint_ids, :3] = xyz

    # Optional: re-triangulate all frames with refined cameras for full consistency.
    if not args.joint_ba_skip_retriangulate:
        for nf in range(n_frames):
            keypoints3d, _ = simple_recon_person(keypoints2d_all[nf], dataset.Pall)
            kp3ds_refined[nf] = keypoints3d

    return kp3ds_refined


def mv1pmf_skel(dataset, check_repro=True, args=None):
    MIN_CONF_THRES = args.thres2d
    no_img = not (args.vis_det or args.vis_repro)
    dataset.no_img = no_img
    kp3ds = []
    keypoints2d_all = []
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
        keypoints2d_all.append(annots['keypoints'].copy())
        if args.vis_det:
            dataset.vis_detections(images, annots, nf, sub_vis=args.sub_vis)
        if args.vis_repro:
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis)

    np.save(join(args.out, "reprojection_error.npy"), np.array(repro_errors))
    log_time(f"Average reprojection error over sequence: {np.nanmean(repro_errors):.2f}px")    # smooth the skeleton
    keypoints2d_all = np.stack(keypoints2d_all)
    kp3ds = np.stack(kp3ds)

    if args.joint_ba:
        log_time("Running joint BA (optimize camera extrinsics + 3D skeleton)...")
        kp3ds = joint_ba_refine(dataset, keypoints2d_all, kp3ds, args)
        repro_errors_ba = []
        for nf in range(kp3ds.shape[0]):
            repro_errors_ba.append(_compute_repro_error(kp3ds[nf], keypoints2d_all[nf], dataset.Pall))
        np.save(join(args.out, "reprojection_error_joint_ba.npy"), np.array(repro_errors_ba))
        log_time(f"Average reprojection error after joint BA: {np.nanmean(repro_errors_ba):.2f}px")

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
    # optimize the human shape
    with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
        body_model = load_model(gender=args.gender, model_type=args.model)
    params = smpl_from_keypoints3d2d(body_model, kp3ds, keypoints2d, bboxes, 
        dataset.Pall, config=dataset.config, args=args,
        weight_shape=weight_shape, weight_pose=weight_pose)
    # write out the results
    dataset.no_img = not (args.vis_smpl or args.vis_repro)
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        param = select_nf(params, nf-start)
        dataset.write_smpl(param, nf)
        if args.write_smpl_full:
            param_full = param.copy()
            param_full['poses'] = body_model.full_poses(param['poses'])
            dataset.write_smpl(param_full, nf, mode='smpl_full')
        if args.write_vertices:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            write_data = [{'id': 0, 'vertices': vertices[0]}]
            dataset.write_vertices(write_data, nf)
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **param)
            dataset.vis_smpl(vertices=vertices[0], faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis, add_back=True)
        if args.vis_repro:
            keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
            kpts_repro = projectN3(keypoints, dataset.Pall)
            dataset.vis_repro(images, kpts_repro, nf=nf, sub_vis=args.sub_vis, mode='repro_smpl')


if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    from easymocap.mytools.debug_utils import log, log_time, mywarn, myerror

    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    parser.add_argument('--joint_ba', action='store_true',
                        help='Jointly optimize camera extrinsics and 3D skeleton before SMPL fitting')
    parser.add_argument('--joint_ba_step', type=int, default=1,
                        help='Frame sampling step for joint BA')
    parser.add_argument('--joint_ba_max_frames', type=int, default=200,
                        help='Maximum number of frames used by joint BA; -1 for all selected frames')
    parser.add_argument('--joint_ba_loss', type=str, default='huber',
                        choices=['linear', 'huber', 'soft_l1', 'cauchy', 'arctan'],
                        help='Robust loss for joint BA')
    parser.add_argument('--joint_ba_f_scale', type=float, default=2.0,
                        help='Robust loss scale for joint BA')
    parser.add_argument('--joint_ba_max_nfev', type=int, default=50,
                        help='Maximum function evaluations for joint BA solver')
    parser.add_argument('--joint_ba_lambda_cam', type=float, default=1e-3,
                        help='L2 regularization weight to keep cameras near initial extrinsics')
    parser.add_argument('--joint_ba_skip_retriangulate', action='store_true',
                        help='Skip re-triangulating all frames after joint BA for speed')
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
    