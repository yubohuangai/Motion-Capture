"""
File: apps/calibration/calib_extri.py
"""
from easymocap.mytools.camera_utils import write_intri
import os
from glob import glob
from os.path import join
import numpy as np
import cv2
from easymocap.mytools import read_intri, write_extri, read_json
from easymocap.mytools.debug_utils import mywarn

def init_intri(path, image):
    camnames = sorted(os.listdir(join(path, image)))
    cameras = {}
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, image, cam, '*.jpg')))
        assert len(imagenames) > 0
        imgname = imagenames[0]
        img = cv2.imread(imgname)
        height, width = img.shape[0], img.shape[1]
        focal = 1.2*max(height, width) # as colmap
        K = np.array([focal, 0., width/2, 0., focal, height/2, 0. ,0., 1.]).reshape(3, 3)
        dist = np.zeros((1, 5))
        cameras[cam] = {
            'K': K,
            'dist': dist
        }
    return cameras


def load_intri(path, image, camnames, intri_arg):
    """
    intri_arg: None -> try join(path, 'intri.yml'); if missing, init_intri heuristic.
    intri_arg: str -> must exist (explicit path from --intri).
    """
    if intri_arg is None:
        p = join(path, 'intri.yml')
        if os.path.isfile(p):
            intri = read_intri(p)
        else:
            mywarn('No intri.yml at {}, using init_intri heuristic'.format(p))
            return init_intri(path, image)
    else:
        assert os.path.isfile(intri_arg), intri_arg
        intri = read_intri(intri_arg)
    if len(intri.keys()) == 1:
        key0 = list(intri.keys())[0]
        for cam in camnames:
            intri[cam] = intri[key0].copy()
    return intri


def apply_intri_distortion_mode(intri, use_distortion):
    """If use_distortion is False, set all cameras' dist to zeros (pinhole model)."""
    if use_distortion:
        return
    for cam in intri:
        d = np.asarray(intri[cam]['dist'])
        intri[cam]['dist'] = np.zeros_like(d)


def solvePnP(k3d, k2d, K, dist, flag, tryextri=False):
    k2d = np.ascontiguousarray(k2d[:, :2])
    # try different initial values:
    if tryextri:
        def closure(rvec, tvec):
            ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, rvec, tvec, True, flags=flag)
            points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
            kpts_repro = points2d_repro.squeeze()
            err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
            return err, rvec, tvec, kpts_repro
        # create a series of extrinsic parameters looking at the origin
        height_guess = 2.1
        radius_guess = 7.
        infos = []
        for theta in np.linspace(0, 2*np.pi, 180):
            st = np.sin(theta)
            ct = np.cos(theta)
            center = np.array([radius_guess*ct, radius_guess*st, height_guess]).reshape(3, 1)
            R = np.array([
                [-st, ct,  0],
                [0,    0, -1],
                [-ct, -st, 0]
            ])
            tvec = - R @ center
            rvec = cv2.Rodrigues(R)[0]
            err, rvec, tvec, kpts_repro = closure(rvec, tvec)
            infos.append({
                'err': err,
                'repro': kpts_repro,
                'rvec': rvec,
                'tvec': tvec
            })
        infos.sort(key=lambda x:x['err'])
        err, rvec, tvec, kpts_repro = infos[0]['err'], infos[0]['rvec'], infos[0]['tvec'], infos[0]['repro']
    else:
        ret, rvec, tvec = cv2.solvePnP(k3d, k2d, K, dist, flags=flag)
        points2d_repro, xxx = cv2.projectPoints(k3d, rvec, tvec, K, dist)
        kpts_repro = points2d_repro.squeeze()
        err = np.linalg.norm(points2d_repro.squeeze() - k2d, axis=1).mean()
    # print(err)
    return err, rvec, tvec, kpts_repro


def relative2world(R_rel, T_rel, R_prev, T_prev):
    """
    Convert relative camera extrinsics (R_rel, T_rel) w.r.t previous camera 
    to world coordinates (same origin as first camera)
    """
    # R_prev, T_prev: previous camera in world coordinates
    # R_rel, T_rel: current camera relative to previous camera
    # World rotation and translation
    R_world = R_rel @ R_prev
    T_world = R_rel @ T_prev + T_rel
    rvec_world = cv2.Rodrigues(R_world)[0]
    return rvec_world, T_world


def sample_list(lst, step):
    if step <= 1:
        return lst
    return lst[::step]


def calib_extri_stereo(path, image, intri_arg, step=6, use_distortion=False):
    camnames = sorted(os.listdir(join(path, image)))
    camnames = [c for c in camnames if os.path.isdir(join(path, image, c))]
    intri = load_intri(path, image, camnames, intri_arg)
    apply_intri_distortion_mode(intri, use_distortion)
    extri = {}

    for ic, cam in enumerate(camnames):
        if ic == 0:
            # World frame = first camera frame (not the calibration board frame).
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            err = 0.0

        elif ic > 0:
            K, dist = intri[cam]['K'], intri[cam]['dist']
            prev_cam = camnames[ic - 1]

            prev_jsons = sorted(glob(join(path, 'chessboard', prev_cam, '*.json')))
            curr_jsons = sorted(glob(join(path, 'chessboard', cam, '*.json')))

            prev_jsons = sample_list(prev_jsons, step)
            curr_jsons = sample_list(curr_jsons, step)

            # match by filename (not index)
            prev_map = {os.path.basename(p): p for p in prev_jsons}
            curr_map = {os.path.basename(p): p for p in curr_jsons}
            common_names = sorted(set(prev_map.keys()) & set(curr_map.keys()))

            if len(common_names) == 0:
                raise RuntimeError(f"No matching chessboard files between {prev_cam} and {cam}")

            objectPoints = []
            imagePoints_prev = []
            imagePoints_curr = []

            used_pairs = 0
            skipped_pairs = 0

            for name in common_names:
                data_prev = read_json(prev_map[name])
                data_curr = read_json(curr_map[name])

                k3d_prev = np.array(data_prev['keypoints3d'], np.float32)
                k2d_prev = np.array(data_prev['keypoints2d'], np.float32)

                k3d_curr = np.array(data_curr['keypoints3d'], np.float32)
                k2d_curr = np.array(data_curr['keypoints2d'], np.float32)

                # keep only visible points
                valid_prev = k2d_prev[:, 2] > 0
                valid_curr = k2d_curr[:, 2] > 0
                valid = valid_prev & valid_curr

                # stereoCalibrate (OpenCV 4.x) uses a DLT path that needs >= 6
                # 3D–2D correspondences per view, not 4.
                if valid.sum() < 6:
                    skipped_pairs += 1
                    continue

                objectPoints.append(k3d_prev[valid])
                imagePoints_prev.append(k2d_prev[valid, :2])
                imagePoints_curr.append(k2d_curr[valid, :2])
                used_pairs += 1
            print(f'[Stereo] {prev_cam} -> {cam}: used_pairs={used_pairs}, skipped_pairs={skipped_pairs}')
            if len(objectPoints) == 0:
                raise RuntimeError(f"No valid stereo pairs for {prev_cam} -> {cam}")

            _, _, _, _, _, R_rel, T_rel, _, _ = cv2.stereoCalibrate(
                objectPoints,
                imagePoints_prev,
                imagePoints_curr,
                intri[prev_cam]['K'],
                intri[prev_cam]['dist'],
                K,
                dist,
                None,
                flags=cv2.CALIB_FIX_INTRINSIC
            )

            # --- Reprojection error in stereo (prev → curr) frame ---
            total_err = 0.0
            total_points = 0

            rvec_rel = cv2.Rodrigues(R_rel)[0]

            for obj_prev, img_pre, img_curr in zip(objectPoints, imagePoints_prev, imagePoints_curr):
                _, rvec_pre, tvec_pre = cv2.solvePnP(obj_prev, img_pre, intri[prev_cam]['K'], intri[prev_cam]['dist'], flags=cv2.SOLVEPNP_ITERATIVE)
                rvec_curr, tvec_curr = cv2.composeRT(rvec_pre, tvec_pre, rvec_rel, T_rel)[:2]

                proj, _ = cv2.projectPoints(
                    obj_prev,
                    rvec_curr,
                    tvec_curr,
                    K,
                    dist
                )
                proj = proj.squeeze()
                err = np.linalg.norm(proj - img_curr, axis=1)
                total_err += err.sum()
                total_points += len(err)

            err = total_err / total_points

            # Convert to world coordinates
            rvec, tvec = relative2world(
                R_rel, T_rel, extri[prev_cam]['R'], extri[prev_cam]['T']
            )

        extri[cam] = {}
        extri[cam]['Rvec'] = rvec
        extri[cam]['R'] = cv2.Rodrigues(rvec)[0]
        extri[cam]['T'] = tvec
        center = - extri[cam]['R'].T @ tvec
        print('{} center => {}, err = {:.3f}'.format(cam, center.squeeze(), err))
    write_intri(join(path, 'intri.yml'), intri)
    write_extri(join(path, 'extri.yml'), extri)


def calib_extri(path, image, intri_arg, image_id, use_distortion=False):
    camnames = sorted(os.listdir(join(path, image)))
    camnames = [c for c in camnames if os.path.isdir(join(path, image, c))]
    intri = load_intri(path, image, camnames, intri_arg)
    apply_intri_distortion_mode(intri, use_distortion)
    extri = {}
    # methods = [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_SQPNP]
    methods = [cv2.SOLVEPNP_ITERATIVE]
    for ic, cam in enumerate(camnames):
        imagenames = sorted(glob(join(path, image, cam, '*{}'.format(args.ext))))
        chessnames = sorted(glob(join(path, 'chessboard', cam, '*.json')))
        # chessname = chessnames[0]
        assert len(chessnames) > 0, cam
        chessname = chessnames[image_id]

        data = read_json(chessname)
        k3d = np.array(data['keypoints3d'], dtype=np.float32)
        k2d = np.array(data['keypoints2d'], dtype=np.float32)
        if k3d.shape[0] != k2d.shape[0]:
            mywarn('k3d {} doesnot match k2d {}'.format(k3d.shape, k2d.shape))
            length = min(k3d.shape[0], k2d.shape[0])
            k3d = k3d[:length]
            k2d = k2d[:length]
        # k3d[:, 0] *= -1
        valididx = k2d[:, 2] > 0
        if valididx.sum() < 4:
            extri[cam] = {}
            rvec = np.zeros((1, 3))
            tvec = np.zeros((3, 1))
            extri[cam]['Rvec'] = rvec
            extri[cam]['R'] = cv2.Rodrigues(rvec)[0]
            extri[cam]['T'] = tvec
            print('[ERROR] Failed to initialize the extrinsic parameters')
            extri.pop(cam)
            continue
        k3d = k3d[valididx]
        k2d = k2d[valididx]
        if args.tryfocal:
            infos = []
            for focal in range(500, 5000, 10):
                dist = intri[cam]['dist']
                K = intri[cam]['K']
                K[0, 0] = focal
                K[1, 1] = focal
                for flag in methods:
                    err, rvec, tvec, kpts_repro = solvePnP(k3d, k2d, K, dist, flag)
                    infos.append({
                        'focal': focal,
                        'err': err,
                        'repro': kpts_repro,
                        'rvec': rvec,
                        'tvec': tvec
                    })
            infos.sort(key=lambda x: x['err'])
            err, rvec, tvec = infos[0]['err'], infos[0]['rvec'], infos[0]['tvec']
            kpts_repro = infos[0]['repro']
            focal = infos[0]['focal']
            intri[cam]['K'][0, 0] = focal
            intri[cam]['K'][1, 1] = focal
        else:
            K, dist = intri[cam]['K'], intri[cam]['dist']
            err, rvec, tvec, kpts_repro = solvePnP(k3d, k2d, K, dist, flag=cv2.SOLVEPNP_ITERATIVE)
        extri[cam] = {}
        extri[cam]['Rvec'] = rvec
        extri[cam]['R'] = cv2.Rodrigues(rvec)[0]
        extri[cam]['T'] = tvec
        center = - extri[cam]['R'].T @ tvec
        print('{} center => {}, err = {:.3f}'.format(cam, center.squeeze(), err))
    write_intri(join(path, 'intri.yml'), intri)
    write_extri(join(path, 'extri.yml'), extri)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument(
        '--intri',
        type=str,
        default=None,
        help='Path to intri YAML. If omitted, uses <path>/intri.yml when present, else init_intri heuristic.',
    )
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--step', type=int, default=6)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tryfocal', action='store_true')
    parser.add_argument('--tryextri', action='store_true')
    parser.add_argument('--image_id', type=int, default=0, help='Image id used for extrinsic calibration')
    parser.add_argument('--stereo', action='store_true', help='Use stereo calibration for adjacent cameras')
    parser.add_argument(
        '--undis',
        action='store_true',
        help='Use distortion coefficients from intri.yml. Default: ignore distortion (zeros).',
    )

    args = parser.parse_args()
    if args.stereo:
        calib_extri_stereo(
            args.path,
            args.image,
            intri_arg=args.intri,
            step=args.step,
            use_distortion=args.undis,
        )
    else:
        calib_extri(
            args.path,
            args.image,
            intri_arg=args.intri,
            image_id=args.image_id,
            use_distortion=args.undis,
        )