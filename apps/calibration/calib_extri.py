"""
File: apps/calibration/calib_extri.py
"""
from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from easymocap.mytools.camera_utils import write_intri
import os
from glob import glob
from os.path import join
import numpy as np
import cv2
from easymocap.mytools import read_intri, write_extri, read_json
from easymocap.mytools.debug_utils import mywarn
from pathlib import Path

# Repo-root default intrinsics (Google Pixel 7 @ 4K); resolved from this file so cwd does not matter.
_DEFAULT_INTRI = str(
    Path(__file__).resolve().parent.parent.parent / "config/calibration/google_pixel_7_4k/intri.yml"
)


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

    # read_intri only loads keys listed in YAML `names`. If that list is shorter
    # than folders under path/image, we would KeyError later unless we expand.
    if len(intri.keys()) == 1:
        key0 = list(intri.keys())[0]
        for cam in camnames:
            intri[cam] = intri[key0].copy()
        return intri

    missing = [c for c in camnames if c not in intri]
    if missing:
        ref = next((c for c in camnames if c in intri), None)
        if ref is None:
            raise RuntimeError(
                'intri cameras {} do not match dataset folders {} under {}/{}.'.format(
                    list(intri.keys()), camnames, path, image))
        mywarn(
            'intri has no K/dist for {}; copying from camera {}. '
            'Add those cameras to YAML `names` (and K_*/dist_*) to avoid this.'.format(
                ', '.join(missing), ref))
        for cam in missing:
            intri[cam] = intri[ref].copy()

    extra = [k for k in intri.keys() if k not in camnames]
    if extra:
        mywarn(
            'intri lists cameras not present under {}/{}: {} (unused for this run).'.format(
                path, image, ', '.join(extra)))

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


def _points_are_collinear(pts, tol=1e-6):
    """Check if a set of 2D or 3D points are (nearly) collinear.
    Uses SVD on the centered point cloud: if the second singular value
    is negligible relative to the first, the points lie on a line."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        return True
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    return s[1] < tol * s[0] if s[0] > tol else True


def _stereo_per_frame_errors(objectPoints, imagePoints_prev, imagePoints_curr,
                             K_prev, dist_prev, K_curr, dist_curr, R_rel, T_rel):
    """Return array of mean reprojection error per frame (prev->curr projection)."""
    rvec_rel = cv2.Rodrigues(R_rel)[0]
    errors = []
    for obj, img_p, img_c in zip(objectPoints, imagePoints_prev, imagePoints_curr):
        _, rv, tv = cv2.solvePnP(obj, img_p, K_prev, dist_prev,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
        rv2, tv2 = cv2.composeRT(rv, tv, rvec_rel, T_rel)[:2]
        proj, _ = cv2.projectPoints(obj, rv2, tv2, K_curr, dist_curr)
        errors.append(float(np.linalg.norm(proj.squeeze() - img_c, axis=1).mean()))
    return np.array(errors)


def calib_extri_stereo(path, image, intri_arg, step=6, use_distortion=False,
                       cameras_filter=None, debug=False, err_threshold=50.0):
    camnames = sorted(os.listdir(join(path, image)))
    camnames = [c for c in camnames if os.path.isdir(join(path, image, c))]
    intri = load_intri(path, image, camnames, intri_arg)
    if cameras_filter:
        camnames = [c for c in camnames if c in cameras_filter]
        print(f'[Stereo] Using camera subset: {camnames}')
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
            used_frame_names = []

            for name in common_names:
                data_prev = read_json(prev_map[name])
                data_curr = read_json(curr_map[name])

                k3d_prev = np.array(data_prev['keypoints3d'], np.float32)
                k2d_prev = np.array(data_prev['keypoints2d'], np.float32)

                k3d_curr = np.array(data_curr['keypoints3d'], np.float32)
                k2d_curr = np.array(data_curr['keypoints2d'], np.float32)

                valid_prev = k2d_prev[:, 2] > 0
                valid_curr = k2d_curr[:, 2] > 0
                valid = valid_prev & valid_curr

                if valid.sum() < 6:
                    skipped_pairs += 1
                    continue

                pts_valid = k3d_prev[valid]
                if _points_are_collinear(pts_valid[:, :2]):
                    if debug:
                        print(f'  [skip] {name}: {int(valid.sum())} points are collinear')
                    skipped_pairs += 1
                    continue

                objectPoints.append(pts_valid)
                imagePoints_prev.append(k2d_prev[valid, :2])
                imagePoints_curr.append(k2d_curr[valid, :2])
                used_frame_names.append(name)
                used_pairs += 1
            print(f'[Stereo] {prev_cam} -> {cam}: used_pairs={used_pairs}, skipped_pairs={skipped_pairs}')
            if len(objectPoints) == 0:
                raise RuntimeError(f"No valid stereo pairs for {prev_cam} -> {cam}")

            # Iterative stereo calibration with outlier rejection
            K_prev_cam = intri[prev_cam]['K']
            dist_prev_cam = intri[prev_cam]['dist']
            keep = list(range(len(objectPoints)))

            for iteration in range(10):
                obj_k = [objectPoints[i] for i in keep]
                imp_k = [imagePoints_prev[i] for i in keep]
                imc_k = [imagePoints_curr[i] for i in keep]

                _, _, _, _, _, R_rel, T_rel, _, _ = cv2.stereoCalibrate(
                    obj_k, imp_k, imc_k,
                    K_prev_cam, dist_prev_cam, K, dist, None,
                    flags=cv2.CALIB_FIX_INTRINSIC
                )

                frame_errors = _stereo_per_frame_errors(
                    obj_k, imp_k, imc_k,
                    K_prev_cam, dist_prev_cam, K, dist, R_rel, T_rel)

                outliers = np.where(frame_errors > err_threshold)[0]
                if len(outliers) == 0:
                    break

                rejected_names = [used_frame_names[keep[i]] for i in outliers]
                rejected_errs = frame_errors[outliers]
                for rn, re in zip(rejected_names, rejected_errs):
                    print(f'  [outlier iter={iteration}] {rn} mean_err={re:.2f} > {err_threshold} => removed')
                keep = [keep[i] for i in range(len(keep)) if i not in outliers]
                if len(keep) == 0:
                    raise RuntimeError(f"All frames rejected as outliers for {prev_cam} -> {cam}")

            if len(keep) < used_pairs:
                print(f'[Stereo] {prev_cam} -> {cam}: kept {len(keep)}/{used_pairs} frames after outlier rejection')

            # Final per-frame errors for reporting
            obj_k = [objectPoints[i] for i in keep]
            imp_k = [imagePoints_prev[i] for i in keep]
            imc_k = [imagePoints_curr[i] for i in keep]
            frame_errors = _stereo_per_frame_errors(
                obj_k, imp_k, imc_k,
                K_prev_cam, dist_prev_cam, K, dist, R_rel, T_rel)

            if debug:
                for fi, ki in enumerate(keep):
                    n_pts = len(obj_k[fi])
                    print(f'  [{used_frame_names[ki]}] n_pts={n_pts}, mean_err={frame_errors[fi]:.2f}')

            total_points = sum(len(o) for o in obj_k)
            err = float(np.sum(frame_errors * np.array([len(o) for o in obj_k])) / total_points)

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
        default=_DEFAULT_INTRI,
        help=(
            'Path to intri YAML. Default: bundled config/calibration/google_pixel_7_4k/intri.yml. '
            'If that file is missing, falls back to <path>/intri.yml or init_intri heuristic.'
        ),
    )
    parser.add_argument('--ext', type=str, default='.jpg')
    parser.add_argument('--step', type=int, default=6)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tryfocal', action='store_true')
    parser.add_argument('--tryextri', action='store_true')
    parser.add_argument('--image_id', type=int, default=0, help='Image id used for extrinsic calibration')
    parser.set_defaults(stereo=True)
    parser.add_argument(
        '--no-stereo',
        action='store_false',
        dest='stereo',
        help='Use single-frame extrinsic calibration instead of adjacent-camera stereo (default: stereo on).',
    )
    parser.add_argument(
        '--undis',
        action='store_true',
        help='Use distortion coefficients from intri.yml. Default: ignore distortion (zeros).',
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=None,
        help='Only calibrate these cameras (e.g. --cameras 02 03)',
    )
    parser.add_argument(
        '--err_threshold',
        type=float,
        default=50.0,
        help='Per-frame mean reprojection error threshold (px) for outlier rejection in stereo mode.',
    )

    args = parser.parse_args()
    if args.intri == _DEFAULT_INTRI and not os.path.isfile(_DEFAULT_INTRI):
        mywarn(
            'Default intri not found at {}; using <path>/intri.yml or init_intri heuristic.'.format(
                _DEFAULT_INTRI
            )
        )
        args.intri = None

    if args.stereo:
        calib_extri_stereo(
            args.path,
            args.image,
            intri_arg=args.intri,
            step=args.step,
            use_distortion=args.undis,
            cameras_filter=args.cameras,
            debug=args.debug,
            err_threshold=args.err_threshold,
        )
    else:
        calib_extri(
            args.path,
            args.image,
            intri_arg=args.intri,
            image_id=args.image_id,
            use_distortion=args.undis,
        )