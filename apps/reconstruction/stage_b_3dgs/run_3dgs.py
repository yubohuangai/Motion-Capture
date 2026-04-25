"""
End-to-end wrapper: export calibration -> run 3D Gaussian Splatting.

Supports single-frame and batch (multi-frame) reconstruction.

Prerequisites:
    Clone and install the official 3DGS repo:
        git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
        cd gaussian-splatting
        pip install submodules/diff-gaussian-rasterization submodules/simple-knn

Usage:
    # Single frame
    python apps/reconstruction/stage_b_3dgs/run_3dgs.py /path/to/data \\
        --frame 0 --output /path/to/recon \\
        --gs_repo /path/to/gaussian-splatting

    # Batch over frames 0-99
    python apps/reconstruction/stage_b_3dgs/run_3dgs.py /path/to/data \\
        --frame_start 0 --frame_end 100 --output /path/to/recon \\
        --gs_repo /path/to/gaussian-splatting

    # With masks and undistortion
    python apps/reconstruction/stage_b_3dgs/run_3dgs.py /path/to/data \\
        --frame 0 --output /path/to/recon --undistort --mask masks \\
        --gs_repo /path/to/gaussian-splatting
"""

import argparse
import os
import subprocess
import sys
from os.path import join


def run_export(data, output, frame, intri, extri, ext, undistort, mask,
               triangulate=True, colmap_bin='colmap', gpu=False):
    """Run export_colmap.py for one frame (lives in stage_a_colmap/)."""
    script = join(os.path.dirname(__file__), '..', 'stage_a_colmap', 'export_colmap.py')
    cmd = [
        sys.executable, script, data,
        '--output', output,
        '--frame', str(frame),
        '--intri', intri,
        '--extri', extri,
        '--ext', ext,
        '--colmap', colmap_bin,
    ]
    if undistort:
        cmd.append('--undistort')
    if mask:
        cmd.extend(['--mask', mask])
    if triangulate:
        cmd.append('--triangulate')
    if gpu:
        cmd.append('--gpu')

    print(f'\n{"="*60}')
    print(f'[run_3dgs] Exporting frame {frame} -> {output}')
    print(f'{"="*60}')
    subprocess.check_call(cmd)


def run_train(gs_repo, source_path, output_path, iterations, gpu):
    """Run 3DGS training."""
    train_script = join(gs_repo, 'train.py')
    if not os.path.exists(train_script):
        print(f'ERROR: {train_script} not found.')
        print(f'Clone the 3DGS repo: git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive')
        sys.exit(1)

    cmd = [
        sys.executable, train_script,
        '-s', source_path,
        '-m', output_path,
        '--iterations', str(iterations),
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print(f'\n{"="*60}')
    print(f'[run_3dgs] Training 3DGS: {" ".join(cmd)}')
    print(f'[run_3dgs] GPU: {gpu}')
    print(f'{"="*60}')
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end: export calibration + run 3D Gaussian Splatting',
    )
    parser.add_argument('data', help='Root data path (images/, intri.yml, extri.yml)')
    parser.add_argument('--output', '-o', required=True,
                        help='Base output directory')
    parser.add_argument('--gs_repo', required=True,
                        help='Path to gaussian-splatting repo root')

    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument('--frame', type=int, default=None,
                             help='Single frame to reconstruct')
    frame_group.add_argument('--frame_start', type=int, default=None,
                             help='First frame for batch mode')

    parser.add_argument('--frame_end', type=int, default=None,
                        help='Last frame (exclusive) for batch mode')
    parser.add_argument('--frame_step', type=int, default=1,
                        help='Frame step for batch mode')

    parser.add_argument('--intri', default='intri.yml')
    parser.add_argument('--extri', default='extri.yml')
    parser.add_argument('--ext', default='.jpg')
    parser.add_argument('--undistort', action='store_true')
    parser.add_argument('--mask', default=None,
                        help='Mask sub-directory name')
    parser.add_argument('--no_triangulate', action='store_true',
                        help='Skip COLMAP triangulation (use empty points3D)')
    parser.add_argument('--colmap', default='colmap',
                        help='Path to COLMAP binary')
    parser.add_argument('--iterations', type=int, default=7000,
                        help='3DGS training iterations (default: 7000)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (default: 0)')
    parser.add_argument('--colmap_gpu', action='store_true',
                        help='Use GPU for COLMAP feature extraction/matching')
    parser.add_argument('--skip_export', action='store_true',
                        help='Skip export step (use existing COLMAP workspace)')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training step (export only)')
    args = parser.parse_args()

    if args.frame is not None:
        frames = [args.frame]
    elif args.frame_start is not None:
        if args.frame_end is None:
            parser.error('--frame_end is required with --frame_start')
        frames = list(range(args.frame_start, args.frame_end, args.frame_step))
    else:
        frames = [0]

    for frame in frames:
        if len(frames) == 1:
            colmap_dir = join(args.output, 'colmap')
            model_dir = join(args.output, 'model')
        else:
            colmap_dir = join(args.output, f'frame_{frame:06d}', 'colmap')
            model_dir = join(args.output, f'frame_{frame:06d}', 'model')

        if not args.skip_export:
            run_export(
                args.data, colmap_dir, frame,
                args.intri, args.extri, args.ext,
                args.undistort, args.mask,
                triangulate=not args.no_triangulate,
                colmap_bin=args.colmap, gpu=args.colmap_gpu,
            )

        if not args.skip_train:
            run_train(args.gs_repo, colmap_dir, model_dir, args.iterations, args.gpu)

    print(f'\n{"="*60}')
    print(f'[run_3dgs] All done. {len(frames)} frame(s) processed.')
    print(f'[run_3dgs] Results at: {args.output}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
