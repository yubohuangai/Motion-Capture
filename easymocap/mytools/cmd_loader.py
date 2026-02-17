'''
  @ Date: 2021-01-15 12:09:27
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-15 11:09:36
  @ FilePath: /EasyMocap/easymocap/mytools/cmd_loader.py
'''
import os
import argparse

def load_parser():
    parser = argparse.ArgumentParser('EasyMocap commond line tools')
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--camera', type=str, default=None)
    parser.add_argument('--annot', type=str, default='annots', help="sub directory name to store the generated annotation files, default to be annots")
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--from_file', type=str, default=None)
    parser.add_argument('--pid', type=int, nargs='+', default=[0],
        help='the person IDs')
    parser.add_argument('--max_person', type=int, default=-1,
        help='maximum number of person')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=100000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    # 
    # keypoints and body model
    # 
    parser.add_argument('--cfg_model', type=str, default=None)
    parser.add_argument('--body', type=str, default='body25', choices=['body15', 'body25', 'h36m', 'bodyhand', 'bodyhandface', 'handl', 'handr', 'total'])
    parser.add_argument('--model', type=str, default='smpl', choices=['smpl', 'smplh', 'smplx', 'manol', 'manor'])
    parser.add_argument('--gender', type=str, default='neutral', 
        choices=['neutral', 'male', 'female'])
    # Input control
    detec = parser.add_argument_group('Detection control')
    detec.add_argument("--thres2d", type=float, default=0.3, 
        help="The threshold for suppress noisy kpts")
    # 
    # Optimization control
    # 
    recon = parser.add_argument_group('Reconstruction control')
    recon.add_argument('--smooth3d', type=int,
        help='the size of window to smooth keypoints3d', default=0)
    recon.add_argument('--MAX_REPRO_ERROR', type=int,
        help='The threshold of reprojection error', default=50)
    recon.add_argument('--MAX_SPEED_ERROR', type=int,
        help='The threshold of reprojection error', default=50)
    recon.add_argument('--robust3d', action='store_true')
    recon.add_argument('--shape_silhouette', action='store_true',
        help='refine shape using silhouette Chamfer loss after initial pose fitting')
    recon.add_argument('--shape_mask_root', type=str, default='masks',
        help='sub directory name for silhouette masks, under sequence root')
    recon.add_argument('--shape_mask_thr', type=int, default=127,
        help='binary threshold for silhouette masks')
    recon.add_argument('--shape_mask_max_points', type=int, default=2000,
        help='maximum sampled silhouette points per frame-view')
    recon.add_argument('--shape_mask_frame_step', type=int, default=1,
        help='use one frame every N frames for silhouette shape refinement')
    recon.add_argument('--shape_silhouette_iters', type=int, default=20,
        help='LBFGS max_iter for shape silhouette refinement')
    recon.add_argument('--shape_silhouette_vert_subsample', type=int, default=1000,
        help='number of mesh vertices sampled for silhouette Chamfer')
    recon.add_argument('--shape_vis_max_mesh_points', type=int, default=4000,
        help='maximum projected mesh points to draw per view in silhouette debug overlays')
    # 
    # visualization part
    # 
    output = parser.add_argument_group('Output control')
    output.add_argument('--vis_det', action='store_true')
    output.add_argument('--vis_repro', action='store_true')
    output.add_argument('--vis_smpl', action='store_true')
    output.add_argument('--vis_shape_silhouette', action='store_true',
        help='save overlays of mask points and reprojected SMPL mesh points')
    output.add_argument('--write_smpl_full', action='store_true')
    parser.add_argument('--write_vertices', action='store_true')
    output.add_argument('--vis_mask', action='store_true')
    output.add_argument('--undis', action='store_true')
    output.add_argument('--sub_vis', type=str, nargs='+', default=[],
        help='the sub folder lists for visualization')
    # 
    # debug
    # 
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_origin', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--no_opt', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--opts',
                        help="Modify config options using the command-line",
                        default=[],
                        nargs='+')
    parser.add_argument('--cfg_opts',
                        help="Modify config options using the command-line",
                        default=[],
                        nargs='+')
    return parser

from os.path import join
def save_parser(args):
    import yaml
    res = vars(args)
    os.makedirs(args.out, exist_ok=True)
    with open(join(args.out, 'exp.yml'), 'w') as f:
        yaml.dump(res, f)

def parse_parser(parser):
    args = parser.parse_args()
    if args.out is None:
        print(' - [Warning] Please specify the output path `--out ${out}`')
        print(' - [Warning] Default to {}/output'.format(args.path))
        args.out = join(args.path, 'output')
    if args.from_file is not None:
        assert os.path.exists(args.from_file), args.from_file
        with open(args.from_file) as f:
            datas = f.readlines()
            subs = [d for d in datas if not d.startswith('#')]
            subs = [d.rstrip().replace('https://www.youtube.com/watch?v=', '') for d in subs]
        newsubs = sorted(os.listdir(join(args.path, 'images')))
        clips = []
        for newsub in newsubs:
            if newsub.split('+')[0] in subs:
                clips.append(newsub)
        for sub in subs:
            if os.path.exists(join(args.path, 'images', sub)):
                clips.append(sub)
        args.sub = clips
    if len(args.sub) == 0 and os.path.exists(join(args.path, 'images')):
        args.sub = sorted(os.listdir(join(args.path, 'images')))
        if args.sub[0].isdigit():
            args.sub = sorted(args.sub, key=lambda x:int(x))
    args.opts = {args.opts[2*i]:float(args.opts[2*i+1]) for i in range(len(args.opts)//2)}
    save_parser(args)
    return args