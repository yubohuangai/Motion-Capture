'''
  @ Date: 2021-04-13 20:12:58
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-06-25 22:14:58
  @ FilePath: /EasyMocapRelease/easymocap/pipeline/weight.py
'''
def load_weight_shape(model, opts):
    if model in ['smpl', 'smplh', 'smplx']:
        weight = {'s3d': 1., 'reg_shapes': 5e-3, 'chamfer': 5e-7}
    elif model == 'mano':
        weight = {'s3d': 1e2, 'reg_shapes': 5e-5}
    else:
        raise NotImplementedError
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight

def load_weight_shape_refine(model, opts):
    """Weights for shape refinement with pose (optimizeShapeWithPose).

    k3d_shape  – direct 3D joint position matching (much richer than bone lengths).
    k2d_shape  – multi-view 2D reprojection (uses image evidence from all cameras).
    chamfer    – silhouette Chamfer loss (captures body width/depth if masks available).
    reg_shapes – L2 prior on betas.
    init_shape – stay close to the initial bone-length-based shape estimate.
    """
    if model in ['smpl', 'smplh', 'smplx']:
        weight = {
            'k3d_shape': 0.,
            'k2d_shape': 1e-4,
            'chamfer': 5e-2,
            'reg_shapes': 5e-3,
            'init_shape': 5e-3,
        }
    elif model == 'mano':
        weight = {
            'k3d_shape': 1e2,
            'k2d_shape': 1e-3,
            'chamfer': 0.,
            'reg_shapes': 5e-5,
            'init_shape': 1e-1,
        }
    else:
        raise NotImplementedError
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight


def load_weight_pose(model, opts):
    if model == 'smpl':
        # weight = {
        #     'k3d': 1., 'reg_poses_zero': 1e-2, 'smooth_body': 2e-1,
        #     'smooth_poses': 1e-1, 'reg_poses': 1e-3,
        #     'k2d': 1e-4
        # }
        weight = {
            'k3d': 1., 'reg_poses_zero': 0, 'smooth_body': 0,
            'smooth_poses': 0, 'reg_poses': 0,
            'k2d': 1e-4
        }
    elif model == 'smplh':
        weight = {
            'k3d': 1., 'k3d_hand': 5.,
            'reg_poses_zero': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1, 'smooth_hand': 1e-3,
            'reg_hand': 1e-4,
            'k2d': 1e-4
        }
    elif model == 'smplx':
        weight = {
            'k3d': 1., 'k3d_hand': 5., 'k3d_face': 2.,
            'reg_poses_zero': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1, 'smooth_hand': 1e-3,
            'reg_hand': 1e-4, 'reg_expr': 1e-2, 'reg_head': 1e-2,
            'k2d': 1e-4
        }
    elif model == 'mano':
        weight = {
            'k3d': 1e2, 'k2d': 2e-3,
            'reg_poses': 1e-3, 'smooth_body': 1e2,
            # 'collision': 1  # If the frame number is too large (more than 1000), then GPU oom
        }
        # weight = {
        #     'k3d': 1., 'k2d': 1e-4,
        #     'reg_poses': 1e-4, 'smooth_body': 0
        # }
    else:
        print(model)
        raise NotImplementedError
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight

def load_weight_pose2d(model, opts):
    if model == 'smpl':
        weight = {
            'k2d': 2e-4, 
            'init_poses': 1e-3, 'init_shapes': 1e-2,
            'smooth_body': 5e-1, 'smooth_poses': 1e-1,
        }
    elif model == 'smplh':
        raise NotImplementedError
    elif model == 'smplx':
        raise NotImplementedError
    else:
        weight = {}
    for key in opts.keys():
        if key in weight.keys():
            weight[key] = opts[key]
    return weight