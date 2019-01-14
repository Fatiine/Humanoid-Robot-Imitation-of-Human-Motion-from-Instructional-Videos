import numpy as np

import transforms3d
from transforms3d import quaternions as qq
from transforms3d import euler
from collections import OrderedDict


def vec_to_quaternion(v):
    n = np.linalg.norm(v)
    v_norm = v / n
    q = qq.axangle2quat(v_norm, n)
    return q


def vec_to_angle(v):
    n = np.linalg.norm(v)
    return n


def smpl_to_deepmimic(poses, trans, cams, proc_param):
    num_img = poses.shape[0]
    frames = np.zeros((num_img, 44))
    for i in range(num_img):
        frames[i] = kinematic_tree(poses[i], trans[i],
                                      cams[i], proc_param[i])
    origin = frames[0, 1:4]
    frames[:, 1:4] = frames[:, 1:4] - origin + np.array([0, 0.9, 0])
    return frames


def root_position(j2d, cam, proc_param):
    img_size = proc_param['img_size']
    undo_scale = 1. / np.array(proc_param['scale'])

    cam_s = cam[0]
    flength = 500.
    tz = flength / (0.5 * img_size * cam_s)
    principal_pt = np.array([img_size, img_size]) / 2.
    start_pt = proc_param['start_pt'] - 0.5 * img_size
    final_principal_pt = (principal_pt + start_pt) * undo_scale
    trans = np.hstack([pp_orig, tz])
    # trans[0], trans[1], trans[2] =  trans[0], trans[1], 0
    return trans


def root_position(j2d, cam, proc_param):
    scale = proc_param['scale']
    orig_size = proc_param['img_size']
    start_pt = proc_param['start_pt'] # - 0.5 * proc_param.crop_image_size
    undo_scale = 1.25 / (150. / scale)

    root = j2d[0]
    root_shifted = (root + start_pt)
    # root_shifted[1] = orig_size[1] - root_shifted[1]
    root_xy = root_shifted * undo_scale

    cam_s = cam[0]
    flength = 1.
    tz = flength / cam_s 
    root_orig = np.hstack([0, -root_xy[1], root_xy[0]])

    print(root_orig)
    return root_orig


def kinematic_tree(poses, j2d, cam, proc_param):

    joints = {
        'Pelvis': 0,
        'Neck': 12,
        'Chest': 3,
        'L_Shoulder': 16, 'L_Elbow': 18,
        'R_Shoulder': 17, 'R_Elbow': 19,
        'L_Hip': 1, 'L_Knee': 4, 'L_Ankle': 7,
        'R_Hip': 2, 'R_Knee': 5, 'R_Ankle': 8
    }

    # LR reverse for deepmimic
    target_joints = {
        'Pelvis': [4, 5, 6, 7],
        'Neck': [12, 13, 14, 15],
        'Chest': [8, 9, 10, 11],
        'L_Shoulder': [39, 40, 41, 42], 'L_Elbow': [43],
        'R_Shoulder': [25, 26, 27, 28], 'R_Elbow': [29],
        'L_Hip': [30, 31, 32, 33], 'L_Knee': [34], 'L_Ankle': [35, 36, 37, 38],
        'R_Hip': [16, 17, 18, 19], 'R_Knee': [20], 'R_Ankle': [21, 22, 23, 24],
    }
    """


    # motions[:, 0] = 0.0625
    # motions[:, 1:4] # root position 
    # motions[:, 4:8]  # root rotation
    # motions[:, 8:12] # chest rotation
    # motions[:, 12:16]  # neck rotation
    # motions[:, 16:20] # right hip rot
    # motions[:, 20]  # right knee
    # motions[:, 21:25]  # right ankle rot
    # motions[:, 25:29] # right shoulder rotation
    # motions[:, 30]  # right elbow
    # motions[:, 30:34]  # left hip rot
    # motions[:, 34]  # left knee
    # motions[:, 35:39]  # left ankle
    # motions[:, 39:43]  # left shoulder rot
    # motions[:, 43] # left elbow rot """

    poses = poses.reshape((-1, 3))
    motions = np.zeros(44)

    r = [0.7071, 0, 0.7071, 0]  # # Quaternion that represents 90 degrees around Y
    # qconjugate(r) = [ 0.7071, 0, -0.7071, 0] 
    motions[1:4] = root_position(j2d, cam, proc_param)
    for joi, num in joints.items():
        x = poses[num]
        # change of basis: SMPL to DeepMimic
        if joi in ['R_Knee', 'L_Knee']:
            q = -vec_to_angle(x)
        elif joi in ['L_Elbow', 'R_Elbow']:
            q = vec_to_angle(x)
        elif joi in ['Pelvis']:
            q = vec_to_quaternion(x)
            q = qq.qmult(r, q)
            q = qq.qmult(q, qq.qconjugate(r))
            q = qq.qmult([0, 1, 0, 0], q)
        elif joi in ['L_Shoulder']:
            q = vec_to_quaternion(x)
            q = qq.qmult(q, [0.7071, 0, 0, 0.7071]) # [0.7071, 0, 0, 0.7071]  Quaternion that represents 90 degrees around Z
            q = qq.qmult(r, q)
            q = qq.qmult(q, qq.qconjugate(r))
        elif joi in ['R_Shoulder']:
            q = vec_to_quaternion(x)
            q = qq.qmult(q, [0.7071, 0, 0, -0.7071]) #[0.7071, 0, 0, -0.7071]  Quaternion that represents 90 degrees around -Z
            q = qq.qmult(r, q)
            q = qq.qmult(q, qq.qconjugate(r))
        else:
            q = vec_to_quaternion(x)
            q = qq.qmult(r, q)
            q = qq.qmult(q, qq.qconjugate(r))
        motions[target_joints[joi]] = q
    return motions