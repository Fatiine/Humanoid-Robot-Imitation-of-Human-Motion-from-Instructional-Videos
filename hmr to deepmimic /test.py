import numpy as np
import cPickle as pk
import json

import transforms3d
from transforms3d import quaternions, euler
from collections import OrderedDict



path = "scythe/scythe_0006/hmr/hmr.pkl"
f = open(path, 'r')
data = pk.load(f)

print(data.keys()) # ['thetas', 'poses', 'cams', 'shapes', 'joint_2d_positions', 'trans']

"""print(data['poses'].shape) # (210, 72) 210 frames, each row contains a 72d SMPL pose vector
# 24 spherical joints, each represented by a 3d axis-angle vector, results the 72d pose vector

print(data['shapes'].shape) # (210, 10) each row contains a 10d SMPL shape vector

print(data['cams'].shape) # (210, 3) each row contains the camera translation
# we assume that the camera is fixed in front of the scene, with identity rotation

# please refer to SMPL project page on how to get 3D joint positions from SMPL pose and shape vectors:
# http://smpl.is.tue.mpg.de/"""
from transforms3d import quaternions as qq


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


def vec_to_quaternion(x):
    th = np.linalg.norm(x)
    x_norm = x / th
    q = qq.axangle2quat(x_norm, th)
    return q


def vec_to_angle(x):
    th = np.linalg.norm(x)
    return th


def smpl_to_deepmimic(q3d, trans, cams):
    num_steps = q3d.shape[0]
    x3d = np.zeros((num_steps, 44))
    for i in range(num_steps):
        x3d[i] = build_kinematic_tree(q3d[i], trans[i],
                                      cams[i])
    origin = x3d[0, 1:4]
    x3d[:, 1:4] = x3d[:, 1:4] - origin + np.array([0, 0.9, 0])
    return x3d


def calcRootTranslation1(j2d, cam, proc_param):
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


def calcRootTranslation(j2d, cam, proc_param):
    scale = proc_param.scale
    orig_size = proc_param.original_image_size

    start_pt = proc_param.crop_start_pt # - 0.5 * proc_param.crop_image_size
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


def build_kinematic_tree(theta, trans, cam):
    """
    z: 3D joint coordinates 14x3
    v: vectors
    """
    # 0 r foot
    # 1 r knee
    # 2 r hip
    # 3 l hip
    # 4 l knee
    # 5 l foot
    # 6 r hand
    # 7 r elbow
    # 8 r shoulder
    # 9 l shoulder
    # 10 l elbow
    # 11 l hand
    # 12 thorax
    # 13 head

    # GYM joints
    # motions[:, 0] = 0.0625
    # motions[:, 4:8] = [1, 0,0,0]    # root rotation
    # motions[:, 8:12] = [1, 0,0,0]   # chest rotation
    # motions[:, 12:16] = [1, 0, 0, 0]  # neck rotation
    # motions[:, 16:20] = [1, 0, 0, 0] # right hip rot
    # motions[:, 20] = [1, 0, 0, 0] # right knee
    # motions[:, 21:25] = [1, 0, 0, 0] # right ankle rot
    # motions[:, 25:29] = [1, 0, 0, 0] # right shoulder rotation
    # motions[:, 30] = [1, 0, 0, 0] # right elbow
    # motions[:, 30:34] = [1, 0, 0, 0] # left hip rot
    # motions[:, 34] = [1, 0, 0, 0] # left knee
    # motions[:, 35:39] = [1, 0, 0, 0] # left ankle
    # motions[:, 39:43] = [1, 0, 0, 0] # left shoulder rot
    # motions[:, 43] = [1, 0, 0, 0] # left elbow rot

    # theta = theta[self.num_cam:(self.num_cam + self.num_theta)]
    theta = theta.reshape((-1, 3))
    z = np.zeros(44)

    r = [0.7071, 0, 0.7071, 0]  # # Quaternion that represents 90 degrees around Y
    # qconjugate(r) = [ 0.7071, 0, -0.7071, 0] 
    z[1:4] = trans #calcRootTranslation(j2d, cam, proc_param)
    for joi, num in joints.items():
        x = theta[num]
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
        z[target_joints[joi]] = q
    return z

def save_motion(data, filename):
        mfile = OrderedDict()
        mfile['Loop'] = 'wrap'
        data[:, 0] = 1
        mfile['Frames'] = data.tolist()
        with open(filename, 'w') as f:
            json.dump(mfile, f)

theta = data['poses']
trans = data['trans']
cams = data['cams']

d = smpl_to_deepmimic(theta, trans, cams)
save_motion(d,"scythe_0006.txt")


