from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os 
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel
from collections import OrderedDict
import json

from smpl_to_deepmimic import smpl_to_deepmimic

flags.DEFINE_string('img_dir', "data/dance/", 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')




def visualize(img, proc_param, joints, verts, cam, i):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(i)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    name = str(i).join(".png")
    plt.savefig(name)

def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


"""def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)"""

def save_motion(data, filename):
    mfile = OrderedDict()
    mfile['Loop'] = 'wrap'
    data[:, 0] = 0.0625
    mfile['Frames'] = data.tolist()
    with open(filename, 'w') as f:
        json.dump(mfile, f)

def main(img_dir):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    num_features = 72
    num_cam = 3
    i = 0
    print(img_dir)
    onlyfiles = [f for f in os.listdir(img_dir)
                 if os.path.isfile(os.path.join(img_dir, f))]
    nbr_img = len(onlyfiles)
    print(nbr_img)
    poses = np.zeros((nbr_img,num_features))
    cams = np.zeros((nbr_img, num_cam))
    trans = np.zeros((nbr_img, 3))
    j2d = np.zeros((nbr_img,19,2))
    proc_params = [{}] * nbr_img
    for file in onlyfiles:
        img_path = os.path.join(img_dir, file)

        input_img, proc_param, img = preprocess_image(img_path)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        joints, verts, cam, joints3d, theta = model.predict(
            input_img, get_theta=True)
        poses[i] = theta[:, num_cam:(num_cam + num_features)]
        j2d[i] = joints
        cams[i] = cam
        proc_params[i] = proc_param
        visualize(img, proc_param, joints[0], verts[0], cams[0], i)
        i += 1

    print(proc_param)
    return poses, cams, j2d, proc_params
        

        #p3d(img, joints3d, joints[0], verts[0], cams[0], proc_param, file)
        #visualize(img, proc_param, joints[0], verts[0], cams[0])

if __name__ == "__main__":
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    img_dir = "data/dance/"
    poses, cams, j2d, proc_param = main(img_dir)

    d = smpl_to_deepmimic(poses,j2d,cams, proc_param)
    save_motion(d,"dance2.txt")


