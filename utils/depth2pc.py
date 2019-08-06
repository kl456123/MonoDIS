# -*- coding: utf-8 -*-
"""
Convert disp/depth map to point cloud
"""

import numpy as np
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import cv2
from utils.view_estimate import draw_3d

# kitti params
baseline = 0.54
p2 = np.asarray([[
    7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
    7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
    1.000000e+00, 2.745884e-03
]]).reshape((3, 4))
# f = p2[0, 0]

# K = p2[:3, :3]
# KT = p2[:, 3]
# K_inv = np.linalg.inv(K)
# T = np.dot(K_inv, KT)
# C = -T

MAX_DEPTH = 100
original_width = 1280
original_height = 384

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1280] = 721.5377

f = width_to_focal[original_width]


def read_img(img_name):
    img = plt.imread(img_name)
    # img = np.asarray(Image.open(img_name))
    return img


def load_npy(file_name):
    pred_disp = np.load(file_name)
    pred_disp = original_width * cv2.resize(
        pred_disp, (original_width, original_height),
        interpolation=cv2.INTER_LINEAR) * original_width/512
    # disp_to_img = scipy.misc.imresize(disp_pp.squeeze(),
    # [original_height, original_width])
    return pred_disp


def disp2pc(img, K_inv, C):
    disp = np.copy(img)
    disp = disp.flatten()
    disp[disp == 0] = MAX_DEPTH
    h, w = img.shape[:2]
    u_index, v_index = np.meshgrid(range(w), range(h))
    u_index = u_index.flatten()
    v_index = v_index.flatten()
    ones = np.ones_like(u_index)
    point_2ds = np.vstack([u_index, v_index, ones])

    depth = f * baseline / disp

    pc = depth[..., np.newaxis] * np.dot(K_inv, point_2ds).T

    # translation(no rotation in kitti)
    pc = pc + C
    return pc


def disp2depth(img):
    disp = np.copy(img)
    disp[disp == 0] = MAX_DEPTH
    depth = f * baseline / disp
    return depth


def pc2bev(pc):
    pc_bev = pc[:, [0, 2]]
    voxel_size = 0.05
    width = 80
    height = 75
    bev_width = int(height / voxel_size)
    bev_height = int(width / voxel_size)
    bev = np.zeros((bev_height, bev_width))
    pc_bev[:, 0] += height / 2

    # voxelize
    pc_bev /= voxel_size
    pc_bev = pc_bev.astype(np.int32)
    area_filter = (pc_bev[:, 1] < bev_width) & (pc_bev[:, 0] < bev_height)
    zeros_filter = (pc_bev[:, 1] >= 0) & (pc_bev[:, 0] >= 0)
    pc_bev = pc_bev[area_filter & zeros_filter]
    bev[pc_bev[:, 0], pc_bev[:, 1]] = 1
    return bev


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def read_calib(calib_path, extend_matrix=True):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape(
        [3, 4])
    P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape(
        [3, 4])
    P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape(
        [3, 4])
    P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape(
        [3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    image_info = {}
    image_info['calib/P0'] = P0
    image_info['calib/P1'] = P1
    image_info['calib/P2'] = P2
    image_info['calib/P3'] = P3
    R0_rect = np.array([float(info)
                        for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect
    image_info['calib/R0_rect'] = rect_4x4
    Tr_velo_to_cam = np.array(
        [float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
    Tr_imu_to_velo = np.array(
        [float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
    image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
    return image_info


def cam2velo(pc, calib_info):
    # import ipdb
    # ipdb.set_trace()
    R0_rect = calib_info['calib/R0_rect']
    Tr_velo_to_cam = calib_info['calib/Tr_velo_to_cam']
    Pr = np.dot(R0_rect, Tr_velo_to_cam)
    pc_velo = np.dot(np.linalg.inv(Pr), pc.T).T
    return pc_velo


def main():
    npy_dir = '/data/object/liangxiong/disparity/data/'
    saved_path = '/data/liangxiong/KITTI/training/pseudo_velodyne'
    calib_dir = '/data/object/training/calib/'
    for idx, file in enumerate(sorted(os.listdir(npy_dir))):
        sample_idx = os.path.splitext(file)[0][:6]
        calib_path = os.path.join(calib_dir, '{}.txt'.format(sample_idx))
        calib_info = read_calib(calib_path)
        p2 = calib_info['calib/P2'][:3, :]
        K = p2[:3, :3]
        KT = p2[:, 3]
        K_inv = np.linalg.inv(K)
        T = np.dot(K_inv, KT)
        C = -T

        img_name = os.path.join(npy_dir, file)
        img = load_npy(img_name)

        pc = disp2pc(img, K_inv, C)
        one = np.ones_like(pc[:, -1:])
        pc = np.concatenate([pc, one], axis=-1)

        velo = cam2velo(pc, calib_info)

        sample_idx = os.path.splitext(file)[0]
        velo.astype(np.float32).tofile(
            os.path.join(saved_path, '{}.bin'.format(sample_idx[:6])))
        sys.stdout.write(
            '\r({}/{}) filename: {}'.format(idx, 7480, sample_idx))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
