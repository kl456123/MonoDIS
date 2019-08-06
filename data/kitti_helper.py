#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: DuanZhixiang(zhixiangduan@deepmotion.ai)
# 3D deepbox helpers

import os
import numpy as np
import math

from PIL import Image
from PIL import ImageDraw

BIN, OVERLAP = 2, 0.1
VEHICLES = ['Car']
CAR_DIMS_AVG = [1.531, 1.629, 3.883]
Y_AVG = 1.71

IMG_PATH = '/home/duan/data/KITTI/training/image_2'
LABEL_PATH = "/home/duan/data/KITTI/training/label_2"
CALIB_PATH = '/home/duan/data/KITTI/training/calib'


def draw_3d_box():
    for img_file in os.listdir(IMG_PATH):
        label_name = os.path.join(LABEL_PATH, img_file.split('.')[0] + '.txt')
        calib_name = os.path.join(CALIB_PATH, img_file.split('.')[0] + '.txt')

        img_path = os.path.join(IMG_PATH, img_file)
        label_path = os.path.join(LABEL_PATH, label_name)
        calib_name = os.path.join(CALIB_PATH, calib_name)

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)


def compute_anchors(angle):
    anchors = []

    wedge = 2. * np.pi / BIN
    l_index = int(angle / wedge)
    r_index = l_index + 1

    if (angle - l_index * wedge) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([l_index, angle - l_index * wedge])

    if (r_index * wedge - angle) < wedge / 2 * (1 + OVERLAP / 2):
        anchors.append([r_index % BIN, angle - r_index * wedge])

    return anchors


def convert_target(obj):
    obj['dims'] -= CAR_DIMS_AVG

    orientation = np.zeros((BIN, 2))
    confidence = np.zeros(BIN)

    alpha = obj['alpha']
    new_alpha = alpha + np.pi / 2.
    if new_alpha < 0:
        new_alpha += 2. * np.pi
    new_alpha -= int(new_alpha / (2. * np.pi) * (2. * np.pi))

    anchors = compute_anchors(new_alpha)
    for anchor in anchors:
        orientation[anchor[0]] = np.array(
            [np.cos(anchor[1]), np.sin(anchor[1])])
        confidence[anchor[0]] = 1.

    confidence /= np.sum(confidence)

    obj['orient'] = orientation
    obj['conf'] = confidence

    return obj


def get_depth_index(depth):
    # Depth range 5 - 80
    depth = max(5, depth)
    if depth < 55.:
        l_index = depth // 5 - 1
        l_error = depth % 5
        l_cos = l_error / 5.0
        l_sin = (1. - l_cos**2)**0.5
        return [l_index, l_cos, l_sin]
    else:
        l_index = 10
        l_error = depth - 55
        l_cos = l_error / (90. - 55)
        l_sin = (1 - l_cos**2)**0.5
        return [l_index, l_cos, l_sin]


def compute_box_3d_in_world(target, p):
    """Takes an target and a project matrix (P) and project the 3D
    bounding box into the world.

    Args:
        target(dict): {'dimension':, 'location':, 'ry':}
        p(numpy.array): 3x4

    Returns:
        (numpy.array): 3x8, coordinates of 8 points in world coordinates.

    """
    rotation_y = target['ry']
    r = [
        math.cos(rotation_y), 0, math.sin(rotation_y), 0, 1, 0,
        -math.sin(rotation_y), 0, math.cos(rotation_y)
    ]
    r = np.array(r).reshape(3, 3)

    h, w, l = target['dimension']

    # The points sequence is 1, 2, 3, 4, 5, 6, 7, 8.
    # Front face: 1, 2, 6, 5; left face: 2, 3, 7, 6
    # Back face: 3, 4, 8, 7; Right face: 4, 1, 5, 8
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h, 0])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0])

    box_points_coords = np.vstack((x_corners, y_corners, z_corners))
    corners_3d = np.dot(r, box_points_coords)
    corners_3d = corners_3d + np.array(target['location']).reshape(3, 1)

    return corners_3d


def compute_box_3d(target, p):
    """Takes an target and a project matrix (P) and project the 3D
    bounding obx into the image plane.

    Args:
        target(dict): {'dimension':, 'location':, 'ry':}
        p(numpy.array): 3x4

    Returns:
        (numpy.array): 2x8, coordinates of 8 points projected to the plane.

    """
    rotation_y = target['ry']
    r = [
        math.cos(rotation_y), 0, math.sin(rotation_y), 0, 1, 0,
        -math.sin(rotation_y), 0, math.cos(rotation_y)
    ]
    r = np.array(r).reshape(3, 3)

    h, w, l = target['dimension']

    # The points sequence is 1, 2, 3, 4, 5, 6, 7, 8.
    # Front face: 1, 2, 6, 5; left face: 2, 3, 7, 6
    # Back face: 3, 4, 8, 7; Right face: 4, 1, 5, 8
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h, 0])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0])

    box_points_coords = np.vstack((x_corners, y_corners, z_corners))
    corners_3d = np.dot(r, box_points_coords)
    corners_3d = corners_3d + np.array(target['location']).reshape(3, 1)
    corners_3d_homo = np.vstack((corners_3d, np.ones(
        (1, corners_3d.shape[1]))))

    corners_2d = np.dot(p, corners_3d_homo)
    corners_2d_xy = corners_2d[:2, :] / corners_2d[2, :]

    return corners_2d_xy.transpose()


def load_projection_matrix(calib_file):
    """Load the camera project matrix."""
    assert os.path.isfile(calib_file)
    with open(calib_file) as f:
        lines = f.readlines()
        line = lines[2]
        line = line.split()
        assert line[0] == 'P2:'
        p = [float(x) for x in line[1:]]
        p = np.array(p).reshape(3, 4)
    return p


def get_alpha(alpha_anchors, alpha_errors):
    """
    Args:
        alpha_anchors(numpy): Nx2
        alpha_errors(numpy): NX2
    """
    alpha_anchors = alpha_anchors.argmax(axis=1).tolist()
    alpha_errors = alpha_errors[:, 0].tolist()
    alphas = []

    for alpha_anchor, alpha_error in zip(alpha_anchors, alpha_errors):
        alpha_error = min(1.0, alpha_error)
        alpha_error = max(-1.0, alpha_error)
        alpha = math.acos(alpha_error)
        if alpha_anchor == 0:
            alpha += 0.5 * np.pi
        else:
            alpha += -.5 * np.pi

        alphas.append(alpha)

    return alphas


def get_depth_coords(length_anchors, length_errors, angles, y_lens):
    """
    Args:
        length_anchors(numpy): Nx11
        length_errors(numpy): Nx2
        y_lens(numpy): Nx2
    """
    length_anchors = length_anchors.argmax(axis=1).tolist()
    length_errors = length_errors[:, 0].tolist()
    angles_cos = angles[:, 0].tolist()
    angles_sin = angles[:, 1].tolist()
    y_lens = y_lens.tolist()

    coords = []

    for anchor, error, angle_cos, angle_sin, y in zip(
            length_anchors, length_errors, angles_cos, angles_sin, y_lens):
        if anchor == 10:
            length = 55. + (90. - 55.) * error
        else:
            length = 5. * (anchor + 1) + 5. * error

        # x = length * angle_cos
        # z = length * angle_sin
        z = length
        x = z * angle_cos / angle_sin

        coords.append([x, y[0] + Y_AVG, z])

    return coords


def get_depth_index_v2(depth):
    # Depth range 4 - 80
    depth = max(4, depth)
    if depth < 60.:
        l_index = (depth - 4) // 2
        l_error = (depth - 4) % 2
        l_cos = l_error / 2.0
        l_sin = (1. - l_cos**2)**0.5
        return [l_index, l_cos, l_sin]
    else:
        l_index = 28
        l_error = depth - 60
        l_cos = l_error / (90. - 60.)
        l_sin = (1 - l_cos**2)**0.5
        return [l_index, l_cos, l_sin]


def process_center_coords(coords):
    # d_x = float(coords[0])
    d_y = float(coords[1])
    d_z = float(coords[2])

    # depth = (d_x**2+d_z**2)**0.5
    depth = d_z
    # angle = math.acos(d_x/depth)

    depth_anchor = get_depth_index(depth)
    d_y_anchor = d_y - Y_AVG
    # angle_anchor = [math.cos(angle), math.sin(angle)]

    return depth_anchor, d_y_anchor


if __name__ == '__main__':
    test_depth = [3.1, 5.4, 33.2, 52.3, 62.3, 75]
    for item in test_depth:
        result = get_depth_index(item)
        print('result is:', result)
