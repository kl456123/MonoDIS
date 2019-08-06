#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: DuanZhixiang(zhixiangduan@deepmotion.ai)
# kitti utils

import numpy as np
# from utils.box_vis import compute_box_3d as compute_box_3dv3
import math


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
    [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def get_lidar_in_image_fov(pc_velo,
                           calib,
                           xmin,
                           ymin,
                           xmax,
                           ymax,
                           return_more=False,
                           clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)

    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
    input: pts_3d: nx3 matrix
    P:      3x4 projection matrix
    output: pts_2d: nx2 matrix

    P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
    => normalize projected_pts_2d(2xn)

    <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
    => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def proj_3dTo2d(pred_boxes_3d, p2):
    """
    Args:
    pred_boxes_3d: shape(N, 3,3,1) (dim,pos,ry)
    """
    num = pred_boxes_3d.shape[0]
    boxes_2d_projs = []

    for i in range(num):
        target = {}
        target['ry'] = pred_boxes_3d[i, -1]
        target['dimension'] = pred_boxes_3d[i, :3]
        target['location'] = pred_boxes_3d[i, 3:6]
        # corners_2d_xy = compute_box_3dv3(target, p2)
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
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = np.array(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

        box_points_coords = np.vstack((x_corners, y_corners, z_corners))
        corners_3d = np.dot(r, box_points_coords)
        corners_3d = corners_3d + np.array(target['location']).reshape(3, 1)
        corners_3d_homo = np.vstack((corners_3d, np.ones(
            (1, corners_3d.shape[1]))))

        corners_2d = np.dot(p2, corners_3d_homo)
        corners_2d_xy = corners_2d[:2, :] / corners_2d[2, :]

        # find the bbox 2d
        corners_2d_xy = corners_2d_xy.reshape(2, 8)
        xmin = corners_2d_xy[0, :].min(axis=0)
        ymin = corners_2d_xy[1, :].min(axis=0)
        xmax = corners_2d_xy[0, :].max(axis=0)
        ymax = corners_2d_xy[1, :].max(axis=0)

        boxes_2d_proj = np.stack([xmin, ymin, xmax, ymax], axis=-1)
        boxes_2d_projs.append(boxes_2d_proj)

    boxes_2d_projs = np.stack(boxes_2d_projs, axis=0)
    return boxes_2d_projs


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        # Camera2 to Imagary camera2
        self.C2IC = [[1, 0, 0, 0], [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 40],
                     [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0]]
        self.C2IC = np.array(self.C2IC).reshape(3, 4)

        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(
            np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = (
            (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = (
            (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[
            2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12],
                  data[13])  # location (x,y,z) in camera coord.
        self.ry = data[
            14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.score = data[15] if data.__len__() == 16 else 0.

        self.box3d = np.array(
            [self.ry, self.h, self.w, self.l, data[11], data[12], data[13]])

    def print_object(self):
        print(('Type, truncation, occlusion, alpha: %s, %d, %d, %f' %
               (self.type, self.truncation, self.occlusion, self.alpha)))
        print(('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' %
               (self.xmin, self.ymin, self.xmax, self.ymax)))
        print(('3d bbox h,w,l: %f, %f, %f' % (self.h, self.w, self.l)))
        print(('3d bbox location, ry: (%f, %f, %f), %f' %
               (self.t[0], self.t[1], self.t[2], self.ry)))


def compute_local_angle(location, ry):
    # [0, pi]
    alpha = np.arctan2(location[2], location[0])
    ry_local = ry - (-alpha)
    return ry_local


# def compute_local_angle(center_2d, p2, ry):
# """
# Args:
# center_2d: shape(N, 2)
# p2: shape(3,4)
# """
# #  import ipdb
# #  ipdb.set_trace()
# M = p2[:, :3]
# center_2d_homo = np.concatenate(
# [center_2d, np.ones_like(center_2d[-1:])], axis=-1)
# direction_vector = np.dot(np.linalg.inv(M), center_2d_homo.T).T
# x_vector = np.array([1, 0, 0])
# cos = np.dot(direction_vector, x_vector.T) / np.linalg.norm(
# direction_vector, axis=-1)
# ray_angle = np.arccos(cos)
# local_angle = ry + ray_angle
# if local_angle > np.pi:
# local_angle = local_angle - 2 * np.pi
# return local_angle


def compute_ray_angle(center_2d, p2):
    M = p2[:, :3]
    center_2d_homo = np.concatenate(
        [center_2d, np.ones_like(center_2d[:, -1:])], axis=-1)
    direction_vector = np.dot(np.linalg.inv(M), center_2d_homo.T).T
    x_vector = np.array([1, 0, 0])
    direction_vector[:, 1] = 0
    cos = np.dot(direction_vector, x_vector) / np.linalg.norm(
        direction_vector, axis=-1)
    ray_angle = np.arccos(cos)
    return ray_angle


def compute_global_angle(center_2d, p2, local_angle):
    """
    Note that just batch is supported
    Args:
        center_2d: shape(N, 2)
        p2: shape(3,4)
    """
    ray_angle = compute_ray_angle(center_2d, p2)
    ry = local_angle + (-ray_angle)
    # if ry < -np.pi:
    # ry += np.pi
    cond = ry < -np.pi
    ry[cond] = ry[cond] + 2 * np.pi
    return ry


def compute_2d_projv2(ry, corners, trans_3d, p):
    """
    Args:
        ry: scalar
        corners: shape(8, 3)
        trans_3d: shape(3)
        p: shape(3,4)
    """
    r = np.stack(
        [np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)],
        axis=-1).reshape(3, 3)
    corners_3d = np.dot(r, corners.T)
    trans_3d = np.repeat(np.expand_dims(trans_3d.T, axis=1), 8, axis=1)
    corners_3d = corners_3d[..., np.newaxis] + trans_3d
    # corners_3d = corners_3d.reshape(3, -1)
    corners_3d_homo = np.vstack((corners_3d, np.ones(
        (1, corners_3d.shape[1]))))

    corners_2d = np.dot(p, corners_3d_homo)
    corners_2d_xy = corners_2d[:2] / corners_2d[2]

    corners_2d_xy = corners_2d_xy.reshape(2, 8)
    xmin = corners_2d_xy[0, :, :].min(axis=0)
    ymin = corners_2d_xy[1, :, :].min(axis=0)
    xmax = corners_2d_xy[0, :, :].max(axis=0)
    ymax = corners_2d_xy[1, :, :].max(axis=0)

    boxes_2d_proj = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes_2d_proj


def compute_2d_proj(ry, corners, trans_3d, p):
    r = np.stack(
        [np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)],
        axis=-1).reshape(3, 3)
    corners_3d = np.dot(r, corners.T)
    trans_3d = np.repeat(np.expand_dims(trans_3d.T, axis=1), 8, axis=1)
    corners_3d = corners_3d[..., np.newaxis] + trans_3d
    # corners_3d = corners_3d.reshape(3, -1)
    corners_3d_homo = np.vstack((corners_3d, np.ones(
        (1, corners_3d.shape[1]))))

    corners_2d = np.dot(p, corners_3d_homo)
    corners_2d_xy = corners_2d[:2] / corners_2d[2]

    corners_2d_xy = corners_2d_xy.reshape(2, 8)
    xmin = corners_2d_xy[0, :, :].min(axis=0)
    ymin = corners_2d_xy[1, :, :].min(axis=0)
    xmax = corners_2d_xy[0, :, :].max(axis=0)
    ymax = corners_2d_xy[1, :, :].max(axis=0)

    boxes_2d_proj = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes_2d_proj


def get_r_2d(line):
    # import ipdb
    # ipdb.set_trace()
    direction = (line[0] - line[1])
    if direction[1] < 0:
        direction = -direction
    cos = direction[0] / np.linalg.norm(direction)
    theta = np.arccos(cos)
    return 1 - theta / np.pi


def direction2angle(x, y):
    # import ipdb
    # ipdb.set_trace()
    ry = np.arccos(x / (np.sqrt(x * x + y * y)))
    if y < 0:
        ry = -ry
    return ry


def get_cls_orient_4(line):
    # import ipdb
    # ipdb.set_trace()

    direction = line[0] - line[1]

    ry = direction2angle(direction[0], direction[1])

    intervals = [[0, np.pi / 2], [np.pi / 2, np.pi], [-np.pi, -np.pi / 2],
                 [-np.pi / 2, 0]]
    for ind, interval in enumerate(intervals):
        if ry >= interval[0] and ry < interval[1]:
            cls = ind
            return cls


def modify_cls_orient(cls_orient, left_side, right_side):
    """
    For special case, classifiy it from common case
    """
    left_dir = (left_side[0] - left_side[1])
    right_dir = (right_side[0] - right_side[1])
    cond = left_dir[0] * right_dir[0] < 0
    if cond:
        return 2
    else:
        return cls_orient


def truncate_box(box_2d, line, normalize=True):
    """
    Args:
        dims_2d:
        line:
    Return: cls_orient:
            reg_orient:
    """
    # import ipdb
    # ipdb.set_trace()
    direction = (line[0] - line[1])
    if direction[0] * direction[1] == 0:
        cls_orient = -1
    else:
        cls_orient = direction[1] / direction[0] > 0
        cls_orient = cls_orient.astype(np.int32)
    # cls_orient = direction[0] > 0
    reg_orient = np.abs(direction)

    # normalize
    if normalize:
        # w, h = dims_2d
        h = box_2d[3] - box_2d[1] + 1
        w = box_2d[2] - box_2d[0] + 1

        reg_orient[0] /= w
        reg_orient[1] /= h
        # reg_orient = np.log(reg_orient)
    return cls_orient, reg_orient


def get_center_side(corners_xy):
    """
    Args:
        location: (3,)
    """
    point0 = corners_xy[0]
    point1 = corners_xy[1]
    point2 = corners_xy[2]
    point3 = corners_xy[3]
    mid0 = (point0 + point1) / 2
    mid1 = (point2 + point3) / 2
    return np.stack([mid0, mid1], axis=0)


def get_center_orient(location, p2, ry):
    """
    Args:
        location: (3,)
    """
    K = p2[:3, :3]
    l = 4

    R = np.stack(
        [np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)],
        axis=-1).reshape(3, 3)

    location1 = location
    location2 = np.dot(R, np.asarray([l, 0, 0])) + location1

    # their projections
    homo_2d_1 = np.dot(K, location1)
    homo_2d_2 = np.dot(K, location2)

    point_2d_1 = homo_2d_1 / homo_2d_1[-1]
    point_2d_2 = homo_2d_2 / homo_2d_2[-1]

    point_2d_1 = point_2d_1[:-1]
    point_2d_2 = point_2d_2[:-1]

    direction = point_2d_2 - point_2d_1

    if direction[0] == 0:
        # boundary line
        cls_orient = -1
    else:
        cls_orient = direction[1] / direction[0] > 0
        cls_orient = cls_orient.astype(np.int32)

    return cls_orient


def get_h_2d(C_3d, dim, P2, box_2d):
    # x,y,z
    # C_3d = np.asarray([-16.53, 2.39, 58.49])
    # h,w,l
    # dim = np.asarray([1.67, 1.87, 3.69])

    bottom_3d = C_3d + np.asarray([0, 0.5 * dim[0], 0])
    top_3d = C_3d - np.asarray([0, 0.5 * dim[0], 0])

    bottom_3d_homo = np.append(bottom_3d, 1)
    top_3d_homo = np.append(top_3d, 1)

    bottom_2d_homo = np.dot(P2, bottom_3d_homo)
    top_2d_homo = np.dot(P2, top_3d_homo)

    lambda_bottom = bottom_2d_homo[-1]
    bottom_2d_homo = bottom_2d_homo / lambda_bottom
    bottom_2d = bottom_2d_homo[:-1]

    lambda_top = top_2d_homo[-1]
    top_2d_homo = top_2d_homo / lambda_top
    top_2d = top_2d_homo[:-1]

    delta_2d = top_2d - bottom_2d

    h = box_2d[3] - box_2d[1] + 1
    return np.abs(delta_2d[-1]) / h


def get_center_2d(C_3d, P2, box_2d):
    C_3d_homo = np.append(C_3d, 1)
    C_2d_homo = np.dot(P2, C_3d_homo)
    C_2d_homo = C_2d_homo / C_2d_homo[-1]
    C_2d = C_2d_homo[:-1]

    # normalize it by using gt box
    h = box_2d[3] - box_2d[1] + 1
    w = box_2d[2] - box_2d[0] + 1
    # x = (box_2d[3] + box_2d[1]) / 2
    # y = (box_2d[2] + box_2d[0]) / 2
    x = box_2d[0]
    y = box_2d[1]
    C_2d_normalized = ((C_2d[0] - x) / w, (C_2d[1] - y) / h)

    return C_2d_normalized


def get_gt_boxes_2d_ground_rect(loc, dim):
    """
    ry=-0.5*np.pi
    z-x coords
    """
    h, w, l = dim
    x, y, z = loc
    zmax = z + 0.5 * l
    zmin = z - 0.5 * l
    xmin = x - 0.5 * w
    xmax = x + 0.5 * w
    return [zmin, xmin, zmax, xmax]


def get_gt_boxes_2d_ground_rect_v2(loc, dim, ry):
    """
    Args:
    Returns:
    """
    h, w, l = dim
    x, y, z = loc

    ry = math.fabs(ry)
    closure_w = l * math.cos(ry) + w * math.sin(ry)
    closure_h = l * math.sin(ry) + w * math.cos(ry)

    zmax = z + 0.5 * closure_h
    zmin = z - 0.5 * closure_h
    xmin = x - 0.5 * closure_w
    xmax = x + 0.5 * closure_w
    return [zmin, xmin, zmax, xmax]


def encode_side_points(line, box_2d_proj):
    """
    Args:
        line: []
        box_2d_proj: []
    """
    # import ipdb
    # ipdb.set_trace()
    xmin, ymin, xmax, ymax = box_2d_proj
    center_proj_x = (xmin + xmax) / 2
    center_proj_y = (ymin + ymax) / 2

    proj_h = ymax - ymin + 1
    proj_w = xmax - xmin + 1

    # center and side points
    center_proj = np.asarray([center_proj_x, center_proj_y])
    dims_proj = np.asarray([proj_w, proj_h])
    point1, point2 = line
    point1_encoded = (point1 - center_proj) / dims_proj
    point2_encoded = (point2 - center_proj) / dims_proj
    return np.concatenate([point1_encoded, point2_encoded], axis=-1)


def encode_bottom_points(corners_xy, box_2d_proj):
    xmin, ymin, xmax, ymax = box_2d_proj
    center_proj_x = (xmin + xmax) / 2
    center_proj_y = (ymin + ymax) / 2

    proj_h = ymax - ymin + 1
    proj_w = xmax - xmin + 1

    # import ipdb
    # ipdb.set_trace()
    # center and side points
    center_proj = np.asarray([center_proj_x, center_proj_y])
    dims_proj = np.asarray([proj_w, proj_h])
    bottom_points = corners_xy[[0, 1, 2, 3]]
    encoded_bottom_points = (bottom_points - center_proj) / dims_proj
    return encoded_bottom_points.reshape(-1)


def generate_keypoint_gt(visible_side, image_shape):
    idx = np.argmax(visible_side[:, 1])
    keypoint = visible_side[idx]
    #  keypoint[0] /= image_shape[1]
    #  keypoint[1] /= image_shape[0]
    return keypoint
