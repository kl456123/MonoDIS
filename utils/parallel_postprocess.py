# -*- coding: utf-8 -*-
import numpy as np
from utils.postprocess import generate_side_points, twopoints2direction


def bmm(a, b):
    """
    Args:
        a: shape(N, M, K)
        b: shape(N, K, S)
    """
    b_T = b.transpose(0, 2, 1)
    return (a[:, :, None, :] * b_T[:, None, :, :]).sum(axis=-1)


def mono_3d_postprocess_bbox(dets_3d, dets_2d, p2):
    """
    May be we can improve performance angle prediction by enumerating
    Args:
        dets_3d: shape(N,4) (hwlry)
        dets_2d: shape(N,5) (xyxyc)
        p2: shape(4,3)
    """
    K = p2[:3, :3]
    K_homo = np.eye(4)
    K_homo[:3, :3] = K

    # K*T
    KT = p2[:, -1]
    T = np.dot(np.linalg.inv(K), KT)

    num = dets_3d.shape[0]
    lines = generate_side_points(dets_2d, dets_3d[:, 3:])
    ry = twopoints2direction(lines, p2)

    zeros = np.zeros_like(ry)
    ones = np.ones_like(ry)
    R = np.stack(
        [
            np.cos(ry), zeros, np.sin(ry), zeros, ones, zeros, -np.sin(ry),
            zeros, np.cos(ry)
        ],
        axis=-1).reshape(num, 3, 3)

    l = dets_3d[:, 2]
    h = dets_3d[:, 0]
    w = dets_3d[:, 1]
    zeros = np.zeros_like(w)
    x_corners = np.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=-1)
    y_corners = np.stack([zeros, zeros, zeros, zeros, -h, -h, -h, -h], axis=-1)
    z_corners = np.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=-1)

    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)

    # after rotation
    #  corners = np.dot(R, corners)

    top_corners = corners[:, -4:]
    bottom_corners = corners[:, :4]
    diag_corners = bottom_corners[:, [2, 3, 0, 1]]
    #  left_side_corners = bottom_corners[:, [1, 2, 3, 0]]
    #  right_side_corners = bottom_corners[:, [3, 0, 1, 2]]

    # meshgrid
    # 4x4x4 in all

    num_top = top_corners.shape[1]
    num_bottom = bottom_corners.shape[1]
    top_index, bottom_index, side_index = np.meshgrid(
        np.arange(num_top), np.arange(num_bottom), np.arange(num_bottom))

    # in object frame
    # 3d points may be in top and bottom side
    # all corners' shape: (N,M,3)
    top_side_corners = top_corners[:, top_index.ravel()]
    bottom_side_corners = bottom_corners[:, bottom_index.ravel()]

    # 3d points may be in left and right side
    # both left and right are not difference here
    left_side_corners = bottom_corners[:, side_index.ravel()]
    right_side_corners = diag_corners[:, side_index.ravel()]

    num_cases = top_side_corners.shape[1]
    # shape(N, 4)
    zeros = np.zeros_like(dets_2d[:, 0])

    coeff_left = np.stack([zeros, zeros, dets_2d[:, 0]], axis=-1) - K[0]
    M = bmm(bmm(K[np.newaxis, ...], R), left_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_left = M[:, :, 0] - M[:, :, 2] * dets_2d[:, 0][..., None]

    coeff_right = np.stack([zeros, zeros, dets_2d[:, 2]], axis=-1) - K[0]
    M = bmm(bmm(K[np.newaxis, ...], R), right_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_right = M[:, :, 0] - M[:, :, 2] * dets_2d[:, 2][..., None]

    coeff_top = np.stack([zeros, zeros, dets_2d[:, 1]], axis=-1) - K[1]
    M = bmm(bmm(K[np.newaxis, ...], R), top_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_top = M[:, :, 1] - M[:, :, 2] * dets_2d[:, 1][..., None]

    coeff_bottom = np.stack([zeros, zeros, dets_2d[:, 3]], axis=-1) - K[1]
    M = bmm(bmm(K[np.newaxis, ...], R), bottom_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_bottom = M[:, :, 1] - M[:, :, 2] * dets_2d[:, 3][..., None]

    # shape(N, 4, 3)
    A = np.stack([coeff_left, coeff_top, coeff_right, coeff_bottom], axis=-2)
    A = np.expand_dims(A, axis=1)
    # shape(N, 64, 4, 3)
    A = np.repeat(A, num_cases, axis=1)
    # shape(N, 64, 4)
    b = np.stack([bias_left, bias_top, bias_right, bias_bottom], axis=-1)

    results_x = []
    errors = []
    for i in range(num):
        for j in range(num_cases):
            res = np.linalg.lstsq(A[i, j], b[i, j])
            results_x.append(res[0] - T)
            if (len(res[1])):
                errors.append(res[1])
            else:
                errors.append(np.zeros(1))

    results_x = np.stack(results_x, axis=0).reshape(num, num_cases, -1)
    errors = np.stack(errors, axis=0).reshape(num, num_cases)
    # idx = errors.argmax(axis=-1)
    # translation = results_x[np.arange(num), idx]
    idx = match(dets_2d, corners, results_x, R, p2)
    translation = results_x[np.arange(num), idx]

    translation = dets_3d[:, -3:]

    return np.concatenate(
        [dets_3d[:, :3], translation, ry[..., None]], axis=-1), None


def match(boxes_2d, corners, trans_3d, r, p):
    """
    Args:
        boxes_2d: shape(N, 4)
        corners: shape(N, 8, 3)
        trans_3d: shape(N, 64,3)
        ry: shape(N, 3, 3)
    """

    corners_3d = bmm(r, corners.transpose(0, 2, 1))
    trans_3d = np.repeat(np.expand_dims(trans_3d, axis=-2), 8, axis=-2)
    corners_3d = np.expand_dims(
        corners_3d.transpose((0, 2, 1)), axis=1) + trans_3d
    corners_3d = corners_3d.reshape(-1, 3)
    corners_3d_homo = np.hstack((corners_3d, np.ones(
        (corners_3d.shape[0], 1))))

    corners_2d = np.dot(p, corners_3d_homo.T)
    corners_2d_xy = corners_2d[:2, :] / corners_2d[2, :]

    corners_2d_xy = corners_2d_xy.reshape(2, -1, 8)
    xmin = corners_2d_xy[0, :, :].min(axis=-1)
    ymin = corners_2d_xy[1, :, :].min(axis=-1)
    xmax = corners_2d_xy[0, :, :].max(axis=-1)
    ymax = corners_2d_xy[1, :, :].max(axis=-1)

    batch_size = boxes_2d.shape[0]
    boxes_2d_proj = np.stack(
        [xmin, ymin, xmax, ymax], axis=-1).reshape(batch_size, -1, 4)
    # import ipdb
    # ipdb.set_trace()
    # import ipdb
    # ipdb.set_trace()
    bbox_overlaps = []
    for i in range(batch_size):
        bbox_overlaps.append(
            py_iou(boxes_2d[i][np.newaxis, ...], boxes_2d_proj[i]))
    bbox_overlaps = np.concatenate(bbox_overlaps, axis=0)
    idx = bbox_overlaps.argmax(axis=-1)
    return idx


def py_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: shape(N,4)
        boxes_b: shape(M,4)
    Returns:
        overlaps: shape(N, M)
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    boxes_a = np.repeat(np.expand_dims(boxes_a, 1), M, axis=1)
    boxes_b = np.repeat(np.expand_dims(boxes_b, 0), N, axis=0)

    xmin = np.maximum(boxes_a[:, :, 0], boxes_b[:, :, 0])
    ymin = np.maximum(boxes_a[:, :, 1], boxes_b[:, :, 1])
    xmax = np.minimum(boxes_a[:, :, 2], boxes_b[:, :, 2])
    ymax = np.minimum(boxes_a[:, :, 3], boxes_b[:, :, 3])

    w = xmax - xmin
    h = ymax - ymin
    w[w < 0] = 0
    h[h < 0] = 0

    inner_area = w * h
    boxes_a_area = py_area(boxes_a)
    boxes_b_area = py_area(boxes_b)

    iou = inner_area / (boxes_a_area + boxes_b_area - inner_area)
    return iou

def py_area(boxes):
    """
    Args:
        boxes: shape(N,M,4)
    """
    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]
    area = width * height
    return area
