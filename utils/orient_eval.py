# -*- coding: utf-8 -*-
"""
Evaluate the orientation in approximately
"""

import numpy as np
import os
from utils.postprocess import py_iou
from numpy.linalg import norm

# some common setting
label_dir = '/data/object/training/label_2'
dets_dir = 'results/data/'
P2 = np.asarray([[
    7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
    7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
    1.000000e+00, 2.745884e-03
]]).reshape((3, 4))
# metric with different distance
dist_intervals = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 1000]]


def read_labels(label_dir, sample_name, classes=['Car']):
    label_file = os.path.join(label_dir, '{}.txt'.format(sample_name))
    with open(label_file, 'r') as f:
        lines = f.readlines()

    box_2ds = []
    box_3ds = []
    for line in lines:
        items = line.strip().split(' ')
        if not items[0] in classes:
            continue
        # 4
        box_2d = items[4:8]

        # 7
        box_3d = items[8:15]

        box_2ds.append(box_2d)
        box_3ds.append(box_3d)

    if len(box_2ds):
        box_2ds = np.stack(box_2ds, axis=0).astype(np.float32)
        box_3ds = np.stack(box_3ds, axis=0).astype(np.float32)
    else:
        box_2ds = np.zeros((0, 4))
        box_3ds = np.zeros((0, 7))
    return box_2ds, box_3ds


def match(box_2ds_det, box_2ds_label):
    bbox_overlaps = py_iou(box_2ds_det, box_2ds_label)

    max_overlaps = np.max(bbox_overlaps, axis=-1)
    assignments = np.argmax(bbox_overlaps, axis=-1)

    iou_thresh = 0.7
    assignments[max_overlaps < iou_thresh] = -1
    return assignments


def get_orient(box_3ds_det, P2):
    K = P2[:3, :3]
    KT = P2[:, -1]
    T = np.dot(np.linalg.inv(K), KT)

    ry = box_3ds_det[:, -1]

    K = P2[:3, :3]
    l = 4
    location = box_3ds_det[:, 3:6]

    zeros = np.zeros_like(ry)
    ones = np.ones_like(ry)
    num = ones.shape[0]
    R = np.stack(
        [
            np.cos(ry), zeros, np.sin(ry), zeros, ones, zeros, -np.sin(ry),
            zeros, np.cos(ry)
        ],
        axis=-1).reshape(num, 3, 3)

    location1 = location
    location2 = np.dot(R, np.asarray([l, 0, 0])) + location1

    # their projections
    homo_2d_1 = np.dot(K, location1.T).T
    homo_2d_2 = np.dot(K, location2.T).T

    point_2d_1 = homo_2d_1 / homo_2d_1[:, -1:]
    point_2d_2 = homo_2d_2 / homo_2d_2[:, -1:]

    point_2d_1 = point_2d_1[:, :-1]
    point_2d_2 = point_2d_2[:, :-1]

    direction = point_2d_2 - point_2d_1

    cls_orient = direction[:, 0] * direction[:, 1] > 0
    cls_orient = cls_orient.astype(np.int32)

    cls_orient[direction[:, 0] == 0] = -1

    return cls_orient


def cls_orient_ap(cls_orient_pred, cls_orient_gt):
    """
    two classes, 1 or 0
    Args:
        cls_orient_pred: shape(N,1)
        cls_orient_gt: shape(N,1) -1 means dont care
    """
    cls_orient_mask = cls_orient_pred == cls_orient_gt
    mask = cls_orient_gt > -1
    cls_orient_mask = cls_orient_mask[mask]

    tp = cls_orient_mask.sum()
    # total = cls_orient_mask.size

    # ap = tp / total
    #  if not ap == 1:
    #  import ipdb
    #  ipdb.set_trace()

    return tp


def mean_distance_error(box_3ds_det, box_3ds_gt):
    loc_det = box_3ds_det[:, 3:6]
    loc_gt = box_3ds_gt[:, 3:6]

    return norm(loc_det - loc_gt, axis=-1)


def mean_depth_error(box_3ds_det, box_3ds_gt):
    loc_det_2d = box_3ds_det[:, 5]
    loc_gt_2d = box_3ds_gt[:, 5]

    return np.abs(loc_det_2d - loc_gt_2d)


def mean_iou_3d():
    pass


def distance_impact_error(box_3ds_det, box_3ds_gt):
    loc_det = box_3ds_det[:, 3:6]
    loc_gt = box_3ds_gt[:, 3:6]

    dist_det = norm(loc_det, axis=-1)
    dist_gt = norm(loc_gt, axis=-1)

    return np.abs(dist_det - dist_gt)


def calc_orient_ap_single(sample_name):

    #  sample_name = '000001'

    box_2ds_label, box_3ds_label = read_labels(label_dir, sample_name)
    box_2ds_det, box_3ds_det = read_labels(dets_dir, sample_name)

    if box_2ds_det.shape[0] * box_2ds_label.shape[0] == 0:
        return 0, np.zeros((5, 3)), np.zeros(1 + 5)

    assignments = match(box_2ds_det, box_2ds_label)

    box_3ds_label_matched = box_3ds_label[assignments]

    matched = assignments > -1
    if not matched.any():
        return 0, np.zeros((5, 3)), np.zeros(1 + 5)
    box_3ds_label_matched = box_3ds_label_matched[matched]
    box_2ds_det = box_2ds_det[matched]
    box_3ds_det = box_3ds_det[matched]

    ##############################
    # statisic
    #############################
    # 1. orient cls ap
    cls_orient_pred = get_orient(box_3ds_det, P2)
    cls_orient_gt = get_orient(box_3ds_label_matched, P2)
    orient_ap = cls_orient_ap(cls_orient_pred, cls_orient_gt)

    dist_stats = []
    nums = [cls_orient_pred.shape[0]]
    for dist_interval in dist_intervals:
        # import ipdb
        # ipdb.set_trace()
        dist_filter = (box_3ds_label_matched[:, 5] >= dist_interval[0]) & (
            box_3ds_label_matched[:, 5] < dist_interval[1])
        num = dist_filter[dist_filter].size
        if num == 0:
            stats = [0, 0, 0]
        else:
            # 2. mean distance error
            MDE = mean_distance_error(box_3ds_det[dist_filter],
                                      box_3ds_label_matched[dist_filter])

            # 3. distance impact error
            MIE = distance_impact_error(box_3ds_det[dist_filter],
                                        box_3ds_label_matched[dist_filter])

            # 4 mean depth error
            MDepE = mean_depth_error(box_3ds_det[dist_filter],
                                     box_3ds_label_matched[dist_filter])

            # 5. 3d iou
            # IOU = mean_iou_3d()
            stats = [MDE.sum(), MIE.sum(), MDepE.sum()]

        dist_stats.append(stats)
        nums.append(num)

    dist_stats = np.asarray(dist_stats)

    return orient_ap, dist_stats, np.asarray(nums)


def main():
    ap = 0
    total = np.zeros(5 + 1)
    dist_stats = np.zeros((5, 3))
    for dets_file in os.listdir(dets_dir):
        sample_name = os.path.splitext(dets_file)[0]
        orient_ap, dist_stat, num = calc_orient_ap_single(sample_name)
        ap += orient_ap
        dist_stats += dist_stat
        total += num

    ap /= total[0]
    for ind, stat in enumerate(dist_stats):
        dist_stats[ind] /= total[ind + 1]

    print('orient ap: {:.4f}'.format(ap))
    for ind, stat in enumerate(dist_stats):
        print('distance interval: [{:.2f}, {:.2f})'.format(
            dist_intervals[ind][0], dist_intervals[ind][1]))
        print('mean distance error: {:.4f}'.format(dist_stats[ind][0]))
        print('distance impact error: {:.4f}'.format(dist_stats[ind][1]))
        print('mean depth error: {:.4f}\n'.format(dist_stats[ind][2]))


if __name__ == '__main__':
    main()
