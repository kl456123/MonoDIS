# -*- coding: utf-8 -*-
"""
This script is just used for merge 2d and 3d detection
"""
import os
import numpy as np
import sys
sys.path.append('.')
from utils.kitti_util import Calibration, Object3d

root_path = '../'

results_dir_3d = os.path.join(root_path, 'Detection_3D/results/box_3d')
results_dir_2d = os.path.join(root_path, 'Detection_2D/results/data')
results_dir_merge = './results/data'
calib_dir = '..//hw_3d/kitti_format/calib'


class Object_3d():
    def __init__(self, sample_name):
        self.sample_name = sample_name
        self._box_3d = self.parse_kitti_3d(sample_name)

    def parse_kitti_3d(self, sample_name):
        pass

    def convert_to_8c(self):
        pass


def parse_kitti_3d(sample_name, results_dir=results_dir_3d):
    file_path = os.path.join(results_dir, '{}.txt'.format(sample_name))
    lines = [line.rstrip() for line in open(file_path)]
    objs = [Object3d(line) for line in lines]

    boxes_3d = [obj.box3d for obj in objs]
    points_3ds = []
    for box_3d in boxes_3d:
        [ry, l, h, w, x, y, z] = box_3d
        xmin = x - 1 / 2 * l
        xmax = x + 1 / 2 * l
        ymin = y - 1 / 2 * h
        ymax = y + 1 / 2 * h
        zmin = z - 1 / 2 * w
        zmax = z + 1 / 2 * w
        points_3d = np.stack(
            np.meshgrid([xmin, xmax], [ymin, ymax], [zmin, zmax]), axis=-1)
        points_3d = points_3d.reshape((-1, 3))
        points_3ds.append(points_3d)

    if len(boxes_3d) == 0:
        return np.zeros((0, 8, 3)), np.zeros((0, 7))
    return np.stack(points_3ds, axis=0), np.stack(boxes_3d, axis=0)


def convert_to_homogeneous(points_3d):
    """
    Args:
    points_3d: shape(N,3)
    Returns:
    homogeneous: shape(N,4)
    """
    # points_homo = np.zeros((points_3d.shape[0], 4))
    # points_homo[:, :3] = points_3d
    return np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1))], axis=-1)


def convert_to_non_homogeneous(points_homo):
    """
    Args:
    points_homo: shape(N,4)
    Returns:
    points_3d: shape(N,3)
    """
    return points_homo[:, :-1] / points_homo[:, -1:]


def parse_kitti_2d(sample_name):
    file_path = os.path.join(results_dir_2d, '{}.txt'.format(sample_name))
    lines = [line.rstrip() for line in open(file_path)]
    objs = [Object3d(line) for line in lines]
    boxes_2d = [obj.box2d for obj in objs]
    scores = [objs[0].score for obj in objs]
    if len(boxes_2d) == 0:
        return np.zeros((0, 4)), np.zeros((0, 5))

    boxes_2d = np.vstack(boxes_2d)
    scores = np.asarray(scores).reshape((-1, 1))
    dets_2d = np.concatenate([boxes_2d, scores], axis=-1)
    return boxes_2d, dets_2d


def points2d_to_bbox(points_2d):
    """
    non homogeneous
    Args:
    points_2d: shape(N,8,2)
    Returns:
    bbox: shape(N,4)
    """

    # shape(N,)
    xmin = np.min(points_2d[:, :, 0], axis=-1)
    xmax = np.max(points_2d[:, :, 0], axis=-1)
    ymin = np.min(points_2d[:, :, 1], axis=-1)
    ymax = np.max(points_2d[:, :, 1], axis=-1)
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


def project_to_image(obj_3d, sample_name=None, calib_matrix=None):
    if sample_name is not None:
        sample_name = 'calib_new'
        calib_file_path = os.path.join(calib_dir, '{}.txt'.format(sample_name))
        calib = Calibration(calib_file_path)

        # shape(3,4)
        calib_matrix = calib.P

    # cast 3d coords to homogeneous coords
    box_8points_3d = convert_to_homogeneous(obj_3d.reshape((-1, 3)))

    # project to image
    box_8points_2d = np.matmul(box_8points_3d, calib_matrix.T)

    # convert back to non_homo
    box_8points_2d = convert_to_non_homogeneous(box_8points_2d)

    # non_homo, so the last axis is 2
    box_8points_2d = box_8points_2d.reshape((-1, 8, 2))
    # find the bbox for the 8 points
    # shape(N,4)
    bbox = points2d_to_bbox(box_8points_2d)
    return bbox


def area_batch(boxes):
    return (boxes[:, :, 3] - boxes[:, :, 1]) * (
        boxes[:, :, 2] - boxes[:, :, 0])


def bbox_overlaps_batch(boxes_a, boxes_b):
    """
    Args:
    boxes_a: shape(N,M,4)
    boxes_b: shape(N,M,4)
    Returns:
    overlaps: shape(N,M)
    """
    xmax = np.minimum(boxes_a[:, :, 2], boxes_b[:, :, 2])
    ymax = np.minimum(boxes_a[:, :, 3], boxes_b[:, :, 3])
    xmin = np.maximum(boxes_a[:, :, 0], boxes_a[:, :, 0])
    ymin = np.maximum(boxes_a[:, :, 1], boxes_b[:, :, 1])
    iw = xmax - xmin
    ih = ymax - ymin
    iw[iw < 0] = 0
    ih[ih < 0] = 0

    area_a = area_batch(boxes_a)
    area_b = area_batch(boxes_b)
    return ih * iw / (area_a + area_b - ih * iw)


def match(obj_2ds, obj_2d_from_3d):
    """
    Note that if no any obj_2d_from_3d is matched with obj_2ds,assign it -1
    Args:
    obj_2ds: shape(N,4)
    obj_2d_from_3d: shape(M,4)
    Returns:
    assignments: shape(N)
    """
    N = obj_2ds.shape[0]
    M = obj_2d_from_3d.shape[0]
    obj_2ds = np.repeat(np.expand_dims(obj_2ds, axis=1), M, axis=1)
    obj_2d_from_3d = np.repeat(
        np.expand_dims(
            obj_2d_from_3d, axis=0), N, axis=0)

    # shape(N,M)
    overlaps = bbox_overlaps_batch(obj_2ds, obj_2d_from_3d)
    assignments = np.argmax(overlaps, axis=1)
    max_overlaps = np.max(overlaps, axis=1)

    # no anyone matchs
    assignments[max_overlaps == 0] = -1
    return assignments


def save_dets_kitti(dets, sample_name):
    label_path = os.path.join(results_dir_merge, sample_name + '.txt')
    class_name = 'Car'
    res_str = []
    kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
    with open(label_path, 'w') as f:
        for det in dets:
            xmin, ymin, xmax, ymax, cf, ry, l, h, w, x, y, z = det
            res_str.append(
                kitti_template.format(class_name, xmin, ymin, xmax, ymax, h, w,
                                      l, x, y, z, ry, cf))
        f.write('\n'.join(res_str))


def merge(sample_name):

    obj_2ds, dets_2ds = parse_kitti_2d(sample_name)
    obj_3ds, obj_3ds_info = parse_kitti_3d(sample_name)
    if obj_2ds.shape[0] == 0 or obj_3ds.shape[0] == 0:
        return

    obj_2d_from_3d = project_to_image(obj_3ds, sample_name)
    assignments = match(obj_2ds, obj_2d_from_3d)

    # merge 2d and 3d according to assignments
    # 1. remove that is not matched
    dets_2ds = dets_2ds[assignments > -1]
    assignments = assignments[assignments > -1]

    # 2. merge obj_3ds_info(rlhwxyz) and dets_2ds(xyxyc)
    # find the matched 3d
    obj_3ds_info = obj_3ds_info[assignments]

    obj = np.concatenate([dets_2ds, obj_3ds_info], axis=-1)

    # 3. save
    save_dets_kitti(obj, sample_name)


def read_data_file(data_file):
    with open(data_file) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def main():
    # merge all
    lines = read_data_file('../hw_3d/kitti_format/val.txt')
    for idx, line in enumerate(lines):
        # try:
        merge(line)
        # except:
        # pass
        #  sys.stdout('skip error')
        sys.stdout.write('\r{}'.format(idx))
        sys.stdout.flush()


if __name__ == '__main__':
    #  sample_name = '000001'
    #  merge(sample_name)
    main()
