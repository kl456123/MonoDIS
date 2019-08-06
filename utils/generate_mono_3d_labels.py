# -*- coding: utf-8 -*-

from utils.box_vis import load_projection_matrix, compute_box_3d
from utils.kitti_util import Object3d
import os
import numpy as np

data_dir = '/data/object/training'


def parse_data_file(data_file_path):
    with open(data_file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def parse_kitti(label_path):
    lines = [line.rstrip() for line in open(label_path)]
    objs = [Object3d(line) for line in lines]

    # 3d
    boxes_3d = [obj.box3d for obj in objs]
    boxes_2d = [obj.box2d for obj in objs]

    if len(boxes_3d) == 0:
        return np.zeros((0, 7))

    if len(boxes_2d) == 0:
        return np.zeros((0, 4))
    return np.stack(boxes_2d, axis=0), np.stack(boxes_3d, axis=0)


def main():
    data_file = 'val.txt'
    data_file_path = os.path.join(data_dir, data_file)

    lines = parse_data_file(data_file_path)
    for sample_name in lines:
        label_path = os.path.join(data_dir,
                                  'label_2/{}.txt'.format(sample_name))
        img_path = os.path.join(data_dir, 'image_2/{}.png'.format(sample_name))
        calib_path = os.path.join(data_dir, 'calib/{}.txt'.format(sample_name))

        # p2
        p2 = load_projection_matrix(calib_path)

        # label
        boxes_2d, boxes_3d = parse_kitti(label_path)

        import ipdb
        ipdb.set_trace()
        for i in range(boxes_3d.shape[0]):
            target = {}
            target['ry'] = boxes_3d[i, 0]
            target['dimension'] = boxes_3d[i, 1:4]
            target['location'] = boxes_3d[i, 4:]

            corners_xy = compute_box_3d(target, p2)
            coords = corners_xy[[0, 1, 3]].reshape(-1)
            label = np.append(coords, corners_xy[4, 1])


if __name__ == '__main__':
    main()
