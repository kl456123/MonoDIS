# -*- coding: utf-8 -*-
"""
use geometry constrain to detection pedestrain in 3d space
"""

import json
import time
import sys
import os
import numpy as np
from PIL import Image

from builder.dataloader_builders.kitti_mono_3d_dataloader_builder import Mono3DKittiDataLoaderBuilder
from utils.postprocess import mono_3d_postprocess_bbox as postprocess
from core.tester import save_dets

PEDESTRIAN_DIMS_MEAN = [1.76070648, 0.66018943, 0.84228437]
CYCLIST_DIMS_MEAN = [1.73720345, 0.5967732, 1.76354641]

DIMS_MEAN = {'Pedestrian': PEDESTRIAN_DIMS_MEAN, 'Cyclist': CYCLIST_DIMS_MEAN}

CLASSES = ['Car', 'Pedestrian', 'Cyclist']


def read_config_json(json_file):
    with open(json_file) as f:
        config = json.load(f)
    return config


def read_labels(label_dir, img_dir):
    for file in os.listdir(img_dir):
        #  import ipdb
        #  ipdb.set_trace()
        file = '{}.txt'.format(os.path.basename(file)[:-4])
        label_file = os.path.join(label_dir, file)
        with open(label_file) as f:
            lines = f.readlines()
            lines = [line.strip().split(',') for line in lines]

        objs_2d = []
        for line in lines:
            obj_class = line[-1]
            if int(obj_class) == 5:
                objs_2d.append(line[:4])

        objs_2d = np.asarray(objs_2d).astype(np.float32)

        img_name = '{}.jpg'.format(label_file[:-4])
        data = {}
        data['img_name'] = img_name
        data['gt_boxes'] = objs_2d
        yield data


def build_dataloader(img_dir):
    for img_name in os.listdir(img_dir):
        img_file = os.path.join(img_dir, img_name)
        img = Image.open(img_file)
        data = {'img_file': img_file}


def main():
    classes = CLASSES[1]
    data_config_file = './configs/pedestrain_kitti_config.json'
    data_config = read_config_json(data_config_file)
    data_config['eval_data_config']['dataset_config']['classes'][0] = classes
    #  import ipdb
    #  ipdb.set_trace()
    data_loader_builder = Mono3DKittiDataLoaderBuilder(
        data_config['eval_data_config'], training=True)
    data_loader = data_loader_builder.build()
    num_samples = len(data_loader)

    label_dir = '/data/liangxiong/pedestrian_data/det_result'
    img_dir = '/data/liangxiong/pedestrian_data/data'
    data_loader = read_labels(label_dir, img_dir)

    p2 = np.asarray([[1057.46, 0, 1002.12, 0], [0, 1057.11, 427.939, 0],
                     [0, 0, 1, 0]]).reshape(3, 4)

    for i, data in enumerate(data_loader):
        start_time = time.time()
        img_file = data['img_name']
        dets = []

        #  gt_boxes = data['gt_boxes'][0].cpu().numpy()
        #  gt_boxes_3d = data['gt_boxes_3d'][0].cpu().numpy()
        #  p2 = data['p2'][0].detach().cpu().numpy()
        gt_boxes = data['gt_boxes']
        if gt_boxes.shape[0] == 0:
            continue

        #  import ipdb
        #  ipdb.set_trace()

        cls_dets_gt = np.concatenate(
            [gt_boxes, np.zeros_like(gt_boxes[:, -1:])], axis=-1)
        mean_dims = np.asarray([DIMS_MEAN[classes]])
        num = gt_boxes.shape[0]
        mean_dims = np.tile(mean_dims, num).reshape(num, -1)

        rcnn_3d_gt, _ = postprocess(mean_dims, cls_dets_gt, p2)
        dets.append(np.concatenate([cls_dets_gt, rcnn_3d_gt], axis=-1))
        save_dets(dets[0], img_file, 'kitti', 'results/data')

        duration_time = time.time() - start_time
        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
