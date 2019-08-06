# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from core.anchor_generators.anchor_generator import AnchorGenerator
from lib.model.rpn.generate_anchors import generate_anchors

from utils.visualize import read_kitti

kitti_label_dir = '/data/object/training/label_2'


def read_boxes_from_label(kitti_label_dir, use_3d=False):
    all_boxes = []
    for label_file in os.listdir(kitti_label_dir):
        label_path = os.path.join(kitti_label_dir, label_file)
        boxes = read_kitti(label_path, pred=False, use_3d=use_3d)
        if boxes.size == 0:
            continue
        all_boxes.append(boxes)

    # x1,y1,x2,y2
    all_boxes = np.concatenate(all_boxes, axis=0)
    return all_boxes[:, :-1]


def analysis(all_boxes, anchors=None):
    w = all_boxes[:, 2] - all_boxes[:, 0]
    h = all_boxes[:, 3] - all_boxes[:, 1]
    r = h / w
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]
    a_r = a_h / a_w

    plt.scatter(np.arange(len(w)), w, color='red')
    plt.scatter(np.arange(len(a_w)), a_w, color='green')

    plt.show()
    plt.scatter(np.arange(len(h)), h, color='red')
    plt.scatter(np.arange(len(a_h)), a_h, color='green')

    plt.show()
    plt.scatter(np.arange(len(r)), r, color='red')
    plt.scatter(np.arange(len(a_r)), a_r, color='green')
    plt.show()


def data_vis(data):
    plt.scatter(np.arange(len(data)), data)
    plt.show()


def read_anchors():
    anchor_generator_config = {
        "base_anchor_size": 16,
        "scales": [4, 8, 16],
        "aspect_ratios": [0.5, 0.8, 1],
        "anchor_stride": [16, 16],
        "anchor_offset": [0, 0]
    }
    anchor_generator = AnchorGenerator(anchor_generator_config)

    anchors = anchor_generator.generate([[1, 1]])
    return anchors[0].cpu().numpy()


if __name__ == '__main__':
    all_boxes = read_boxes_from_label(kitti_label_dir)
    anchors = read_anchors()
    # all_boxes = np.concatenate([all_boxes, anchors])
    analysis(all_boxes, anchors=anchors)
