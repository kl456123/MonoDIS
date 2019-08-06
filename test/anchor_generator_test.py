# -*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append('.')
from core.anchor_generators.anchor_generator import AnchorGenerator
from lib.model.rpn.generate_anchors import generate_anchors
from utils.visualize import visualize_bbox, read_img, shift_bbox

anchor_generator_config = {
    "base_anchor_size": 1,
    "scales": [4],
    "aspect_ratios": [1],
    "anchor_stride": [16, 16],
    "anchor_offset": [0, 0]
}
anchor_generator = AnchorGenerator(anchor_generator_config)

anchors = anchor_generator.generate([[24, 80]])
# print(anchors)

expect_anchors = generate_anchors(
    base_size=anchor_generator_config['base_anchor_size'],
    ratios=np.array(anchor_generator_config['aspect_ratios']),
    scales=np.array(anchor_generator_config['scales']))

img = read_img('/data/object/training/image_2/000117.png')


def vis_help(anchors, expect_anchors):
    # shift_bbox(anchors, translation=(200, 200))
    # shift_bbox(expect_anchors, translation=(800, 200))
    # anchors = np.concatenate([anchors, expect_anchors], axis=0)
    visualize_bbox(img, anchors)


vis_help(anchors[0], expect_anchors)
# print(expect_anchors)

import matplotlib.pyplot as plt


def data_vis(data):
    plt.scatter(np.arange(len(data)), data)
    plt.show()


def analysis(all_boxes):
    w = all_boxes[:, 2] - all_boxes[:, 0]
    h = all_boxes[:, 3] - all_boxes[:, 1]
    data_vis(w)
    data_vis(h)


# analysis(anchors[0])
