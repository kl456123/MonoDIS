# -*- coding: utf-8 -*-

from data.transforms import kitti_bev_transform, kitti_transform


def build(transform_config):
    """
    all kinds of transform are generated here according to its type
    """
    dataset_type = transform_config
    if dataset_type == 'kitti_bev':
        kitti_bev_transform(transform_config)
    elif dataset_type == 'kitti':
        kitti_transform(transform_config)
    else:
        raise ValueError('unknown tranform type')
