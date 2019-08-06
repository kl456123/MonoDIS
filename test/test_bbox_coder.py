# -*- coding: utf-8 -*-
import numpy as np

import torch
from core.bbox_coders.bbox_3d_coder import BBox3DCoder
from data.datasets.kitti_mono_3d import Mono3DKittiDataset
from utils.drawer import ImageVisualizer
import data.transforms.kitti_transform as trans

from utils import geometry_utils

normal_mean = np.asarray([0.485, 0.456, 0.406])
normal_van = np.asarray([0.229, 0.224, 0.225])


def build_dataset():
    dataset_config = {
        "classes": ["Car"],
        "cache_bev": False,
        "dataset_file": "train.txt",
        "root_path": "/data",
        "use_proj_2d": False,
        "use_rect_v2": False
    }

    trans_cfg = {
        "crop_size": [384, 1280],
        "normal_mean": [0.485, 0.456, 0.406],
        "normal_van": [0.229, 0.224, 0.225],
        "random_blur": 0,
        "random_brightness": 10,
        "resize_range": [0.2, 0.4]
    }
    transform = trans.Compose([
        # trans.RandomHorizontalFlip(),
        # trans.Resize(trans_cfg['crop_size']),
        # trans.RandomHSV(),
        trans.Boxes3DTo2D(),
        trans.ToTensor(),
        trans.Normalize(trans_cfg['normal_mean'], trans_cfg['normal_van'])
    ])

    dataset = Mono3DKittiDataset(dataset_config, transforms=transform)

    return dataset


def build_visualizer():
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=True,
        save_dir=save_dir)
    return visualizer


def test_bbox_coder():

    bbox_coder = BBox3DCoder({})
    dataset = build_dataset()

    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=True,
        save_dir=save_dir)

    for sample in dataset:
        mean_dims = torch.from_numpy(sample['mean_dims'][None])
        label_boxes_3d = sample['gt_boxes_3d']
        label_boxes_2d = sample['gt_boxes']
        label_classes = sample['gt_labels']
        p2 = torch.from_numpy(sample['p2'])
        bbox_coder.mean_dims = mean_dims

        encoded_corners_2d = bbox_coder.encode_batch_bbox(
            label_boxes_3d, label_boxes_2d, label_classes, p2)

        # side_lines = encoded_corners_2d[:, 16:20]

        # encoded_corners_2d = torch.cat(
        # [
        # encoded_corners_2d[:, :6], encoded_corners_2d[:, 6:11],
        # encoded_corners_2d[:, 10:11], encoded_corners_2d[:, 11:16],
        # encoded_corners_2d[:, 15:16]
        # ],
        # dim=-1)

        decoded_corners_2d = bbox_coder.decode_batch_bbox(
            encoded_corners_2d, label_boxes_2d, p2)

        boxes_3d = torch.cat(
            [
                decoded_corners_2d[:, 6:9], decoded_corners_2d[:, 3:6],
                decoded_corners_2d[:, -1:]
            ],
            dim=-1)
        # boxes_3d = decoded_corners_2d
        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(boxes_3d)

        corners_3d = corners_3d.cpu().detach().numpy()

        # import ipdb
        # ipdb.set_trace()
        # image_path = sample[]
        image_path = sample['img_name']
        image = sample['img'].permute(1, 2, 0).cpu().detach().numpy()
        image = image.copy()
        image = image * normal_van + normal_mean
        # image = None
        # corners_2d = torch.cat([side_lines] * 4, dim=-1).view(-1, 8, 2)
        # corners_2d = corners_2d.cpu().detach().numpy()
        visualizer.render_image_corners_2d(
            image_path, image, corners_3d=corners_3d, p2=p2)


if __name__ == '__main__':
    test_bbox_coder()
