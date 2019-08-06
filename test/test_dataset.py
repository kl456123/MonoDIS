# -*- coding: utf-8 -*-
import torch
import numpy as np
from test.test_bbox_coder import build_dataset
from utils.drawer import ImageVisualizer
from utils import geometry_utils


def main():

    normal_mean = np.asarray([0.485, 0.456, 0.406])
    normal_van = np.asarray([0.229, 0.224, 0.225])
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
        label_boxes_3d = sample['gt_boxes_3d']
        label_boxes_2d = sample['gt_boxes']
        label_classes = sample['gt_labels']
        p2 = torch.from_numpy(sample['p2'])
        image_path = sample['img_name']

        label_boxes_3d = torch.cat(
            [
                label_boxes_3d[:, 3:6], label_boxes_3d[:, :3],
                label_boxes_3d[:, 6:]
            ],
            dim=-1)
        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(label_boxes_3d)
        image = sample['img'].permute(1, 2, 0).cpu().detach().numpy()
        image = image.copy()
        image = image * normal_van + normal_mean
        # import ipdb
        # ipdb.set_trace()
        corners_3d = corners_3d.cpu().detach().numpy()
        visualizer.render_image_corners_2d(
            image_path, image, corners_3d=corners_3d, p2=p2)

if __name__=='__main__':
    main()
