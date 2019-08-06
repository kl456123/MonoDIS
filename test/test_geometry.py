# -*- coding: utf-8 -*-

import torch
import numpy as np
from utils import geometry_utils
from test.test_bbox_coder import build_dataset


def test_geometry():

    dataset = build_dataset()
    for sample in dataset:
        # img_name = sample['img_name']
        # if img_name =='/data/object/training/image_2/001017.png':
            # import ipdb
            # ipdb.set_trace()
        # else:
            # continue

        label_boxes_3d = sample['gt_boxes_3d']
        p2 = torch.from_numpy(sample['p2'])
        label_boxes_3d = torch.cat(
            [
                label_boxes_3d[:, 3:6], label_boxes_3d[:, :3],
                label_boxes_3d[:, 6:]
            ],
            dim=-1)

        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        front_mid = corners_3d[:, [0, 1]].mean(dim=1)
        rear_mid = corners_3d[:, [2, 3]].mean(dim=1)
        points_3d = torch.cat([rear_mid, front_mid], dim=0)
        points_2d = geometry_utils.torch_points_3d_to_points_2d(points_3d, p2)

        lines = points_2d.contiguous().view(2, -1, 2).permute(
            1, 0, 2).contiguous().view(-1, 4)
        # import ipdb
        # ipdb.set_trace()
        ry_pred1 = geometry_utils.torch_pts_2d_to_dir_3d_v2(
            lines.unsqueeze(0), p2.unsqueeze(0))[0]
        # ry_pred2 = geometry_utils.torch_dir_to_angle()
        # deltas = points_3d[1]-points_3d[0]
        # ry_pred2 = -torch.atan2(deltas[2], deltas[0])
        ry_gt = label_boxes_3d[:, -1]
        height = label_boxes_3d[:, 1]
        ry_gt[height<0] = geometry_utils.reverse_angle(ry_gt[height<0])
        cond = torch.abs(ry_pred1 - ry_gt) < 1e-4
        assert cond.all(), '{} error {} {}'.format(sample['img_name'], ry_gt, ry_pred1)


if __name__ == '__main__':
    test_geometry()
