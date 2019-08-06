# -*- coding: utf-8 -*-
"""
Pytorch version projector
"""

import torch


class Projector(object):
    def __init__(self):
        pass

    @classmethod
    def proj_point_3to2img(cls, pts_3d, p2):
        ones = torch.ones_like(pts_3d[:, -1:])
        pts_3d_homo = torch.cat([pts_3d, ones], dim=-1).transpose(1, 0)
        pts_2d_homo = p2.matmul(pts_3d_homo).transpose(2, 1)

        pts_2d_homo = pts_2d_homo / pts_2d_homo[:, :, -1:]
        return pts_2d_homo[:, :, :-1]

    @classmethod
    def proj_box_3to2gp():
        pass

    @classmethod
    def proj_box_3to2img(cls, target, p2):

        dims = target['dimension']
        locations = target['location']
        ry = target['ry']
        h = dims[:, 0]
        w = dims[:, 1]
        l = dims[:, 2]

        # 3d bounding box corners
        zeros = torch.zeros_like(l)
        x_corners = torch.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=-1)
        y_corners = torch.stack(
            [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=-1)
        z_corners = torch.stack(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            dim=-1)

        # rotation matrix
        c = torch.cos(ry)
        s = torch.sin(ry)
        ones = torch.ones_like(c)
        zeros = torch.zeros_like(c)
        R = torch.stack(
            [c, zeros, s, zeros, ones, zeros, -s, zeros, c], dim=-1).view(-1,
                                                                          3, 3)

        corners_3d = torch.stack(
            [x_corners, y_corners, z_corners], dim=-1).type_as(locations)
        # shape(N,3,8)
        corners_3d = torch.bmm(R, corners_3d.permute(
            0, 2, 1)) + locations.unsqueeze(-1)
        corners_3d = corners_3d.permute(0, 2, 1).contiguous()

        corners_2d = cls.proj_point_3to2img(corners_3d.view(-1, 3), p2).view(
            -1, 8, 2)

        # find the min bbox
        # corners_2d: shape(N, 8, 2)
        xmin, _ = torch.min(corners_2d[:, :, 0], dim=-1)
        ymin, _ = torch.min(corners_2d[:, :, 1], dim=-1)
        xmax, _ = torch.max(corners_2d[:, :, 0], dim=-1)
        ymax, _ = torch.max(corners_2d[:, :, 1], dim=-1)
        boxes_2d = torch.stack([xmin, ymin, xmax, ymax], dim=-1)

        return boxes_2d
