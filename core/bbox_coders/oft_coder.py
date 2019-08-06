# -*- coding: utf-8 -*-
import torch
import math
from core.ops import get_angle
from torch.nn import functional as F

from core.avod import box_3d_encoder
from core.avod import box_4c_encoder


class OFTBBoxCoder(object):
    def __init__(self, coder_config):
        self.etha = coder_config['etha']
        self.dim_mean = coder_config['dim_mean']

    def encode_batch_bbox(self, voxel_centers, gt_boxes_3d):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            voxel_centers: shape(M,3)
            gt_boxes_3d: shape(N,M,7), (dim,pos,ry)
        """

        # encode dim
        h_3d_mean, w_3d_mean, l_3d_mean = self.dim_mean

        target_h_3d = torch.log(gt_boxes_3d[:, :, 0] / h_3d_mean)
        target_w_3d = torch.log(gt_boxes_3d[:, :, 1] / w_3d_mean)
        target_l_3d = torch.log(gt_boxes_3d[:, :, 2] / l_3d_mean)
        targets_dim = torch.stack(
            [target_h_3d, target_w_3d, target_l_3d], dim=-1)

        # encode pos
        targets_pos = (gt_boxes_3d[:, :, 3:6] - voxel_centers) / self.etha

        # encode angle
        # ry = gt_boxes_3d[:, :, -1]
        # targets_angle = torch.stack([torch.cos(ry), torch.sin(ry)], dim=-1)
        targets_angle = gt_boxes_3d[:, :, -1:]

        targets = torch.cat([targets_dim, targets_pos, targets_angle], dim=-1)
        return targets

    def decode_batch_bbox(self, voxel_centers, targets):

        # decode dim
        h_3d_mean, w_3d_mean, l_3d_mean = self.dim_mean
        targets_dim = targets[:, :, :3]
        h_3d = torch.exp(targets_dim[:, :, 0]) * h_3d_mean
        w_3d = torch.exp(targets_dim[:, :, 1]) * w_3d_mean
        l_3d = torch.exp(targets_dim[:, :, 2]) * l_3d_mean

        decoded_dims = torch.stack([h_3d, w_3d, l_3d], dim=-1)

        # decode pos
        targets_pos = targets[:, :, 3:6]
        decoded_pos = voxel_centers + targets_pos * self.etha

        # decode angle
        # targets_angle = targets[:, :, 6:]
        # ry = torch.atan2(targets_angle[:, :, 1],
        # targets_angle[:, :, 0]).unsqueeze(-1)

        return torch.cat([decoded_dims, decoded_pos], dim=-1)

    def encode_batch_labels(self, voxel_centers, gt_boxes_3d):
        """
        Args:
            voxel_centers: shape(N, 3)
            gt_labels: shape(num_batch, M)
            gt_boxes_3d: shape(num_batch, M, 7)
        """
        pos = gt_boxes_3d[:, :, 3:6]

        pos = pos.unsqueeze(2)

        gt_x = pos[:, :, :, 0]
        gt_z = pos[:, :, :, 2]

        voxel_centers = voxel_centers.unsqueeze(0).unsqueeze(0)

        voxel_x = voxel_centers[:, :, :, 0]
        voxel_z = voxel_centers[:, :, :, 2]

        # shape(num_batch, M, N)
        scores_map = torch.exp(-(torch.pow((gt_x - voxel_x), 2) + torch.pow(
            (gt_z - voxel_z), 2)) / (2 * self.etha * self.etha))
        scores_map = scores_map.max(dim=1)[0]
        return scores_map

    def decode_batch_angle_multibin(self, targets, bin_centers, num_bins):
        """
        Args:
            targets: shape(N, 3)
        """
        # find the best angle
        angles = targets.view(-1, num_bins, 4)
        angles_cls = F.softmax(angles[:, :, :2], dim=-1)
        _, angles_cls_argmax = torch.max(angles_cls[:, :, 1], dim=-1)
        row = torch.arange(
            0, angles_cls_argmax.shape[0]).type_as(angles_cls_argmax)
        angles_oritations = angles[:, :, 2:][row, angles_cls_argmax]

        # decode
        bin_centers = bin_centers[angles_cls_argmax]
        theta = get_angle(angles_oritations[:, 1], angles_oritations[:, 0])
        theta = bin_centers + theta
        return theta.unsqueeze(0).unsqueeze(-1)

    def decode_batch_angle_box_4c(self, targets):
        # fake ground plane
        targets = targets[0]
        ground_plane = torch.tensor([0, -1, 0, 1.6]).type_as(targets)
        targets_box_3d = box_4c_encoder.torch_box_4c_to_box_3d(targets,
                                                               ground_plane)
        return targets_box_3d[:, -1:].unsqueeze(0)

    def encode_batch_angle_box_4c(self, targets):
        targets = targets[0]
        # fake ground plane
        ground_plane = torch.tensor([0, -1, 0, 1.6]).type_as(targets)

        # convert targets to targets_box_3d
        pos = targets[:, 3:6]
        h = targets[:, 0]
        w = targets[:, 1]
        l = targets[:, 2]
        dims = torch.stack([l, w, h], dim=-1)
        targets_box_3d = torch.cat([pos, dims, targets[:, 6:]], dim=-1)

        # targets_box_4c
        targets_box_4c = box_4c_encoder.torch_box_3d_to_box_4c(targets_box_3d,
                                                               ground_plane)
        # targets_box_4c_fake
        targets_anchor = box_3d_encoder.torch_box_3d_to_anchor(targets_box_3d)
        targets_box_3d_fake = torch.cat([targets_anchor, targets[:, 6:]],
                                        dim=-1)
        targets_box_4c_fake = box_4c_encoder.torch_box_3d_to_box_4c(
            targets_box_3d_fake, ground_plane)

        #offsets, 10 columns
        offsets = box_4c_encoder.torch_box_4c_to_offsets(targets_box_4c,
                                                         targets_box_4c_fake)
        return offsets.unsqueeze(0)
