# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from core.ops import get_angle
import torch.nn.functional as F


class MultiBinNewLoss(nn.Module):
    def __init__(self, num_bins=4, overlaps=1 / 6 * math.pi, angle_offset=0):
        super().__init__()
        self.num_bins = num_bins
        self.angle_cls_loss = nn.CrossEntropyLoss(reduce=False)
        self.overlaps = overlaps
        self.offset = angle_offset

        # import ipdb
        # ipdb.set_trace()
        interval = 2 * math.pi / self.num_bins

        # bin centers
        bin_centers = torch.arange(0, self.num_bins) * interval
        bin_centers = bin_centers.cuda()
        bin_centers += self.offset
        cond = bin_centers > math.pi
        bin_centers[cond] = bin_centers[cond] - 2 * math.pi

        self.max_deltas = (interval + overlaps) / 2
        # self.left = bin_centers - 1 / 2 * interval
        # self.right = bin_centers + 1 / 2 * interval
        self.bin_centers = bin_centers

    def generate_cls_targets(self, local_angle):
        """
        local_angle ranges from [-pi, pi]
        """
        angle_dist = -torch.cos(local_angle - self.bin_centers)
        #  angle_dist = -math.cos(self.max_deltas)
        _, cls_targets = torch.min(angle_dist, dim=-1)
        #  cls_targets = cos_deltas >= min_deltas

        return cls_targets.long()

    def decode_angle(self, encoded_angle):
        pass

    def forward(self, preds, targets):
        """
        data format of preds: num_bins * (conf, sin, cos)
        data format of targets: local_angle
        Args:
            preds: shape(N,num*4)
            targets: shape(N,1)
        """
        # import ipdb
        # ipdb.set_trace()
        preds = preds.contiguous().view(-1, self.num_bins, 3)
        # targets[targets < 0] = targets[targets < 0] + 2 * math.pi

        # generate cls target
        cls_targets = self.generate_cls_targets(targets)

        # cls loss
        angle_cls_loss = self.angle_cls_loss(preds[:, :, 0].contiguous(),
                                             cls_targets.view(-1))
        #  angle_cls_loss = angle_cls_loss.view(-1, self.num_bins)
        # change from sum to mean
        #  angle_cls_loss = angle_cls_loss.mean(dim=-1)

        # residual loss
        # reg_targets = self.generate_reg_targets(targets)
        theta = get_angle(preds[:, :, 2], preds[:, :, 1])
        col_inds = cls_targets.detach()
        #  row = torch.arange(
        #  0, angles_cls_argmax.shape[0]).type_as(angles_cls_argmax)
        row_inds = torch.arange(0, col_inds.shape[0]).type_as(col_inds)
        angle_reg_loss = -torch.cos(targets - self.bin_centers - theta)
        angle_reg_loss = angle_reg_loss[row_inds, col_inds]
        # num_covered = angle_reg_weights.sum(dim=-1)
        # angle_reg_loss = angle_reg_loss.sum(dim=-1)

        total_loss = angle_cls_loss + angle_reg_loss
        # total_loss = angle_cls_loss

        # import ipdb
        # ipdb.set_trace()
        # some stats for meansure the cls precision
        angle_cls_preds = preds[:, :, 0].detach()
        angle_cls_probs = F.softmax(angle_cls_preds, dim=-1)
        _, cls_preds = torch.max(angle_cls_probs, dim=-1)
        # angle_cls_probs[angle_cls_probs >= 0.5] = 1
        # angle_cls_probs[angle_cls_probs < 0.5] = 0
        tp_mask = cls_preds == cls_targets

        return total_loss, tp_mask
