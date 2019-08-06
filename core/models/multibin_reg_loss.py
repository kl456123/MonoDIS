# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from core.ops import get_angle
import torch.nn.functional as F


class MultiBinRegLoss(nn.Module):
    def __init__(self, num_bins=4, overlaps=1 / 6 * math.pi):
        super().__init__()
        self.num_bins = num_bins
        self.overlaps = overlaps

        # import ipdb
        # ipdb.set_trace()
        interval = 2 * math.pi / self.num_bins

        # bin centers
        bin_centers = torch.arange(0, self.num_bins) * interval
        bin_centers = bin_centers.cuda()
        cond = bin_centers > math.pi
        bin_centers[cond] = bin_centers[cond] - 2 * math.pi

        self.max_deltas = (interval + overlaps) / 2
        self.bin_centers = bin_centers

        self.l2_loss = nn.MSELoss(reduce=False)

    def generate_cls_targets(self, local_angle):
        """
        local_angle ranges from [-pi, pi]
        """
        # cls_targets = (local_angle >= self.left) & (local_angle < self.right)
        deltas = torch.abs(local_angle - self.bin_centers)
        cls_targets = (deltas <= self.max_deltas) | (
            deltas > 2 * math.pi - self.max_deltas)
        return cls_targets.long()

    def decode_angle(self, encoded_angle):
        pass

    def forward(self, preds, local_angle):
        """
        data format of preds: (N, num_bins * 2)
        data format of targets: local_angle
        Args:
            preds: shape(N,num*4)
            targets: shape(N,1)
        """
        # import ipdb
        # ipdb.set_trace()
        # generate prob target
        prob_targets = -torch.cos(self.bin_centers + local_angle)
        prob_targets[prob_targets < 0] = 0

        probs = F.softmax(preds.view(-1, self.num_bins, 2), dim=-1)

        l2_loss = self.l2_loss(probs[:, :, 1], prob_targets)
        return l2_loss.mean(dim=-1)
