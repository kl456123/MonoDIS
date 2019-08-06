# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
from core.ops import get_angle
import torch.nn.functional as F


class MultiBinLoss(nn.Module):
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
        bin_centers = torch.arange(0, self.num_bins).float() * interval
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
        cos_deltas = torch.cos(local_angle - self.bin_centers)
        min_deltas = math.cos(self.max_deltas)
        cls_targets = cos_deltas >= min_deltas

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
        #  import ipdb
        #  ipdb.set_trace()
        preds = preds.contiguous().view(-1, self.num_bins, 4)
        # targets[targets < 0] = targets[targets < 0] + 2 * math.pi

        # generate cls target
        cls_targets = self.generate_cls_targets(targets)

        # cls loss
        angle_cls_loss = self.angle_cls_loss(
            preds[:, :, :2].contiguous().view(-1, 2), cls_targets.view(-1))
        angle_cls_loss = angle_cls_loss.view(-1, self.num_bins)
        # change from sum to mean
        angle_cls_loss = angle_cls_loss.mean(dim=-1)

        # residual loss
        # reg_targets = self.generate_reg_targets(targets)
        theta = get_angle(preds[:, :, 3], preds[:, :, 2])
        angle_reg_weights = cls_targets.detach().float()
        angle_reg_loss = -angle_reg_weights * torch.cos(
            targets - self.bin_centers - theta)
        num_covered = angle_reg_weights.sum(dim=-1)
        angle_reg_loss = 1 / num_covered * angle_reg_loss.sum(dim=-1)

        total_loss = angle_cls_loss + angle_reg_loss
        # total_loss = angle_cls_loss

        # import ipdb
        # ipdb.set_trace()
        # some stats for meansure the cls precision
        angle_cls_preds = preds[:, :, :2].detach()
        angle_cls_probs = F.softmax(angle_cls_preds, dim=-1)
        angle_cls_probs[angle_cls_probs >= 0.5] = 1
        angle_cls_probs[angle_cls_probs < 0.5] = 0
        tp_mask = 1 - (angle_cls_probs[:, :, 1].long() ^ cls_targets)

        return total_loss, tp_mask
