# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super().__init__()


class WeightedSmoothL1Loss(nn.modules.loss.SmoothL1Loss):
    def __init__(self, reduction='elementwise_mean'):
        super().__init__(reduction='none')
        self.reduction = reduction

    def forward(self, input, target, weight):
        loss = super().forward(input, target)
        # dont need backward for weights,
        # it cames from input
        loss *= weight.detach()
        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()


class SharpL2Loss(Loss):
    def forward(self, reg, reg_targets):
        reg_deltas = torch.abs(reg - reg_targets)
        sign = reg_deltas < 1
        return sign.float() * 0.5 * torch.pow(reg_deltas, 2) + (
            ~sign).float() * (1 / 3 + torch.pow(reg_deltas, 3) + 1 / 6)


class ClusterLoss(Loss):
    def forward(self, sample_batch):
        """
        calculate loss of cluster
        Args:
            sample_batch: shape(N,M)
        Returns:
            a scalar of loss
        """
        if sample_batch.numel() == 0:
            return torch.tensor(0).type_as(sample_batch)
        batch_size = sample_batch.shape[0]
        mean_vector = sample_batch.mean(dim=0)
        mean_vector = mean_vector.expand(batch_size, -1)
        return torch.norm(sample_batch - mean_vector, 2, dim=-1).mean()
