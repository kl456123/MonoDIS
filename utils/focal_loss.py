#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: YangMaoke, DuanZhixiang({maokeyang, zhixiangduan}@deepmotion.ai)
# Focal loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def one_hot_embeding(labels, num_classes):
    """Embeding labels to one-hot form.

    Args:
        labels(LongTensor): class labels
        num_classes(int): number of classes
    Returns:
        encoded labels, sized[N, #classes]

    """

    y = torch.eye(num_classes)  # [D, D]
    labels_array = labels.cpu().numpy()
    return y[labels]  # [N, D]


class FocalLoss(nn.Module):
    def __init__(self, num_classes=21):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        """Focal loss

        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
        Returns:
            (tensor): focal loss

        """

        alpha = 0.25
        gamma = 2

        t = one_hot_embeding(y.data.cpu(), self.num_classes)
        t = Variable(t).cuda()  # [N, 20]

        logit = F.softmax(x)
        logit = logit.clamp(1e-7, 1.-1e-7)
        conf_loss_tmp = -1 * t.float() * torch.log(logit)
        conf_loss_tmp = alpha * conf_loss_tmp * (1-logit)**gamma
        conf_loss = conf_loss_tmp.sum()

        return conf_loss

    def forward(self, cls_preds, cls_targets, is_print=False):
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds(tensor): predicted locations, sized [batch_size, #anchors, 4].
          loc_targets(tensor): encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds(tensor): predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets(tensor): encoded target labels, sized [batch_size, #anchors].
        Returns:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).

        """

        pos = (cls_targets == 1)  # [N,#anchors]
        num_pos = pos.data.long().sum()

        pos_neg = cls_targets > -1  # exclude ignored anchors
        masked_cls_preds = cls_preds.view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        num_pos_neg = pos_neg.data.long().sum()

        num_pos_neg = max(1.0, num_pos_neg)
        num_pos = max(1.0, num_pos)
        if is_print:
            print(('cls_loss: %.3f' % (cls_loss.data[0] / num_pos)))
        loss = cls_loss / num_pos

        return loss
