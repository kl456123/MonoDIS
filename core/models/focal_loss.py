#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: YangMaoke, DuanZhixiang({maokeyang, zhixiangduan}@deepmotion.ai)
# Focal loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot_embeding(labels, num_classes):
    """Embeding labels to one-hot form.

    Args:
        labels(LongTensor): class labels
        num_classes(int): number of classes
    Returns:
        encoded labels, sized[N, #classes]

    """

    y = torch.eye(num_classes)  # [D, D]
    return y[labels]  # [N, D]


class FocalLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 alpha=0.5,
                 gamma=0,
                 auto_alpha=False,
                 clip_alpha=40):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.auto_alpha = auto_alpha
        self.clip_alpha = clip_alpha

    def focal_loss(self, x, y):
        """Focal loss

        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
        Returns:
            (tensor): focal loss

        """

        t = one_hot_embeding(y.data.cpu(), self.num_classes)
        t = Variable(t).cuda()  # [N, 20]

        logit = F.softmax(x, dim=-1)
        logit = logit.clamp(1e-7, 1. - 1e-7)
        conf_loss_tmp = -1 * t.float() * torch.log(logit)

        # use gamma to balance easy and hard
        conf_loss_tmp = conf_loss_tmp * (1 - logit)**self.gamma

        if self.auto_alpha:
            alpha = self.calculate_alpha(t, conf_loss_tmp.detach())
        else:
            if isinstance(self.alpha, (float, int)):
                fg_num_classes = self.num_classes - 1
                alpha = torch.FloatTensor([self.alpha] + [(
                    1 - self.alpha) / fg_num_classes] * fg_num_classes)
            else:
                alpha = torch.FloatTensor(self.alpha)

            # insance check
            assert len(
                alpha
            ) == self.num_classes, 'alpha should be determined for each classes'

        # expand it like as logit
        alpha = alpha.expand_as(conf_loss_tmp).type_as(conf_loss_tmp)

        # use alpha to balance fg and bg
        conf_loss = alpha * conf_loss_tmp
        conf_loss = conf_loss.sum(dim=-1)

        return conf_loss

    def calculate_alpha(self, t, conf_loss):
        """
        Args:
            conf_loss: shape(N,D)
        """
        if conf_loss.requires_grad:
            conf_loss = conf_loss.detach()

        conf_loss = conf_loss.sum(dim=-1)

        # reweight conf loss for balance pos and neg(auto adjust alpha)
        total_loss = conf_loss.sum()
        neg_loss = conf_loss[t[:, 0] == 1].sum()
        # make sure total loss is the same
        neg_alpha = 0.5 * total_loss / neg_loss
        pos_alpha = 0.5 * total_loss / (total_loss - neg_loss)
        # clip alpha by thresh(it matters)
        pos_alpha = torch.clamp(pos_alpha, min=0, max=self.clip_alpha)
        neg_alpha = torch.clamp(neg_alpha, min=0, max=self.clip_alpha)
        # print(
        # "neg_alpha/pos_alpha({:.4f}/{:.4f})".format(neg_alpha, pos_alpha))
        # neg_alpha = 0.25
        # pos_alpha = 1 - neg_alpha

        fg_num_classes = self.num_classes - 1
        alpha = torch.FloatTensor([neg_alpha] + [pos_alpha / fg_num_classes] *
                                  fg_num_classes)
        return alpha

    def forward(self, cls_preds, cls_targets, ignored_label=-1,
                is_print=False):
        """
        Args:
            cls_preds: shape(N[N1,N2,...],num_classes)
            cls_targets: shape(N[N1,N2,...])
        """

        # shape(N*K,num_classes)
        masked_cls_preds = cls_preds.view(-1, self.num_classes)
        # shape(N*K,)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets)

        # ignore loss of ignroed_label
        ignored_mask = cls_targets == ignored_label
        cls_loss[ignored_mask] = 0

        # shape(N,K)
        cls_loss = cls_loss.view_as(cls_targets)

        if is_print:
            print(('cls_loss: %.3f' % (cls_loss.mean())))
        return cls_loss
