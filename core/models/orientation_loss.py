# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class OrientationLoss(nn.Module):
    def __init__(self, split_loss=False):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=-1)
        self.reg_loss = nn.MSELoss(reduce=False)
        self.split_loss = split_loss

    def forward(self, preds, targets):
        """
        Args:
            preds: shape()
            targets: shape(N, 3), (cls_orient(1), reg_orient(2),h_2d(1),c_2d(2))
        """
        # import ipdb
        # ipdb.set_trace()

        # cls loss
        cls_orient = targets[:, 0].long()
        cls_preds = preds[:, :3]
        cls_loss = self.cls_loss(cls_preds, cls_orient)

        # depth_ind_targets = targets[:, 3].long()
        # depth_ind_preds = preds[:, 4:4 + 11]
        # depth_ind_loss = self.cls_loss(depth_ind_preds, depth_ind_targets)

        # cls_orient_4s = targets[:, 7].long()
        # cls_orient_4s_preds = preds[:, 8:12]
        # cls_4s_loss = self.cls_loss(cls_orient_4s_preds, cls_orient_4s)

        # # import ipdb
        # # ipdb.set_trace()
        # center_orients = targets[:, 8].long()
        # center_orients_preds = preds[:, 12:]
        # center_orient_loss = self.cls_loss(center_orients_preds,
        # center_orients)

        # reg loss
        reg_orient = torch.cat([targets[:, 1:3]], dim=-1)
        reg_preds = torch.cat([preds[:, 3:5]], dim=-1)
        reg_loss = self.reg_loss(reg_preds, reg_orient)
        if self.split_loss:
            loss_dict = {
                'cls_orient_loss': cls_loss,
                'reg_orient_loss': reg_loss.sum(dim=-1),
                # 'depth_ind_loss': depth_ind_loss
                # 'h_2d_loss': reg_loss[:, 2],
                # 'c_2d_loss': reg_loss[:, 3:5].sum(dim=-1),
                # 'r_2d_loss': reg_loss[:, 5],
                # 'cls_orient_4_loss': cls_4s_loss,
                # 'center_orient': center_orient_loss
            }
            return loss_dict
        else:
            orient_loss = reg_loss.sum(dim=-1) + cls_loss
            return orient_loss
