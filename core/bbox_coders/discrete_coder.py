# -*- coding: utf-8 -*-
import torch


class DiscreteBBoxCoder(object):
    def __init__(self, coder_config):
        self.iou_anchors = coder_config['iou_anchors']
        self.iou_intervals = coder_config['iou_intervals']
        self.num_levels = len(self.iou_intervals)

    def encode_reg(self, reg_targets):
        """
        Args:
            reg_targets: shape(N,M)
        Returns:
            encoded_reg_targets: shape(N,M,4)
        """
        iou_anchors = self.iou_anchors
        num_levels = len(iou_anchors)
        res = []
        for i in range(num_levels):
            res.append(reg_targets - iou_anchors[i])
        return torch.stack(res, dim=-1)

    def _label_mask(self, iou_reg_targets, i):
        iou_intervals = self.iou_intervals
        interval = iou_intervals[i]
        return (iou_reg_targets > interval[0]) & (
            iou_reg_targets < interval[1])

    def encode_cls(self, iou_reg_targets):
        """
        Args:
            iou_reg_targets: shape(N,M)
        """
        # iou_anchors = [0.05, 0.25, 0.55, 0.85]
        iou_intervals = self.iou_intervals

        iou_scores_targets = torch.zeros_like(iou_reg_targets)
        for i in range(len(iou_intervals)):
            iou_scores_targets[self._label_mask(iou_reg_targets, i)] = i
        return iou_scores_targets.long()

    def decode_batch(self, iou_cls, iou_reg):
        """
        decode iou from cls and reg predictions
        Args:
            iou_cls: shape(N,M,4)
            iou_reg: shape(N,M,4)
        Returns:
            decoded_iou: shape(N,M)
        """
        # import ipdb
        # ipdb.set_trace()
        if iou_cls.dim() == 2:
            iou_cls = iou_cls.unsqueeze(dim=0)
            size = iou_cls.shape[1]
        else:
            size = iou_cls.shape[:2]
        if iou_reg.dim() == 2:
            iou_reg = iou_reg.unsqueeze(dim=0)
        iou_anchors = self.iou_anchors
        num_levels = iou_cls.shape[-1]
        decoded_iou = []
        for i in range(num_levels):
            decoded_iou.append(iou_cls[:, :, i] *
                               (iou_anchors[i] + iou_reg[:, :, i]))
        decoded_iou = torch.stack(decoded_iou, dim=-1)
        decoded_iou = decoded_iou.sum(dim=-1)
        return decoded_iou.view(size)
