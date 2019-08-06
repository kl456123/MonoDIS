# -*- coding: utf-8 -*-

import torch


class DeltaCoder(object):
    def __init__(self, coder_config):
        pass

    def decode_batch(self, deltas, boxes):
        """
        Args:
            deltas: shape(N,K*A,4)
            boxes: shape(N,K*A,4)
        Returns:
        """
        if boxes.dim() == 3:
            pass
        elif boxes.dim() == 2:
            boxes = boxes.expand_as(deltas)
        else:
            raise ValueError("The dimension of boxes should be 3 or 2")
        widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
        xmin = deltas[:, :, 0] * widths + boxes[:, :, 0]
        ymin = deltas[:, :, 1] * heights + boxes[:, :, 1]
        xmax = deltas[:, :, 2] * widths + boxes[:, :, 2]
        ymax = deltas[:, :, 3] * heights + boxes[:, :, 3]
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

    def encode_batch(self, ex_rois, gt_rois):
        """
        """
        if ex_rois.dim() == 2:
            ex_rois = ex_rois.expand_as(gt_rois)

        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
        xmin_target = (gt_rois[:, :, 0] - ex_rois[:, :, 0]) / ex_widths
        ymin_target = (gt_rois[:, :, 1] - ex_rois[:, :, 1]) / ex_heights
        xmax_target = (gt_rois[:, :, 2] - ex_rois[:, :, 2]) / ex_widths
        ymax_target = (gt_rois[:, :, 3] - ex_rois[:, :, 3]) / ex_heights
        return torch.stack(
            [xmin_target, ymin_target, xmax_target, ymax_target], dim=-1)
