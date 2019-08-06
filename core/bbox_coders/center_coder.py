# -*- coding: utf-8 -*-
import torch


class CenterCoder(object):
    def __init__(self, coder_config):
        self.bbox_normalize_targets_precomputed = torch.tensor(
            coder_config['bbox_normalize_targets_precomputed'])
        if self.bbox_normalize_targets_precomputed:
            self.bbox_normalize_means = torch.tensor(
                coder_config['bbox_normalize_means'])
            self.bbox_normalize_stds = torch.tensor(
                coder_config['bbox_normalize_stds'])

    def decode_batch(self, deltas, boxes):
        if self.bbox_normalize_targets_precomputed:
            dtype = deltas.type()
            # Optionally normalize targets by a precomputed mean and stdev
            deltas = (
                deltas * self.bbox_normalize_stds.expand_as(deltas).type(dtype)
                + self.bbox_normalize_means.expand_as(deltas).type(dtype))

        reg_targets_batch = self._decode_batch(deltas, boxes)

        return reg_targets_batch

    def _decode_batch(self, deltas, boxes):
        """
        Args:
            deltas: shape(N,K*A,4)
            boxes: shape(N,K*A,4)
        """
        if boxes.dim() == 3:
            pass
        elif boxes.dim() == 2:
            boxes = boxes.expand_as(deltas)
        else:
            raise ValueError("The dimension of boxes should be 3 or 2")
        widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0::4]
        dy = deltas[:, :, 1::4]
        dw = deltas[:, :, 2::4]
        dh = deltas[:, :, 3::4]

        pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
        pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
        pred_w = torch.exp(dw) * widths.unsqueeze(2)
        pred_h = torch.exp(dh) * heights.unsqueeze(2)

        pred_boxes = deltas.clone()
        # x1
        pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def encode_batch(self, bboxes, assigned_gt_boxes):
        reg_targets_batch = self._encode_batch(bboxes, assigned_gt_boxes)

        if self.bbox_normalize_targets_precomputed:
            dtype = reg_targets_batch.type()
            # Optionally normalize targets by a precomputed mean and stdev
            reg_targets_batch = (
                (reg_targets_batch - self.bbox_normalize_means.expand_as(
                    reg_targets_batch).type(dtype)) /
                self.bbox_normalize_stds.expand_as(reg_targets_batch).type(
                    dtype))

        return reg_targets_batch

    def _encode_batch(self, ex_rois, gt_rois):

        if ex_rois.dim() == 2:
            ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
            ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
            ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
            ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

            gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
            gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
            gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
            gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

            targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -1).expand_as(gt_ctr_x)
                          ) / ex_widths
            targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -1).expand_as(gt_ctr_y)
                          ) / ex_heights
            targets_dw = torch.log(gt_widths /
                                   ex_widths.view(1, -1).expand_as(gt_widths))
            targets_dh = torch.log(gt_heights / ex_heights.view(
                1, -1).expand_as(gt_heights))

        elif ex_rois.dim() == 3:
            ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
            ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
            ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
            ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

            gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
            gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
            gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
            gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

            targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = torch.log(gt_widths / ex_widths)
            targets_dh = torch.log(gt_heights / ex_heights)
        else:
            raise ValueError('ex_roi input dimension is not correct.')

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh),
                              2)

        return targets
