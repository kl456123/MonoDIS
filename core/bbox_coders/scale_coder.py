# -*- coding: utf-8 -*-

import torch


class ScaleCoder(object):
    def __init__(self, coder_config):
        self.bbox_normalize_targets_precomputed = torch.tensor(
            coder_config['bbox_normalize_targets_precomputed'])
        if self.bbox_normalize_targets_precomputed:
            self.bbox_normalize_means = torch.tensor(
                coder_config['bbox_normalize_means'])
            self.bbox_normalize_stds = torch.tensor(
                coder_config['bbox_normalize_stds'])

    def encode_batch(self, anchors, gt_boxes):
        """
        Args:
            anchors: shape(num,4) or shape(N,num,4),
                one row refers to (x1,y1,x2,y2)
            gt_boxes: shape(N,num,4) (x1,y1,x2,y2)
        Returns:
            encoded_boxes: shape(N,num,4)
        """
        if anchors.dim() == 2:
            anchors = anchors.expand_as(gt_boxes)

        if anchors.dim() == 3:
            anchor_ws_half = (anchors[:, :, 2] - anchors[:, :, 0] + 1) / 2
            anchor_hs_half = (anchors[:, :, 3] - anchors[:, :, 1] + 1) / 2
            anchor_ctr_x = (anchors[:, :, 0] + anchors[:, :, 2]) / 2
            anchor_ctr_y = (anchors[:, :, 3] + anchors[:, :, 1]) / 2

            target_lefts = torch.log(
                (anchor_ctr_x - gt_boxes[:, :, 0]) / anchor_ws_half)
            target_rights = torch.log(
                (gt_boxes[:, :, 2] - anchor_ctr_x) / anchor_ws_half)
            target_tops = torch.log(
                (anchor_ctr_y - gt_boxes[:, :, 1]) / anchor_hs_half)
            target_downs = torch.log(
                (gt_boxes[:, :, 3] - anchor_ctr_y) / anchor_hs_half)

            targets = torch.stack(
                (target_lefts, target_rights, target_tops, target_downs),
                dim=2)
            return targets
        else:
            raise ValueError('the first input dimension is not correct.')

    def decode_batch(self, deltas, boxes):
        """
        Args:
            deltas: shape(N,num,4)
            boxes: shape(N,num,4)
        Returns:
            decoded_boxes: shape(N,num,4)
        """
        # import ipdb
        # ipdb.set_trace()
        if boxes.dim() == 3:
            pass
        elif boxes.dim() == 2:
            boxes = boxes.expand_as(deltas)
        else:
            raise ValueError("The dimension of boxes should be 3 or 2")

        boxes_ws_half = (boxes[:, :, 2] - boxes[:, :, 0] + 1) / 2 + 1
        boxes_hs_half = (boxes[:, :, 3] - boxes[:, :, 1] + 1) / 2 + 1
        boxes_ctr_x = (boxes[:, :, 0] + boxes[:, :, 2]) / 2
        boxes_ctr_y = (boxes[:, :, 1] + boxes[:, :, 3]) / 2

        decoded_lefts = torch.exp(deltas[:, :, 0]) * boxes_ws_half
        decoded_rights = torch.exp(deltas[:, :, 1]) * boxes_ws_half
        decoded_tops = torch.exp(deltas[:, :, 2]) * boxes_hs_half
        decoded_downs = torch.exp(deltas[:, :, 3]) * boxes_hs_half

        decoded_xmin = boxes_ctr_x - decoded_lefts
        decoded_ymin = boxes_ctr_y - decoded_tops
        decoded_xmax = boxes_ctr_x + decoded_rights
        decoded_ymax = boxes_ctr_y + decoded_downs

        decoded = torch.stack(
            (decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax), dim=2)
        return decoded
