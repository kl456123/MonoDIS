# -*- coding: utf-8 -*-

import torch


class DistanceSimilarityCalc(object):
    def compare_batch(self, anchors, gt_boxes):
        """
        Args:
            anchors:(N,4) or (batch_size,N,4)
            gt_boxes:(batch_size,M,4)
        Returns:
            match_quality_matrix: (batch_size,N,M)
        """
        # import ipdb
        # ipdb.set_trace()
        batch_size = gt_boxes.shape[0]
        M = gt_boxes.shape[1]
        if anchors.dim() == 2:
            N = anchors.shape[0]
            anchors = anchors.expand(batch_size, N, 4)
        if anchors.dim() == 3:
            anchors = anchors.view(batch_size, N, 1, 4).expand(batch_size, N,
                                                               M, 4)
            gt_boxes = gt_boxes.view(batch_size, 1, M, 4).expand(batch_size, N,
                                                                 M, 4)
            anchors_boxes_x = (anchors[:, :, :, 0] + anchors[:, :, :, 2]) / 2
            anchors_boxes_y = (anchors[:, :, :, 1] + anchors[:, :, :, 3]) / 2

            gt_ctr_x = (gt_boxes[:, :, :, 0] + gt_boxes[:, :, :, 2]) / 2
            gt_ctr_y = (gt_boxes[:, :, :, 1] + gt_boxes[:, :, :, 3]) / 2

            distance = torch.sqrt(
                (gt_ctr_x - anchors_boxes_x) * (gt_ctr_y - anchors_boxes_y))
            return distance
        else:
            raise ValueError('incorrect anchors dim!')
