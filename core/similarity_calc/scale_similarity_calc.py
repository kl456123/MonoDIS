# -*- coding: utf-8 -*-

import torch


class ScaleSimilarityCalc(object):
    def compare(self, anchors, gt_boxes):
        pass

    def compare_batch(self, anchors, gt_boxes, use_dist=True):
        """
        Args:
            anchors:(N,4) or (batch_size,N,4)
            gt_boxes:(batch_size,M,4)
        Returns:
            match_quality_matrix: (batch_size,N,M)
        """
        batch_size = gt_boxes.shape[0]
        M = gt_boxes.shape[1]

        if anchors.dim() == 2:
            N = anchors.shape[0]
            anchors = anchors.view(1, N, 4).expand(batch_size, N, 4)

        if anchors.dim() == 3:
            N = anchors.shape[1]
            # get center of anchors
            anchors = anchors.view(batch_size, N, 1, 4).expand(batch_size, N,
                                                               M, 4)
            ctr_x = (anchors[:, :, :, 2] + anchors[:, :, :, 0]) / 2
            ctr_y = (anchors[:, :, :, 3] + anchors[:, :, :, 1]) / 2

            query_boxes = gt_boxes.view(batch_size, 1, M, 4).expand(batch_size,
                                                                    N, M, 4)
            x_cond = (ctr_x > query_boxes[:, :, :, 0]) & (
                ctr_x < query_boxes[:, :, :, 2])
            y_cond = (ctr_y > query_boxes[:, :, :, 1]) & (
                ctr_y < query_boxes[:, :, :, 3])
            cond = x_cond & y_cond
            # for debug
            # assert cond.sum()

            if use_dist:
                # calculate the distance between centers of gt boxes and anchors
                gt_ctr_x = (gt_boxes[:, :, 0] + gt_boxes[:, :, 2]) / 2
                gt_ctr_y = (gt_boxes[:, :, 1] + gt_boxes[:, :, 3]) / 2
                distance = torch.sqrt((ctr_x - gt_ctr_x) * (ctr_x - gt_ctr_x) +
                                      (ctr_y - gt_ctr_y) * (ctr_y - gt_ctr_y))
                # use 1e5 to repreasent inf
                theta = 1e-5
                match_quality_matrix = 1 / (distance + theta)
                match_quality_matrix[~cond] = 0
                return match_quality_matrix
        else:
            raise ValueError("dim of anchors is not correct!")

        match_quality_matrix = torch.zeros(batch_size, N, M)
        match_quality_matrix[cond] = 1
        return match_quality_matrix
