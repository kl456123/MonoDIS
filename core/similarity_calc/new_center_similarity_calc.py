#!/usr/bin/env python
# encoding: utf-8

import torch


class NewCenterSimilarityCalc(object):
    def __init__(self):
        pass

    def compare(self, anchors, gt_boxes):
        """
        anchors: (N, 4) ndarray of float
        gt_boxes: (K, 4) ndarray of float

        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        N = anchors.size(0)
        K = gt_boxes.size(0)

        gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
                         (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)).view(1, K)

        # anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) *
        # (anchors[:, 3] - anchors[:, 1] + 1)).view(N, 1)

        boxes = anchors.view(N, 1, 4).expand(N, K, 4)
        query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

        iw = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2]) - torch.max(
            boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, 3], query_boxes[:, :, 3]) - torch.max(
            boxes[:, :, 1], query_boxes[:, :, 1]) + 1)
        ih[ih < 0] = 0

        # ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / gt_boxes_area

        return overlaps

    def compare_batch(self, anchors, gt_boxes):
        """
        anchors: (N, 4) ndarray of float
        gt_boxes: (b, K, 5) ndarray of float

        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        batch_size = gt_boxes.size(0)

        if anchors.dim() == 2:

            N = anchors.size(0)
            K = gt_boxes.size(1)

            anchors = anchors.view(1, N, 4).expand(batch_size, N,
                                                   4).contiguous()
            gt_boxes = gt_boxes[:, :, :4].contiguous()

            gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
            gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
            gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

            anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
            anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
            # anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size,
            # N, 1)

            gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
            anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

            boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K,
                                                             4)
            query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size,
                                                                    N, K, 4)

            iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
                  torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
                  torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
            ih[ih < 0] = 0
            # ua = anchors_area + gt_boxes_area - (iw * ih)
            overlaps = iw * ih / gt_boxes_area

            # mask the overlap here.
            overlaps.masked_fill_(
                gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K),
                0)
            overlaps.masked_fill_(
                anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N,
                                                                K), -1)

        elif anchors.dim() == 3:
            N = anchors.size(1)
            K = gt_boxes.size(1)

            if anchors.size(2) == 4:
                anchors = anchors[:, :, :4].contiguous()
            else:
                anchors = anchors[:, :, 1:5].contiguous()

            gt_boxes = gt_boxes[:, :, :4].contiguous()

            gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
            gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
            gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

            anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
            anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
            # anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size,
            # N, 1)

            gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
            anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

            boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K,
                                                             4)
            query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size,
                                                                    N, K, 4)

            iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
                  torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
            iw[iw < 0] = 0

            ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
                  torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
            ih[ih < 0] = 0
            # ua = anchors_area + gt_boxes_area - (iw * ih)

            overlaps = iw * ih / gt_boxes_area

            # mask the overlap here.
            overlaps.masked_fill_(
                gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K),
                0)
            overlaps.masked_fill_(
                anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N,
                                                                K), -1)
        else:
            raise ValueError('anchors input dimension is not correct.')

        return overlaps
