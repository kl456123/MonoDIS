# -*- coding: utf-8 -*-
import torch
from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc


def outside_window_filter(boxes, im_shape):
    # import ipdb
    # ipdb.set_trace()

    # find the total outside boxes
    out_filter = (boxes[:, :, 2] < 0) | (boxes[:, :, 3] < 0) | (
        boxes[:, :, 0] >= im_shape[0][1]) | (boxes[:, :, 1] >= im_shape[0][0])
    in_filter = ~out_filter
    return in_filter


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes = boxes.clone()
    boxes[boxes < 0] = 0

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    size = boxes.size()
    boxes = boxes.view(-1, 4)

    boxes[:, 0][boxes[:, 0] > batch_x] = batch_x
    boxes[:, 1][boxes[:, 1] > batch_y] = batch_y
    boxes[:, 2][boxes[:, 2] > batch_x] = batch_x
    boxes[:, 3][boxes[:, 3] > batch_y] = batch_y

    boxes = boxes.view(size)

    return boxes


def window_filter(anchors, window, allowed_border=0):
    """
    Args:
        anchors: Tensor(N,4) ,xxyy format
        window: length-2 list (h,w)
    Returns:
        keep: Tensor(N,) bool type
    """
    keep = ((anchors[:, 0] >= -allowed_border) &
            (anchors[:, 1] >= -allowed_border) &
            (anchors[:, 2] < window[1] + allowed_border) &
            (anchors[:, 3] < window[0]) + allowed_border)
    return keep


def size_filter(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
    hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
    mask = (ws >= min_size) & (hs >= min_size)
    return mask


def area(boxes):
    """
    Get Area of batch bboxes
    Args:
        boxes: shape(N,M,4)
    """
    w = boxes[:, :, 2] - boxes[:, :, 0] + 1
    h = boxes[:, :, 3] - boxes[:, :, 1] + 1
    return w * h


def iou(boxes, gt_boxes):
    """
    Args:
        boxes: shape(N,M,4)
        gt_boxes: shape(N,M,4)
    """
    boxes_area = area(boxes)
    gt_boxes_area = area(gt_boxes)
    intersection_boxes = intersection(boxes, gt_boxes)
    area_intersection = area(intersection_boxes)
    return area_intersection / (boxes_area + gt_boxes_area - area_intersection)


def intersection(bbox, gt_boxes):
    """
    Args:
        bbox: shape(N,M,4)
        gt_boxes: shape(N,M,4)
    Returns:
        intersection: shape(N,M,4)
    """
    xmin = torch.max(gt_boxes[:, :, 0], bbox[:, :, 0])
    ymin = torch.max(gt_boxes[:, :, 1], bbox[:, :, 1])
    xmax = torch.min(gt_boxes[:, :, 2], bbox[:, :, 2])
    ymax = torch.min(gt_boxes[:, :, 3], bbox[:, :, 3])

    # if no intersection
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    cond = (w < 0) | (h < 0)
    # xmin[cond] = 0
    # xmax[cond] = 0
    # ymin[cond] = 0
    # ymax[cond] = 0
    inter_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    inter_boxes[cond] = 0
    return inter_boxes


def super_nms(bboxes, nms_thresh=0.8, nms_num=2, loop_time=1):
    """
    Args:
        bboxes: shape(N,4)
    """
    similarity_calc = CenterSimilarityCalc()
    # expand batch dim
    bboxes = bboxes.unsqueeze(0)
    # shape(N,M,M)
    match_quality_matrix = similarity_calc.compare_batch(bboxes, bboxes)
    # squeeze back
    # shape(M,M)
    match_quality_matrix = match_quality_matrix[0]
    bboxes = bboxes[0]

    match_mask = match_quality_matrix > nms_thresh
    match_quality_matrix[match_mask] = 1
    match_quality_matrix[~match_mask] = 0

    for i in range(loop_time):
        # shape(M,)
        match_num = match_quality_matrix.sum(dim=-1)
        remain_unmatch_inds = torch.nonzero(match_num <= nms_num + 1)[:, 0]
        match_quality_matrix = match_quality_matrix.transpose(0, 1)
        match_quality_matrix[remain_unmatch_inds] = 0
        match_quality_matrix = match_quality_matrix.transpose(0, 1)

    match_num = match_quality_matrix.sum(dim=-1)
    # exclude self
    remain_match_inds = torch.nonzero(match_num > (nms_num + 1))
    if remain_match_inds.numel():
        remain_match_inds = remain_match_inds[:, 0]

    return remain_match_inds


def horizontal_flip_box(boxes, image_width):
    xmin = boxes[:, 0]
    xmax = boxes[:, 1]
    new_xmin = image_width - xmax
    new_xmax = image_width - xmin
    boxes = boxes.copy()
    boxes[:, 0] = new_xmin
    boxes[:, 2] = new_xmax

    return boxes
