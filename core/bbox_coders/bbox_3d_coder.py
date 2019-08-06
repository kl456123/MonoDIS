# -*- coding: utf-8 -*-
import torch
import math
from core.ops import get_angle
from torch.nn import functional as F

from utils import geometry_utils
from core.utils import tensor_utils
from utils import encoder_utils


class BBox3DCoder(object):
    def __init__(self, coder_config):
        self.mean_dims = None

    def decode_batch(self, deltas, boxes):
        """
        Args:
            deltas: shape(N,K*A,4)
            boxes: shape(N,K*A,4)
        """
        pass
        # if boxes.dim() == 3:

    # pass
    # elif boxes.dim() == 2:
    # boxes = boxes.expand_as(deltas)
    # else:
    # raise ValueError("The dimension of boxes should be 3 or 2")
    # widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    # heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    # ctr_x = boxes[:, :, 0] + 0.5 * widths
    # ctr_y = boxes[:, :, 1] + 0.5 * heights

    # dx = deltas[:, :, 0::4]
    # dy = deltas[:, :, 1::4]
    # dw = deltas[:, :, 2::4]
    # dh = deltas[:, :, 3::4]

    # pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    # pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    # pred_w = torch.exp(dw) * widths.unsqueeze(2)
    # pred_h = torch.exp(dh) * heights.unsqueeze(2)

    # pred_boxes = deltas.clone()
    # # x1
    # pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # # y1
    # pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # # x2
    # pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # # y2
    # pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    # return pred_boxes

    # def encode_batch(self, bboxes, assigned_gt_boxes):
    # reg_targets_batch = self._encode_batch(bboxes, assigned_gt_boxes)

    # return reg_targets_batch

    def encode_batch(self, boxes_2d, coords):
        """
        Note that bbox_3d is just some points in image about 3d bbox
        Args:
            bbox_2d: shape(N,4)
            bbox_3d: shape(N,7)
        """
        center_x = (boxes_2d[:, 2] + boxes_2d[:, 0]) / 2
        center_y = (boxes_2d[:, 3] + boxes_2d[:, 1]) / 2
        center = torch.stack([center_x, center_y], dim=-1)
        w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
        h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)
        dims = torch.stack([w, h], dim=-1)

        bbox_3d = coords[:, :-1].view(-1, 3, 2)
        bbox_3d = (bbox_3d - center.unsqueeze(1)) / dims.unsqueeze(1)
        y = (coords[:, -1:] - center[:, 1:]) / dims[:, 1:]
        coords = torch.cat([bbox_3d.view(-1, 6), y], dim=-1)

        # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh),
        # 2)

        return coords

    def encode_batch_dims(self, boxes_2d, dims):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            boxes_2d: shape(N,)
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """
        w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
        h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)

        target_h = torch.log(dims[:, 0] / h)
        target_w = torch.log(dims[:, 1] / w)
        target_l = torch.log(dims[:, 2] / w)

        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        target_h_3d = (dims[:, 3] - h_3d_mean) / h_3d_std
        target_w_3d = (dims[:, 4] - w_3d_mean) / w_3d_std
        target_l_3d = (dims[:, 5] - l_3d_mean) / l_3d_std
        targets = torch.stack(
            [
                target_h, target_w, target_l, target_h_3d, target_w_3d,
                target_l_3d
            ],
            dim=-1)
        return targets

    def reorder_lines(self, lines):

        # sort by x
        _, order = torch.sort(lines[..., 0], dim=-1)

        lines = tensor_utils.multidim_index(lines, order)
        return lines

    def find_nearest_between_two_lines(self, lines, lines_2d):
        lines = lines.view(-1, 2, 2, 3)
        lines_2d = lines_2d.view(-1, 2, 2, 2)
        mid_points = lines.mean(dim=2)  # (N, 2, 3)
        dist = mid_points.norm(dim=-1)

        _, visible_index = torch.min(dist, dim=-1)
        row = torch.arange(visible_index.numel()).type_as(visible_index)
        # may be one of them or may be none of them
        near_side = lines_2d[row, visible_index]

        # import ipdb
        # ipdb.set_trace()
        # calc visible
        left_slope = geometry_utils.torch_line_to_orientation(
            lines_2d[:, 0, 0], lines_2d[:, 0, 1])
        right_slope = geometry_utils.torch_line_to_orientation(
            lines_2d[:, 1, 0], lines_2d[:, 1, 1])
        visible_cond = left_slope * right_slope > 0

        near_side = self.reorder_lines(near_side)

        return near_side, visible_cond

    def encode_batch_bbox(self, gt_boxes_3d, proposals, assigned_gt_labels,
                          p2):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """
        # import ipdb
        # ipdb.set_trace()
        location = gt_boxes_3d[:, 3:6]
        dims = gt_boxes_3d[:, :3]
        ry = gt_boxes_3d[:, 6:]

        # ray_angle = -torch.atan2(location[:, 2], location[:, 0])
        # local_ry = ry - ray_angle.unsqueeze(-1)
        center_depth = location[:, -1:]
        center_2d = geometry_utils.torch_points_3d_to_points_2d(location, p2)

        targets = torch.cat([dims, ry, center_depth, center_2d], dim=-1)
        return targets

    def encode_batch_angle(self, dims, assigned_gt_labels):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """

        bg_mean_dims = torch.zeros_like(self.mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, self.mean_dims], dim=1)
        assigned_mean_dims = mean_dims[0][assigned_gt_labels].float()
        assigned_std_dims = torch.ones_like(assigned_mean_dims)
        targets = (dims[:, :3] - assigned_mean_dims) / assigned_std_dims

        targets = torch.cat([targets, dims[:, 3:]], dim=-1)
        return targets

    def encode_batch_keypoint(self, keypoint, num_intervals, rois_batch):
        x = keypoint[:, 0]
        keypoint_type = keypoint[:, 2].long()

        rois = rois_batch[0, :, 1:]

        num_bbox = rois.shape[0]
        x_start = rois[:, 0]
        w = rois[:, 2] - rois[:, 0] + 1
        x_stride = w / num_intervals
        x_offset = torch.round((x - x_start) / x_stride).long()
        keypoint_gt = torch.zeros((num_bbox, 4 * 28)).type_as(rois_batch)
        x_index = keypoint_type * 28 + x_offset
        row_ind = torch.arange(0, num_bbox).type_as(x_index)
        keypoint_gt[row_ind, x_index] = 1
        return keypoint_gt

    def decode_batch_dims(self, targets, rois_batch):
        """
        Args:
            boxes_2d: shape(N,)
            targets: shape(N,)
        """
        # dims
        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        h_3d = targets[:, 0] * h_3d_std + h_3d_mean
        w_3d = targets[:, 1] * w_3d_std + w_3d_mean
        l_3d = targets[:, 2] * l_3d_std + l_3d_mean

        # rois w and h
        rois = rois_batch[0, :, 1:]
        w = rois[:, 2] - rois[:, 0] + 1
        h = rois[:, 3] - rois[:, 1] + 1
        x = (rois[:, 2] + rois[:, 0]) / 2
        y = (rois[:, 3] + rois[:, 1]) / 2
        centers = torch.stack([x, y], dim=-1)
        dims = torch.stack([w, h], dim=-1)

        # import ipdb
        # ipdb.set_trace()
        points = centers.unsqueeze(1) + targets[:, 3:].view(
            -1, 4, 2) * dims.unsqueeze(1)
        # point2 = centers + targets[:, 5:7] * dims

        # cls orient
        # cls_orient = targets[:, 3:5]
        # cls_orient = F.softmax(cls_orient, dim=-1)
        # cls_orient, cls_orient_argmax = torch.max(cls_orient, dim=-1)

        # reg_orient = targets[:, 5:7]

        # decode h_2d
        # h_2d = torch.exp(targets[:, 7]) * h

        # decode c_2d
        # c_2d_x = targets[:, 8] * w + x
        # c_2d_y = targets[:, 9] * h + y

        bbox = torch.stack([h_3d, w_3d, l_3d], dim=-1)
        return torch.cat([bbox, points[:, 1], points[:, 2]], dim=-1)
        # return torch.cat([bbox, targets[:, 3:]], dim=-1)

    def decode_batch_depth(self, targets):
        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        h_3d = targets[:, 0] * h_3d_std + h_3d_mean
        w_3d = targets[:, 1] * w_3d_std + w_3d_mean
        l_3d = targets[:, 2] * l_3d_std + l_3d_mean

        cls_orient = targets[:, 3:5]
        cls_orient = F.softmax(cls_orient, dim=-1)
        cls_orient, cls_orient_argmax = torch.max(cls_orient, dim=-1)

        reg_orient = targets[:, 5:7]

        bbox = torch.stack([h_3d, w_3d, l_3d], dim=-1)
        orient = torch.stack(
            [
                cls_orient_argmax.type_as(reg_orient), reg_orient[:, 0],
                reg_orient[:, 1]
            ],
            dim=-1)

        # decode location
        # depth_ind_preds = targets[:, 7:7 + 11]
        # depth_ind_preds = F.softmax(depth_ind_preds, dim=-1)
        # _, depth_ind_preds_argmax = torch.max(depth_ind_preds, dim=-1)

        # depth_ind = depth_ind_preds_argmax.float().unsqueeze(-1)

        # return torch.cat([bbox, orient, depth_ind, targets[:, 7 + 11:]],
        # dim=-1)
        return torch.cat([bbox, orient, targets[:, 7:]], dim=-1)

    def decode_ry(self, encoded_ry_preds, points1, proposals_xywh, p2):
        # import ipdb
        # ipdb.set_trace()
        cos = torch.cos(encoded_ry_preds[:, 0]) * proposals_xywh[:, 2] / 7
        sin = torch.sin(encoded_ry_preds[:, 0]) * proposals_xywh[:, 3] / 7
        points2_x = points1[:, 0] - cos
        points2_y = points1[:, 1] - sin
        points2 = torch.stack([points2_x, points2_y], dim=-1)
        lines = torch.cat([points1, points2], dim=-1)
        ry = geometry_utils.torch_pts_2d_to_dir_3d(
            lines.unsqueeze(0), p2.unsqueeze(0))[0]
        return ry.unsqueeze(-1)

    def decode_batch_bbox(self, targets, proposals, p2):
        # import ipdb
        # ipdb.set_trace()

        p2 = p2.float()
        mean_dims = torch.tensor([1.8, 1.8, 3.7]).type_as(proposals)
        dims_pred = torch.exp(targets[:, :3]) * mean_dims
        encoded_ry_preds = targets[:, 3:4]
        center_depth_pred = targets[:, 4:5]
        center_2d_pred = encoder_utils.decode_points(targets[:, 5:7],
                                                     proposals)

        location = geometry_utils.torch_points_2d_to_points_3d(
            center_2d_pred, center_depth_pred, p2)

        # ray_angle = -torch.atan2(location[:, 2], location[:, 0])
        # ry_pred = local_ry_pred + ray_angle.unsqueeze(-1)
        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
            proposals.unsqueeze(0))[0]
        ry_pred = self.decode_ry(encoded_ry_preds, center_2d_pred,
                                 proposals_xywh, p2)

        return torch.cat([dims_pred, location, ry_pred], dim=-1)

    def decode_batch_angle(self, targets, bin_centers=None):
        """
        Args:
            targets: shape(N, 3)
        """
        bg_mean_dims = torch.zeros_like(self.mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, self.mean_dims], dim=1).float()
        # assigned_mean_dims = mean_dims[0][pred_labels].float()
        std_dims = torch.ones_like(mean_dims)
        #  targets = (dims[:, :3] - assigned_mean_dims) / assigned_std_dims
        bbox = targets[:, :-2].view(targets.shape[0], -1,
                                    3) * std_dims + mean_dims
        bbox = bbox.view(targets.shape[0], -1)

        # ry
        # sin = targets[:, -2]
        # cos = targets[:, -1]
        theta = get_angle(targets[:, -1], targets[:, -2])
        if bin_centers is not None:
            theta = bin_centers + theta
            # theta = bin_centers
            # theta = -torch.acos(targets[:, 3]) - bin_centers

        # cond_pos = (cos < 0) & (sin > 0)
        # cond_neg = (cos < 0) & (sin < 0)
        # theta[cond_pos] = math.pi - theta[cond_pos]
        # theta[cond_neg] = -math.pi - theta[cond_neg]

        # ry = torch.atan(sin / cos)
        # cond = cos < 0
        # cond_pos = sin > 0
        # cond_neg = sin < 0
        # ry[cond & cond_pos] = ry[cond & cond_pos] + math.pi
        # ry[cond & cond_neg] = ry[cond & cond_neg] - math.pi

        return torch.cat([bbox, theta.unsqueeze(-1)], dim=-1)
        # bbox = torch.stack([h_3d, w_3d, l_3d, theta], dim=-1)
        # return bbox
