# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.rpn_model import RPNModel
from core.models.focal_loss import FocalLoss
from core.models.multibin_loss import MultiBinLoss
from core.models.multibin_reg_loss import MultiBinRegLoss
from core.models.orientation_loss import OrientationLoss
from lib.model.roi_layers import ROIAlign
#  from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from core.filler import Filler
from core.mono_3d_target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler
from core.profiler import Profiler
from utils.visualizer import FeatVisualizer

import functools

from utils import geometry_utils
from utils import encoder_utils
from test.test_bbox_coder import build_visualizer


class Mono3DFinalPlusFasterRCNN(Model):
    def forward(self, feed_dict):
        self.target_assigner.bbox_coder_3d.mean_dims = feed_dict['mean_dims']
        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        if self.training:
            self.pre_subsample(prediction_dict, feed_dict)
        rois_batch = prediction_dict['rois_batch']

        # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        # shape(N,C,1,1)
        second_pooled_feat = self.feature_extractor.second_stage_feature(
            pooled_feat)

        second_pooled_feat = second_pooled_feat.mean(3).mean(2)

        rcnn_cls_scores = self.rcnn_cls_preds(second_pooled_feat)
        rcnn_bbox_preds = self.rcnn_bbox_preds(second_pooled_feat)
        rcnn_3d = self.rcnn_3d_pred(second_pooled_feat)

        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][
            proposals_order]

        ###################################
        # 3d training
        ###################################

        prediction_dict['rcnn_3d'] = rcnn_3d

        if not self.training:
            rcnn_3d = self.target_assigner.bbox_coder_3d.decode_batch_bbox(
                rcnn_3d, rois_batch[0, :, 1:], feed_dict['p2'][0])

            prediction_dict['rcnn_3d'] = rcnn_3d

        return prediction_dict

    def pre_forward(self):
        pass

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_preds, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_preds, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = ResNetFeatureExtractor(
            self.feature_extractor_config)
        self.rpn_model = RPNModel(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = ROIAlign(
                (self.pooling_size, self.pooling_size), 1.0 / 16.0, 2)
        elif self.pooling_mode == 'ps':
            self.rcnn_pooling = PSRoIPool(7, 7, 1.0 / 16, 7, self.n_classes)
        elif self.pooling_mode == 'psalign':
            raise NotImplementedError('have not implemented yet!')
        elif self.pooling_mode == 'deformable_psalign':
            raise NotImplementedError('have not implemented yet!')
        # self.rcnn_cls_pred = nn.Conv2d(2048, self.n_classes, 3, 1, 1)
        self.rcnn_cls_preds = nn.Linear(self.in_channels, self.n_classes)
        if self.class_agnostic:
            self.rcnn_bbox_preds = nn.Linear(self.in_channels, 4)
        else:
            self.rcnn_bbox_preds = nn.Linear(self.in_channels,
                                             4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(self.n_classes)
        else:
            self.rcnn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        # self.rcnn_3d_pred = nn.Linear(c, 3 + 4 + 11 + 2 + 1)
        if self.class_agnostic_3d:
            self.rcnn_3d_pred = nn.Linear(self.in_channels, 3 + 1 + 3)
        else:
            self.rcnn_3d_pred = nn.Linear(self.in_channels,
                                          3 * self.n_classes + 1 + 3)

        self.rcnn_3d_loss = OrientationLoss(split_loss=True)
        self.l1_loss = nn.L1Loss(reduce=False)
        self.l2_loss = nn.MSELoss(reduce=False)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduce=False)

    def init_param(self, model_config):
        self.in_channels = model_config.get('ndin', 2048)
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes) + 1
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.class_agnostic_3d = model_config['class_agnostic_3d']
        self.crop_resize_with_max_pool = model_config[
            'crop_resize_with_max_pool']
        self.truncated = model_config['truncated']

        self.use_focal_loss = model_config['use_focal_loss']
        self.subsample_twice = model_config['subsample_twice']
        self.rcnn_batch_size = model_config['rcnn_batch_size']

        # some submodule config
        self.feature_extractor_config = model_config[
            'feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        # sampler
        self.sampler = BalancedSampler(model_config['sampler_config'])

        # self.reduce = model_config.get('reduce')
        self.reduce = True

        self.visualizer = FeatVisualizer()

        self.num_bins = 4

        # more accurate bbox for 3d prediction
        # if self.train_3d:
        # fg_thresh = 0.6
        # else:
        # fg_thresh = 0.5
        # model_config['target_assigner_config']['fg_thresh'] = fg_thresh

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        self.profiler = Profiler()

        self.h_cat = False

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        # gt_boxes_proj = feed_dict['gt_boxes_proj']
        gt_labels = feed_dict['gt_labels']

        # shape(N,7)
        gt_boxes_3d = feed_dict['gt_boxes_3d']

        # orient
        # cls_orient = torch.unsqueeze(feed_dict['cls_orient'], dim=-1).float()
        # reg_orient = feed_dict['reg_orient']
        # orient = torch.cat([cls_orient, reg_orient], dim=-1)

        # depth = gt_boxes_3d[:, :, 5:6]
        # c_2ds = feed_dict['c_2d']
        p2 = feed_dict['p2']

        # gt_boxes_3d = torch.cat(
        # [gt_boxes_3d[:, :, :3], orient, depth, c_2ds], dim=-1)

        ##########################
        # assigner
        ##########################
        rcnn_cls_targets, rcnn_reg_targets,\
            rcnn_cls_weights, rcnn_reg_weights,\
            rcnn_reg_targets_3d, rcnn_reg_weights_3d = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_boxes_3d, gt_labels, p2)

        ##########################
        # subsampler
        ##########################
        cls_criterion = None
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size,
            pos_indicator,
            indicator=indicator,
            criterion=cls_criterion)
        rcnn_cls_weights = rcnn_cls_weights[batch_sampled_mask]
        rcnn_reg_weights = rcnn_reg_weights[batch_sampled_mask]
        rcnn_reg_weights_3d = rcnn_reg_weights_3d[batch_sampled_mask]
        num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=-1)
        # check
        assert num_cls_coeff, 'bug happens'
        # assert num_reg_coeff, 'bug happens'
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones_like(num_reg_coeff)

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict[
            'rcnn_reg_weights_3d'] = rcnn_reg_weights_3d / num_reg_coeff.float(
            )
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets_3d'] = rcnn_reg_targets_3d[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

    def squeeze_bbox_preds(self, rcnn_bbox_preds, rcnn_cls_targets, out_c=4):
        """
        squeeze rcnn_bbox_preds from shape (N, 4 * num_classes) to shape (N, 4)
        Args:
            rcnn_bbox_preds: shape(N, num_classes, 4)
            rcnn_cls_targets: shape(N, 1)
        """
        rcnn_bbox_preds = rcnn_bbox_preds.view(-1, self.n_classes, out_c)
        batch_size = rcnn_bbox_preds.shape[0]
        offset = torch.arange(0, batch_size) * rcnn_bbox_preds.size(1)
        rcnn_cls_targets = rcnn_cls_targets + offset.type_as(rcnn_cls_targets)
        rcnn_bbox_preds = rcnn_bbox_preds.contiguous().view(
            -1, out_c)[rcnn_cls_targets]
        return rcnn_bbox_preds

    def calc_ry_loss(self, ry_pred, ry_gt, image_shape, proposals):
        """
        Args:
            ry_pred: shape(N, 6)
            ry_gt: shape(N, 5)
        """

        # import ipdb
        # ipdb.set_trace()
        visible = ry_gt[:, 4:]

        cls_loss = self.rcnn_cls_loss(ry_pred[:, 4:], visible.view(-1).long())
        encoded_points_gt = ry_gt[:, :2]

        lines = encoder_utils.decode_points(encoded_points_gt, proposals)
        visibility = self.calc_points_visibility(
            lines.view(-1, 2), image_shape).unsqueeze(-1)
        # image truncated and self-occlusion
        point_loss = self.l2_loss(ry_pred[:, :2],
                                  ry_gt[:, :2]) * visibility.float()
        angle_loss = self.l2_loss(ry_pred[:, 2:4], ry_gt[:, 2:4]) * visible
        ry_loss = torch.cat(
            [point_loss, angle_loss,
             cls_loss.unsqueeze(-1)], dim=-1)

        return ry_loss
        # theta_loss = self.l2_loss(ry_)
        # return torch.cat([reg_loss, theta_loss], dim=-1)

        # return torch.cat([reg_loss, cls_loss.unsqueeze(-1)], dim=-1)

    def calc_points_visibility(self, points, image_shape):
        image_shape = torch.tensor([0, 0, image_shape[1], image_shape[0]])
        image_shape = image_shape.type_as(points).view(1, 4)
        image_filter = geometry_utils.torch_window_filter(
            points.unsqueeze(0), image_shape, deltas=200)[0]
        return image_filter

    def calc_local_corners(self, dims, ry):
        h = dims[:, 0]
        w = dims[:, 1]
        l = dims[:, 2]
        zeros = torch.zeros_like(l).type_as(l)

        zeros = torch.zeros_like(ry[:, 0])
        ones = torch.ones_like(ry[:, 0])
        cos = torch.cos(ry[:, 0])
        sin = torch.sin(ry[:, 0])
        cos = cos
        sin = sin

        rotation_matrix = torch.stack(
            [cos, zeros, sin, zeros, ones, zeros, -sin, zeros, cos],
            dim=-1).reshape(-1, 3, 3)

        x_corners = torch.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=0)
        y_corners = torch.stack(
            [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=0)
        z_corners = torch.stack(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            dim=0)

        # shape(N, 3, 8)
        box_points_coords = torch.stack(
            (x_corners, y_corners, z_corners), dim=0)
        # rotate and translate
        # shape(N, 3, 8)
        corners_3d = torch.bmm(rotation_matrix,
                               box_points_coords.permute(2, 0, 1))

        return corners_3d.permute(0, 2, 1)

    def decode_bbox(self, center_2d, center_depth, dims, ry, p2):
        # location
        location = geometry_utils.torch_points_2d_to_points_3d(
            center_2d, center_depth, p2)

        # local corners
        local_corners = self.calc_local_corners(dims, ry)

        # global corners
        # global_corners = (
        # location.view(N, M, 1, 3) + local_corners.view(N, M, 8, 3)).view(
        # N, M, -1)
        global_corners = location[:, None] + local_corners
        return global_corners.contiguous().view(-1, 24)

    def decode_ry(self, encoded_ry_preds, points1, proposals_xywh, p2):
        # slope, encoded_points = torch.split(encoded_ry_preds, [1, 2], dim=-1)
        # import ipdb
        # ipdb.set_trace()
        # slope = slope * proposals_xywh[:, :, 3:4] / (
        # proposals_xywh[:, :, 2:3] + 1e-7)
        # points1 = encoded_points * proposals_xywh[:, :,
        # 2:] + proposals_xywh[:, :, :2]
        # norm = torch.norm(encoded_ry_preds, dim=-1)
        # cos = encoded_ry_preds[:, 0] / norm
        # sin = encoded_ry_preds[:, 1] / norm
        cos = torch.cos(encoded_ry_preds[:, 0]) * proposals_xywh[:, 2] / 7
        sin = torch.sin(encoded_ry_preds[:, 0]) * proposals_xywh[:, 3] / 7
        points2_x = points1[:, 0] - cos
        points2_y = points1[:, 1] - sin
        points2 = torch.stack([points2_x, points2_y], dim=-1)
        lines = torch.cat([points1, points2], dim=-1)
        ry = geometry_utils.torch_pts_2d_to_dir_3d(
            lines.unsqueeze(0), p2.unsqueeze(0))[0]
        return ry.unsqueeze(-1)

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        #  import ipdb
        #  ipdb.set_trace()

        loss_dict = {}

        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))
        # targets and weights
        rcnn_cls_weights = prediction_dict['rcnn_cls_weights']
        rcnn_reg_weights = prediction_dict['rcnn_reg_weights']

        rcnn_cls_targets = prediction_dict['rcnn_cls_targets']
        rcnn_reg_targets = prediction_dict['rcnn_reg_targets']

        # classification loss
        rcnn_cls_scores = prediction_dict['rcnn_cls_scores']

        rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores, rcnn_cls_targets)
        rcnn_cls_loss *= rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)

        # bounding box regression L1 loss
        rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        #
        if not self.class_agnostic:
            rcnn_bbox_preds = self.squeeze_bbox_preds(rcnn_bbox_preds,
                                                      rcnn_cls_targets)
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        ######################################
        # 3d loss
        ######################################

        train_3d = True
        p2 = feed_dict['p2'][0]
        proposals = prediction_dict['rois_batch'][0, :, 1:]
        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
            proposals.unsqueeze(0))[0]
        mean_dims = torch.tensor([1.8, 1.8, 3.7]).type_as(proposals)
        # visualizer = build_visualizer()
        if train_3d:
            rcnn_reg_weights_3d = prediction_dict['rcnn_reg_weights_3d']
            rcnn_reg_targets_3d = prediction_dict['rcnn_reg_targets_3d']
            rcnn_3d = prediction_dict['rcnn_3d']

            dims_pred = torch.exp(rcnn_3d[:, :3]) * mean_dims
            center_depth_pred = rcnn_3d[:, 4:5]
            center_2d_pred = encoder_utils.decode_points(
                rcnn_3d[:, 5:7], proposals)

            # gt
            dims_gt = rcnn_reg_targets_3d[:, :3]
            ry_gt = rcnn_reg_targets_3d[:, 3:4]
            center_depth_gt = rcnn_reg_targets_3d[:, 4:5]
            center_2d_gt = rcnn_reg_targets_3d[:, 5:7]

            global_corners_gt = self.decode_bbox(center_2d_gt, center_depth_gt,
                                                 dims_gt, ry_gt, p2)

            ry_pred = self.decode_ry(rcnn_3d[:, 3:4], center_2d_gt,
                                     proposals_xywh, p2)
            # gt_boxes_3d = feed_dict['gt_boxes_3d'][0]
            # gt_boxes_3d = torch.cat(
            # [gt_boxes_3d[:, 3:6], gt_boxes_3d[:, :3], gt_boxes_3d[:, 6:]],
            # dim=-1)
            # global_corners_gt = geometry_utils.torch_boxes_3d_to_corners_3d(
            # gt_boxes_3d)
            # global_corners_gt = global_corners_gt[rcnn_reg_weights_3d>0]

            # corners_3d = global_corners_gt.view(-1, 8,
            # 3).cpu().detach().numpy()
            # image_path = feed_dict['img_name'][0]
            # image = feed_dict['img'][0].permute(1, 2, 0).cpu().detach().numpy()
            # image = image.copy()
            # normal_mean = np.asarray([0.485, 0.456, 0.406])
            # normal_van = np.asarray([0.229, 0.224, 0.225])
            # image = image * normal_van + normal_mean
            # p2 = p2.cpu().detach().numpy()
            # visualizer.render_image_corners_2d(
            # image_path, image, corners_3d=corners_3d[:10], p2=p2)

            # import ipdb
            # ipdb.set_trace()
            for index, item in enumerate(
                [('center_2d_loss', center_2d_pred), ('center_depth_loss',
                                                      center_depth_pred),
                 ('dims', dims_pred), ('ry', ry_pred)]):
                args_gt = [center_2d_gt, center_depth_gt, dims_gt, ry_gt, p2]
                args_gt[index] = item[1]
                loss_name = item[0]
                global_corners_preds = self.decode_bbox(*args_gt)

                # temp = global_corners_preds[rcnn_reg_weights_3d > 0]

                # corners_3d = temp.view(-1, 8, 3).cpu().detach().numpy()
                # image_path = feed_dict['img_name'][0]
                # image = feed_dict['img'][0].permute(1, 2,
                # 0).cpu().detach().numpy()
                # image = image.copy()
                # normal_mean = np.asarray([0.485, 0.456, 0.406])
                # normal_van = np.asarray([0.229, 0.224, 0.225])
                # image = image * normal_van + normal_mean
                # p2_np = p2.cpu().detach().numpy()

                # visualizer.render_image_corners_2d(
                # image_path, image, corners_3d=corners_3d[:10], p2=p2_np)
                loss = self.smooth_l1_loss(1 / 3.0 * global_corners_preds,
                                           1 / 3.0 * global_corners_gt).sum(
                                               dim=-1) * rcnn_reg_weights_3d
                loss_dict[loss_name] = loss.sum() * 0.5 * 3

        return loss_dict
