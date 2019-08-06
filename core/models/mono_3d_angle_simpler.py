# -*- coding: utf-8 -*-
"""
This model is mainly used for estimating angle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.rpn_model import RPNModel
from core.models.focal_loss import FocalLoss
from core.models.multibin_loss import MultiBinLoss
from core.models.multibin_reg_loss import MultiBinRegLoss
from core.models.orientation_loss import OrientationLoss
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from core.filler import Filler
from core.mono_3d_target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler
from utils.visualizer import FeatVisualizer

import math

import functools


class Mono3DAngleSimplerFasterRCNN(Model):
    def forward(self, feed_dict):
        #  import ipdb
        #  ipdb.set_trace()
        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        # proposals = prediction_dict['proposals_batch']
        # shape(N,num_proposals,5)
        # pre subsample for reduce consume of memory
        if self.training:
            self.pre_subsample(prediction_dict, feed_dict)
        rois_batch = prediction_dict['rois_batch']

        # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        ###################################
        # 3d training
        ###################################
        # mono_3d_pooled_feat = self.feature_extractor.third_stage_feature(
        # pooled_feat.detach())
        # mono_3d_pooled_feat = mono_3d_pooled_feat.mean(3).mean(2)
        # rcnn_3d = self.rcnn_3d_preds_new(mono_3d_pooled_feat)

        # prediction_dict['rcnn_3d'] = rcnn_3d

        # shape(N,C,1,1)
        pooled_feat = self.feature_extractor.second_stage_feature(pooled_feat)

        rcnn_cls_scores_map = self.rcnn_cls_pred(pooled_feat)
        rcnn_cls_scores = rcnn_cls_scores_map.mean(3).mean(2)
        saliency_map = F.softmax(rcnn_cls_scores_map, dim=1)
        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        pooled_feat = pooled_feat * saliency_map[:, 1:, :, :]
        # shape(N,C)
        if self.reduce:
            pooled_feat = pooled_feat.mean(3).mean(2)
        else:
            pooled_feat = pooled_feat.view(self.rcnn_batch_size, -1)

        rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat)
        #  rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat)

        #  rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        rcnn_3d_dims = self.rcnn_dims_pred(pooled_feat)
        rcnn_3d_angles = self.rcnn_angle_pred(pooled_feat).view(
            -1, self.num_bins, 2)
        rcnn_3d_angles_cls = self.rcnn_angle_conf_pred(pooled_feat).view(
            -1, self.num_bins, 2)
        rcnn_3d_angles_cls_reg = torch.cat(
            [rcnn_3d_angles_cls, rcnn_3d_angles], dim=-1).view(
                -1, self.num_bins * 4)

        rcnn_3d = torch.cat([rcnn_3d_dims, rcnn_3d_angles_cls_reg], dim=-1)
        prediction_dict['rcnn_3d'] = rcnn_3d

        prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][
            proposals_order]

        if not self.training:
            # import ipdb
            # ipdb.set_trace()
            dims = rcnn_3d[:, :3]
            angles = rcnn_3d[:, 3:].view(-1, self.num_bins, 4)
            angles_cls = F.softmax(angles[:, :, :2], dim=-1)
            _, angles_cls_argmax = torch.max(angles_cls[:, :, 1], dim=-1)
            row = torch.arange(
                0, angles_cls_argmax.shape[0]).type_as(angles_cls_argmax)
            angles_oritations = angles[:, :, 2:][row, angles_cls_argmax]
            rcnn_3d = torch.cat([dims, angles_oritations], dim=-1)
            #  import ipdb
            #  ipdb.set_trace()
            rcnn_3d = self.target_assigner.bbox_coder_3d.decode_batch_angle(
                rcnn_3d, self.rcnn_3d_loss.bin_centers[angles_cls_argmax])
            prediction_dict['rcnn_3d'] = rcnn_3d

        return prediction_dict

    def pre_forward(self):
        # params
        if self.train_3d and self.training and not self.train_2d:
            self.freeze_modules()
            # for parameter in self.feature_extractor.third_stage_feature.parameters(
            # ):
            # parameter.requires_grad = True
            # for param in self.rcnn_3d_preds_new.parameters():
            # param.requires_grad = True

            for param in self.rcnn_angle_conf_pred.parameters():
                param.requires_grad = True
            for param in self.rcnn_angle_pred.parameters():
                param.requires_grad = True
            for param in self.rcnn_dims_pred.parameters():
                param.requires_grad = True
            self.freeze_bn(self)
        # self.unfreeze_bn(self.feature_extractor.third_stage_feature)

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = ResNetFeatureExtractor(
            self.feature_extractor_config)
        self.rpn_model = RPNModel(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = RoIAlignAvg(self.pooling_size,
                                            self.pooling_size, 1.0 / 16.0)
        elif self.pooling_mode == 'ps':
            self.rcnn_pooling = PSRoIPool(7, 7, 1.0 / 16, 7, self.n_classes)
        elif self.pooling_mode == 'psalign':
            raise NotImplementedError('have not implemented yet!')
        elif self.pooling_mode == 'deformable_psalign':
            raise NotImplementedError('have not implemented yet!')
        #  self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        self.rcnn_cls_pred = nn.Conv2d(2048, self.n_classes, 3, 1, 1)
        if self.reduce:
            in_channels = 2048
        else:
            in_channels = 2048 * 4 * 4
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(in_channels, 4)
        else:
            self.rcnn_bbox_pred = nn.Linear(in_channels, 4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(2)
        else:
            self.rcnn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        # some 3d statistic
        # some 2d points projected from 3d
        # self.rcnn_3d_preds_new = nn.Linear(in_channels, 3 + 4 * self.num_bins)

        self.rcnn_3d_loss = MultiBinLoss(num_bins=self.num_bins)

        # dims
        self.rcnn_dims_pred = nn.Sequential(
            * [nn.Linear(in_channels, 256), nn.ReLU(), nn.Linear(256, 3)])

        # angle
        self.rcnn_angle_pred = nn.Sequential(* [
            nn.Linear(in_channels, 256), nn.ReLU(), nn.Linear(
                256, self.num_bins * 2)
        ])

        # angle conf
        self.rcnn_angle_conf_pred = nn.Sequential(* [
            nn.Linear(in_channels, 256), nn.ReLU(), nn.Linear(
                256, self.num_bins * 2)
        ])

    def init_param(self, model_config):
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.crop_resize_with_max_pool = model_config[
            'crop_resize_with_max_pool']
        self.truncated = model_config['truncated']

        self.use_focal_loss = model_config['use_focal_loss']
        self.subsample_twice = model_config['subsample_twice']
        self.rcnn_batch_size = model_config['rcnn_batch_size']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        # sampler
        self.sampler = BalancedSampler(model_config['sampler_config'])

        self.reduce = True

        self.visualizer = FeatVisualizer()

        self.num_bins = 12

        self.train_3d = True

        # self.train_2d = not self.train_3d
        self.train_2d = False

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']
        #  gt_boxes_3d = feed_dict['coords']
        #  dims_2d = feed_dict['dims_2d']
        # use local angle
        #  oritations = feed_dict['local_angle_oritation']
        local_angle = feed_dict['local_angle']

        # shape(N,7)
        gt_boxes_3d = feed_dict['gt_boxes_3d']

        # orient

        # here just concat them
        # dims and their projection

        gt_boxes_3d = torch.cat([gt_boxes_3d[:, :, :3], local_angle], dim=-1)

        ##########################
        # assigner
        ##########################
        rcnn_cls_targets, rcnn_reg_targets,\
            rcnn_cls_weights, rcnn_reg_weights,\
            rcnn_reg_targets_3d, rcnn_reg_weights_3d = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_boxes_3d, gt_labels )

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
        assert num_reg_coeff, 'bug happens'

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict[
            'rcnn_reg_weights_3d'] = rcnn_reg_weights_3d / num_reg_coeff.float()
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets_3d'] = rcnn_reg_targets_3d[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}

        if self.train_2d:
            # submodule loss
            loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))
            # targets and weights
            rcnn_cls_weights = prediction_dict['rcnn_cls_weights']
            rcnn_reg_weights = prediction_dict['rcnn_reg_weights']

            rcnn_cls_targets = prediction_dict['rcnn_cls_targets']
            rcnn_reg_targets = prediction_dict['rcnn_reg_targets']

            # classification loss
            rcnn_cls_scores = prediction_dict['rcnn_cls_scores']
            rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores,
                                               rcnn_cls_targets)
            rcnn_cls_loss *= rcnn_cls_weights
            rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)

            # bounding box regression L1 loss
            rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
            rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                                 rcnn_reg_targets).sum(dim=-1)
            rcnn_bbox_loss *= rcnn_reg_weights
            rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

            loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
            loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        ######################################
        # 3d loss
        ######################################

        rcnn_reg_weights_3d = prediction_dict['rcnn_reg_weights_3d']
        rcnn_reg_targets_3d = prediction_dict['rcnn_reg_targets_3d']
        rcnn_3d = prediction_dict['rcnn_3d']
        if self.train_3d:

            # dims
            rcnn_3d_loss_dims = self.rcnn_bbox_loss(
                rcnn_3d[:, :3], rcnn_reg_targets_3d[:, :3]).sum(dim=-1)

            # angles
            rcnn_angle_loss, angle_tp_mask = self.rcnn_3d_loss(
                rcnn_3d[:, 3:], rcnn_reg_targets_3d[:, 3:])

            rcnn_3d_loss = rcnn_3d_loss_dims * rcnn_reg_weights_3d
            rcnn_3d_loss = rcnn_3d_loss.sum(dim=-1)

            rcnn_angle_loss = rcnn_angle_loss * rcnn_reg_weights_3d
            rcnn_angle_loss = rcnn_angle_loss.sum(dim=-1)

            loss_dict['rcnn_3d_loss'] = rcnn_3d_loss
            loss_dict['rcnn_angle_loss'] = rcnn_angle_loss

            # angles stats
            angle_tp_mask = angle_tp_mask[rcnn_reg_weights_3d > 0]
            angles_tp_num = angle_tp_mask.int().sum().item()
            angles_all_num = angle_tp_mask.numel()
        else:
            angles_all_num = 0
            angles_tp_num = 0

        # store all stats in target assigner
        self.target_assigner.stat.update({
            'angle_num_tp': torch.tensor(0),
            'angle_num_all': 1,

            # stats of orient
            'orient_tp_num': 0,
            'orient_tp_num2': 0,
            'orient_tp_num3': 0,
            'orient_all_num3': 0,
            # 'orient_pr': orient_pr,
            'orient_all_num': 0,
            'orient_tp_num4': 0,
            'orient_all_num4': 0,
            'cls_orient_2s_all_num': angles_all_num,
            'cls_orient_2s_tp_num': angles_tp_num

            #  'angles_tp_num': angles_tp_num,
            #  'angles_all_num': angles_all_num
        })
        # import ipdb
        # ipdb.set_trace()

        return loss_dict
