# -*- coding: utf-8 -*-

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
from core.profiler import Profiler
from utils.visualizer import FeatVisualizer

import functools
from core.ops import b_inv

import numpy as np


class Mono3DBetterFasterRCNN(Model):
    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        rois_batch = prediction_dict['rois_batch']
        if self.training:
            second_stage_auxiliary_dict, rois_batch = self.pre_subsample(
                rois_batch, feed_dict, self.rcnn_batch_size)

        # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        # shape(N,C,1,1)
        pooled_feat = self.feature_extractor.second_stage_feature(pooled_feat)

        rcnn_cls_scores_map = self.rcnn_cls_pred(pooled_feat)
        rcnn_cls_scores = rcnn_cls_scores_map.mean(3).mean(2)
        saliency_map = F.softmax(rcnn_cls_scores_map, dim=1)
        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        pooled_feat = pooled_feat * saliency_map[:, 1:, :, :]

        reduced_pooled_feat = pooled_feat.mean(3).mean(2)

        rcnn_bbox_preds = self.rcnn_bbox_pred(reduced_pooled_feat)
        # rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat)

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
        if not self.training or self.train_3d:
            rcnn_bbox_preds = rcnn_bbox_preds.detach()
            final_bbox = self.target_assigner.bbox_coder.decode_batch(
                rcnn_bbox_preds.unsqueeze(0), rois_batch[:, :, 1:])
            final_rois_inds = torch.zeros_like(final_bbox[:, :, -1:])
            final_rois_batch = torch.cat([final_rois_inds, final_bbox], dim=-1)

            if self.training:
                third_stage_auxiliary_dict, final_rois_batch = self.pre_subsample(
                    final_rois_batch, feed_dict, self.rcnn_batch_size//2)

            # shape(M,C,7,7)
            mono_3d_pooled_feat = self.rcnn_pooling(
                base_feat, final_rois_batch.view(-1, 5))

            mono_3d_pooled_feat = self.feature_extractor.third_stage_feature(
                mono_3d_pooled_feat)
            mono_3d_pooled_feat = mono_3d_pooled_feat.mean(3).mean(2)

            rcnn_3d = self.rcnn_3d_pred(mono_3d_pooled_feat)

            prediction_dict['rcnn_3d'] = rcnn_3d

            if not self.training:
                rcnn_3d = self.target_assigner.bbox_coder_3d.decode_batch_bbox(
                    rcnn_3d, final_rois_batch[0,:,1:])

                prediction_dict['rcnn_3d'] = rcnn_3d

        # put all auxiliary_dict into prediction_dict
        if self.training:
            prediction_dict.update(second_stage_auxiliary_dict)
            if self.train_3d:
                prediction_dict[
                    'rcnn_reg_weights_3d'] = third_stage_auxiliary_dict[
                        'rcnn_reg_weights_3d']
                prediction_dict[
                    'rcnn_reg_targets_3d'] = third_stage_auxiliary_dict[
                        'rcnn_reg_targets_3d']

        return prediction_dict

    def pre_forward(self):
        pass
        # params
        #  if self.train_3d and self.training and not self.train_2d:

    #  self.freeze_modules()
    #  for parameter in self.feature_extractor.third_stage_feature.parameters(
    #  ):
    #  parameter.requires_grad = True
    #  for param in self.rcnn_3d_pred.parameters():
    #  param.requires_grad = True
    #  self.freeze_bn(self)
    #  self.unfreeze_bn(self.feature_extractor.third_stage_feature)

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
        self.rcnn_cls_pred = nn.Conv2d(2048, self.n_classes, 3, 1, 1)
        # self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
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

        self.rcnn_3d_pred = nn.Linear(in_channels, 3 + 4)

        self.rcnn_3d_loss = OrientationLoss(split_loss=True)

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

        # self.reduce = model_config.get('reduce')
        self.reduce = True

        self.visualizer = FeatVisualizer()

        self.train_3d = True

        #  self.train_2d = True

        # more accurate bbox for 3d prediction
        #  if self.train_3d:
        #  fg_thresh = 0.6
        #  else:
        #  fg_thresh = 0.5
        #  model_config['target_assigner_config']['fg_thresh'] = fg_thresh

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        self.profiler = Profiler()

        self.h_cat = False

    def pre_subsample(self, rois_batch, feed_dict, batch_size):
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        # shape(N,7)
        gt_boxes_3d = feed_dict['gt_boxes_3d']
        used =  feed_dict['used']

        # orient
        cls_orient = torch.unsqueeze(feed_dict['cls_orient'], dim=-1).float()
        reg_orient = feed_dict['reg_orient']
        orient = torch.cat([cls_orient, reg_orient], dim=-1)

        gt_boxes_3d = torch.cat([gt_boxes_3d[:, :, :3], orient, used], dim=-1)

        ##########################
        # assigner
        ##########################
        rcnn_cls_targets, rcnn_reg_targets,\
            rcnn_cls_weights, rcnn_reg_weights,\
            rcnn_reg_targets_3d, rcnn_reg_weights_3d = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_boxes_3d, gt_labels)

        ##########################
        # subsampler
        ##########################
        cls_criterion = None
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            batch_size,
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

        auxiliary_dict = {}

        auxiliary_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        auxiliary_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        auxiliary_dict[
            'rcnn_reg_weights_3d'] = rcnn_reg_weights_3d / num_reg_coeff.float()
        auxiliary_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        auxiliary_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]
        auxiliary_dict['rcnn_reg_targets_3d'] = rcnn_reg_targets_3d[
            batch_sampled_mask]

        # update rois_batch
        rois_batch = rois_batch[batch_sampled_mask].view(rois_batch.shape[0],
                                                         -1, 5)
        return auxiliary_dict, rois_batch

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
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
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        ######################################
        # 3d loss
        ######################################
        if self.train_3d:

            rcnn_reg_weights_3d = prediction_dict['rcnn_reg_weights_3d']
            rcnn_reg_targets_3d = prediction_dict['rcnn_reg_targets_3d']
            rcnn_3d = prediction_dict['rcnn_3d']

            # dims
            rcnn_3d_loss_dims = self.rcnn_bbox_loss(
                rcnn_3d[:, :3], rcnn_reg_targets_3d[:, :3]).sum(dim=-1)

            # angles
            res = self.rcnn_3d_loss(rcnn_3d[:, 3:], rcnn_reg_targets_3d[:, 3:])
            for res_loss_key in res:
                tmp = res[res_loss_key] * rcnn_reg_weights_3d
                res[res_loss_key] = tmp.sum(dim=-1)
            loss_dict.update(res)

            rcnn_3d_loss = rcnn_3d_loss_dims * rcnn_reg_weights_3d
            rcnn_3d_loss = rcnn_3d_loss.sum(dim=-1)

            loss_dict['rcnn_3d_loss'] = rcnn_3d_loss

            # stats of orients
            cls_orient_preds = rcnn_3d[:, 3:5]
            cls_orient = rcnn_reg_targets_3d[:, 3]
            _, cls_orient_preds_argmax = torch.max(cls_orient_preds, dim=-1)
            orient_tp_mask = cls_orient.type_as(
                cls_orient_preds_argmax) == cls_orient_preds_argmax
            mask = (rcnn_reg_weights_3d > 0) & (rcnn_reg_targets_3d[:, 3] > -1)
            orient_tp_mask = orient_tp_mask[mask]
            orient_tp_num = orient_tp_mask.int().sum().item()
            orient_all_num = orient_tp_mask.numel()

            # store all stats in target assigner
            self.target_assigner.stat.update({
                # 'angle_num_tp': torch.tensor(0),
                # 'angle_num_all': 1,

                # stats of orient
                'orient_tp_num': orient_tp_num,
                # 'orient_tp_num2': orient_tp_num2,
                # 'orient_tp_num3': orient_4s_tp_num,
                # 'orient_all_num3': orient_all_num3,
                # 'orient_pr': orient_pr,
                'orient_all_num': orient_all_num,
                # 'orient_tp_num4': orient_tp_num4,
                # 'orient_all_num4': orient_all_num4,
                #  'cls_orient_2s_all_num': depth_ind_all_num,
                #  'cls_orient_2s_tp_num': depth_ind_tp_num
            })

        return loss_dict
