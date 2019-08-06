# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.iou_rpn_model import IoURPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg

from core.filler import Filler
from core.LED_target_assigner import LEDTargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler

import functools


class IoUFasterRCNN(Model):
    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()

        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})
        # batch_size = base_feat.shape[0]

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

        # shape(N,C,1,1)
        pooled_feat = self.feature_extractor.second_stage_feature(pooled_feat)
        ########################################
        # semantic map
        ########################################
        rcnn_cls_scores_map = self.rcnn_cls_pred(pooled_feat)
        rcnn_cls_scores = rcnn_cls_scores_map.mean(3).mean(2)
        saliency_map = F.softmax(rcnn_cls_scores_map, dim=1)
        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)
        # shape(N,C)
        rcnn_bbox_feat = pooled_feat * saliency_map[:, 1:, :, :]
        rcnn_bbox_feat = rcnn_bbox_feat.mean(3).mean(2)

        rcnn_bbox_preds = self.rcnn_bbox_pred(rcnn_bbox_feat)

        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        # iou
        rcnn_iou = self.rcnn_iou(rcnn_bbox_feat)
        rcnn_iou = F.sigmoid(rcnn_iou)

        if self.use_iox:
            # iog
            rcnn_iog = self.rcnn_iog(rcnn_bbox_feat)
            rcnn_iog = F.sigmoid(rcnn_iog)

            # iod
            rcnn_iod = self.rcnn_iog(rcnn_bbox_feat)
            rcnn_iod = F.sigmoid(rcnn_iod)

            rcnn_iou_indirect = self.calculate_iou(rcnn_iog, rcnn_iod)
            rcnn_iou_final = (1 - self.alpha
                              ) * rcnn_iou_indirect + self.alpha * rcnn_iou
            prediction_dict['rcnn_iog'] = rcnn_iog
            prediction_dict['rcnn_iod'] = rcnn_iod
        else:
            # use iou directly
            rcnn_iou_final = rcnn_iou

        rcnn_fg_probs_final = rcnn_cls_probs[:, 1:] * torch.exp(-torch.pow(
            (1 - rcnn_iou_final[:, 1:]), 2) / self.theta)

        prediction_dict['rcnn_cls_probs'] = torch.cat(
            [rcnn_fg_probs_final, rcnn_fg_probs_final], dim=-1)
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores
        prediction_dict['rcnn_iou'] = rcnn_iou

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][0][
            proposals_order]

        return prediction_dict

    def calculate_iou(self, iog, iod):
        mask = ~(iod == 0)
        iou_indirect = torch.zeros_like(iog)
        iod = iod[mask]
        iog = iog[mask]
        iou_indirect[mask] = (iod * iog) / (iod + iog - iod * iog)
        return iou_indirect

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

        # freeze module
        # self.freeze_modules()
        # # unfreeze some layers
        # unfreeze_params = [

    # self.rpn_model.rpn_iou.bias, self.rpn_model.rpn_iou.weight,
    # self.rcnn_iou.bias, self.rcnn_iou.weight
    # ]
    # for param in unfreeze_params:
    # param.requires_grad = True

    def init_modules(self):
        self.feature_extractor = ResNetFeatureExtractor(
            self.feature_extractor_config)
        self.rpn_model = IoURPNModel(self.rpn_config)
        self.rcnn_pooling = RoIAlignAvg(self.pooling_size, self.pooling_size,
                                        1.0 / 16.0)
        self.rcnn_cls_pred = nn.Conv2d(2048, self.n_classes, 3, 1, 1)
        in_channels = 2048
        self.rcnn_iou = nn.Linear(in_channels, self.n_classes)
        self.rcnn_iog = nn.Linear(in_channels, self.n_classes)
        self.rcnn_iod = nn.Linear(in_channels, self.n_classes)

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
        self.rcnn_iou_loss = nn.MSELoss(reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

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
        self.theta = 1.0
        self.alpha = 0.6

        self.use_focal_loss = model_config['use_focal_loss']
        self.subsample_twice = model_config['subsample_twice']
        self.rcnn_batch_size = model_config['rcnn_batch_size']
        self.iou_criterion = model_config['iou_criterion']
        self.use_iox = model_config['use_iox']
        # self.use_cls_pred = model_config['use_cls_pred']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        # assigner
        self.target_assigner = LEDTargetAssigner(
            model_config['target_assigner_config'])

        # sampler
        # self.sampler = HardNegativeSampler(model_config['sampler_config'])
        if self.iou_criterion:
            self.sampler = DetectionSampler(model_config['sampler_config'])
        else:
            self.sampler = BalancedSampler(model_config['sampler_config'])

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        ##########################
        # assigner
        ##########################
        #  import ipdb
        #  ipdb.set_trace()
        rcnn_cls_targets, rcnn_reg_targets, rcnn_cls_weights, rcnn_reg_weights = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_labels)

        ##########################
        # subsampler
        ##########################
        if self.iou_criterion:
            cls_criterion = self.target_assigner.matcher.assigned_overlaps_batch
        else:
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
        num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=-1)
        # check
        assert num_cls_coeff, 'bug happens'
        assert num_reg_coeff, 'bug happens'

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        # import ipdb
        # ipdb.set_trace()
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

        if not self.training:
            # used for track
            proposals_order = prediction_dict['proposals_order']

            prediction_dict['proposals_order'] = proposals_order[
                batch_sampled_mask]

        # iou targets
        rcnn_iou_targets = self.target_assigner.matcher.assigned_overlaps_batch
        prediction_dict['rcnn_iou_targets'] = rcnn_iou_targets[
            batch_sampled_mask]

        # iog targets
        rcnn_iog_targets = self.target_assigner.matcher.assigned_iog_batch
        prediction_dict['rcnn_iog_targets'] = rcnn_iog_targets[
            batch_sampled_mask]

        # iod targets
        rcnn_iod_targets = self.target_assigner.matcher.assigned_iod_batch
        prediction_dict['rcnn_iod_targets'] = rcnn_iod_targets[
            batch_sampled_mask]

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

        # iou loss
        rcnn_iou = prediction_dict['rcnn_iou'][:, 1]
        rcnn_iou_targets = prediction_dict['rcnn_iou_targets']
        rcnn_iou = torch.exp(rcnn_iou)
        rcnn_iou_targets = torch.exp(rcnn_iou_targets)
        rcnn_iou_loss = self.rcnn_iou_loss(rcnn_iou, rcnn_iou_targets)
        rcnn_iou_loss *= rcnn_cls_weights
        rcnn_iou_loss = rcnn_iou_loss.sum(dim=-1)

        if self.use_iox:
            # iog loss
            rcnn_iog = prediction_dict['rcnn_iog'][:, 1]
            rcnn_iog_targets = prediction_dict['rcnn_iog_targets']
            rcnn_iog = torch.exp(rcnn_iog)
            rcnn_iog_targets = torch.exp(rcnn_iog_targets)
            rcnn_iog_loss = self.rcnn_iou_loss(rcnn_iog, rcnn_iog_targets)
            rcnn_iog_loss *= rcnn_cls_weights
            rcnn_iog_loss = rcnn_iog_loss.sum(dim=-1)

            # iod loss
            rcnn_iod = prediction_dict['rcnn_iod'][:, 1]
            rcnn_iod_targets = prediction_dict['rcnn_iod_targets']
            rcnn_iod = torch.exp(rcnn_iod)
            rcnn_iod_targets = torch.exp(rcnn_iod_targets)
            rcnn_iod_loss = self.rcnn_iou_loss(rcnn_iod, rcnn_iod_targets)
            rcnn_iod_loss *= rcnn_cls_weights
            rcnn_iod_loss = rcnn_iod_loss.sum(dim=-1)

            loss_dict['rcnn_iod_loss'] = rcnn_iod_loss
            loss_dict['rcnn_iog_loss'] = rcnn_iog_loss

        # classification loss
        rcnn_cls_scores = prediction_dict['rcnn_cls_scores']
        # exp
        # rcnn_cls_scores = torch.exp(rcnn_cls_scores)
        # rcnn_cls_targets = torch.exp(rcnn_cls_targets)

        rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores, rcnn_cls_targets)
        rcnn_cls_loss *= rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)

        # bounding box regression L1 loss
        rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights
        # rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        # loss weights has no gradients
        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss
        loss_dict['rcnn_iou_loss'] = rcnn_iou_loss

        # add rcnn_cls_targets to get the statics of rpn
        # loss_dict['rcnn_cls_targets'] = rcnn_cls_targets

        return loss_dict
