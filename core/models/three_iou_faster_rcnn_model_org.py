# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.three_iou_rpn_model_org import RPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg

from core.filler import Filler
from core.three_iou_target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler

import functools


class OrgThreeIoUFasterRCNN(Model):
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
        # no necessary for iou
        pooled_feat = pooled_feat.mean(3).mean(2)
        rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat)
        # rcnn_cls_scores = rcnn_cls_scores_map.mean(3).mean(2)
        # saliency_map = F.softmax(rcnn_cls_scores_map, dim=1)
        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)
        # shape(N,C)
        # rcnn_bbox_feat = pooled_feat * saliency_map[:, 1:, :, :]
        # rcnn_bbox_feat = rcnn_bbox_feat.mean(3).mean(2)

        rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat)
        # shape(N,C)
        # pooled_feat = pooled_feat.mean(3).mean(2)

        # rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat)
        # rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat)

        # rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][0][
            proposals_order]

        return prediction_dict

    #  def unloaded_parameters(self):
    #  return ['rcnn_cls_pred.bias', 'rcnn_cls_pred.weight']

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
        self.rcnn_pooling = RoIAlignAvg(self.pooling_size, self.pooling_size,
                                        1.0 / 16.0)
        # self.rcnn_cls_pred = nn.Conv2d(2048, self.n_classes, 3, 1, 1)
        self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(2048, 4)
        else:
            self.rcnn_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # loss module
        # if self.use_focal_loss:
        # self.rcnn_cls_loss = FocalLoss(2)
        # else:
        # self.rcnn_cls_loss = functools.partial(
        # F.cross_entropy, reduce=False)
        self.rcnn_cls_loss = nn.MSELoss(reduce=False)

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

        self.use_focal_loss = model_config['use_focal_loss']
        self.subsample_twice = model_config['subsample_twice']
        self.rcnn_batch_size = model_config['rcnn_batch_size']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        # sampler
        # self.bbox_sampler = DetectionSampler({'fg_fraction': 0.5})
        self.bbox_sampler = HardNegativeSampler({'fg_fraction': 1})
        # self.bbox_sampler = BalancedSampler({'fg_fraction': 1})
        self.iou_sampler = BalancedSampler(model_config['sampler_config'])

    def pre_subsample(self, prediction_dict, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        ##########################
        # assigner
        ##########################
        # import ipdb
        # ipdb.set_trace()
        rcnn_cls_targets, rcnn_reg_targets, rcnn_cls_weights, rcnn_reg_weights = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_labels)

        ##########################
        # double subsampler
        ##########################

        cls_criterion = None

        # bbox subsample
        pos_indicator = rcnn_reg_weights > 0
        # indicator = pos_indicator
        cls_criterion = self.target_assigner.matcher.assigned_overlaps_batch
        bbox_batch_sampled_mask = self.bbox_sampler.subsample_batch(
            self.rcnn_batch_size,
            pos_indicator,
            criterion=cls_criterion)
        # rcnn_cls_weights = rcnn_cls_weights[bbox_batch_sampled_mask]
        # num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)
        # ignore when bbox loss is not necessary
        bbox_batch_sampled_mask[rcnn_reg_weights == 0] = 0
        rcnn_reg_weights = rcnn_reg_weights[bbox_batch_sampled_mask]
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=-1)
        assert num_reg_coeff, 'bug happens'

        # iou subsample
        # balanced subsample
        pos_indicator = rcnn_cls_targets > 0
        iou_batch_sampled_mask = self.iou_sampler.subsample_batch(
            self.rcnn_batch_size, pos_indicator)
        # rcnn_cls_weights = rcnn_cls_weights[iou_batch_sampled_mask]
        # num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)

        # check

        # make sure iou optimized when bbox optimized
        iou_batch_sampled_mask |= bbox_batch_sampled_mask
        rcnn_cls_weights = rcnn_cls_weights[iou_batch_sampled_mask]
        num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)
        assert num_cls_coeff, 'bug happens'

        batch_sampled_mask = iou_batch_sampled_mask

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            iou_batch_sampled_mask]
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            bbox_batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

        if not self.training:
            # used for track
            proposals_order = prediction_dict['proposals_order']

            prediction_dict['proposals_order'] = proposals_order[
                batch_sampled_mask]
        # prediction_dict['iou_batch_sampled_mask'] = iou_batch_sampled_mask
        prediction_dict['bbox_batch_sampled_mask'] = bbox_batch_sampled_mask[
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

        # mask
        # iou_batch_sampled_mask = prediction_dict['iou_batch_sampled_mask']
        bbox_batch_sampled_mask = prediction_dict['bbox_batch_sampled_mask']

        # classification loss
        rcnn_cls_scores = prediction_dict['rcnn_cls_probs'][:, 1]
        # exp
        # rcnn_cls_scores = torch.exp(rcnn_cls_scores)
        # rcnn_cls_targets = torch.exp(rcnn_cls_targets)
        rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores, rcnn_cls_targets)
        rcnn_cls_loss *= rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)

        # bounding box regression L1 loss
        rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        rcnn_bbox_preds = rcnn_bbox_preds[bbox_batch_sampled_mask]
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights
        # rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        # loss weights has no gradients
        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        return loss_dict
