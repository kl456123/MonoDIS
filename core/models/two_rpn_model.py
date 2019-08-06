# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
# from core.models.rpn_model import RPNModel
from core.models.first_rpn_model import FirstRPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg

from core.filler import Filler
from core.target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.models.feature_extractor_model import FeatureExtractor
from core.samplers.balanced_sampler import BalancedSampler

from lib.model.utils.net_utils import _smooth_l1_loss
from lib.model.rpn.proposal_target_layer_tworpn import _ProposalTargetLayer as ProposalTargetTwoRPN

from utils import box_ops


class TwoRPNModel(Model):
    def clean_base_feat(self, base_feat, rois_batch, gt_boxes=None):
        """
        Args:
            base_feat: shape(N,C,H,W)
            rois: shape(N,M,5)
        Returns:
            clean_feat: shape(N,C,H,W)
        """
        batch_size = rois_batch.shape[0]
        upsampled_feat = self.upsample(base_feat)
        rois_batch = rois_batch[:, :, 1:]
        if gt_boxes is not None:
            rois_batch = torch.cat([rois_batch, gt_boxes], dim=1)

        # round it first
        rois_batch = rois_batch.int()

        # filter small rois
        rois_batch = rois_batch.view(-1, 4)
        # import ipdb
        # ipdb.set_trace()
        keep = box_ops.size_filter(rois_batch, 16)
        rois_batch = rois_batch[keep].view(batch_size, -1, 4)

        rois_per_img = rois_batch.shape[1]
        mask = torch.zeros(upsampled_feat.shape[0], upsampled_feat.shape[2],
                           upsampled_feat.shape[3])
        for i in range(batch_size):
            # copy
            rois = rois_batch[i]
            for j in range(rois_per_img):
                roi = rois[j]
                mask[i, roi[1]:roi[3], roi[0]:roi[2]] = 1

        upsampled_feat *= mask.unsqueeze(1).type_as(upsampled_feat)
        clean_feat = F.upsample(
            upsampled_feat, size=base_feat.shape[-2:], mode='bilinear')

        return clean_feat

    def second_rpn_bbox_select(self, second_rpn_bbox_pred, proposals_order):
        """
        Args:
            proposals_order: shape(batch_size,rois_per_img)
            second_rpn_bbox_pred: shape(N,A*4,H,W)
        Returns:
            res_batch: shape (batch_size,rois_per_img,4)
        """
        batch_size = second_rpn_bbox_pred.shape[0]
        proposals_order = proposals_order.view(batch_size, -1)
        second_rpn_bbox_pred = second_rpn_bbox_pred.permute(
            0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        res_batch = second_rpn_bbox_pred.new(batch_size,
                                             proposals_order.shape[1], 4)
        for i in range(batch_size):
            bbox_single = second_rpn_bbox_pred[i]
            order_single = proposals_order[i]
            res_batch[i] = bbox_single[order_single]
        return res_batch.view(-1, 4)

    def second_rpn_cls_select(self, second_rpn_cls_score, proposals_order):
        """
        note that nclasses =2 here
        Args:
            proposals_order: shape(batch_size,rois_per_img)
            second_rpn_cls_score: shape(N,2*A,H,W)
        Returns:
            res_batch: shape(batch_size,rois_per_img,2)
        """
        batch_size = second_rpn_cls_score.shape[0]

        proposals_order = proposals_order.view(batch_size, -1)
        h, w = second_rpn_cls_score.shape[-2:]
        second_rpn_cls_score = second_rpn_cls_score.view(
            batch_size, self.n_classes, -1, h,
            w).permute(0, 3, 4, 2,
                       1).contiguous().view(batch_size, -1, self.n_classes)
        res_batch = second_rpn_cls_score.new(
            batch_size, proposals_order.shape[1], self.n_classes)
        for i in range(batch_size):
            cls_single = second_rpn_cls_score[i]
            order_single = proposals_order[i]
            res_batch[i] = cls_single[order_single]
        return res_batch.view(-1, self.n_classes)

    def forward(self, feed_dict):
        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})
        # batch_size = base_feat.shape[0]

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        if self.training:
            self.pre_subsample(prediction_dict, feed_dict)
        rois_batch = prediction_dict['rois_batch']

        gt_boxes = feed_dict['gt_boxes']
        cleaned_feat = self.clean_base_feat(base_feat, rois_batch, gt_boxes)
        second_rpn_conv1 = F.relu(
            self.second_rpn_conv(cleaned_feat), inplace=True)

        # cls
        # shape(N,2*A,H,W)
        second_rpn_cls_scores = self.second_rpn_cls_score(second_rpn_conv1)

        # reg
        # shape(N,A*4,H,W)
        second_rpn_bbox_pred = self.second_rpn_bbox_pred(second_rpn_conv1)

        proposals_order = prediction_dict['proposals_order']
        # mask select
        proposals_order = proposals_order.long()
        second_rpn_bbox_pred = self.second_rpn_bbox_select(
            second_rpn_bbox_pred, proposals_order)
        second_rpn_cls_scores = self.second_rpn_cls_select(
            second_rpn_cls_scores, proposals_order)

        if self.training:
            second_rpn_cls_prob = 0
        else:
            second_rpn_cls_prob = F.softmax(second_rpn_cls_scores, dim=1)

        second_rpn_anchors = prediction_dict['anchors'][0][proposals_order]
        prediction_dict['second_rpn_anchors'] = second_rpn_anchors
        prediction_dict.update({
            'rcnn_cls_probs': second_rpn_cls_prob,
            'rcnn_bbox_preds': second_rpn_bbox_pred,
            'rcnn_cls_scores': second_rpn_cls_scores,
        })

        return prediction_dict

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = FeatureExtractor(
            self.feature_extractor_config)
        self.rpn_model = FirstRPNModel(self.rpn_config)
        # self.rcnn_pooling = RoIAlignAvg(self.pooling_size, self.pooling_size,
        # 1.0 / 16.0)
        # self.l2loss = nn.MSELoss(reduce=False)

        self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(2048, 4)
        else:
            self.rcnn_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(2)
        else:
            self.rcnn_cls_loss = F.cross_entropy

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        self.din = 1024
        self.num_anchors = 9
        self.nc_bbox_out = 4 * self.num_anchors
        self.nc_score_out = self.num_anchors * 2
        # self.second_rpn_conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
        self.second_rpn_conv = self.make_second_rpn_conv()
        self.second_rpn_cls_score = nn.Conv2d(1024, self.nc_score_out, 1, 1, 0)
        self.second_rpn_bbox_pred = nn.Conv2d(1024, self.nc_bbox_out, 1, 1, 0)

        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear')

    def make_second_rpn_conv(self):
        layers = []
        layers.append(nn.Conv2d(self.din, 512, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=False))
        layers.append(nn.Conv2d(512, self.din, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(self.din))
        return nn.Sequential(*layers)

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        ##########################
        # assigner
        ##########################
        rcnn_cls_targets, rcnn_reg_targets, rcnn_cls_weights, rcnn_reg_weights = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_labels)

        ##########################
        # subsampler
        ##########################
        pos_indicator = rcnn_cls_targets > 0
        critation = rcnn_cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size, pos_indicator, criterion=critation)
        rcnn_cls_weights = rcnn_cls_weights[batch_sampled_mask]
        rcnn_reg_weights = rcnn_reg_weights[batch_sampled_mask]
        num_cls_coeff = rcnn_cls_weights.type(torch.cuda.ByteTensor).sum(
            dim=-1)
        num_reg_coeff = rcnn_reg_weights.type(torch.cuda.ByteTensor).sum(
            dim=-1)
        if num_cls_coeff == 0:
            num_cls_coeff = torch.ones([]).type_as(num_cls_coeff)
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

        # used for track
        proposals_order = prediction_dict['proposals_order']

        prediction_dict['proposals_order'] = proposals_order[batch_sampled_mask]

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
        # self.sampler = HardNegativeSampler(model_config['sampler_config'])
        self.sampler = BalancedSampler(model_config['sampler_config'])

        # coder
        self.bbox_coder = self.target_assigner.bbox_coder

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
        # rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        # loss weights has no gradients
        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        # add rcnn_cls_targets to get the statics of rpn
        loss_dict['rcnn_cls_targets'] = rcnn_cls_targets

        return loss_dict
