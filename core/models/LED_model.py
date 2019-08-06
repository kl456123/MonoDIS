# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.LED_rpn_model import LEDRPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from core.filler import Filler
from core.LED_target_assigner import LEDTargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler
from core.loss import SharpL2Loss
from core.bbox_coders.discrete_coder import DiscreteBBoxCoder

import functools


class LEDFasterRCNN(Model):
    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()

        # feed_dict['input_size'] = torch.stack(img_shapes, dim=0)
        feed_dict['input_size'] = feed_dict['im_info']

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
        # shape(N,C)
        if self.reduce:
            pooled_feat = pooled_feat.mean(3).mean(2)
        else:
            pooled_feat = pooled_feat.view(self.rcnn_batch_size, -1)

        rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat)
        rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat)
        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)
        # import ipdb
        # ipdb.set_trace()
        iou, iou_scores, iou_reg = self.iou_pred(pooled_feat)
        iog, iog_scores, iog_reg = self.iog_pred(pooled_feat)
        iod, iod_scores, iod_reg = self.iod_pred(pooled_feat)

        iou = self.iox_clip(iou)
        iog = self.iox_clip(iog)
        iod = self.iox_clip(iod)

        # import ipdb
        # ipdb.set_trace()
        iou_indirect = self.calculate_iou(iog, iod)
        iou_final = (1 - self.alpha) * iou_indirect + self.alpha * iou
        if self.use_cls_pred:

            rcnn_fg_probs_final = rcnn_cls_probs[:, 1] * torch.exp(-torch.pow(
                (1 - iou_final), 2) / self.theta)
        else:
            rcnn_fg_probs_final = iou_final

        prediction_dict['rcnn_cls_probs'] = torch.stack(
            [rcnn_fg_probs_final, rcnn_fg_probs_final], dim=-1)
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores
        # prediction_dict['rcnn_iou_final'] = iou_final

        prediction_dict['rcnn_iou_reg'] = iou_reg
        prediction_dict['rcnn_iou_scores'] = iou_scores
        prediction_dict['rcnn_iod_reg'] = iod_reg
        prediction_dict['rcnn_iod_scores'] = iod_scores
        prediction_dict['rcnn_iog_reg'] = iog_reg
        prediction_dict['rcnn_iog_scores'] = iog_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][0][
            proposals_order]

        return prediction_dict

    def iox_clip(self, iox):
        iox = iox.clone()
        iox[iox < 0] = 0
        iox[iox > 1] = 1
        return iox

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

        Filler.normal_init(self.rcnn_coarse_map_conv_iod, 0, 0.001,
                           self.truncated)
        Filler.normal_init(self.rcnn_fine_map_conv_iod, 0, 0.001,
                           self.truncated)

        Filler.normal_init(self.rcnn_coarse_map_conv_iou, 0, 0.001,
                           self.truncated)

        Filler.normal_init(self.rcnn_fine_map_conv_iou, 0, 0.001,
                           self.truncated)
        Filler.normal_init(self.rcnn_fine_map_conv_iog, 0, 0.001,
                           self.truncated)
        Filler.normal_init(self.rcnn_coarse_map_conv_iog, 0, 0.001,
                           self.truncated)

        # freeze all first
        self.freeze_modules()

        # unfreeze some modules
        self.rpn_model.unfreeze_modules()
        self.unfreeze_modules()

    def unfreeze_modules(self):
        unfreeze_modules = [
            self.rcnn_coarse_map_conv_iod.bias,
            self.rcnn_fine_map_conv_iod.bias,
            self.rcnn_coarse_map_conv_iog.bias,
            self.rcnn_fine_map_conv_iog.bias,
            self.rcnn_coarse_map_conv_iou.bias,
            self.rcnn_fine_map_conv_iou.bias,
            self.rcnn_coarse_map_conv_iod.weight,
            self.rcnn_fine_map_conv_iod.weight,
            self.rcnn_coarse_map_conv_iog.weight,
            self.rcnn_fine_map_conv_iog.weight,
            self.rcnn_coarse_map_conv_iou.weight,
            self.rcnn_fine_map_conv_iou.weight
        ]
        for module in unfreeze_modules:
            module.requires_grad = True

    def init_modules(self):
        self.feature_extractor = ResNetFeatureExtractor(
            self.feature_extractor_config)
        self.rpn_model = LEDRPNModel(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = RoIAlignAvg(self.pooling_size,
                                            self.pooling_size, 1.0 / 16.0)
        elif self.pooling_mode == 'ps':
            self.rcnn_pooling = PSRoIPool(7, 7, 1.0 / 16, 7, self.n_classes)
        elif self.pooling_mode == 'psalign':
            raise NotImplementedError('have not implemented yet!')
        elif self.pooling_mode == 'deformable_psalign':
            raise NotImplementedError('have not implemented yet!')
        self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        if self.reduce:
            in_channels = 2048
        else:
            in_channels = 2048 * 4 * 4
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(in_channels, 4)
        else:
            self.rcnn_bbox_pred = nn.Linear(in_channels, 4 * self.n_classes)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        # pred for iox
        self.rcnn_coarse_map_conv_iou = nn.Linear(2048, 4)
        self.rcnn_fine_map_conv_iou = nn.Linear(2048, 4)

        self.rcnn_coarse_map_conv_iog = nn.Linear(2048, 4)
        self.rcnn_fine_map_conv_iog = nn.Linear(2048, 4)

        self.rcnn_coarse_map_conv_iod = nn.Linear(2048, 4)
        self.rcnn_fine_map_conv_iod = nn.Linear(2048, 4)

        # loss for iox
        if self.use_sharpL2:
            self.reg_loss = SharpL2Loss()
        else:
            self.reg_loss = nn.MSELoss(reduce=False)
        self.cls_loss = nn.CrossEntropyLoss(reduce=False)

        # cls loss
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(2)
        else:
            self.rcnn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

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
        self.rpn_config = model_config['rpn_config']
        self.theta = 1.0

        self.use_focal_loss = model_config['use_focal_loss']
        self.subsample_twice = model_config['subsample_twice']
        self.rcnn_batch_size = model_config['rcnn_batch_size']
        self.use_sigmoid = model_config.get('use_sigmoid')
        self.use_sharpL2 = model_config['use_sharpL2']
        self.use_cls_pred = model_config['use_cls_pred']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']

        # assigner
        self.target_assigner = LEDTargetAssigner(
            model_config['target_assigner_config'])

        # sampler
        self.sampler = BalancedSampler(model_config['sampler_config'])

        self.reduce = True

        self.alpha = 0.6
        # self.iou_anchors = [0.05, 0.25, 0.55, 0.85]
        # self.iou_lengths = [0.05, 0.15, 0.15, 0.15]
        # self.iou_intervals = [[0, 0.1], [0.1, 0.4], [0.4, 0.7], [0.7, 1.0]]
        self.iox_bbox_coder = DiscreteBBoxCoder(
            model_config['iox_coder_config'])

    def iou_pred(self, rcnn_conv):
        return self.iox_pred(rcnn_conv, self.rcnn_coarse_map_conv_iou,
                             self.rcnn_fine_map_conv_iou)

    def iog_pred(self, rcnn_conv):
        return self.iox_pred(rcnn_conv, self.rcnn_coarse_map_conv_iog,
                             self.rcnn_fine_map_conv_iog)

    def iod_pred(self, rcnn_conv):
        return self.iox_pred(rcnn_conv, self.rcnn_coarse_map_conv_iod,
                             self.rcnn_fine_map_conv_iod)

    def iox_pred(self, rcnn_conv, rcnn_coarse_map_conv, rcnn_fine_map_conv):
        batch_size = rcnn_conv.shape[0]
        coarse_map = rcnn_coarse_map_conv(rcnn_conv)
        fine_map = rcnn_fine_map_conv(rcnn_conv)

        coarse_map_reshape = coarse_map.view(batch_size, 4)
        iou_level_probs = F.softmax(coarse_map_reshape, dim=1)
        iou_level_probs = iou_level_probs.view_as(coarse_map)
        if self.use_sigmoid:
            # normalize it
            iou_reg = 2 * F.sigmoid(fine_map) - 1
        else:
            iou_reg = fine_map
        iou_cls = iou_level_probs
        decoded_iou = self.iox_bbox_coder.decode_batch(iou_cls, iou_reg)

        # used for cls and reg loss
        iou_cls_scores = coarse_map
        return decoded_iou, iou_cls_scores, iou_reg

    def calculate_iou(self, iog, iod):
        mask = ~(iod == 0)
        iou_indirect = torch.zeros_like(iog)
        iod = iod[mask]
        iog = iog[mask]
        iou_indirect[mask] = (iod * iog) / (iod + iog - iod * iog)
        return iou_indirect

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
        iou_targets = self.target_assigner.matcher.assigned_overlaps_batch
        iou_cls_targets = self.iox_bbox_coder.encode_cls(iou_targets)
        iou_reg_targets = self.iox_bbox_coder.encode_reg(iou_targets)

        prediction_dict['rcnn_iou_cls_targets'] = iou_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_iou_reg_targets'] = iou_reg_targets[
            batch_sampled_mask]

        # iod targets
        iod_targets = self.target_assigner.matcher.assigned_iod_batch
        iod_cls_targets = self.iox_bbox_coder.encode_cls(iod_targets)
        iod_reg_targets = self.iox_bbox_coder.encode_reg(iod_targets)

        prediction_dict['rcnn_iod_cls_targets'] = iod_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_iod_reg_targets'] = iod_reg_targets[
            batch_sampled_mask]

        # iog targets
        iog_targets = self.target_assigner.matcher.assigned_iog_batch
        iog_cls_targets = self.iox_bbox_coder.encode_cls(iog_targets)
        iog_reg_targets = self.iox_bbox_coder.encode_reg(iog_targets)

        prediction_dict['rcnn_iog_cls_targets'] = iog_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_iog_reg_targets'] = iog_reg_targets[
            batch_sampled_mask]

    def iox_loss(self, iou_scores, iou_cls_targets, iou_reg, iou_reg_targets):
        iou_cls_loss = self.cls_loss(iou_scores, iou_cls_targets)
        iou_reg_loss = self.reg_loss(iou_reg, iou_reg_targets).sum(dim=-1)
        return iou_cls_loss.mean(), iou_reg_loss.mean()

    def iou_loss(self, prediction_dict):
        return self.iox_loss(prediction_dict['rcnn_iou_scores'],
                             prediction_dict['rcnn_iou_cls_targets'],
                             prediction_dict['rcnn_iou_reg'],
                             prediction_dict['rcnn_iou_reg_targets'])

    def iog_loss(self, prediction_dict):
        return self.iox_loss(prediction_dict['rcnn_iog_scores'],
                             prediction_dict['rcnn_iog_cls_targets'],
                             prediction_dict['rcnn_iog_reg'],
                             prediction_dict['rcnn_iog_reg_targets'])

    def iod_loss(self, prediction_dict):
        return self.iox_loss(prediction_dict['rcnn_iod_scores'],
                             prediction_dict['rcnn_iod_cls_targets'],
                             prediction_dict['rcnn_iod_reg'],
                             prediction_dict['rcnn_iod_reg_targets'])

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}

        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

        # iou loss
        iou_cls_loss, iou_reg_loss = self.iou_loss(prediction_dict)

        # iog loss
        iog_cls_loss, iog_reg_loss = self.iog_loss(prediction_dict)

        # iod loss
        iod_cls_loss, iod_reg_loss = self.iod_loss(prediction_dict)

        # total_loss = [
        # iou_cls_loss, iou_reg_loss, iog_cls_loss, iog_reg_loss,
        # iod_reg_loss, iod_cls_loss
        # ]

        # classification loss
        if self.use_cls_pred:
            rcnn_cls_weights = prediction_dict['rcnn_cls_weights']
            rcnn_cls_targets = prediction_dict['rcnn_cls_targets']
            rcnn_cls_scores = prediction_dict['rcnn_cls_scores']
            rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores,
                                               rcnn_cls_targets)
            rcnn_cls_loss *= rcnn_cls_weights
            rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)
            loss_dict['rcnn/cls_loss'] = rcnn_cls_loss

        loss_dict['rcnn/iou_cls_loss'] = iou_cls_loss
        loss_dict['rcnn/iou_reg_loss'] = iou_reg_loss
        loss_dict['rcnn/iog_cls_loss'] = iog_cls_loss
        loss_dict['rcnn/iog_reg_loss'] = iog_reg_loss
        loss_dict['rcnn/iod_reg_loss'] = iod_reg_loss
        loss_dict['rcnn/iod_cls_loss'] = iod_cls_loss
        # iox_loss = 0
        # for loss in total_loss:
        # if torch.isnan(loss).byte().any():
        # import ipdb
        # ipdb.set_trace()
        # iox_loss += loss

        # bbox regression loss
        rcnn_reg_weights = prediction_dict['rcnn_reg_weights']
        rcnn_reg_targets = prediction_dict['rcnn_reg_targets']
        rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        # loss weights has no gradients
        # loss_dict['rcnn_cls_loss'] = iox_loss
        loss_dict['rcnn/bbox_loss'] = rcnn_bbox_loss

        # add rcnn_cls_targets to get the statics of rpn
        # loss_dict['rcnn_reg_targets'] = rcnn_reg_weights

        return loss_dict
