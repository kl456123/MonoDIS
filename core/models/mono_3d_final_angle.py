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


class Mono3DFinalAngleFasterRCNN(Model):
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
            if self.class_agnostic_3d:
                orient = rcnn_3d[:, 3:]
                dims = rcnn_3d[:, :3]
            else:
                orient = rcnn_3d[:, 3 * self.n_classes:]
                dims = rcnn_3d[:, :3 * self.n_classes]
            angles = orient.view(-1, self.num_bins, 4)
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
            self.rcnn_pooling = ROIAlign((self.pooling_size,
                                          self.pooling_size), 1.0 / 16.0, 2)
        elif self.pooling_mode == 'ps':
            self.rcnn_pooling = PSRoIPool(7, 7, 1.0 / 16, 7, self.n_classes)
        elif self.pooling_mode == 'psalign':
            raise NotImplementedError('have not implemented yet!')
        elif self.pooling_mode == 'deformable_psalign':
            raise NotImplementedError('have not implemented yet!')
        # self.rcnn_cls_pred = nn.Conv2d(2048, self.n_classes, 3, 1, 1)
        self.rcnn_cls_preds = nn.Linear(2048, self.n_classes)
        if self.reduce:
            in_channels = 2048
        else:
            in_channels = 2048 * 4 * 4
        if self.class_agnostic:
            self.rcnn_bbox_preds = nn.Linear(in_channels, 4)
        else:
            self.rcnn_bbox_preds = nn.Linear(in_channels, 4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(self.n_classes)
        else:
            self.rcnn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        # self.rcnn_3d_pred = nn.Linear(c, 3 + 4 + 11 + 2 + 1)
        if self.class_agnostic_3d:
            self.rcnn_3d_pred = nn.Linear(in_channels, 3 + 4 * self.num_bins)
        else:
            self.rcnn_3d_pred = nn.Linear(
                in_channels, 3 * self.n_classes + 4 * self.num_bins)

        #  self.rcnn_3d_loss = OrientationLoss(split_loss=True)
        self.rcnn_3d_loss = MultiBinLoss(num_bins=self.num_bins)

    def init_param(self, model_config):
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
        self.feature_extractor_config = model_config['feature_extractor_config']
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

        rcnn_reg_weights_3d = prediction_dict['rcnn_reg_weights_3d']
        rcnn_reg_targets_3d = prediction_dict['rcnn_reg_targets_3d']
        rcnn_3d = prediction_dict['rcnn_3d']

        if not self.class_agnostic_3d:
            dims_pred = rcnn_3d[:, :3 * self.n_classes]
            dims_pred = self.squeeze_bbox_preds(dims_pred, rcnn_cls_targets, 3)
            orient_pred = rcnn_3d[:, 3 * self.n_classes:]
        else:
            dims_pred = rcnn_3d[:, :3]
            orient_pred = rcnn_3d[:, 3:]
        # dims
        rcnn_3d_loss_dims = self.rcnn_bbox_loss(
            dims_pred, rcnn_reg_targets_3d[:, :3]).sum(dim=-1)

        # angles
        rcnn_angle_loss, angle_tp_mask = self.rcnn_3d_loss(
            orient_pred, rcnn_reg_targets_3d[:, 3:])
        # angles
        #  res = self.rcnn_3d_loss(rcnn_3d[:, 3:], rcnn_reg_targets_3d[:, 3:6])
        #  for res_loss_key in res:
        #  tmp = res[res_loss_key] * rcnn_reg_weights_3d
        #  res[res_loss_key] = tmp.sum(dim=-1)
        #  loss_dict.update(res)

        rcnn_3d_loss = rcnn_3d_loss_dims * rcnn_reg_weights_3d
        rcnn_3d_loss = rcnn_3d_loss.sum(dim=-1)

        rcnn_angle_loss = rcnn_angle_loss * rcnn_reg_weights_3d
        rcnn_angle_loss = rcnn_angle_loss.sum(dim=-1)

        loss_dict['rcnn_3d_loss'] = rcnn_3d_loss
        loss_dict['rcnn_angle_loss'] = rcnn_angle_loss

        # stats of orients
        angle_tp_mask = angle_tp_mask[rcnn_reg_weights_3d > 0]
        angles_tp_num = angle_tp_mask.int().sum().item()
        angles_all_num = angle_tp_mask.numel()
        #  cls_orient_preds = rcnn_3d[:, 3:5]
        #  cls_orient = rcnn_reg_targets_3d[:, 3]
        #  _, cls_orient_preds_argmax = torch.max(cls_orient_preds, dim=-1)
        #  orient_tp_mask = cls_orient.type_as(
        #  cls_orient_preds_argmax) == cls_orient_preds_argmax
        #  mask = (rcnn_reg_weights_3d > 0) & (rcnn_reg_targets_3d[:, 3] > -1)
        #  orient_tp_mask = orient_tp_mask[mask]
        #  orient_tp_num = orient_tp_mask.int().sum().item()
        #  orient_all_num = orient_tp_mask.numel()

        # gt_boxes_proj = feed_dict['gt_boxes_proj']

        self.target_assigner.stat.update({
            'cls_orient_2s_all_num': angles_all_num,
            'cls_orient_2s_tp_num': angles_tp_num
            # 'angle_num_tp': torch.tensor(0),
            # 'angle_num_all': 1,

            # stats of orient
            #  'orient_tp_num': orient_tp_num,
            # 'orient_tp_num2': orient_tp_num2,
            #  'orient_tp_num3': orient_tp_num3,
            # 'orient_all_num3': orient_all_num3,
            # 'orient_pr': orient_pr,
            #  'orient_all_num': orient_all_num,
            #  'orient_all_num3': orient_all_num3,
            # 'orient_tp_num4': orient_tp_num4,
            # 'orient_all_num4': orient_all_num4,
            #  'cls_orient_2s_all_num': depth_ind_all_num,
            #  'cls_orient_2s_tp_num': depth_ind_tp_num
        })

        return loss_dict
