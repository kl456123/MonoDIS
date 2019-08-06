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
from core.stereo_target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler
from utils.visualizer import FeatVisualizer
from core.models import common_blocks
from core.profiler import Profiler

import functools


class KeyPointPredictor(nn.Module):
    def __init__(self, inplane, output=4):
        super().__init__()

        layers = []
        for i in range(6):
            layers.append(common_blocks.conv3x3_bn_relu(inplane, 256))
            inplane = 256

        # upsample
        deconv = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        bn = nn.BatchNorm2d(256)
        relu = nn.ReLU()
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        conv = nn.Conv2d(256, output, 1, 1, 0)
        layers.extend([deconv, bn, relu, upsample, conv])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class KeyPointPredictor2(nn.Module):
    def __init__(self, inplane, output=4):
        super().__init__()

        layers = []
        for i in range(8):
            layers.append(common_blocks.conv3x3_bn_relu(inplane, 512))
            inplane = 512

        # upsample
        deconv = nn.ConvTranspose2d(512, 512, 2, 2, 0)
        bn = nn.BatchNorm2d(512)
        relu = nn.ReLU()
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        conv = nn.Conv2d(512, output, 1, 1, 0)
        layers.extend([deconv, bn, relu, upsample, conv])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Mono3DSimplerFasterRCNN(Model):
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

        if self.training:
            self.pre_subsample(prediction_dict, feed_dict)
        rois_batch = prediction_dict['rois_batch']

        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))
        mask_pooled_feat = self.mask_rcnn_pooling(base_feat,
                                                  rois_batch.view(-1, 5))

        pooled_feat = self.feature_extractor.second_stage_feature(pooled_feat)

        #  common_pooled_feat = pooled_feat

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
        keypoint_heatmap = self.keypoint_predictor(mask_pooled_feat)
        keypoint_scores = keypoint_heatmap.view(-1, 56 * 56)
        keypoint_probs = F.softmax(keypoint_scores, dim=-1)

        prediction_dict['keypoint_probs'] = keypoint_probs
        prediction_dict['keypoint_scores'] = keypoint_scores

        # import ipdb
        # ipdb.set_trace()
        rcnn_3d = self.rcnn_3d_pred(reduced_pooled_feat)
        prediction_dict['rcnn_3d'] = rcnn_3d
        if not self.training:
            #  import ipdb
            #  ipdb.set_trace()
            #  _, keypoint_peak_pos = keypoint_probs.max(dim=-1)
            keypoints = self.keypoint_coder.decode_keypoint_heatmap(
                rois_batch[0, :, 1:], keypoint_probs.view(-1, 4, 56 * 56))
            prediction_dict['keypoints'] = keypoints

        return prediction_dict

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)


    def modify_feature_extractor(self):
        from torchvision.models.resnet import Bottleneck
        layer4 = self._make_layer(Bottleneck, 512, 3, stride=1)
        self.feature_extractor.second_stage_feature = layer4

    def init_modules(self):
        self.feature_extractor = ResNetFeatureExtractor(
            self.feature_extractor_config)

        self.modify_feature_extractor()
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
        self.mask_rcnn_pooling = RoIAlignAvg(14, 14, 1.0 / 16.0)
        # self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
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
        self.rcnn_kp_loss = functools.partial(
            F.cross_entropy, reduce=False, ignore_index=-1)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        # some 3d statistic
        # some 2d points projected from 3d
        self.rcnn_3d_pred = nn.Linear(in_channels, 3)

        # self.rcnn_3d_loss = MultiBinLoss(num_bins=self.num_bins)
        # self.rcnn_3d_loss = MultiBinRegLoss(num_bins=self.num_bins)
        self.rcnn_3d_loss = OrientationLoss(split_loss=True)

        self.keypoint_predictor = KeyPointPredictor2(1024)

    def _make_layer(self, block, planes, blocks, stride=1):
        inplanes  = 1024
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

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

        self.num_bins = 4

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])
        self.keypoint_coder = self.target_assigner.keypoint_coder

        self.profiler = Profiler()

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        # shape(N,7)
        gt_boxes_3d = feed_dict['gt_boxes_3d']

        keypoint_gt = feed_dict['keypoint_gt']

        # import ipdb
        # ipdb.set_trace()
        gt_boxes_3d = torch.cat([gt_boxes_3d[:, :, :3], keypoint_gt], dim=-1)

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

        # keypoint heatmap loss
        # keypoint_gt = feed_dict['keypoint_gt']
        #  import ipdb
        #  ipdb.set_trace()
        rcnn_reg_targets_3d = prediction_dict['rcnn_reg_targets_3d']
        rcnn_reg_weights_3d = prediction_dict['rcnn_reg_weights_3d']
        keypoint_scores = prediction_dict['keypoint_scores']
        keypoint_gt = rcnn_reg_targets_3d[:, 3:].contiguous().view(-1, 2)
        keypoint_weights = keypoint_gt[:, 1]
        keypoint_pos = keypoint_gt[:, 0]
        keypoint_pos[keypoint_weights == 0] = -1
        keypoint_loss = self.rcnn_kp_loss(keypoint_scores, keypoint_pos.long())
        keypoint_loss = keypoint_loss.view(
            -1, 4) * rcnn_reg_weights_3d.unsqueeze(-1)
        #  keypoint_loss = keypoint_loss * keypoint_weights
        loss_dict['keypoint_loss'] = keypoint_loss.sum(dim=-1).sum(dim=-1)

        # dims loss
        rcnn_3d = prediction_dict['rcnn_3d']
        rcnn_3d_loss = self.rcnn_bbox_loss(rcnn_3d, rcnn_reg_targets_3d[:, :3])
        rcnn_3d_loss = rcnn_3d_loss * rcnn_reg_weights_3d.sum(dim=-1)
        loss_dict['rcnn_3d_loss'] = rcnn_3d_loss.sum(dim=-1).sum(dim=-1)

        return loss_dict
