# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from core.model import Model
from core.anchor_generators.anchor_generator import AnchorGenerator
# from core.samplers.hard_negative_sampler import HardNegativeSampler
# from core.samplers.balanced_sampler import BalancedSampler
from core.samplers.detection_sampler import DetectionSampler
from core.target_assigner import TargetAssigner
from core.filler import Filler
from core.models.focal_loss import FocalLoss
from core.loss import ClusterLoss

from utils import box_ops
from lib.model.nms.nms_wrapper import nms
import functools


class LossRPNModel(Model):
    def init_param(self, model_config):
        self.in_channels = model_config['din']
        self.post_nms_topN = model_config['post_nms_topN']
        self.pre_nms_topN = model_config['pre_nms_topN']
        self.nms_thresh = model_config['nms_thresh']
        self.use_score = model_config['use_score']
        self.rpn_batch_size = model_config['rpn_batch_size']
        self.use_focal_loss = model_config['use_focal_loss']

        # sampler
        # self.sampler = HardNegativeSampler(model_config['sampler_config'])
        # self.sampler = BalancedSampler(model_config['sampler_config'])
        self.sampler = DetectionSampler(model_config['sampler_config'])

        # anchor generator
        self.anchor_generator = AnchorGenerator(
            model_config['anchor_generator_config'])
        self.num_anchors = self.anchor_generator.num_anchors
        self.nc_bbox_out = 4 * self.num_anchors
        self.nc_score_out = self.num_anchors * 2

        # target assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        # bbox coder
        self.bbox_coder = self.target_assigner.bbox_coder

        self.use_iou = model_config.get('use_iou')

    def init_weights(self):
        self.truncated = False

        Filler.normal_init(self.rpn_conv, 0, 0.01, self.truncated)
        Filler.normal_init(self.rpn_cls_score, 0, 0.01, self.truncated)
        Filler.normal_init(self.rpn_bbox_pred, 0, 0.01, self.truncated)

    def init_modules(self):
        # define the convrelu layers processing input feature map
        self.rpn_conv = nn.Conv2d(self.in_channels, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.rpn_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer

        if self.use_score:
            bbox_feat_channels = 512 + 2
            self.nc_bbox_out /= self.num_anchors
        else:
            bbox_feat_channels = 512
        self.rpn_bbox_pred = nn.Conv2d(bbox_feat_channels, self.nc_bbox_out, 1,
                                       1, 0)

        # bbox
        self.rpn_bbox_loss = nn.modules.loss.SmoothL1Loss(reduce=False)

        # cls
        if self.use_focal_loss:
            self.rpn_cls_loss = FocalLoss(2)
        else:
            self.rpn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

        self.cluster_loss = ClusterLoss()

    def generate_proposal(self, rpn_cls_probs, anchors, rpn_bbox_preds,
                          im_info):
        # TODO create a new Function
        """
        Args:
            rpn_cls_probs: FloatTensor,shape(N,2*num_anchors,H,W)
            rpn_bbox_preds: FloatTensor,shape(N,num_anchors*4,H,W)
            anchors: FloatTensor,shape(N,4,H,W)

        Returns:
            proposals_batch: FloatTensor, shape(N,post_nms_topN,4)
            fg_probs_batch: FloatTensor, shape(N,post_nms_topN)
        """
        # assert len(
        # rpn_bbox_preds) == 1, 'just one feature maps is supported now'
        # rpn_bbox_preds = rpn_bbox_preds[0]
        anchors = anchors[0]
        # do not backward
        anchors = anchors
        rpn_cls_probs = rpn_cls_probs.detach()
        rpn_bbox_preds = rpn_bbox_preds.detach()

        batch_size = rpn_bbox_preds.shape[0]
        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous()
        # shape(N,H*W*num_anchors,4)
        rpn_bbox_preds = rpn_bbox_preds.view(batch_size, -1, 4)
        # apply deltas to anchors to decode
        # loop here due to many features maps
        # proposals = []
        # for rpn_bbox_preds_single_map, anchors_single_map in zip(
        # rpn_bbox_preds, anchors):
        # proposals.append(
        # self.bbox_coder.decode(rpn_bbox_preds_single_map,
        # anchors_single_map))
        # proposals = torch.cat(proposals, dim=1)

        proposals = self.bbox_coder.decode_batch(rpn_bbox_preds, anchors)

        # filer and clip
        proposals = box_ops.clip_boxes(proposals, im_info)

        # fg prob
        fg_probs = rpn_cls_probs[:, self.num_anchors:, :, :]
        fg_probs = fg_probs.permute(0, 2, 3, 1).contiguous().view(batch_size,
                                                                  -1)

        # sort fg
        _, fg_probs_order = torch.sort(fg_probs, dim=1, descending=True)

        # fg_probs_batch = torch.zeros(batch_size,
        # self.post_nms_topN).type_as(rpn_cls_probs)
        proposals_batch = torch.zeros(batch_size, self.post_nms_topN,
                                      4).type_as(rpn_bbox_preds)
        proposals_order = torch.zeros(
            batch_size, self.post_nms_topN).fill_(-1).type_as(fg_probs_order)

        for i in range(batch_size):
            proposals_single = proposals[i]
            fg_probs_single = fg_probs[i]
            fg_order_single = fg_probs_order[i]
            # pre nms
            if self.pre_nms_topN > 0:
                fg_order_single = fg_order_single[:self.pre_nms_topN]
            proposals_single = proposals_single[fg_order_single]
            fg_probs_single = fg_probs_single[fg_order_single]

            # nms
            keep_idx_i = nms(
                torch.cat((proposals_single, fg_probs_single.unsqueeze(1)), 1),
                self.nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            # post nms
            if self.post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self.post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            fg_probs_single = fg_probs_single[keep_idx_i]
            fg_order_single = fg_order_single[keep_idx_i]

            # padding 0 at the end.
            num_proposal = keep_idx_i.numel()
            proposals_batch[i, :num_proposal, :] = proposals_single
            # fg_probs_batch[i, :num_proposal] = fg_probs_single
            proposals_order[i, :num_proposal] = fg_order_single
        return proposals_batch, proposals_order

    def forward(self, bottom_blobs):
        base_feat = bottom_blobs['base_feat']
        batch_size = base_feat.shape[0]
        gt_boxes = bottom_blobs['gt_boxes']
        im_info = bottom_blobs['im_info']

        # rpn conv
        rpn_conv = F.relu(self.rpn_conv(base_feat), inplace=True)

        # rpn cls score
        # shape(N,2*num_anchors,H,W)
        rpn_cls_scores = self.rpn_cls_score(rpn_conv)

        # rpn cls prob shape(N,2*num_anchors,H,W)
        rpn_cls_score_reshape = rpn_cls_scores.view(batch_size, 2, -1)
        rpn_cls_probs = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_probs = rpn_cls_probs.view_as(rpn_cls_scores)
        # import ipdb
        # ipdb.set_trace()

        # rpn bbox pred
        # shape(N,4*num_anchors,H,W)
        if self.use_score:
            # shape (N,2,num_anchoros*H*W)
            rpn_cls_scores = rpn_cls_score_reshape.permute(0, 2, 1)
            rpn_bbox_preds = []
            for i in range(self.num_anchors):
                rpn_bbox_feat = torch.cat(
                    [rpn_conv, rpn_cls_scores[:, ::self.num_anchors, :, :]],
                    dim=1)
                rpn_bbox_preds.append(self.rpn_bbox_pred(rpn_bbox_feat))
            rpn_bbox_preds = torch.cat(rpn_bbox_preds, dim=1)
        else:
            # get rpn offsets to the anchor boxes
            rpn_bbox_preds = self.rpn_bbox_pred(rpn_conv)
            # rpn_bbox_preds = [rpn_bbox_preds]

        # generate anchors
        feature_map_list = [base_feat.size()[-2:]]
        anchors = self.anchor_generator.generate(feature_map_list)

        ###############################
        # Proposal
        ###############################
        # note that proposals_order is used for track transform of propsoals
        proposals_batch, proposals_order = self.generate_proposal(
            rpn_cls_probs, anchors, rpn_bbox_preds, im_info)
        batch_idx = torch.arange(batch_size).view(batch_size, 1).expand(
            -1, proposals_batch.shape[1]).type_as(proposals_batch)
        rois_batch = torch.cat((batch_idx.unsqueeze(-1), proposals_batch),
                               dim=2)

        if self.training:
            rois_batch = self.append_gt(rois_batch, gt_boxes)

        rpn_cls_scores = rpn_cls_scores.view(batch_size, 2, -1,
                                             rpn_cls_scores.shape[2],
                                             rpn_cls_scores.shape[3])
        rpn_cls_scores = rpn_cls_scores.permute(
            0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)

        # postprocess
        rpn_cls_probs = rpn_cls_probs.view(
            batch_size, 2, -1, rpn_cls_probs.shape[2], rpn_cls_probs.shape[3])
        rpn_cls_probs = rpn_cls_probs.permute(0, 3, 4, 2, 1).contiguous().view(
            batch_size, -1, 2)
        predict_dict = {
            'proposals_batch': proposals_batch,
            'rpn_cls_scores': rpn_cls_scores,
            'rois_batch': rois_batch,
            'anchors': anchors,

            # used for loss
            'rpn_bbox_preds': rpn_bbox_preds,
            'rpn_cls_probs': rpn_cls_probs,
            'proposals_order': proposals_order,
        }

        return predict_dict

    def append_gt(self, rois_batch, gt_boxes):
        ################################
        # append gt_boxes to rois_batch for losses
        ################################
        # may be some bugs here
        gt_boxes_append = torch.zeros(gt_boxes.shape[0], gt_boxes.shape[1],
                                      5).type_as(gt_boxes)
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        # cat gt_boxes to rois_batch
        rois_batch = torch.cat([rois_batch, gt_boxes_append], dim=1)
        return rois_batch

    def loss(self, prediction_dict, feed_dict):
        # loss for cls
        loss_dict = {}

        gt_boxes = feed_dict['gt_boxes']

        anchors = prediction_dict['anchors']

        assert len(anchors) == 1, 'just one feature maps is supported now'
        num_gt = gt_boxes.shape[1]
        anchors = anchors[0]

        #################################
        # target assigner
        ################################
        # no need gt labels here,it just a binary classifcation problem
        #  import ipdb
        #  ipdb.set_trace()
        rpn_cls_targets, rpn_reg_targets, \
            rpn_cls_weights, rpn_reg_weights = \
            self.target_assigner.assign(anchors, gt_boxes, gt_labels=None)

        ################################
        # subsample
        ################################

        pos_indicator = rpn_reg_weights > 0
        indicator = rpn_cls_weights > 0

        if self.use_iou:
            cls_criterion = self.target_assigner.matcher.assigned_overlaps_batch
        else:
            rpn_cls_probs = prediction_dict['rpn_cls_probs'][:, :, 1]
            cls_criterion = rpn_cls_probs

        batch_sampled_mask = self.sampler.subsample_batch(
            self.rpn_batch_size,
            pos_indicator,
            criterion=cls_criterion,
            indicator=indicator)
        batch_sampled_mask = batch_sampled_mask.type_as(rpn_cls_weights)
        rpn_cls_weights = rpn_cls_weights * batch_sampled_mask
        rpn_reg_weights = rpn_reg_weights * batch_sampled_mask
        num_cls_coeff = (rpn_cls_weights > 0).sum(dim=1)
        num_reg_coeff = (rpn_reg_weights > 0).sum(dim=1)
        # check
        #  assert num_cls_coeff, 'bug happens'
        #  assert num_reg_coeff, 'bug happens'
        if num_cls_coeff == 0:
            num_cls_coeff = torch.ones([]).type_as(num_cls_coeff)
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        # cls loss
        # shape(N,M,2)
        rpn_cls_score = prediction_dict['rpn_cls_scores']
        # rpn_cls_loss = self.rpn_cls_loss(rpn_cls_score, rpn_cls_targets)
        rpn_cls_loss = self.rpn_cls_loss(
            rpn_cls_score.view(-1, 2), rpn_cls_targets.view(-1))
        rpn_cls_loss = rpn_cls_loss.view_as(rpn_cls_weights)
        rpn_cls_loss *= rpn_cls_weights
        rpn_cls_loss = rpn_cls_loss.sum(dim=1) / num_cls_coeff.float()

        # bbox loss
        # shape(N,num,4)
        rpn_bbox_preds = prediction_dict['rpn_bbox_preds']
        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous()
        # shape(N,H*W*num_anchors,4)
        rpn_bbox_preds = rpn_bbox_preds.view(rpn_bbox_preds.shape[0], -1, 4)
        rpn_reg_loss = self.rpn_bbox_loss(rpn_bbox_preds, rpn_reg_targets)
        rpn_reg_loss *= rpn_reg_weights.unsqueeze(-1).expand(-1, -1, 4)
        rpn_reg_loss = rpn_reg_loss.view(rpn_reg_loss.shape[0], -1).sum(
            dim=1) / num_reg_coeff.float()

        # cluster loss
        match = self.target_assigner.matcher.assignments[0]
        #  rpn_cls_feat_single = rpn_cls_score[0]
        rpn_bbox_pred_single = rpn_bbox_preds[0]
        #  cluster_cls_loss = 0
        cluster_bbox_loss = 0
        # if num_gt == 1:

        for i in range(num_gt):
            #  cls_feat = rpn_cls_feat_single[match == i]
            #  cluster_cls_loss += self.cluster_loss(cls_feat)
            bbox_feat = rpn_bbox_pred_single[match == i]
            cluster_bbox_loss += self.cluster_loss(bbox_feat)

        # if cluster_cls_loss.item() == 0:
        # import ipdb
        # ipdb.set_trace()
        loss_dict['rpn/cls_loss'] = rpn_cls_loss
        loss_dict['rpn/bbox_loss'] = rpn_reg_loss
        loss_dict['rpn/cluster_bbox_loss'] = cluster_bbox_loss
        #  loss_dict['rpn/cluster_cls_loss'] = cluster_cls_loss
        return loss_dict
