#!/usr/bin/env python
# encoding: utf-8



import torch
import torch.nn as nn
import numpy as np
from core.target_assigner import TargetAssigner
from core.sampler import Sampler


class AnchorTargetLayer(nn.Module):
    """
    """

    def __init__(self, layer_config):
        super().__init__()
        # some parameters
        self.rpn_positive_weight = layer_config['rpn_positive_weight']
        self.rpn_negative_overlaps = layer_config['rpn_negative_overlaps']
        self.rpn_positive_overlaps = layer_config['rpn_positive_overlaps']
        self.rpn_batch_size = layer_config['rpn_batch_size']
        # subsample score and iou or subsample score only
        self.subsample_twice = layer_config['subsample_twice']
        self.subsample_type = layer_config['subsample_type']

        self.target_assigner = TargetAssigner()
        self.sampler = Sampler(self.subsample_type)

    def forward(self, anchors, rpn_cls_score, gt_boxes, gt_labels):
        """
        Subsample and generate samples for training
        Args:
            rpn_cls_score, used for subsample
            gt_boxes, gt boxes,shape(N,M,4)
            anchors, shape(K,4)
            im_info, info of image size and ratios
        Returns:
            bbox_weights: weights for box regression
            cls_weights: wegihts for cls
            labels: labels for each anchors
            bbox_targets: bbox regression target for each anchors
        """
        ######################
        # assignments
        ######################
        cls_targets, reg_targets, cls_weights, reg_weights = self.target_assigner.assign(
            anchors, gt_boxes, gt_labels)

        ##########################
        # subsampler
        ##########################
        if self.subsample_twice:
            # subsample both
            cls_batch_sampled_mask = self.sampler.subsample(
                cls_weights,
                self.rpn_batch_size,
                cls_targets.type(torch.ByteTensor),
                critation=rpn_cls_score)
            cls_weights *= cls_batch_sampled_mask
            reg_batch_sampled_mask = self.sampler.subsample(
                reg_weights, self.rpn_batch_size)
            reg_weights *= reg_batch_sampled_mask
        else:
            # subsample score only
            batch_sampled_mask = self.sampler.subsample(
                cls_weights,
                self.rpn_batch_size,
                cls_targets.type(torch.ByteTensor),
                critation=rpn_cls_score)
            cls_weights = cls_weights * batch_sampled_mask
            reg_weights = reg_weights * batch_sampled_mask

        output = {}
        output['cls_targets'] = cls_targets
        output['reg_targets'] = reg_targets
        output['cls_weights'] = cls_weights
        output['reg_weights'] = reg_weights

        return output
