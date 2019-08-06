#!/usr/bin/env python
# encoding: utf-8

import torch

# core classes

from core.analyzer import Analyzer

# builder
from builder import matcher_builder
from builder import bbox_coder_builder
from builder import similarity_calc_builder


class TargetAssigner(object):
    def __init__(self, assigner_config):

        # some compositions
        self.similarity_calc = similarity_calc_builder.build(
            assigner_config['similarity_calc_config'])
        self.bbox_coder = bbox_coder_builder.build(
            assigner_config['coder_config'])
        self.matcher = matcher_builder.build(assigner_config['matcher_config'])
        self.analyzer = Analyzer()

        # cls thresh
        self.fg_thresh_cls = assigner_config['fg_thresh_cls']
        self.bg_thresh_cls = assigner_config['bg_thresh_cls']

        # bbox thresh
        self.fg_thresh_reg = assigner_config['fg_thresh_reg']
        # self.bg_thresh_reg = assigner_config['bg_thresh_reg']

    @property
    def stat(self):
        return self.analyzer.stat

    def assign(self, bboxes, gt_boxes, gt_labels=None, cls_prob=None):
        """
        Assign each bboxes with label and bbox targets for training

        Args:
            bboxes: shape(N,K,4), encoded by xxyy
            gt_boxes: shape(N,M,4), encoded likes as bboxes
        """

        # usually IoU overlaps is used as metric
        bboxes = bboxes.detach()
        match_quality_matrix = self.similarity_calc.compare_batch(bboxes,
                                                                  gt_boxes)

        # match 0.7 for truly recall calculation
        #  if self.fg_thresh_cls < 0.7:
        fake_match = self.matcher.match_batch(match_quality_matrix, 0.7)
        self.analyzer.analyze(fake_match, gt_boxes.shape[1])

        #################################
        # handle cls
        #################################
        #  cls_match = self.matcher.match_batch(match_quality_matrix,
        #  self.fg_thresh_cls)
        #  iou_assigned_overlaps_batch = self.matcher.assigned_overlaps_batch

        #  # assign classification targets
        #  cls_targets = self._assign_classification_targets(cls_match, gt_labels)
        #  cls_targets = iou_assigned_overlaps_batch

        #  # create classification weights
        #  cls_weights = self._create_classification_weights(
        #  cls_assigned_overlaps_batch)

        #  cls_targets[cls_match == -1] = 0

        # as for cls weights, ignore according to bg_thresh
        #  if self.bg_thresh_cls > 0:
        #  ignored_bg = (cls_assigned_overlaps_batch > self.bg_thresh_cls) & (
        #  cls_match == -1)
        #  cls_weights[ignored_bg] = 0

        ##################################
        # handle reg
        ##################################
        reg_match = self.matcher.match_batch(match_quality_matrix,
                                             self.fg_thresh_reg)
        reg_assigned_overlaps_batch = self.matcher.assigned_overlaps_batch

        # assign regression targets
        reg_targets = self._assign_regression_targets(reg_match, bboxes,
                                                      gt_boxes)

        # create regression weights
        reg_weights = self._create_regression_weights(
            reg_assigned_overlaps_batch)

        reg_targets[reg_match == -1] = 0
        reg_weights[reg_match == -1] = 0

        ##################################
        # handle iou
        ##################################
        iou_targets = reg_assigned_overlaps_batch.clone()
        iou_weights = self._create_classification_weights(
            reg_assigned_overlaps_batch)
        # suppress bg
        iou_targets[reg_assigned_overlaps_batch < self.bg_thresh_cls] = 0
        iou_targets[reg_assigned_overlaps_batch > self.fg_thresh_cls] = 1

        return iou_targets, reg_targets, iou_weights, reg_weights

    def _create_regression_weights(self, assigned_overlaps_batch):
        """
        Args:
        assigned_overlaps_batch: shape (num_batch,num_boxes)
        Returns:
        reg_weights: shape(num_batch,num_boxes,4)
        """
        #  gamma = 2
        #  return torch.pow(1 - assigned_overlaps_batch, gamma).detach()
        return torch.ones_like(assigned_overlaps_batch)

    def _create_classification_weights(self, assigned_overlaps_batch):
        """
        All samples can be used for calculating loss,So reserve all.
        """
        cls_weights = torch.ones_like(assigned_overlaps_batch)
        return cls_weights

    def _assign_regression_targets(self, match, bboxes, gt_boxes):
        """
        Args:
            match: Tensor(num_batch,num_boxes)
            gt_boxes: Tensor(num_batch,num_gt_boxes,4)
        Returns:
            reg_targets: Tensor(num_batch,num_boxes,4)
        """
        # shape(num_batch,num_boxes,4)
        batch_size = gt_boxes.shape[0]
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        match += offset.view(batch_size, 1).type_as(match)
        assigned_gt_boxes = gt_boxes.view(-1, 4)[match.view(-1)].view(
            batch_size, -1, 4)
        reg_targets_batch = self.bbox_coder.encode_batch(bboxes,
                                                         assigned_gt_boxes)
        # no need grad_fn
        return reg_targets_batch

    def _assign_classification_targets(self, match, gt_labels):
        """
        Just return the countpart labels
        Note that use zero to represent background labels
        For the first stage, generate binary labels, For the second stage
        generate countpart gt_labels
        """
        # binary labels classifcation
        if gt_labels is None:
            # consider it as binary classification problem
            return self._generate_binary_labels(match)

        # multiple labels classification
        batch_size = match.shape[0]
        offset = torch.arange(0, batch_size) * gt_labels.size(1)
        match += offset.view(batch_size, 1).type_as(match)
        cls_targets_batch = gt_labels.view(-1)[match.view(-1)].view(
            batch_size, match.shape[1])
        return cls_targets_batch

    def _generate_binary_labels(self, match):
        gt_labels_batch = torch.ones_like(match).long()
        return gt_labels_batch
