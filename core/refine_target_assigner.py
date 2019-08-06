#!/usr/bin/env python
# encoding: utf-8

import torch

# core classes

from core.analyzer import Analyzer

# builder
from builder import matcher_builder
from builder import bbox_coder_builder
from builder import similarity_calc_builder


class RefineTargetAssigner(object):
    def __init__(self, assigner_config):

        # some compositions
        self.similarity_calc = similarity_calc_builder.build(
            assigner_config['similarity_calc_config'])
        self.bbox_coder = bbox_coder_builder.build(
            assigner_config['coder_config'])
        self.matcher = matcher_builder.build(assigner_config['matcher_config'])
        self.analyzer = Analyzer()

        self.fg_thresh = assigner_config['fg_thresh']
        self.bg_thresh = assigner_config['bg_thresh']
        # self.clobber_positives = assigner_config['clobber_positives']

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
        # import ipdb
        # ipdb.set_trace()

        # usually IoU overlaps is used as metric
        bboxes = bboxes.detach()
        match_quality_matrix = self.similarity_calc.compare_batch(bboxes,
                                                                  gt_boxes)

        # match
        # shape(N,K)
        match = self.matcher.match_batch(match_quality_matrix, self.fg_thresh)

        self.analyzer.analyze(match, gt_boxes.shape[1])

        # get assigned infomation
        # shape (num_batch,num_boxes)
        assigned_overlaps_batch = self.matcher.assigned_overlaps_batch

        # assign regression targets
        reg_targets = self._assign_regression_targets(match, bboxes, gt_boxes)

        # assign classification targets
        #  cls_targets = self._assign_classification_targets(match, gt_labels,
        #  assigned_overlaps_batch)
        cls_targets = assigned_overlaps_batch.clone()

        # create regression weights
        reg_weights = self._create_regression_weights(assigned_overlaps_batch)

        # create classification weights
        cls_weights = self._create_classification_weights(
            assigned_overlaps_batch)

        ####################################
        # postprocess
        ####################################
        # match == -1 means unmatched
        reg_targets[match == -1] = 0
        cls_targets[match == -1] = 0
        reg_weights[match == -1] = 0

        # as for cls weights, ignore according to bg_thresh
        if self.bg_thresh > 0:
            ignored_bg = (assigned_overlaps_batch > self.bg_thresh) & (
                match == -1)
            cls_weights[ignored_bg] = 0

        return cls_targets, reg_targets, cls_weights, reg_weights

    def _create_regression_weights(self, assigned_overlaps_batch):
        """
        Args:
        assigned_overlaps_batch: shape (num_batch,num_boxes)
        Returns:
        reg_weights: shape(num_batch,num_boxes,4)
        """
        #  gamma = 2
        #  return torch.pow(1 - assigned_overlaps_batch, gamma).detach()
        #  return torch.ones_like(assigned_overlaps_batch
        return assigned_overlaps_batch.clone()

    def _create_classification_weights(self, assigned_overlaps_batch):
        """
        All samples can be used for calculating loss,So reserve all.
        """
        #  cls_weights = torch.ones_like(assigned_overlaps_batch)
        #  return cls_weights
        return assigned_overlaps_batch.clone()

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

    def _assign_classification_targets(self, match, gt_labels,
                                       match_quality_matrix):
        """
        Just return the countpart labels
        Note that use zero to represent background labels
        For the first stage, generate binary labels, For the second stage
        generate countpart gt_labels
        """
        # binary labels classifcation
        # if gt_labels is None:
        # consider it as binary classification problem
        return self._generate_binary_labels(match, match_quality_matrix)

        # multiple labels classification
        # TODO(as for multiple cls ,match_quality_matrix should also be
        # considered)
        # batch_size = match.shape[0]
        # offset = torch.arange(0, batch_size) * gt_labels.size(1)
        # match += offset.view(batch_size, 1).type_as(match)
        # cls_targets_batch = gt_labels.view(-1)[match.view(-1)].view(

    # batch_size, match.shape[1])
    # return cls_targets_batch

    def _generate_binary_labels(self, match, match_quality_matrix):
        """
        Select cls_target from matrix according to match
        Args:
            match: shape(N,K)
            match_quality_matrix: shape(N,K,M)
        """
        num = match.numel()
        row = torch.arange(0, num).type_as(match)
        M = match_quality_matrix.shape[2]
        cls_targets_batch = match_quality_matrix.view(
            -1, M)[row, match.view(-1)].view_as(match)
        return cls_targets_batch
