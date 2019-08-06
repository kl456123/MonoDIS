#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn.functional as F

# core classes

from core.analyzer import Analyzer

# builder
from builder import matcher_builder
from builder import bbox_coder_builder
from builder import similarity_calc_builder

from core.similarity_calc.scale_similarity_calc import ScaleSimilarityCalc
from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc


class ThreshDistanceTargetAssigner(object):
    def __init__(self, assigner_config):

        # some compositions
        self.reg_similarity_calc = similarity_calc_builder.build(
            assigner_config['similarity_calc_config'])

        # make sure match boxes when they are enough close for classification
        # self.cls_similarity_calc = CenterSimilarityCalc()
        # make sure more bbox can be optimized
        # self.reg_similarity_calc = ScaleSimilarityCalc()

        self.bbox_coder = bbox_coder_builder.build(
            assigner_config['coder_config'])
        self.matcher = matcher_builder.build(assigner_config['matcher_config'])
        self.analyzer = Analyzer()

        self.fg_thresh = assigner_config['fg_thresh']
        assert self.fg_thresh <= 0.25
        self.bg_thresh = assigner_config['bg_thresh']
        assert self.bg_thresh > 0.25
        # self.clobber_positives = assigner_config['clobber_positives']

    @property
    def stat(self):
        return self.analyzer.stat

    def assign(self, bboxes, gt_boxes, gt_labels=None, cls_prob=None):
        """
        match policy depends on  reg_match_matrix
        cls targets depends on cls_match_quality_matrix
        Args:
        bboxes: shape(N,K,4), encoded by xxyy
        gt_boxes: shape(N,M,4), encoded likes as bboxes
        """
        # import ipdb
        # ipdb.set_trace()

        # usually IoU overlaps is used as metric
        bboxes = bboxes.detach()
        # just use IoU here
        # shape(N,K,M)
        # cls_match_quality_matrix = self.cls_similarity_calc.compare_batch(
        # bboxes, gt_boxes)
        distance_matrix = self.reg_similarity_calc.compare_batch(bboxes,
                                                                 gt_boxes)
        assigned_distance, argmax_distance = torch.max(distance_matrix, dim=-1)

        reg_targets = self._assign_regression_targets(argmax_distance, bboxes,
                                                      gt_boxes)
        criterion = reg_targets * reg_targets
        # delta_x and delta_y
        criterion = criterion[:, :, :2]
        unmatched_mask = (criterion[:, :, 0] >= self.fg_thresh) | (
            criterion[:, :, 1] >= self.fg_thresh)
        argmax_distance[unmatched_mask] = -1
        match = argmax_distance

        # shape(N,K)
        # match = self.matcher.match_batch(reg_match_quality_matrix,
        # self.fg_thresh)

        # some statistics about result of match
        # self.analyzer.analyze(match, gt_boxes.shape[1])

        # get assigned infomation
        # shape (num_batch,num_boxes)
        # assigned_overlaps_batch = self.matcher.assigned_overlaps_batch

        # num = match.numel()
        # M = distance_matrix.shape[2]
        # row = torch.arange(0, num).type_as(match)
        # assigned_distance = distance.view(
        # -1, M)[row, match.view(-1)].view_as(match)

        #######################
        # reg
        #######################

        # create regression weights
        reg_weights = self._create_regression_weights(assigned_distance)

        #######################
        # cls
        #######################
        # create classification weights
        cls_weights = self._create_classification_weights(assigned_distance)

        # assign classification targets
        cls_targets = self._assign_classification_targets(match, gt_labels)

        ####################################
        # postprocess
        ####################################
        # match == -1 means unmatched
        reg_targets[match == -1] = 0
        cls_targets[match == -1] = 0
        reg_weights[match == -1] = 0

        # as for cls weights, ignore according to bg_thresh
        ignored_bg = (assigned_distance < self.bg_thresh) & (match == -1)
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
        #  return torch.ones_like(assigned_overlaps_batch)
        # return 10 * (F.sigmoid(assigned_overlaps_batch) - 0.5)
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
        cls_targets_batch = torch.ones_like(match)
        cls_targets_batch[match == -1] = 0
        return cls_targets_batch
