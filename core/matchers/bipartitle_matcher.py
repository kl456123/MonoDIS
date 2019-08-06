#!/usr/bin/env python
# encoding: utf-8

from core.matcher import Matcher
import torch


class BipartitleMatcher(Matcher):
    def __init__(self, matcher_config):
        super().__init__()

        # self.pos_thresh = matcher_config['fg_thresh']
        # self.neg_thresh = matcher_config['bg_thresh']
        # self.thresh = matcher_config['thresh']
        # self.clobber_positives = matcher_config['clobber_positives']

    def match(self, match_quality_matrix, thresh):
        """
        Match each bboxes with gt_boxes,if no any gt boxes match
        with bbox, assign match idx of it with -1.
        And make sure all gt boxes can be matched

        1. calculate overlaps
        2. each anchors match gt_boxes which has max overlaps with
        3. do the same thing for gt_boxes

        Args:
            match_quality_matrix: usually overlaps is used
        """

        #################################
        # match all bboxes
        ################################
        # shape(N,M)
        # overlaps = bbox_overlaps(bboxes, gt_boxes)
        overlaps = match_quality_matrix

        # shape(N,1)
        max_overlaps, argmax_overlaps = torch.max(overlaps, dim=1)
        argmax_overlaps[max_overlaps <= thresh] = -1
        # matched_gt = torch.ones(overlaps.shape[1])
        # matched_gt[argmax_overlaps[max_overlaps<thresh]] = 0

        # shape(1,M)
        gt_max_overlaps, argmax_gt_overlaps = torch.max(overlaps, dim=0)
        # just protect those have matched already
        gt_max_overlaps[argmax_overlaps[max_overlaps >= thresh]] = 0

        ##################################
        # make sure all gt has been matched
        #################################
        assignments_overlaps = torch.zeros_like(overlaps)

        # filter no overlaps
        row_inds = argmax_gt_overlaps[gt_max_overlaps > 0].view(-1)
        # col_inds = torch.arange(match_quality_matrix.shape[1])[
        # gt_max_overlaps.view(-1) > 0].type_as(row_inds)
        col_inds = torch.nonzero(gt_max_overlaps.view(-1) > 0).view(-1)
        gt_max_overlaps = gt_max_overlaps[gt_max_overlaps > 0].view(-1)

        assignments_overlaps[row_inds, col_inds] = gt_max_overlaps

        # shape(N,1)
        max_assignments_overlaps, gt_assignments = torch.max(
            assignments_overlaps, dim=1)
        gt_assignments = gt_assignments[max_assignments_overlaps > 0]

        ##########################
        # final assignment results
        ##########################
        # shape(N,1)
        # copy from bboxes assignments result
        assignments = torch.tensor(argmax_overlaps)

        # combined with gt assignments
        assignments[max_assignments_overlaps > 0] = gt_assignments

        # shape(N,)
        assigned_overlaps = overlaps[torch.arange(assignments.numel()).type_as(
            assignments), assignments]
        assigned_overlaps[max_overlaps < thresh] = max_overlaps[max_overlaps <
                                                                thresh]
        # shape of both of them is (N,)

        self._assigned_overlaps = assigned_overlaps

        return assignments.view(-1)
