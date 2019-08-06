# -*- coding: utf-8 -*-

import torch


class Analyzer(object):
    def __init__(self):
        # init value
        self.stat = {
            'num_det': 1,
            'num_tp': 0,
            'matched_thresh': 0,
            'recall_thresh': 0,

            # stats for angles
            'angle_num_tp': torch.tensor(0),
            'angle_num_all': 1,

            # stats of orient
            'orient_tp_num': 0,
            'orient_tp_num2': 0,
            'orient_tp_num3': 0,
            'orient_all_num3': 0,
            # 'orient_pr': orient_pr,
            'orient_all_num': 0,
            'orient_tp_num4': 0,
            'orient_all_num4': 0,
            'cls_orient_2s_all_num': 0,
            'cls_orient_2s_tp_num': 0
        }
        self.match = None
        self.append_gt = True

    def analyze(self, match, num_gt):
        """
        analyze result from match,calculate AP and AR
        note that -1 means it is not matched
        Args:
        match: tensor(N,M)

        """
        self.match = match
        assert match.shape[0] == 1
        match = match[0]
        match = match[match > -1]
        gt_mask = torch.zeros(num_gt).type_as(match)
        if self.append_gt:
            gt_mask[match[:-num_gt]] = 1
        else:
            gt_mask[match] = 1
        matched = gt_mask.sum().item()
        recall = matched / num_gt
        self.stat.update({
            'matched': matched,
            'num_gt': num_gt,
            'recall': recall
        })
        # return self.stat
        # print('matched_gt/all_gt/average recall({}/{}/{}): '.format(

    # matched, num_gt, recall))
    # num_all_samples = match.numel()
    # num_unmatched_samples = torch.nonzero(

    # match[match == -1]).view(-1).numel()
    # num_matched_samples = num_all_samples - num_unmatched_samples
    # print('match rate: ', num_matched_samples / num_all_samples)

    def analyze_ap(self,
                   match,
                   rcnn_cls_probs,
                   num_gt,
                   thresh=0.5):
        # import ipdb
        # ipdb.set_trace()
        if match.dim() == 2:
            assert match.shape[0] == 1
            match = match[0]
        if rcnn_cls_probs.dim() == 2:
            assert rcnn_cls_probs.shape[0] == 1
            rcnn_cls_probs = rcnn_cls_probs[0]
        num_det = torch.nonzero(rcnn_cls_probs > thresh).numel()
        num_tp = torch.nonzero((rcnn_cls_probs > thresh) & (match > -1
                                                            )).numel()

        match_thresh = match[(rcnn_cls_probs > thresh) & (match > -1)]

        gt_mask = torch.zeros(num_gt).type_as(match_thresh)
        if self.append_gt:
            gt_mask[match_thresh[:-num_gt]] = 1
        else:
            gt_mask[match_thresh] = 1
        matched_thresh = gt_mask.sum().item()
        recall_thresh = matched_thresh / num_gt
        # if num_det:
        # precision = num_tp/num_det
        # else:
        # precision = 0
        self.stat.update({
            'num_tp': num_tp,
            'num_det': num_det,

            # recall after thresh
            'matched_thresh': matched_thresh,
            'recall_thresh': recall_thresh
        })
