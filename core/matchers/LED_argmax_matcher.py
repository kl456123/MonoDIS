# -*- coding: utf-8 -*-

import torch

from core.matcher import Matcher


class LEDArgmaxMatcher(Matcher):
    def __init__(self, matcher_config):
        super().__init__()
        # self.thresh = matcher_config['thresh']

    def match(self, match_quality_matrix, thresh):
        """
        For each sample box,find the gt idx that has max overlaps with it,
        means matched
        """

        max_overlaps, argmax_overlaps = torch.max(match_quality_matrix, dim=1)
        idx = torch.arange(
            argmax_overlaps.numel()).type_as(argmax_overlaps), argmax_overlaps
        assigned_iog = self.iog_match_quality_matrix[idx]

        assigned_iod = self.iod_match_quality_matrix[idx]
        argmax_overlaps[max_overlaps < thresh] = -1

        # shape of both of them is (N,)

        self._assigned_overlaps = max_overlaps
        self._assigned_iod = assigned_iod
        self._assigned_iog = assigned_iog
        return argmax_overlaps
