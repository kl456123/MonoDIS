# -*- coding: utf-8 -*-

import torch

from core.matcher import Matcher


class ArgmaxMatcher(Matcher):
    def __init__(self, matcher_config):
        super().__init__()
        # self.thresh = matcher_config['thresh']

    def match(self, match_quality_matrix, thresh):
        """
        For each sample box,find the gt idx that has max overlaps with it,
        means matched
        """

        max_overlaps, argmax_overlaps = torch.max(match_quality_matrix, dim=1)
        argmax_overlaps[max_overlaps <= thresh] = -1

        self._assigned_overlaps = max_overlaps
        return argmax_overlaps
