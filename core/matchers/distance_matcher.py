#!/usr/bin/env python
# encoding: utf-8

from core.matcher import Matcher
import torch

INF = 1e5


class ScaleMatcher(Matcher):
    def __init__(self, matcher_config):
        super().__init__()

    def match(self, match_quality_matrix, thresh):
        """
        match each bbox with gts
        if center of bbox is in the inner of gt box,
        bbox matchs with gt box

        note that all gt boxes can be matched if this method is used

        Args:
            match_quality_matrix: shape(N,M)
        Returns:
            match: shape(N)
        """

        max_quality, argmax_quality = torch.max(match_quality_matrix, dim=-1)
        argmax_quality[max_quality <= thresh] = -1

        self._assigned_overlaps = max_quality
        return argmax_quality
