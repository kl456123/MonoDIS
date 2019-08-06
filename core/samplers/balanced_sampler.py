# -*- coding: utf-8 -*-

import torch
import numpy as np
from core.sampler import Sampler


class BalancedSampler(Sampler):
    def __init__(self, sampler_config):
        super().__init__(sampler_config)
        """
        Note that the scores here is the critation of samples,it can be
        confidence or IoU e,c
        """
        # self.fg_fraction = sampler_config['fg_fraction']

    def subsample(self,
                  num_samples,
                  pos_indicator,
                  criterion=None,
                  indicator=None):
        """
        Args:
            num_samples
        """
        num_fg = int(num_samples * self.fg_fraction)
        pos_indicator = indicator & pos_indicator
        fg_inds = torch.nonzero(pos_indicator).view(-1)
        sum_fg = fg_inds.numel()
        if sum_fg > num_fg:
            rand_num = torch.from_numpy(
                np.random.permutation(fg_inds.size(0))).type_as(fg_inds).long()
            fg_inds = fg_inds[rand_num[:num_fg]]

        num_bg = num_samples - fg_inds.numel()

        # subsample negative labels if we have too many
        neg_indicator = indicator & ~pos_indicator
        bg_inds = torch.nonzero(neg_indicator).view(-1)
        sum_bg = bg_inds.numel()
        if sum_bg > num_bg:
            rand_num = torch.from_numpy(
                np.random.permutation(bg_inds.size(0))).type_as(bg_inds).long()
            bg_inds = bg_inds[rand_num[:num_bg]]

        keep = torch.cat([fg_inds, bg_inds], dim=0)

        sample_mask = torch.zeros_like(pos_indicator)
        sample_mask[keep] = 1

        return sample_mask
