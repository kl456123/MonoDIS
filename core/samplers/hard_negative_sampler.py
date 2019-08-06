# -*- coding: utf-8 -*-

from core.sampler import Sampler
import torch


class HardNegativeSampler(Sampler):
    def __init__(self, sampler_config):
        super().__init__(sampler_config)
        """
        Note that the scores here is the critation of samples,it can be
        confidence or IoU e,c
        """

    # def subsample(self,
    # num_samples,
    # pos_indicator,
    # criterion=None,
    # indicator=None):
    # sorted_loss, order = torch.sort(criterion, descending=True)
    # keep = order[:num_samples]
    # sample_mask = torch.zeros_like(pos_indicator)
    # sample_mask[keep] = 1
    # return sample_mask

    def subsample(self,
                  num_samples,
                  pos_indicator,
                  criterion=None,
                  indicator=None):
        fg_num_for_sample = int(self.fg_fraction * num_samples)
        pos_indicator = indicator & pos_indicator
        fg_inds = torch.nonzero(pos_indicator).view(-1)
        if fg_inds.numel() > fg_num_for_sample:
            sorted_scores, order = torch.sort(criterion, descending=False)
            fg_order = order[pos_indicator[order]]
            fg_inds = fg_order[:fg_num_for_sample]

        # the remain is bg
        bg_num_for_sample = num_samples - fg_inds.numel()
        neg_indicator = indicator & ~pos_indicator
        bg_inds = torch.nonzero(neg_indicator).view(-1)

        if bg_inds.numel() > bg_num_for_sample:
            sorted_scores, order = torch.sort(criterion, descending=True)
            bg_order = order[neg_indicator[order]]
            bg_inds = bg_order[:bg_num_for_sample]

        # if not enough samples,oversample from fg
        if fg_inds.numel() + bg_inds.numel() < num_samples:
            raise ValueError(
                "can not subsample enough samples,please check it!")

        keep = torch.cat([fg_inds, bg_inds], dim=0)

        sample_mask = torch.zeros_like(pos_indicator)
        sample_mask[keep] = 1

        return sample_mask
