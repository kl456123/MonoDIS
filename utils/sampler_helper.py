# -*- coding: utf-8 -*-

from core.samplers.hard_negative_sampler import HardNegativeSampler


def subsample(overlaps, samples_mask, sampler_config, scores=None):
    """
    Here samples_mask indicate one sample can be used or not
    """
    if scores is not None:
        critation = scores
    else:
        critation = overlaps
    sampler = HardNegativeSampler(sampler_config)
    thresh = sampler_config['thresh']
    pos_mask = samples_mask and (overlaps > thresh)
    sampler.subsample(samples_mask, pos_mask, critation)


if __name__ == '__main__':
    sampler_config = {'fg_fraction': 0.3, 'num_samples': 100}
