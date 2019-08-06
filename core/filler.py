# -*- coding: utf-8 -*-


class Filler(object):
    @staticmethod
    def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            if m.bias is not None:
                m.bias.data.zero_()
