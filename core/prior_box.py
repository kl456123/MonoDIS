from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # number of priors for feature map location (either 4 or 6)
        self.input_size = cfg['input_size']
        self.num_priors = len(cfg['aspect_ratios'])
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

    def forward(self, feature_maps):
        mean = []
        # import ipdb
        # ipdb.set_trace()
        for k, f in enumerate(feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                # unit center x,y
                x_stride = self.input_size[1] / f[1]
                y_stride = self.input_size[0] / f[0]
                cx = (j + 0.5) * x_stride
                cy = (i + 0.5) * y_stride

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios:
                    # mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        mean = torch.Tensor(mean).view(-1, 4)

        # xywh convert xyxy
        xmin = mean[:, 0] - 0.5 * mean[:, 2]
        xmax = mean[:, 0] + 0.5 * mean[:, 2]
        ymin = mean[:, 1] - 0.5 * mean[:, 3]
        ymax = mean[:, 1] + 0.5 * mean[:, 3]
        output = torch.stack([xmin, ymin, xmax, ymax], dim=-1)

        if self.clip:
            output[:, ::2].clamp_(max=self.input_size[1], min=0)
            output[:, 1::2].clamp_(max=self.input_size[0], min=0)
        return output.cuda()
