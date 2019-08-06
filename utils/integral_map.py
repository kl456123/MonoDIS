# -*- coding: utf-8 -*-

import torch


class IntegralMapGenerator(object):
    @staticmethod
    def generate(input_map):
        """
        Args:
            input_map: shape(NCHW)
        """
        integral_map = input_map.clone()

        integral_map = integral_map.cumsum(dim=-1)
        integral_map = integral_map.cumsum(dim=-2)

        return integral_map

    @staticmethod
    def calc(integral_map, bbox_2d, min_area=1):
        """
        Args:
            bbox_2d: shape(N, 4)
        """
        # import ipdb
        # ipdb.set_trace()
        F = integral_map
        # be sure integral number first
        # used_coords_filter = (bbox_2d[:, 0] >= 0) & (bbox_2d[:, 1] >= 0) & (
            # bbox_2d[:, 2] <= 1) & (bbox_2d[:, 3] <= 1)
        # import ipdb
        # ipdb.set_trace()
        # unused_coords_filter = torch.nonzero(~used_coords_filter)

        bbox_2d[:, ::2].clamp_(min=0, max=1)
        bbox_2d[:, 1::2].clamp_(min=0, max=1)

        bbox_2d[:, ::2] = bbox_2d[:, ::2] * (F.shape[3] - 1)
        bbox_2d[:, 1::2] = bbox_2d[:, 1::2] * (F.shape[2] - 1)

        bbox_2d = torch.round(bbox_2d).long()

        xmin = bbox_2d[:, 0]
        ymin = bbox_2d[:, 1]
        xmax = bbox_2d[:, 2]
        ymax = bbox_2d[:, 3]
        area = (ymax - ymin) * (xmax - xmin)

        # import ipdb
        # ipdb.set_trace()

        # used_area_filter = area >= min_area
        unused_area_filter = area < min_area
        area[unused_area_filter] = 1
        inds_filter = torch.nonzero(unused_area_filter)[:, 0]
        # # F[:, :, inds_filter, inds_filter] = 0

        # import ipdb
        # ipdb.set_trace()
        # F = F.permute(2, 3, 0, 1).contiguous()
        # res = (F[ymin, xmin, :, :] + F[ymax, xmax, :, :] - F[ymin, xmax, :, :]
        # - F[ymax, xmin, :, :])
        # res = res.permute(1, 2, 0).contiguous()

        # # filter
        # xmin = xmin[used_area_filter]
        # ymin = ymin[used_area_filter]
        # xmax = xmax[used_area_filter]
        # ymax = ymax[used_area_filter]

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        res = (F[:, :, ymin, xmin] + F[:, :, ymax, xmax] - F[:, :, ymin, xmax]
               - F[:, :, ymax, xmin]) / area.type_as(F)
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        # res = F[:, :, ymin, xmin]
        mask = torch.ones_like(res)
        mask[:, :, inds_filter] = 0
        # mask[:, :, unused_coords_filter] = 0

        res = res * mask

        return res
