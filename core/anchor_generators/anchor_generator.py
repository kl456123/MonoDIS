# -*- coding: utf-8 -*-

import torch

import core.ops as ops


class AnchorGenerator(object):
    def __init__(self, anchor_generator_config):
        """
        """
        self.base_anchor_size = torch.tensor(
            anchor_generator_config['base_anchor_size'],
            dtype=torch.float32).cuda()
        self.scales = torch.tensor(
            anchor_generator_config['scales'], dtype=torch.float32).cuda()
        self.aspect_ratios = torch.tensor(
            anchor_generator_config['aspect_ratios'],
            dtype=torch.float32).cuda()
        # self.anchor_stride = torch.tensor(
        # anchor_generator_config['anchor_stride'],
        # dtype=torch.float32).cuda()

        self.anchor_offset = torch.tensor(
            anchor_generator_config['anchor_offset'],
            dtype=torch.float32).cuda()

        if anchor_generator_config.get('use_pyramid'):
            self.num_anchors = self.aspect_ratios.numel()
        else:
            self.num_anchors = self.aspect_ratios.numel() * self.scales.numel()

        # self.input_size = anchor_generator_config['input_size']

    def generate_pyramid(self, feature_map_list, im_shape):
        anchors_list = []
        scales = self.scales.view(len(self.scales), -1).expand(
            (-1, len(self.aspect_ratios)))

        for stage, feature_map_shape in enumerate(feature_map_list):
            anchors_list.append(
                self._generate(feature_map_shape, scales[stage],
                               self.aspect_ratios, im_shape))

        return torch.cat(anchors_list, dim=0)

    def generate(self, feature_map_list, im_shape):
        """
        Args:
            feature_map_list, list of (stride, ratio)
        Returns:
            anchors
        """
        anchors_list = []
        scales, aspect_ratios = ops.meshgrid(self.scales, self.aspect_ratios)
        for feature_map_shape in feature_map_list:
            anchors_list.append(
                self._generate(feature_map_shape, scales, aspect_ratios,
                               im_shape))

        return torch.cat(anchors_list, dim=0)

    def _generate(self, feature_map_shape, scales, aspect_ratios, im_shape):
        """
        """
        # shape(A,)

        ratios_sqrt = torch.sqrt(aspect_ratios)
        heights = scales * ratios_sqrt * self.base_anchor_size
        widths = scales / ratios_sqrt * self.base_anchor_size
        anchor_stride = [
            im_shape[0] / feature_map_shape[0],
            im_shape[1] / feature_map_shape[1]
        ]

        y_ctrs = torch.arange(feature_map_shape[0]).cuda() * anchor_stride[
            0] + self.anchor_offset[0]
        x_ctrs = torch.arange(feature_map_shape[1]).cuda() * anchor_stride[
            1] + self.anchor_offset[1]

        x_ctrs = x_ctrs.float()
        y_ctrs = y_ctrs.float()

        # meshgrid
        # shape(H*W,)
        x_ctrs, y_ctrs = ops.meshgrid(x_ctrs, y_ctrs)

        # shape(K*A,)
        heights, x_ctrs = ops.meshgrid(heights, x_ctrs)
        widths, y_ctrs = ops.meshgrid(widths, y_ctrs)

        xmin = x_ctrs - 0.5 * (widths - 1)
        ymin = y_ctrs - 0.5 * (heights - 1)
        xmax = x_ctrs + 0.5 * (widths - 1)
        ymax = y_ctrs + 0.5 * (heights - 1)

        # (x,y,h,w)
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)
