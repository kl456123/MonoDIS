# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch


class KeyPointCoder(object):
    def __init__(self, coder_config):
        self.heatmap_size = coder_config['size']

    def encode_keypoint_heatmap(self, bboxes, keypoints):
        """
        Note that it is not necessary to convert maps to spatial format
        Args:
            keypoints: shape(N,K,2), here N refers to num of instances, K refers to num of joints
            rois: shape(N, 4)
        Returns:
            keypoint_heatmap: shape(N, K, m*m)
        """
        # import ipdb
        # ipdb.set_trace()
        spatial_dims = self.heatmap_size[0] * self.heatmap_size[1]
        heatmaps_shape = (keypoints.shape[0], keypoints.shape[1], spatial_dims)
        heatmaps = torch.zeros(heatmaps_shape).type_as(bboxes)

        peak_pos = self._calculate_peak_pos(bboxes, keypoints)
        # filter inside of the window
        inside_filter = (peak_pos[:, :, 0] < self.heatmap_size[1]) & (
            peak_pos[:, :, 1] < self.heatmap_size[0]) & (
                peak_pos[:, :, 0] >= 0) & (peak_pos[:, :, 1] >= 0)

        # assign peak to heatmaps
        heatmaps = heatmaps.view(-1, spatial_dims)
        peak_pos = peak_pos.view(-1, 2)
        inside_filter = inside_filter.view(-1)

        # import ipdb
        # ipdb.set_trace()
        peak_offsets = peak_pos[:, 1] * self.heatmap_size[1] + peak_pos[:, 0]
        peak_weights = torch.zeros_like(peak_offsets).type_as(peak_offsets)
        peak_weights[inside_filter] = 1
        peak = torch.stack([peak_offsets, peak_weights], dim=-1)
        peak = peak.view(-1, 4 * 2).float()
        return peak
        #  row_inds = torch.arange(0, heatmaps.shape[0]).type_as(peak_offsets)

        #  heatmaps[row_inds[inside_filter], peak_offsets[inside_filter]] = 1.0
        #  heatmaps = heatmaps.view(heatmaps_shape)

        #  return heatmaps

    def _calculate_peak_pos(self, bboxes, keypoints):
        bboxes = bboxes.unsqueeze(1)
        # convert to (w,h)
        heatmap_size = (self.heatmap_size[1], self.heatmap_size[0])
        heatmap_size = torch.tensor(heatmap_size).type_as(bboxes)

        bboxes_w = bboxes[:, :, 2] - bboxes[:, :, 0]
        bboxes_h = bboxes[:, :, 3] - bboxes[:, :, 1]
        # note that (w,h) here
        bboxes_dim = torch.stack([bboxes_w, bboxes_h], dim=-1)

        # shape(N,K,2)
        peak_pos_norm = (keypoints - bboxes[:, :, :2]) / bboxes_dim
        peak_pos = (peak_pos_norm * heatmap_size).long()
        return peak_pos

    def decode_keypoint_heatmap(self,
                                bboxes,
                                keypoint_heatmap,
                                pixel_offsets=0.5):
        """
        Args:
            rois: shape(N,4)
            keypoint_heatmap: shape(N,K,m*m)
        Returns:
            keypoints: shape(N,K,2)
        """
        _, peak_pos = keypoint_heatmap.max(dim=-1)
        peak_pos_y = peak_pos / self.heatmap_size[1]
        peak_pos_x = peak_pos % self.heatmap_size[1]
        peak_pos = torch.stack([peak_pos_x, peak_pos_y], dim=-1).float()

        heatmap_size = (self.heatmap_size[1], self.heatmap_size[0])
        heatmap_size = torch.tensor(heatmap_size).type_as(peak_pos)
        peak_pos_norm = (peak_pos + pixel_offsets) / heatmap_size

        bboxes = bboxes.unsqueeze(1)
        bboxes_w = bboxes[:, :, 2] - bboxes[:, :, 0]
        bboxes_h = bboxes[:, :, 3] - bboxes[:, :, 1]
        # note that (w,h) here
        bboxes_dim = torch.stack([bboxes_w, bboxes_h], dim=-1)

        offsets = peak_pos_norm * bboxes_dim

        keypoints = offsets + bboxes[:, :, :2]
        return keypoints
