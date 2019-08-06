# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch


class Matcher(ABC):
    def __init__(self):
        # information is generated during matching
        self._assigned_overlaps = None
        self._assigned_overlaps_batch = None

    @property
    def assignments(self):
        return self._assignments

    @property
    def assigned_overlaps_batch(self):
        return self._assigned_overlaps_batch

    @property
    def assigned_iod_batch(self):
        return self._assigned_iod_batch

    @property
    def assigned_iog_batch(self):
        return self._assigned_iog_batch

    @abstractmethod
    def match(self, match_quality_matrix, thresh):
        pass

    def match_batch(self, match_quality_matrix_batch, thresh):
        """
        batch version of assign function
        Args:
            match_quality_matrix_batch: shape(N,num_boxes,num_gts)
        """
        batch_size = match_quality_matrix_batch.shape[0]
        assignments = []
        overlaps = []
        iog = []
        iod = []

        for i in range(batch_size):
            # shape(K)
            if hasattr(self, 'iog_match_quality_matrix_batch'):
                self.iog_match_quality_matrix = self.iog_match_quality_matrix_batch[
                    i]
            if hasattr(self, 'iod_match_quality_matrix_batch'):
                self.iod_match_quality_matrix = self.iod_match_quality_matrix_batch[
                    i]
            assignments_per_img = self.match(match_quality_matrix_batch[i],
                                             thresh)
            assignments.append(assignments_per_img)
            overlaps.append(self._assigned_overlaps)
            if hasattr(self, '_assigned_iog'):
                iog.append(self._assigned_iog)
            if hasattr(self, '_assigned_iod'):
                iod.append(self._assigned_iod)

        # shape(N,num_boxes)
        assignments = torch.stack(assignments)
        # shape(N,num_boxes)
        overlaps = torch.stack(overlaps)
        self._assigned_overlaps_batch = overlaps
        if len(iog):
            self._assigned_iog_batch = torch.stack(iog)
        if len(iod):
            self._assigned_iod_batch = torch.stack(iod)

        self._assignments = assignments
        return assignments
