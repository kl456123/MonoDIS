# -*- coding: utf-8 -*-

from data.datasets.kitti_bev import KITTIBEVDataset

from builder.dataloader_builder import DataLoaderBuilder
import data.transforms.kitti_transform as trans


class KITTIBEVDataLoaderBuilder(DataLoaderBuilder):
    def build_dataset(self):
        """
        dataset_config, tranform_config and transform can be used
        """
        self.dataset = KITTIBEVDataset(self.dataset_config, self.transform)
        return self.dataset

    def build_transform(self):
        """
        tranform_config can be used
        """
        if self.training:
            all_trans = [trans.BEVRandomHorizontalFlip(), trans.BEVToTensor()]
        else:
            all_trans = [trans.BEVToTensor()]

        self.transform = trans.Compose(all_trans)
        return self.transform
