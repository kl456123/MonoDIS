# -*- coding: utf-8 -*-

from data.datasets.kitti_mono_3d import Mono3DKittiDataset

from builder.dataloader_builder import DataLoaderBuilder
import data.transforms.kitti_transform as trans


class Mono3DKittiDataLoaderBuilder(DataLoaderBuilder):
    def build_dataset(self):
        """
        dataset_config, tranform_config and transform can be used
        """
        self.dataset = Mono3DKittiDataset(self.dataset_config, self.transform,
                                          self.training)
        tmp = self.dataset[0]
        return self.dataset

    def build_transform(self):
        """
        tranform_config can be used
        """
        trans_cfg = self.tranform_config
        if self.training:
            self.transform = trans.Compose([
                trans.RandomHorizontalFlip(),
                # trans.RandomSampleCrop(trans_cfg['resize_range'][0],
                                       # trans_cfg['resize_range'][1]),
                trans.Resize(trans_cfg['crop_size']),
                trans.RandomHSV(),
                trans.Boxes3DTo2D(),
                trans.ToTensor(),
                trans.Normalize(trans_cfg['normal_mean'],
                                trans_cfg['normal_van'])
            ])
        else:
            self.transform = trans.Compose([
                trans.Resize(trans_cfg['crop_size']),
                trans.ToTensor(), trans.Normalize(trans_cfg['normal_mean'],
                                                  trans_cfg['normal_van'])
            ])
        return self.transform
