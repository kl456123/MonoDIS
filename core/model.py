# -*- coding: utf-8 -*-

import torch.nn as nn


class Model(nn.Module):
    featmaps_dict = {}

    def __init__(self, model_config):
        super(Model, self).__init__()
        # store model_config
        self.init_param(model_config)

        # use config to set up model
        self.init_modules()

        # init weights
        self.init_weights()

        # freeze modules
        #  self.freeze_modules()

    def add_feat(self, key, value):
        self.featmaps_dict[key] = value

    def get_feat(self, key=None):
        if key is None:
            return self.featmaps_dict
        return self.featmaps_dict[key]

    def loss(self):
        pass

    def init_weights(self):
        pass

    def init_modules(self):
        pass

    def init_param(self, model_config):
        pass

    def pre_forward(self):
        pass

    def freeze_modules(self):
        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def freeze_bn(module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def unfreeze_bn(module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

    def unloaded_parameters(self):
        return []
