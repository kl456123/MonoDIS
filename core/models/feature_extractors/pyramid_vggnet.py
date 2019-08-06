# -*- coding: utf-8 -*-

from core.model import Model
import torch.nn as nn
import torch.nn.functional as F
from core.layers.l2_norm import L2Norm
import torch


class PyramidVggnetExtractor(Model):
    def forward(self, x):
        # import ipdb
        # ipdb.set_trace()
        source_layers = list()

        # extract source layers used for prediction
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base_feat[k](x)

        s = self.l2_norm(x)
        source_layers.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base_feat)):
            x = self.base_feat[k](x)
        source_layers.append(x)

        # extras layers
        for k, v in enumerate(self.extras_layers):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                source_layers.append(x)
        return source_layers

    def init_param(self, model_config):
        self.base_cfg = [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512
        ]
        self.extras_cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
        self.input_channels = model_config['img_channels']
        self.din = model_config['din']
        self.pretrained_path = model_config['pretrained_model']
        self.pretrained = model_config['pretrained']

    def init_modules(self):
        self.base_feat = self.make_base()
        self.extras_layers = self.make_extras()
        self.l2_norm = L2Norm(512, 20)
        # loc_layers, conf_layers = self.make_multibox(base_feat, extras_layers)

        # make list be modules
        # self.base_feat = nn.Sequential(*base_feat)
        # self.extras_layers = nn.Sequential(*extras_layers)
        # self.loc_layers = nn.Sequential(*loc_layers)
        # self.conf_layers = nn.Sequential(*conf_layers)

    def make_base(self, batch_norm=False):
        cfg = self.base_cfg
        i = self.input_channels
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [
                    nn.MaxPool2d(
                        kernel_size=2, stride=2, ceil_mode=True)
                ]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [
                        conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [
            pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)
        ]
        return nn.ModuleList(layers)

    def make_extras(self):
        # Extra layers added to VGG for feature scaling
        cfg = self.extras_cfg
        i = self.din
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [
                        nn.Conv2d(
                            in_channels,
                            cfg[k + 1],
                            kernel_size=(1, 3)[flag],
                            stride=2,
                            padding=1)
                    ]
                else:
                    layers += [
                        nn.Conv2d(
                            in_channels, v, kernel_size=(1, 3)[flag])
                    ]
                flag = not flag
            in_channels = v
        return nn.ModuleList(layers)

    def init_weights(self):
        if self.pretrained:
            pretrained_model = torch.load(self.pretrained_path)
            self.base_feat.load_state_dict(pretrained_model)
