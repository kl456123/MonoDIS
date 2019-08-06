# -*- coding: utf-8 -*-
"""
use resnet18 to construct the topdown network and front end
"""

import torch.nn as nn
import torch

from torchvision import models
from core.model import Model
from torchvision.models.resnet import Bottleneck
import copy


class OFTNetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        self.model_path = model_config['pretrained_model']
        self.dout_base_model = 1024
        self.pretrained = model_config['pretrained']
        self.class_agnostic = model_config['class_agnostic']
        self.classes = model_config['classes']
        self.img_channels = model_config['img_channels']

        self.use_cascade = model_config.get('use_cascade')
        self.separate_feat = model_config.get('separate_feat')

        self.use_img_feat = model_config.get('use_img_feat')

    def init_modules(self):
        resnet = models.resnet18()
        if self.training and self.pretrained:
            print(("Loading pretrained weights from %s" % (self.model_path)))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({
                k: v
                for k, v in list(state_dict.items())
                if k in resnet.state_dict()
            })

        base_features = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        ]

        # if not image(e.g lidar)
        if not self.img_channels == 3:
            self.first_layer = nn.Conv2d(
                self.img_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            base_features[0] = self.first_layer

        # img feauture
        img_feature1 = nn.Sequential(*base_features)
        img_feature2 = resnet.layer3
        img_feature3 = resnet.layer4
        self.img_features = nn.ModuleList(
            [img_feature1, img_feature2, img_feature3])

        # bev feature
        # topdown_features = [resnet.layer3, resnet.layer4]
        self.bev_feature = self._make_topdown_network()

        if self.use_img_feat is not None:
            conv = nn.Conv2d(256, 64, 3, 1, 1)
            bn = nn.BatchNorm2d(64)
            relu = nn.ReLU()
            layer2 = copy.deepcopy(resnet.layer2)
            self.img_feat_extractor = nn.Sequential(* [conv, bn, relu, layer2])

    def _make_topdown_network(self):
        self.inplanes = 256
        return self._make_layer(Bottleneck, 256, 3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img_input):
        img_features = [self.img_features[0](img_input)]
        for i in range(1, 3):
            last_feauters = img_features[-1]
            img_features.append(self.img_features[i](last_feauters))
        return img_features
