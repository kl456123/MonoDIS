# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

from torchvision import models
from core.model import Model
import copy
import os
from .resnet18_pruned import resnet18


class ResNetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        net_arch_map = {
            'res50': 'resnet50-19c8e357.pth',
            'res18': 'resnet18-5c106cde.pth',
            'res18_pruned': 'resnet18_pruned0.5.pth'
        }
        self.model_map = {
            'res50': models.resnet50,
            'res18': models.resnet18,
            'res18_pruned': resnet18
        }
        self.net_arch = model_config.get('net_arch', 'res50')
        model_name = net_arch_map[self.net_arch]
        #  self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.model_path = os.path.join(model_config['pretrained_path'],
                                       model_name)
        self.dout_base_model = 1024
        self.pretrained = model_config['pretrained']
        self.img_channels = model_config['img_channels']

        self.use_cascade = model_config.get('use_cascade')
        # self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.separate_feat = model_config.get('separate_feat')

    def init_modules(self):
        resnet = self.model_map[self.net_arch]()
        # self.model_path = '/node01/jobs/io/pretrained/resnet50-19c8e357.pth'
        if self.training and self.pretrained:
            print(("Loading pretrained weights from %s" % (self.model_path)))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({
                k: v
                for k, v in list(state_dict.items())
                if k in resnet.state_dict()
            })

        if self.net_arch == 'res18_pruned':
            base_features = [
                resnet.conv1, resnet.bn1, resnet.maxpool, resnet.layer1,
                resnet.layer2, resnet.layer3
            ]
        else:
            base_features = [
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3
            ]

        if self.separate_feat:
            base_features = base_features[:-1]
            self.first_stage_cls_feature = resnet.layer3
            self.first_stage_bbox_feature = copy.deepcopy(resnet.layer3)

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

        self.first_stage_feature = nn.Sequential(*base_features)

        self.second_stage_feature = nn.Sequential(resnet.layer4)
        if self.use_cascade:
            self.third_stage_feature = copy.deepcopy(self.second_stage_feature)
