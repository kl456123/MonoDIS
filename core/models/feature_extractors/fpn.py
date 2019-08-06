# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from core.model import Model
from torchvision.models import resnet50
import torch


class FPNFeatureExtractor(Model):
    def init_param(self, model_config):
        self.model_path = model_config['pretrained_model']
        self.pretrained = model_config['pretrained']
        self.pooled_size = model_config['pooling_size']

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def init_modules(self):
        resnet = resnet50()
        layer1 = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1
        ]
        if self.training and self.pretrained:
            print(("Loading pretrained weights from %s" % (self.model_path)))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({
                k: v
                for k, v in list(state_dict.items())
                if k in resnet.state_dict()
            })
        # bottom-up layers
        # self.layer0 = nn.Sequential(*layer0)
        self.layer2 = nn.Sequential(*layer1)
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4

        # lateral layers
        self.lateral4 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.lateral3 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral2 = nn.Conv2d(256, 256, 1, 1, 0)

        # smooth layers
        # self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth4 = nn.Conv2d(256, 256, 3, 1, 1)

        # special for layer5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        self.maxpool2d = nn.MaxPool2d(1, stride=2)

        self.second_stage_feature = nn.Sequential(*[
            nn.Conv2d(
                256,
                1024,
                kernel_size=self.pooled_size,
                stride=self.pooled_size,
                padding=0), nn.ReLU(True), nn.Conv2d(
                    1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        ])

    def init_weights(self):
        pass

    def first_stage_feature(self, inputs):
        # bottom up
        c2 = self.layer2(inputs)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        # top down
        p5 = self.toplayer(c5)
        p4 = self.upsample_add(p5, self.lateral4(c4))
        p4 = self.smooth4(p4)
        p3 = self.upsample_add(p4, self.lateral3(c3))
        p3 = self.smooth3(p3)
        p2 = self.upsample_add(p3, self.lateral2(c2))
        p2 = self.smooth2(p2)
        # p6 = self.maxpool2d(p5)
        rpn_feature_maps = [p4]
        mrcnn_feature_maps = [p2, p3, p4, p5]
        return rpn_feature_maps, mrcnn_feature_maps

    # def second_stage_feature(self, inputs):
    # pass
