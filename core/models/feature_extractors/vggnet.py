# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

from torchvision import models
from core.model import Model


class VGGFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        #  self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.mode_path = model_config['pretained_model']
        self.dout_base_model = 512
        self.pretrained = model_config['pretrained']
        self.class_agnostic = model_config['class_agnostic']
        self.classes = model_config['classes']
        self.img_channels = model_config['img_channels']

    def init_modules(self):
        vggnet = models.vgg16()
        # self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        # self.model_path = '/node01/jobs/io/pretrained/resnet50-19c8e357.pth'
        if self.training and self.pretrained:
            print(("Loading pretrained weights from %s" % (self.model_path)))
        state_dict = torch.load(self.model_path)
        vggnet.load_state_dict({
            k: v
            for k, v in list(state_dict.items()) if k in vggnet.state_dict()
        })
        base_features = list(vggnet.features._modules.values())

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

        self.first_stage_feature = nn.Sequential(*base_features[:-1])

        self.second_stage_feature = nn.Sequential(*list(
            vggnet.classifier._modules.values())[:-1])
