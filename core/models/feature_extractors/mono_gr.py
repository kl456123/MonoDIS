# -*- coding: utf-8 -*-

from core.model import Model
from torchvision import models
import torch.nn as nn


class BufferZone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class ConvbnReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MonoGRFeatureExtractor(Model):
    def forward(self):
        pass

    def init_param(self, model_config):
        pass

    def init_modules(self):
        # backbone
        vggnet = models.vgg16()
        features = vggnet.features
        conv4 = features[:-2]
        pool5 = features[-1]

        # buffer zone
