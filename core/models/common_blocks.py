# -*- coding: utf-8 -*-

import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    conv = conv3x3(in_planes, out_planes, stride=1)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU()
    return nn.Sequential(* [conv, bn, relu])
