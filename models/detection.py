import math
import torch
import numpy
import itertools

from torch import nn
bn_momentum = 0.1


class Detection(nn.Module):
    def __init__(self):
        super(Detection, self).__init__()

    def forward(self, _input):
        pass

    def load_pretrained_weight(self, net):
        pass


class MultiBoxLayer(nn.Module):
    def __init__(self, cfg):
        super(MultiBoxLayer, self).__init__()
        self.num_classes = cfg['num_classes']
        self.num_anchors = cfg['num_anchors']
        self.num_features = cfg['num_features']

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.num_features)):
            self.loc_layers.append(
                nn.Conv2d(
                    self.num_features[i],
                    self.num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            self.conf_layers.append(
                nn.Conv2d(
                    self.num_features[i],
                    self.num_anchors[i] * self.num_classes,
                    kernel_size=3,
                    padding=1))

    def forward(self, features):
        y_locs = []
        y_confs = []
        for i, x in enumerate(features):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        return loc_preds, conf_preds


class UnifiedMultiBoxLayer(nn.Module):
    def __init__(self, num_classes, num_anchor):
        super(UnifiedMultiBoxLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_layer = nn.Sequential(
            GNBottleneck(
                inplanes=256, planes=64),
            GNBottleneck(
                inplanes=256, planes=64),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.num_anchors * 4,
                kernel_size=1))

        self.ry_layer = nn.Sequential(
            GNBottleneck(
                inplanes=256, planes=64),
            GNBottleneck(
                inplanes=256, planes=64),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.num_anchors * 2,
                kernel_size=1))
        self.conf_layer = nn.Sequential(
            GNBottleneck(
                inplanes=256, planes=64),
            GNBottleneck(
                inplanes=256, planes=64),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.num_anchors * self.num_classes,
                kernel_size=1))

    def forward(self, features):
        y_locs = []
        rys = []
        y_confs = []

        for x in enumerate(features):
            y_loc = self.loc_layer(x[1])
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_ry = self.ry_layer(x[1])
            N = y_ry.size(0)
            y_ry = y_ry.permute(0, 2, 3, 1).contiguous()
            y_ry = y_ry.view(N, -1, 2)
            rys.append(y_ry)

            y_conf = self.conf_layer(x[1])
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)
        ry_preds = torch.cat(rys, 1)

        return loc_preds, ry_preds, conf_preds


class DepthWiseMultiBoxLayer(nn.Module):
    def __init__(self, num_classes, num_anchor):
        super(DepthWiseMultiBoxLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_layer = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.num_anchors * 4,
                kernel_size=1))
        self.conf_layer = nn.Sequential(
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                256, 256, kernel_size=3, stride=1, padding=1),
            GroupBatchnorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.num_anchors * self.num_classes,
                kernel_size=1))

    def forward(self, features):
        y_locs = []
        y_confs = []

        for x in enumerate(features):
            y_loc = self.loc_layer(x[1])
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_conf = self.conf_layer(x[1])
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        return loc_preds, conf_preds


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class InstanceBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(InstanceBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GNBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = GroupBatchnorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = GroupBatchnorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = GroupBatchnorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num=16, eps=1e-10):
        super(GroupBatchnorm2d, self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)

        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta


class SSDPriorBox(object):
    """
        * Compute priorbox coordinates in center-offset form for each source feature map.
    """

    def __init__(self):
        super(SSDPriorBox, self).__init__()

    def __call__(self, cfg):
        self.image_size = cfg['input_shape']
        self.aspect_ratios = cfg['aspect_ratio']
        self.default_ratio = cfg['default_ratio']
        self.output_stride = cfg['output_stride']
        self.clip = True

        scale_w = self.image_size[0]
        scale_h = self.image_size[1]
        steps_w = [s / scale_w for s in self.output_stride]
        steps_h = [s / scale_h for s in self.output_stride]
        sizes = self.default_ratio
        aspect_ratios = self.aspect_ratios

        feature_map_w = [
            int(math.floor(scale_w / s)) for s in self.output_stride
        ]
        feature_map_h = [
            int(math.floor(scale_h / s)) for s in self.output_stride
        ]
        assert len(feature_map_h) == len(feature_map_w)
        num_layers = len(feature_map_h)

        boxes = []
        for i in range(num_layers):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            for h, w in itertools.product(
                    list(range(fm_h)), list(range(fm_w))):
                cx = (w + 0.5) * steps_w[i]
                cy = (h + 0.5) * steps_h[i]

                s = sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(sizes[i] * sizes[i + 1])
                boxes.append((cx, cy, s, s))

                s = sizes[i]
                for ar in aspect_ratios[i]:
                    boxes.append(
                        (cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append(
                        (cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

        boxes = numpy.array(boxes, dtype=float)
        boxes = torch.from_numpy(boxes).float()  # back to torch land
        if self.clip:
            boxes.clamp_(min=0., max=1.)
        return boxes


class DenseSSDPriorBox(object):
    """
        * Compute priorbox coordinates in center-offset form for each source feature map.
    """

    def __init__(self):
        super(DenseSSDPriorBox, self).__init__()

    def __call__(self, cfg):
        self.image_size = cfg['input_shape']
        self.aspect_ratios = cfg['aspect_ratio']
        self.default_ratio = cfg['default_ratio']
        self.output_stride = cfg['output_stride']
        self.clip = True

        scale_w = self.image_size[0]
        scale_h = self.image_size[1]
        steps_w = [s / scale_w for s in self.output_stride]
        steps_h = [s / scale_h for s in self.output_stride]
        sizes = self.default_ratio
        aspect_ratios = self.aspect_ratios

        # feature_map_w = [200, 50, 25, 13]
        # feature_map_h = [88, 44, 22, 11]
        feature_map_w = [200, 100]
        feature_map_h = [176, 88]
        assert len(feature_map_h) == len(feature_map_w)
        num_layers = len(feature_map_h)

        boxes = []
        for i in range(num_layers):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            for h, w in itertools.product(
                    list(range(fm_h)), list(range(fm_w))):
                cx = (w + 0.5) * steps_w[i]
                cy = (h + 0.5) * steps_h[i]

                s = sizes[i]
                boxes.append((cx, cy, s, s))

                for ar in aspect_ratios[i]:
                    boxes.append(
                        (cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append(
                        (cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

                s = math.sqrt(sizes[i] * sizes[i + 1])
                boxes.append((cx, cy, s, s))

                for ar in aspect_ratios[i]:
                    boxes.append(
                        (cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append(
                        (cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

        boxes = numpy.array(boxes, dtype=float)
        boxes = torch.from_numpy(boxes).float()  # back to torch land
        if self.clip:
            boxes.clamp_(min=0., max=1.)
        return boxes


class RevertedResDetBlock(nn.Sequential):
    def __init__(self, input_num, out_num, stride=2, expand_ratio=2):
        super(RevertedResDetBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_num,
                out_channels=input_num * expand_ratio,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(
                num_features=input_num * expand_ratio, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                in_channels=input_num * expand_ratio,
                out_channels=input_num * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=input_num * expand_ratio,
                bias=False),
            nn.BatchNorm2d(
                num_features=input_num * expand_ratio, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                in_channels=input_num * expand_ratio,
                out_channels=out_num,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(
                num_features=out_num, momentum=bn_momentum), )

    def forward(self, _x):
        feature = super(RevertedResDetBlock, self).forward(_x)
        return feature


class DetFeatureBlock(nn.Sequential):
    def __init__(self, input_num, out_num, is_pooling=True, is_upsample=False):
        super(DetFeatureBlock, self).__init__()
        if is_pooling:
            self.add_module('pool1', nn.AvgPool2d(kernel_size=2, stride=2))
        if is_upsample:
            self.add_module(
                'upsample', nn.Upsample(
                    scale_factor=2, mode='bilinear'))

        # self.add_module('norm1', nn.BatchNorm2d(num_features=input_num)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                in_channels=input_num,
                out_channels=input_num,
                kernel_size=3,
                padding=1,
                groups=input_num)),

        # self.add_module('norm2', nn.BatchNorm2d(num_features=input_num)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            nn.Conv2d(
                in_channels=input_num, out_channels=out_num, kernel_size=1))

    def forward(self, _input):
        feature = super(DetFeatureBlock, self).forward(_input)
        return feature


class ResDetFeatureBlock(nn.Sequential):
    def __init__(self, input_num, out_num, is_pooling=True, is_upsample=False):
        super(ResDetFeatureBlock, self).__init__()
        if is_pooling:
            self.add_module(
                'pool1',
                nn.Conv2d(
                    in_channels=input_num,
                    out_channels=input_num,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=input_num))
        if is_upsample:
            self.add_module(
                'upsample', nn.Upsample(
                    scale_factor=2, mode='bilinear'))

        self.add_module(
            'conv1',
            nn.Conv2d(
                in_channels=input_num,
                out_channels=out_num,
                kernel_size=1,
                bias=False))
        self.add_module(
            'block1', GNBottleneck(
                inplanes=out_num, planes=out_num / 4)),
        self.add_module(
            'block2', GNBottleneck(
                inplanes=out_num, planes=out_num / 4)),

    def forward(self, _input):
        feature = super(ResDetFeatureBlock, self).forward(_input)

        return feature


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num, group_num=16, eps=1e-10):
        super(GroupBatchnorm2d, self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, self.group_num, -1)

        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)

        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)

        return x * self.gamma + self.beta
