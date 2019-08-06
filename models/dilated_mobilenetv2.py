import torch.nn as nn

from collections import OrderedDict

bn_momentum = 0.1

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup, momentum=bn_momentum),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup, momentum=bn_momentum),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel_size=3, stride=stride, padding=dilation,
                      dilation=dilation, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=bn_momentum),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class DilatedMobileNetV2(nn.Module):

    def __init__(self, width_mult=1., output_stride=8):
        super(DilatedMobileNetV2, self).__init__()
        self.num_features = 320
        self.scale_factor = output_stride / 8
        scale = self.scale_factor
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, scale, 2 / scale],
            [6, 96, 3, 1, 2 / scale],
            [6, 160, 3, 1, 4 / scale],
            [6, 320, 1, 1, 4 / scale],
        ]

        input_channel = int(32 * width_mult)
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, dilate in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, dilation=dilate))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, dilation=dilate))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)

    def get_num_features(self):
        return self.num_features

    def load_pretrained_weight(self, net):
        new_state_dict = OrderedDict()
        for k, v in list(net.items()):
            k_name = k.split('.')
            if k_name[0] == 'features' and float(k_name[1]) < 18:
                new_state_dict[k] = v
        model_dict = self.state_dict()
        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)
