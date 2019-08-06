import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    # Add pruning_rate in function BasicBlock()
    def __init__(self, inplanes, planes, pruning_rate, stride=1, dilation=1, padding=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride, dilation, padding)
        self.pruned_channel_planes = int(planes - math.floor(planes * pruning_rate))
        self.conv1 = conv3x3(inplanes, self.pruned_channel_planes, stride, dilation, padding)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(self.pruned_channel_planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv3x3(self.pruned_channel_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation, stride=stride,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    # Add pruning_rate in function __init__
    def __init__(self, block, layers, pruning_rate):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], pruning_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], pruning_rate, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], pruning_rate, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], pruning_rate, stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, pruning_rate, stride=1, dilation=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, pruning_rate, stride, dilation, padding, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, pruning_rate, dilation=dilation, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        C1 = F.relu(self.bn1(self.conv1(x)), inplace=True) # [H/2, W/2, 64]
        x = self.maxpool(C1) # [H/4, W/4, 64]
        C2 = self.layer1(x) # [H/4, W/4, 64]
        C3 = self.layer2(C2) # [H/8, W/8, 128]
        C4 = self.layer3(C3) # [H/16, W/16, 256]
        C5 = self.layer4(C4) # [H/32, W/32, 512]
        return C4, C5

    def load_pretrained_weight(self, net):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in net.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2], 0.5)  # set pruning_rate=0.5
    return model

if __name__ == '__main__':
    net = resnet18()
    weights = torch.load("pretrained/resnet18_pruned0.5.pth")
    net.load_pretrained_weight(weights)
    x = torch.randn(1, 3, 384, 768).cuda()
    net.eval().cuda()
    c4, c5 = net(x)
    print(c4.size(), c5.size())
