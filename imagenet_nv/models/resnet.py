import torch.nn as nn, math
from .layers import *

__all__ = ['ResNet', 'fa_resnet50']


def conv(in_planes, out_planes, ks=3, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=ks, stride=stride, padding=ks//2, bias=False)

def bn(ni, init_zero=False):
    m = nn.BatchNorm2d(ni, momentum=0.01)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m

def conv_bn_relu(in_planes, out_planes, ks=3, stride=1, init_zero=False):
    return nn.Sequential(conv(in_planes, out_planes, ks=ks, stride=stride),
        bn(out_planes, init_zero=init_zero),
        nn.ReLU(inplace=True))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.features = nn.Sequential(
            conv_bn_relu(inplanes, planes, ks=1),
            conv_bn_relu(planes, planes, stride=stride),
            conv(planes, planes*self.expansion, ks=1))
        self.bn3 = bn(planes*self.expansion)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)
        out = self.features(x)
        out = self.bn3(out)
        out += residual
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, init=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        layers = [conv(3, 64, ks=7, stride=2), bn(64), nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        layers += self._make_layer(block, 64, layers[0])
        layers += self._make_layer(block, 128, layers[1], stride=2)
        layers += self._make_layer(block, 256, layers[2], stride=2)
        layers += self._make_layer(block, 512, layers[3], stride=2)
        layers += [nn.AvgPool2d(7, stride=1), Flatten(),
                   nn.Linear(512 * block.expansion, num_classes)]
        self.features = nn.Sequential(*layers)

        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = (None if stride == 1 and self.inplanes == planes * block.expansion
            else nn.Sequential(
                conv(self.inplanes, planes * block.expansion, ks=1, stride=stride),
                bn(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample))]
        self.inplanes = planes * block.expansion
        return layers + [block(self.inplanes, planes)) for i in range(1, blocks)]

    def forward(self, x): return self.features(x)


def fa_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

