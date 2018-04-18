import torch.nn as nn, math, torch.nn.functional as F
from .layers import *

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
    def __init__(self, inplanes, planes, stride=1, expansion=4, downsample=None, bn_final=False, bn_zero=False):
        super(Bottleneck, self).__init__()
        self.expansion,self.downsample,self.bn_final = expansion,downsample,bn_final
        self.features = nn.Sequential(
            conv_bn_relu(inplanes, planes, ks=1),
            conv_bn_relu(planes, planes, stride=stride),
            conv(planes, planes*self.expansion, ks=1))
        self.bn3 = bn(planes*self.expansion, init_zero=bn_zero)

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)
        out = self.features(x)
        if not self.bn_final: out = self.bn3(out)
        out += residual
        if self.bn_final: out = self.bn3(out)
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, layer_szs, num_classes=1000, init=True, expansion=4, k=1, bn_final=False, bn_zero=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.expansion,self.bn_final,self.bn_zero = expansion,bn_final,bn_zero

        layers = [conv(3, 64, ks=7, stride=2), bn(64), nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        for i,layer_sz in enumerate(layer_szs):
            layers += self._make_layer(block, int(64*(2**i)*k), layer_sz, stride=1 if i==0 else 2)
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(512 * self.expansion, num_classes)]
        self.features = nn.Sequential(*layers)

        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = (None if stride == 1 and self.inplanes == planes * self.expansion
            else nn.Sequential(
                conv(self.inplanes, planes * self.expansion, ks=1, stride=stride),
                bn(planes * self.expansion)))

        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        expansion=self.expansion, bn_final=self.bn_final, bn_zero=self.bn_zero)]
        self.inplanes = planes * self.expansion
        return layers + [block(self.inplanes, planes,
                         expansion=self.expansion, bn_final=self.bn_final, bn_zero=self.bn_zero)
                         for i in range(1, blocks)]

    def forward(self, x): return self.features(x)


def fa_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
def bnzero_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], bn_zero=True)
def bnfinal_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], bn_final=True)
def noinit_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], init=False)
def fa5_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 3, 4, 3], expansion=5, bn_final=True)
def fa4_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], bn_final=True, bn_zero=True)
def w15_resnet50(pretrained=False, **kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], bn_final=True, bn_zero=True, k=1.5)

