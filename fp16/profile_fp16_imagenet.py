from fastai.dataloader import *
from fastai.dataset import *
from fastai.transforms import *
from fastai.models import *
from fastai.conv_learner import *

DIR = Path('data/imagenet/')
TRAIN_CSV='train.csv'

from pathlib import Path

arch = resnet34
tfms = tfms_from_model(arch, 256, aug_tfms=transforms_side_on)
bs = 128

data = ImageClassifierData.from_csv(DIR, 'train1', DIR/TRAIN_CSV, tfms=tfms, bs=bs)

# m = ConvnetBuilder(resnet34, data.c, data.is_multi, data.is_reg)
# # models = ConvnetBuilder(resnet34, data.c, data.is_multi, data.is_reg)
# # m.model = network_to_half(m.model)
# learner = ConvLearner(data, m)

# learner.fit(0.5,1,cycle_len=1)



import torch
import torch.nn as nn


class tofp16(nn.Module):
    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def copy_in_params(net, params):
    net_params = list(net.parameters())
    for i in range(len(params)):
        net_params[i].data.copy_(params[i].data)


def set_grad(params, params_with_grad):

    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
        param.grad.data.copy_(param_w_grad.grad.data)


def BN_convert_float(module):
    '''
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))


m16 = ConvnetBuilder(resnet34, data.c, data.is_multi, data.is_reg)
m16.model = network_to_half(m16.model)
learner16 = ConvLearner(data, m16)

learner16.fit(0.5,1,cycle_len=1)