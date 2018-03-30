import torch
from torchvision.models import vgg16,densenet121,resnet152
from time import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
import torchvision.models as models
torch.backends.cudnn.benchmark=True
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print('cuda version=', torch.version.cuda)
print('cudnn version=', torch.backends.cudnn.version())

# for arch in ['densenet121', 'vgg16', 'resnet152']:
for arch in ['vgg16']:
    model   = models.__dict__[arch]().cuda()
    loss   = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                        momentum=0.9,
                                        weight_decay=1e-5)
    durations = []
    num_runs = 100

    for i in range(num_runs + 1):
        x = torch.rand(16, 3, 224, 224)
        x_var = torch.autograd.Variable(x).cuda()
        target = Variable(torch.LongTensor(16).fill_(1).cuda())
        torch.cuda.synchronize()
        t1 = time()
        out = model(x_var)
        err = loss(out, target)
        err.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time()

        # treat the initial run as warm up and don't count
        if i > 0:
            durations.append(t2 - t1)

    print('{} FP 32 avg over {} runs: {} ms'.format(arch, len(durations), sum(durations) / len(durations) * 1000)) 

    model   = models.__dict__[arch]().cuda().half()
    loss   = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                        momentum=0.9,
                                        weight_decay=1e-5)
    durations = []
    num_runs = 100

    for i in range(num_runs + 1):
        x = torch.rand(16, 3, 224, 224)
        x_var = torch.autograd.Variable(x).cuda().half()
        target = Variable(torch.LongTensor(16).fill_(1).cuda())
        torch.cuda.synchronize()
        t1 = time()
        out = model(x_var)
        err = loss(out, target)
        err.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time()

        # treat the initial run as warm up and don't count
        if i > 0:
            durations.append(t2 - t1)

    print('{} FP 16 avg over {} runs: {} ms'.format(arch, len(durations), sum(durations) / len(durations) * 1000)) 