##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import torch
import torch.nn as nn

#from ..nn import SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ['accuracy', 'MixUpWrapper', 'get_selabel_vector']

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MixUpWrapper(object):
    def __init__(self, alpha, num_classes, dataloader, device):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device

    def mixup_loader(self, loader):
        def mixup(alpha, num_classes, data, target):
            with torch.no_grad():
                bs = data.size(0)
                c = np.random.beta(alpha, alpha)
                perm = torch.randperm(bs).cuda()

                md = c * data + (1-c) * data[perm, :]
                mt = c * target + (1-c) * target[perm, :]
                return md, mt

        for input, target in loader:
            input, target = input.cuda(self.device), target.cuda(self.device)
            target = torch.nn.functional.one_hot(target, self.num_classes)
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)


def get_selabel_vector(target, nclass):
    r"""Get SE-Loss Label in a batch
    Args:
        predict: input 4D tensor
        target: label 3D tensor (BxHxW)
        nclass: number of categories (int)
    Output:
        2D tensor (BxnClass)
    """
    batch = target.size(0)
    tvect = torch.zeros(batch, nclass)
    for i in range(batch):
        hist = torch.histc(target[i].data.float(), 
                           bins=nclass, min=0,
                           max=nclass-1)
        vect = hist>0
        tvect[i] = vect
    return tvect
