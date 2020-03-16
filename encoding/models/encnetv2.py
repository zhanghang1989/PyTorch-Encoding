###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import torch
from torch.autograd import Variable

import torch.nn as nn
from torch.nn.functional import upsample

from .base import BaseNet
from .fcn import FCNHead
from .encnet import EncModule
from ..nn import PyramidPooling

__all__ = ['EncNetV2', 'get_encnetv2', 'get_encnetv2_resnet152_ade']

class EncNetV2(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, lateral=False,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(EncNetV2, self).__init__(nclass, backbone, aux, se_loss,
                                       norm_layer=norm_layer, **kwargs)
        self.head = EncHead(2048, nclass, se_loss=se_loss, lateral=lateral,
                            norm_layer=norm_layer, up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        
        x = list(self.head(c4))
        x[0] = upsample(x[0], imsize, **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, imsize, **self._up_kwargs)
            x.append(auxout)

        return tuple(x)


class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, lateral=True,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(
            PyramidPooling(in_channels, norm_layer, up_kwargs=up_kwargs),
            nn.Conv2d(in_channels*2, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))
        self.se_loss = se_loss

    def forward(self, x):
        x = self.conv5(x)
        outs = list(self.encmodule(x))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)


def get_encnetv2(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                   root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnetv2(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import get_segmentation_dataset
    data = get_segmentation_dataset(dataset)
    model = EncNetV2(data.num_class, backbone=backbone, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('encnetv2_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_encnetv2_resnet152_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""EncNet-PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnetv2_152_ade(pretrained=True)
    >>> print(model)
    """
    return get_encnetv2('ade20k', 'resnet152', pretrained)
