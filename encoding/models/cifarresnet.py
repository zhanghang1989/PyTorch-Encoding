##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
from torch.autograd import Variable
from ..nn import View, RFConv2d

__all__ = ['cifar_resnet20']

class Basicblock(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, conv_layer=None, norm_layer=nn.BatchNorm2d):
        super(Basicblock, self).__init__()
        if inplanes != planes or stride !=1 :
            self.downsample = True
            self.residual_layer = conv_layer(inplanes, planes,
                                            kernel_size=1, stride=stride,
                                            bias=False)
        else:
            self.downsample = False
        conv_block=[]
        conv_block+=[norm_layer(inplanes),
                     nn.ReLU(inplace=True),
                     conv_layer(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                     norm_layer(planes),
                     nn.ReLU(inplace=True),
                     conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                     ]
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, input):
        if self.downsample:
            residual = self.residual_layer(input)
        else:
            residual = input
        return residual + self.conv_block(input)


class Bottleneck(nn.Module):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        if inplanes != planes*self.expansion or stride !=1 :
            self.downsample = True
            self.residual_layer = conv_layer(inplanes, 
                planes * self.expansion, kernel_size=1, stride=stride,
                bias=False)
        else:
            self.downsample = False
        conv_block = []
        conv_block += [norm_layer(inplanes),
                       nn.ReLU(inplace=True),
                       conv_layer(inplanes, planes, kernel_size=1, 
                           stride=1, bias=False)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       conv_layer(planes, planes, kernel_size=3, 
                           stride=stride, padding=1, bias=False)]
        conv_block += [norm_layer(planes),
                       nn.ReLU(inplace=True),
                       conv_layer(planes, planes * self.expansion, 
                           kernel_size=1, stride=1, bias=False)]
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        if self.downsample:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)
       

class CIFAR_ResNet(nn.Module):
    def __init__(self, block=Basicblock, num_blocks=[2,2,2], width_factor = 1, 
                 num_classes=10, conv_layer=None, norm_layer=torch.nn.BatchNorm2d):
        super(CIFAR_ResNet, self).__init__()
        self.expansion = block.expansion

        self.inplanes = int(width_factor * 16)
        strides = [1, 2, 2]
        model = []
        # Conv_1
        model += [conv_layer(3, self.inplanes, kernel_size=3, padding=1, bias=False),
                  norm_layer(self.inplanes),
                  nn.ReLU(inplace=True)]
        # Residual units
        model += [self._residual_unit(block, self.inplanes, num_blocks[0],
                                      strides[0], conv_layer=conv_layer,
                                      norm_layer=norm_layer)]
        for i in range(2):
            model += [self._residual_unit(
                block, int(2*self.inplanes/self.expansion),
                num_blocks[i+1], strides[i+1],
                conv_layer=conv_layer,
                norm_layer=norm_layer)]

        # Last conv layer
        model += [norm_layer(self.inplanes),
                  nn.ReLU(inplace=True),
                  nn.AvgPool2d(8),
                  View(-1, self.inplanes),
                  nn.Linear(self.inplanes, num_classes)]
        self.model = nn.Sequential(*model)

    def _residual_unit(self, block, planes, n_blocks, stride, conv_layer, norm_layer):
        strides = [stride] + [1]*(n_blocks-1)
        layers = []
        for i in range(n_blocks):
            layers += [block(self.inplanes, planes, strides[i],
                             conv_layer=conv_layer,
                             norm_layer=norm_layer)]
            self.inplanes = self.expansion*planes
        return nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


def cifar_resnet20(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a CIFAR ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CIFAR_ResNet(Bottleneck, [3, 3, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('cifar_resnet20', root=root)), strict=False)
    return model
