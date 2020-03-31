##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Rectify function"""
import torch
from torch.autograd import Function

from .. import lib

__all__ = ['rectify']

class _rectify(Function):
    @staticmethod
    def forward(ctx, y, x, kernel_size, stride, padding, dilation, average):
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.average = average
        if x.is_cuda:
            lib.gpu.conv_rectify(y, x, kernel_size, stride, padding, dilation, average)
        else:
            raise NotImplementedError
        ctx.mark_dirty(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_variables
        if x.is_cuda:
            lib.gpu.conv_rectify(grad_y, x, ctx.kernel_size, ctx.stride,
                                 ctx.padding, ctx.dilation, ctx.average)
        else:
            raise NotImplementedError
        ctx.mark_dirty(grad_y)
        return grad_y, None, None, None, None, None, None

rectify = _rectify.apply
