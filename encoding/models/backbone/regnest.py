import numpy as np
import torch.nn as nn
from ...nn import GlobalAvgPool2d, ConvBnAct

__all__ = ['RegNeSt', 'regnetx_4g', 'regnest_4g']

# code modified from https://github.com/signatrix/regnet
class AnyNeSt(nn.Module):
    def __init__(self, ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width,
                 radix, stride, rectified_conv, rectify_avg, avg_down):
        super().__init__()
        for block_width, bottleneck_ratio, group_width in zip(ls_block_width, ls_bottleneck_ratio, ls_group_width):
            assert block_width % (bottleneck_ratio * group_width) == 0
        self.net = nn.Sequential()
        prev_block_width = 32
        self.net.add_module("stem", ConvBnAct(3, prev_block_width, kernel_size=3, stride=2, padding=1, bias=False))

        for i, (num_blocks, block_width, bottleneck_ratio, group_width) in \
                enumerate(zip(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width)):
            self.net.add_module("stage_{}".format(i),
                                Stage(num_blocks, prev_block_width, block_width,
                                      bottleneck_ratio, group_width=group_width,
                                      radix=radix, stride=stride,
                                      rectified_conv=rectified_conv, 
                                      rectify_avg=rectify_avg,
                                      avg_down=avg_down))
            prev_block_width = block_width

        self.net.add_module("pool", GlobalAvgPool2d())
        self.net.add_module("fc", nn.Linear(ls_block_width[-1], 1000))

    def forward(self, x):
        x = self.net(x)
        return x


class RegNeSt(AnyNeSt):
    def __init__(self, initial_width, slope, quantized_param, network_depth, bottleneck_ratio, group_width,
                 radix=0, stride=2, rectified_conv=False, rectify_avg=False, avg_down=False):
        # We need to derive block width and number of blocks from initial parameters.
        parameterized_width = initial_width + slope * np.arange(network_depth)  # From equation 2
        parameterized_block = np.log(parameterized_width / initial_width) / np.log(quantized_param)  # From equation 3
        parameterized_block = np.round(parameterized_block)
        quantized_width = initial_width * np.power(quantized_param, parameterized_block)
        # We need to convert quantized_width to make sure that it is divisible by 8
        quantized_width = 8 * np.round(quantized_width / 8)
        ls_block_width, ls_num_blocks = np.unique(quantized_width.astype(np.int), return_counts=True)
        # At this points, for each stage, the above-calculated block width could be incompatible to group width
        # due to bottleneck ratio. Hence, we need to adjust the formers.
        # Group width could be swapped to number of groups, since their multiplication is block width
        ls_group_width = np.array([min(group_width, block_width // bottleneck_ratio) for block_width in ls_block_width])
        ls_block_width = np.round(ls_block_width // bottleneck_ratio / group_width) * group_width
        ls_group_width = ls_group_width.astype(np.int) * bottleneck_ratio
        ls_bottleneck_ratio = [bottleneck_ratio for _ in range(len(ls_block_width))]
        ls_group_width  = ls_group_width.tolist()
        ls_block_width = ls_block_width.astype(np.int).tolist()

        super().__init__(ls_num_blocks, ls_block_width, ls_bottleneck_ratio, ls_group_width,
                         radix=radix, stride=stride, rectified_conv=rectified_conv, rectify_avg=rectify_avg,
                         avg_down=avg_down)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_ratio, group_width,
                 stride, radix, rectified_conv, rectify_avg,
                 avg_down):
        super(Bottleneck, self).__init__()
        inter_channels = out_channels // bottleneck_ratio
        groups = inter_channels // group_width

        self.conv1 = ConvBnAct(in_channels, inter_channels, kernel_size=1, bias=False)
        self.conv2 = ConvBnAct(inter_channels, inter_channels, kernel_size=3, stride=stride,
                                      groups=groups, padding=1, bias=False, radix=radix,
                                      rectify=rectified_conv, rectify_avg=rectify_avg)
        self.conv3 = ConvBnAct(inter_channels, out_channels, kernel_size=1, bias=False, act=False)
        if stride != 1 or in_channels != out_channels:
            if avg_down:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    ConvBnAct(in_channels, out_channels, kernel_size=1, stride=1, bias=False, act=False))
            else:
                self.shortcut = ConvBnAct(in_channels, out_channels, kernel_size=1, stride=stride, bias=False, act=False)
        else:
            self.shortcut = None
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        if self.shortcut is not None:
            x2 = self.shortcut(x)
        else:
            x2 = x
        x = self.relu(x1 + x2)
        return x


class Stage(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, bottleneck_ratio, group_width,
                 stride, radix, rectified_conv, rectify_avg, avg_down):
        super().__init__()
        self.blocks = nn.Sequential()
        self.blocks.add_module("block_0", Bottleneck(in_channels, out_channels, bottleneck_ratio, group_width,
                                                     stride=stride, radix=radix, rectified_conv=rectified_conv,
                                                     rectify_avg=rectify_avg, avg_down=avg_down))
        for i in range(1, num_blocks):
            self.blocks.add_module("block_{}".format(i),
                                   Bottleneck(out_channels, out_channels, bottleneck_ratio, group_width,
                                              stride=1, radix=radix, rectified_conv=rectified_conv,
                                              rectify_avg=rectify_avg, avg_down=avg_down))

    def forward(self, x):
        x = self.blocks(x)
        return x

def regnetx_4g(pretrained=False, root='~/.encoding/models', **kwargs):
    bottleneck_ratio = 1
    group_width = 40
    initial_width = 96
    slope = 38.65
    quantized_param = 2.43
    network_depth = 23
    model = RegNeSt(initial_width, slope, quantized_param, network_depth,
                    bottleneck_ratio, group_width, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('regnetx_4g', root=root)), strict=True)
    return model

def regnest_4g(pretrained=False, root='~/.encoding/models', **kwargs):
    radix = 2
    bottleneck_ratio = 1
    group_width = 40
    initial_width = 96
    slope = 38.65
    quantized_param = 2.43
    network_depth = 23
    model = RegNeSt(initial_width, slope, quantized_param, network_depth,
                    bottleneck_ratio, group_width, radix=radix,
                    avg_down=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('regnetx_4g', root=root)), strict=True)
    return model
