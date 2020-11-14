import math

from mmcv.cnn import build_conv_layer, build_norm_layer

from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet
from torch.nn.modules.utils import _pair, _single
from mmcv.ops.deform_conv import deform_conv2d

import torch
from torch import nn

#############################################################################################
class SKDConv(nn.Conv2d):
    def __init__(self, channels, stride, M=2, reduction=16):
        super(SKDConv, self).__init__(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            bias=False)

        self.deform_groups = 1

        self.conv_offset_3 = nn.Conv2d(
            channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)

        self.conv_offset_5 = nn.Conv2d(
            channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=2,
            dilation=2,
            bias=True)

        self.init_offset()

        self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.M = M

        self.att_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * M, 1, bias=False)
        )
        self.att_s = nn.Conv2d(1, 2, 3, 1, 1, bias=False)

    def init_offset(self):
        self.conv_offset_3.weight.data.zero_()
        self.conv_offset_3.bias.data.zero_()
        self.conv_offset_5.weight.data.zero_()
        self.conv_offset_5.bias.data.zero_()

    def forward(self, x):
        offset_3 = self.conv_offset_3(x)
        offset_5 = self.conv_offset_5(x)

        splited = list()
        splited.append(deform_conv2d(x, offset_3, self.weight, self.stride, self.padding,
                       self.dilation, self.groups, self.deform_groups))
        for i in range(1, self.M):
            self.padding = 1 + i
            self.dilation = 1 + i
            weight = self.weight + self.weight_diff
            splited.append(deform_conv2d(x, offset_5, weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deform_groups))

        self.padding = 1
        self.dilation = 1

        feats = sum(splited)
        att_c = self.att_c(feats.contiguous())
        att_c = att_c.reshape(x.size(0), self.M, x.size(1))
        att_c = att_c.softmax(dim=1)
        att_c = att_c.reshape(x.size(0), -1, 1, 1)
        att_c = torch.split(att_c, x.size(1), dim=1)

        att_c = sum([w * s for w, s in zip(att_c, splited)])

        att_s = self.att_s(torch.max(feats, dim=1, keepdim=True)[0])
        att_s = att_s.softmax(dim=1)
        att_s = torch.split(att_s, 1, dim=1)

        att_s = sum([w * s for w, s in zip(att_s, splited)])

        #return (att_c + att_s) / 2
        #return torch.where(att_c > att_s, att_c, att_s)
        return att_c


#############################################################################################


class SKConv(nn.Conv2d):
    def __init__(self, channels, stride, M=2, reduction=16):
        super(SKConv, self).__init__(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            bias=False)
        # self.conv1 = nn.ModuleList([])
        # for i in range(M):
        #     self.conv1.append(nn.Sequential(
        #         nn.Conv2d(channels, channels, kernel_size=3, padding=1+i, dilation=1+i, bias=False),
        #         nn.BatchNorm2d(channels),
        #         nn.ReLU()
        #     ))
        self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.M = M

        self.att_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels * M, 1, bias=False)
        )
        self.att_s = nn.Conv2d(1, 2, 3, 1, 1, bias=False)
        # self.att_3d = nn.Sequential(
        #     nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        splited = list()
        splited.append(super(SKConv, self).conv2d_forward(x, self.weight))
        for i in range(1, self.M):
            self.padding = 1 + i
            self.dilation = 1 + i
            weight = self.weight + self.weight_diff
            splited.append(super(SKConv, self).conv2d_forward(x, weight))

        self.padding = 1
        self.dilation = 1

        feats = sum(splited)
        att_c = self.att_c(feats.contiguous())
        att_c = att_c.reshape(x.size(0), self.M, x.size(1))
        att_c = att_c.softmax(dim=1)
        att_c = att_c.reshape(x.size(0), -1, 1, 1)
        att_c = torch.split(att_c, x.size(1), dim=1)

        att_c = sum([w * s for w, s in zip(att_c, splited)])

        att_s = self.att_s(torch.max(feats, dim=1, keepdim=True)[0])
        att_s = att_s.softmax(dim=1)
        att_s = torch.split(att_s, 1, dim=1)

        att_s = sum([w * s for w, s in zip(att_s, splited)])

        #return (att_c + att_s) / 2
        #return torch.where(att_c > att_s, att_c, att_s)
        return att_s

        #
        # att_3d = self.att_3d(feats_c+feats_s)
        #
        # return att_3d * feats_c + (1-att_3d) * feats_s


class Bottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 **kwargs):
        """Bottleneck block for ResNeXt.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            # self.conv2 = build_conv_layer(
            #     self.conv_cfg,
            #     width,
            #     width,
            #     kernel_size=3,
            #     stride=self.conv2_stride,
            #     padding=self.dilation,
            #     dilation=self.dilation,
            #     groups=groups,
            #     bias=False)
            self.conv2 = SKDConv(width, self.conv2_stride)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                self.dcn,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)


@BACKBONES.register_module()
class ResNeXtAT(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        self.groups = groups
        self.base_width = base_width
        super(ResNeXtAT, self).__init__(**kwargs)

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``"""
        return ResLayer(
            groups=self.groups,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)
