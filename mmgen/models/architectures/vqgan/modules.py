# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
from mmcv.cnn.bricks import Linear, build_activation_layer, build_norm_layer, build_conv_layer

from mmgen.registry import MODULES

def nonlinearity(x):
    """Sigmoid function.

    Args:
        x (torch.Tensor): Input feature map.

    Returns:
        torch.Tensor: Output feature map.
    """
    activate = build_activation_layer(dict(type='Sigmoid'))
    return x*activate(x)

def Normalize(in_channels):
    """Normalize function.

    Args:
        in_channels (int): The channel number of the input feature map.

    Returns:
        norm(torch.FloatTensor): Output feature map.
    """
    norm_cfg = dict(type='GN', num_groups=32, eps=1e-6, affine=True)
    _, norm = build_norm_layer(norm_cfg, in_channels)
    return norm


@MODULES.register_module()
class DiffusionDownsample(nn.Module):
    """Downsampling operation. Support padding, average
    pooling and convolution for downsample operation.

    Ref: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
    
    Args:
        in_channels (int): Number of channels of the input feature map to be
            downsampled.
        with_conv (bool, optional): Whether use convolution operation for
            downsampling.  Defaults to `True`.
    """
    def __init__(self,
                 in_channels, 
                 with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = build_conv_layer(None,
                                        in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        """Forward function for downsampling operation.
        Args:
            x (torch.Tensor): Feature map to downsample.

        Returns:
            torch.Tensor: Feature map after downsampling.
        """
        if self.with_conv:
            # do asymmetric padding
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


@MODULES.register_module()
class DiffusionResnetBlock(nn.Module):
    """Resblock for the diffusion model. If `in_channels` not equals to
    `out_channels`, a learnable shortcut with conv layers will be added.
    
    Ref: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py

    Args:
        in_channels (int): Number of channels of the input feature map.
        out_channels (int, optional): Number of output channels of the
            ResBlock. If not defined, the output channels will equal to the
            `in_channels`. Defaults to `None`.
        conv_shortcut (bool): Whether to use conv_shortcut in
            convolution layers. Defaults to `False`.
        dropout (float): Probability of the dropout layers.
        temb_channels (int): Number of channels of the input embedding.
    """
    def __init__(self,
                 *,
                 in_channels, 
                 out_channels=None, 
                 conv_shortcut=False, 
                 dropout, 
                 temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = build_conv_layer(None,
                                    in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        if temb_channels > 0:
            self.temb_proj = Linear(temb_channels,
                                    out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = build_conv_layer(None,
                                    out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = build_conv_layer(None,
                                                    in_channels,
                                                    out_channels,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1)
            else:
                self.nin_shortcut = build_conv_layer(None,
                                                    in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        """Forward function.
        Args:
            x (torch.Tensor): Input feature map tensor.
            temb (torch.Tensor): Shared time embedding.
        Returns:
            torch.Tensor : Output feature map tensor.
        """
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

