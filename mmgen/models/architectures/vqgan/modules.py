# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import Linear, build_conv_layer, build_norm_layer
from mmgen.registry import MODULES


@MODULES.register_module()
class DiffusionResnetBlock(nn.Module):
    # yapf: disable
    """Resblock for the diffusion model. If `in_channels` not equals to
    `out_channels`, a learnable shortcut with conv layers will be added.

    Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules

    Args:
        in_channels (int): Number of channels of the input feature map.
        out_channels (int, optional): Number of output channels of the
            ResBlock. If not defined, the output channels will equal to the
            `in_channels`. Defaults to `None`.
        conv_shortcut (bool, optional): Whether to use conv_shortcut in
            convolution layers. Defaults to `False`.
        dropout (float): Probability of the dropout layers.
        temb_channels (int, optional): Number of channels of the input time embedding.
                            Defaults to `512`.
        norm_cfg (dict, optional): Config for the norm of output layer.
            Defaults to dict(type='BN').
    """

    def __init__(self,
                 *,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout,
                 temb_channels=512,
                 norm_cfg=dict(type='GN', num_groups=32, eps=1e-6,
                               affine=True)):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.silu = nn.SiLU()

        self.norm1 = build_norm_layer(norm_cfg, in_channels)
        self.conv1 = build_conv_layer(None,
                                      in_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        if temb_channels > 0:
            self.temb_proj = Linear(temb_channels, out_channels)
        self.norm2 = build_norm_layer(norm_cfg, out_channels)
        self.dropout = nn.Dropout(dropout)
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
        h = self.norm1(x)
        h = self.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
