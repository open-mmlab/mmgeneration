from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ACTIVATION_LAYERS
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer
from mmcv.cnn.utils import constant_init

from mmgen.models.builder import MODULES, build_module


class EmbedSequential(nn.Sequential):
    """A sequential module that passes timestep embeddings to the children that
    support it as an extra input.

    Modified from
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py#L35
    """

    def forward(self, x, y):
        for layer in self:
            if isinstance(layer, DenoisingResBlock):
                x = layer(x, y)
            else:
                x = layer(x)
        return x


@ACTIVATION_LAYERS.register_module()
class SiLU(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        # return F.silu(x, inplace=self.inplace)
        return torch.sigmoid(x) * x


@MODULES.register_module()
class MultiHeadAttention(nn.Module):
    """An attention block allows spatial position to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.  # noqa

    Args:
        in_channels (int): Channels of the input feature map.
        num_heads (int, optional): Number of heads in the attention.
        norm_cfg (dict, optional): Config for normalization layer. Default
            to ``dict(type='GN', num_groups=32)``
    """

    def __init__(self,
                 in_channels,
                 num_heads=1,
                 norm_cfg=dict(type='GN', num_groups=32)):
        super().__init__()
        self.num_heads = num_heads
        _, self.norm = build_norm_layer(norm_cfg, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.proj = nn.Conv1d(in_channels, in_channels, 1)

    @staticmethod
    def QKVAttention(qkv):
        channel = qkv.shape[1] // 3
        q, k, v = torch.chunk(qkv, 3, dim=1)
        scale = 1 / np.sqrt(np.sqrt(channel))
        weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = torch.einsum('bts,bcs->bct', weight, v)
        return weight

    def forward(self, x):
        """Forward function for multi head attention.
        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Feature map after attention.
        """
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.QKVAttention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj(h)
        return (h + x).reshape(b, c, *spatial)

    def init_weights(self):
        constant_init(self.proj, 0)


@MODULES.register_module()
class TimeEmbedding(nn.Module):
    """Time embedding layer, reference to Two level embedding. First embedding
    time by an embedding function, then feed to neural networks.

    Args:
        in_channels (int): The channel number of the input feature map.
        embedding_channels (int): The channel number of the output embedding.
        act_cfg (dict, optional): Config for activation layer. Defaults to
            ``dict(type='SiLU', inplace=False)``.
    """

    def __init__(self,
                 in_channels,
                 embedding_channels,
                 embedding_mode='sin',
                 embedding_cfg=None,
                 act_cfg=dict(type='SiLU', inplace=False)):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(in_channels, embedding_channels),
            build_activation_layer(act_cfg),
            nn.Linear(embedding_channels, embedding_channels))

        # add `dim` to embedding config
        embedding_cfg_ = dict(dim=in_channels)
        if embedding_cfg is not None:
            embedding_cfg_.update(embedding_cfg)
        if embedding_mode.upper() == 'SIN':
            self.embedding_fn = partial(self.sinusodial_embedding,
                                        **embedding_cfg_)
        else:
            raise ValueError()

    @staticmethod
    def sinusodial_embedding(timesteps, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """

        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """Forward function for time embedding layer.
        Args:
            t (torch.Tensor): Input timesteps. Shape as [TODO:]

        Returns:
            torch.Tensor: Timesteps embedding, shape as [TODO:].

        """
        return self.blocks(self.embedding_fn(t))


@MODULES.register_module()
class DenoisingResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 embedding_channels,
                 use_scale_shift_norm,
                 dropout,
                 out_channels=None,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        _norm_cfg = deepcopy(norm_cfg)

        _, norm_1 = build_norm_layer(_norm_cfg, in_channels)
        conv_1 = [
            norm_1,
            build_activation_layer(act_cfg),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        ]
        self.conv_1 = nn.Sequential(*conv_1)

        norm_with_embedding_cfg = dict(
            in_channels=out_channels,
            embedding_channels=embedding_channels,
            use_scale_shift=use_scale_shift_norm,
            norm_cfg=_norm_cfg)
        self.norm_with_embedding = build_module(
            dict(type='NormWithEmbedding'),
            default_args=norm_with_embedding_cfg)

        conv_2 = [
            build_activation_layer(act_cfg),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        ]
        self.conv_2 = nn.Sequential(*conv_2)

        assert shortcut_kernel_size in [
            1, 3
        ], ('Only support `1` and `3` for `shortcut_kernel_size`, but '
            f'receive {shortcut_kernel_size}.')

        self.learnable_shortcut = out_channels != in_channels

        if self.learnable_shortcut:
            shortcut_padding = 1 if shortcut_kernel_size == 3 else 0
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                shortcut_kernel_size,
                padding=shortcut_padding)

    def forward_shortcut(self, x):
        if self.learnable_shortcut:
            return self.shortcut(x)
        return x

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.
            y (torch.Tensor): Shared time embedding or shared label embedding.

        Returns:
            torch.Tensor : Output feature map tensor.
        """
        shortcut = self.forward_shortcut(x)
        x = self.conv_1(x)
        x = self.norm_with_embedding(x, y)
        x = self.conv_2(x)
        return x + shortcut

    def init_weights(self):
        # apply zero init to last conv layer
        constant_init(self.conv_2[-1], 0)


@MODULES.register_module()
class NormWithEmbedding(nn.Module):
    """Nornalization with embedding layer. If use_scale_shift == True,
    embedding results would be chunked and used to re-shift and re-scale
    normalization results. Otherwise, embedding results would directly add to
    input of normalization layer.

    Args:
        in_channels (int)
        embedding_channels (int)
        norm_cfg ()
    """

    def __init__(self,
                 in_channels,
                 embedding_channels,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 use_scale_shift=True):
        super().__init__()
        self.use_scale_shift = use_scale_shift
        _, self.norm = build_norm_layer(norm_cfg, in_channels)

        embedding_output = in_channels * 2 if use_scale_shift else in_channels
        self.embedding_layer = nn.Sequential(
            build_activation_layer(act_cfg),
            nn.Linear(embedding_channels, embedding_output))

    def forward(self, x, y):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature map tensor.
            y (torch.Tensor): Shared time embedding or shared label embedding.

        Returns:
            torch.Tensor : Output feature map tensor.
        """
        embedding = self.embedding_layer(y)[:, :, None, None]
        if self.use_scale_shift:
            scale, shift = torch.chunk(embedding, 2, dim=1)
            x = self.norm(x)
            x = x * (1 + scale) + shift
        else:
            x = self.norm(x + embedding)
        return x


@MODULES.register_module()
class DenoisingDownsample(nn.Module):

    def __init__(self, in_channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.downsample = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
        else:
            self.downsample = nn.AvgPool2d(stride=2)

    def forward(self, x):
        return self.downsample(x)


@MODULES.register_module()
class DenoisingUpsample(nn.Module):

    def __init__(self, in_channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if hasattr(self, 'conv'):
            x = self.conv(x)
        return x
