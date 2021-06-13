from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import build_activation_layer, build_upsample_layer
from torch.nn import Parameter
from torch.nn.modules.batchnorm import SyncBatchNorm
from torch.nn.utils import spectral_norm

from mmgen.models.builder import MODULES


class SNConvModule(ConvModule):
    # TODO:
    """[summary]

    Args:
        with_spectral_norm (bool, optional): [description]. Defaults to
            False.
        spectral_norm_cfg ([type], optional): [description]. Defaults to
            None.
    """

    def __init__(self,
                 *args,
                 with_spectral_norm=False,
                 spectral_norm_cfg=None,
                 **kwargs):
        super().__init__(*args, with_spectral_norm=False, **kwargs)
        self.with_spectral_norm = with_spectral_norm
        self.spectral_norm_cfg = deepcopy(
            spectral_norm_cfg) if spectral_norm_cfg else dict()

        if self.with_spectral_norm:
            self.conv = spectral_norm(self.conv, **self.spectral_norm_cfg)


@MODULES.register_module()
class BigGANGenResBlock(nn.Module):
    # TODO:
    """[summary]

    Args:
        in_channels ([type]): [description]
        out_channels ([type]): [description]
        dim_after_concat ([type]): [description]
        conv_cfg ([type], optional): [description]. Defaults to dict(type=
            'Conv2d').
        shortcut_cfg ([type], optional): [description]. Defaults to dict(
                type='Conv2d').
        act_cfg ([type], optional): [description]. Defaults to dict(type=
            'ReLU').
        upsample_cfg ([type], optional): [description]. Defaults to dict(
                type='nearest', scale_factor=2).
        sn_eps ([type], optional): [description]. Defaults to 1e-6.
        with_spectral_norm (bool, optional): [description]. Defaults to
            True.
        label_input (bool, optional): [description]. Defaults to False.
        auto_sync_bn (bool, optional): [description]. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim_after_concat,
                 conv_cfg=dict(type='Conv2d'),
                 shortcut_cfg=dict(type='Conv2d'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 sn_eps=1e-6,
                 with_spectral_norm=True,
                 label_input=False,
                 auto_sync_bn=True):
        super().__init__()
        self.activation = build_activation_layer(act_cfg)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.with_upsample = upsample_cfg is not None
        if self.with_upsample:
            self.upsample_layer = build_upsample_layer(self.upsample_cfg)
        self.learnable_sc = in_channels != out_channels or self.with_upsample
        if self.learnable_sc:
            self.shortcut = SNConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm,
                spectral_norm_cfg=dict(eps=sn_eps))
        # Here in_channels of BigGANGenResBlock equal to output_dim of ccbn
        self.bn1 = BigGANConditionBN(
            in_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            label_input=label_input,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)
        self.bn2 = BigGANConditionBN(
            out_channels,
            dim_after_concat,
            sn_eps=sn_eps,
            label_input=label_input,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)

        self.conv1 = SNConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))

        self.conv2 = SNConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))

    def forward(self, x, y):
        # TODO:
        """[summary]

        Args:
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        x0 = self.bn1(x, y)
        x0 = self.activation(x0)
        if self.with_upsample:
            x0 = self.upsample_layer(x0)
            x = self.upsample_layer(x)
        x0 = self.conv1(x0)
        x0 = self.bn2(x0, y)
        x0 = self.activation(x0)
        x0 = self.conv2(x0)
        if self.learnable_sc:
            x = self.shortcut(x)
        return x0 + x


@MODULES.register_module()
class BigGANConditionBN(nn.Module):
    """[summary]

    Args:
        num_features ([type]): [description]
        input_dim ([type]): [description]
        bn_eps ([type], optional): [description]. Defaults to 1e-5.
        sn_eps ([type], optional): [description]. Defaults to 1e-6.
        momentum (float, optional): [description]. Defaults to 0.1.
        label_input (bool, optional): [description]. Defaults to False.
        with_spectral_norm (bool, optional): [description]. Defaults to
            True.
        auto_sync_bn (bool, optional): [description]. Defaults to True.
    """

    def __init__(self,
                 num_features,
                 input_dim,
                 bn_eps=1e-5,
                 sn_eps=1e-6,
                 momentum=0.1,
                 label_input=False,
                 with_spectral_norm=True,
                 auto_sync_bn=True):
        super().__init__()
        assert num_features > 0 and input_dim > 0
        # Prepare gain and bias layers
        if not label_input:
            self.gain = nn.Linear(input_dim, num_features, bias=False)
            self.bias = nn.Linear(input_dim, num_features, bias=False)
        else:
            self.gain = nn.Embedding(input_dim, num_features)
            self.bias = nn.Embedding(input_dim, num_features)

        # please pay attention if shared_embedding is False
        if with_spectral_norm:
            self.gain = spectral_norm(self.gain, eps=sn_eps)
            self.bias = spectral_norm(self.bias, eps=sn_eps)

        self.bn = nn.BatchNorm2d(
            num_features, eps=bn_eps, momentum=momentum, affine=False)

        if auto_sync_bn and dist.is_initialized():
            self.bn = SyncBatchNorm.convert_sync_batchnorm(self.bn)

    def forward(self, x, y):
        """[summary]

        Args:
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Calculate class-conditional gains and biases
        gain = (1. + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias


@MODULES.register_module()
class SelfAttentionBlock(nn.Module):
    """[summary]

    Args:
        in_channels ([type]): [description]
        with_spectral_norm (bool, optional): [description]. Defaults to
            True.
        sn_eps ([type], optional): [description]. Defaults to 1e-6.
    """

    def __init__(self, in_channels, with_spectral_norm=True, sn_eps=1e-6):
        super(SelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.theta = SNConvModule(
            self.in_channels,
            self.in_channels // 8,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))
        self.phi = SNConvModule(
            self.in_channels,
            self.in_channels // 8,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))
        self.g = SNConvModule(
            self.in_channels,
            self.in_channels // 2,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))
        self.o = SNConvModule(
            self.in_channels // 2,
            self.in_channels,
            kernel_size=1,
            padding=0,
            bias=False,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))
        # Learnable gain parameter
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        # TODO:
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.in_channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.in_channels // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(-1, self.in_channels // 2,
                                                    x.shape[2], x.shape[3]))
        return self.gamma * o + x


@MODULES.register_module()
class BigGANDiscResBlock(nn.Module):
    # TODO:
    """[summary]

    Args:
        in_channels ([type]): [description]
        out_channels ([type]): [description]
        conv_cfg ([type], optional): [description]. Defaults to dict(type=
            'Conv2d').
        shortcut_cfg ([type], optional): [description]. Defaults to dict(
                type='Conv2d').
        act_cfg ([type], optional): [description]. Defaults to dict(type=
            'ReLU', inplace=False).
        sn_eps ([type], optional): [description]. Defaults to 1e-6.
        with_downsample (bool, optional): [description]. Defaults to True.
        with_spectral_norm (bool, optional): [description]. Defaults to
            True.
        head_block (bool, optional): [description]. Defaults to False.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_cfg=dict(type='Conv2d'),
        shortcut_cfg=dict(type='Conv2d'),
        act_cfg=dict(type='ReLU', inplace=False),
        sn_eps=1e-6,
        with_downsample=True,
        with_spectral_norm=True,
        head_block=False,
    ):
        super().__init__()
        self.activation = build_activation_layer(act_cfg)
        self.with_downsample = with_downsample
        self.head_block = head_block
        if self.with_downsample:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.learnable_sc = in_channels != out_channels or self.with_downsample
        if self.learnable_sc:
            self.shortcut = SNConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm,
                spectral_norm_cfg=dict(eps=sn_eps))

        self.conv1 = SNConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))

        self.conv2 = SNConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps))

    def forward_sc(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.head_block:
            if self.with_downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.shortcut(x)
        else:
            if self.learnable_sc:
                x = self.shortcut(x)
            if self.with_downsample:
                x = self.downsample(x)
        return x

    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.head_block:
            x0 = x
        else:
            x0 = self.activation(x)
        x0 = self.conv1(x0)
        x0 = self.activation(x0)
        x0 = self.conv2(x0)
        if self.with_downsample:
            x0 = self.downsample(x0)
        x1 = self.forward_sc(x)
        return x0 + x1
