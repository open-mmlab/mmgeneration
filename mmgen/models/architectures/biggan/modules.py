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


@MODULES.register_module()
class BigGANGenResBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim_after_concat,
                 conv_cfg=dict(type='Conv2d'),
                 shortcut_cfg=dict(type='Conv2d'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 with_spectral_norm=True,
                 shared_embedding=True,
                 auto_sync_bn=True):

        super().__init__()
        self.activation = build_activation_layer(act_cfg)
        self.upsample_cfg = upsample_cfg
        self.with_upsample = upsample_cfg is not None
        if self.with_upsample:
            self.upsample_layer = build_upsample_layer(self.upsample_cfg)
        self.learnable_sc = in_channels != out_channels or self.with_upsample
        if self.learnable_sc:
            self.shortcut = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm)
        # Here in_channels of BigGANGenResBlock equal to output_dim of ccbn
        self.bn1 = BigGANConditionBatchNorm(
            in_channels,
            dim_after_concat,
            shared_embedding=shared_embedding,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)
        self.bn2 = BigGANConditionBatchNorm(
            out_channels,
            dim_after_concat,
            shared_embedding=shared_embedding,
            with_spectral_norm=with_spectral_norm,
            auto_sync_bn=auto_sync_bn)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)

        self.conv2 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)

    def forward(self, x, y):
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
class BigGANConditionBatchNorm(nn.Module):

    def __init__(self,
                 num_features,
                 input_dim,
                 bn_eps=1e-5,
                 sn_eps=1e-6,
                 momentum=0.1,
                 shared_embedding=True,
                 with_spectral_norm=True,
                 auto_sync_bn=True):
        super(BigGANConditionBatchNorm, self).__init__()
        assert num_features > 0 and input_dim > 0
        # Prepare gain and bias layers
        if shared_embedding:
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
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias


@MODULES.register_module()
class SelfAttentionBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 conv_cfg=dict(type='Conv2d'),
                 with_spectral_norm=True):
        super(SelfAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.theta = ConvModule(
            self.in_channels,
            self.in_channels // 8,
            kernel_size=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)
        self.phi = ConvModule(
            self.in_channels,
            self.in_channels // 8,
            kernel_size=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)
        self.g = ConvModule(
            self.in_channels,
            self.in_channels // 2,
            kernel_size=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)
        self.o = ConvModule(
            self.in_channels // 2,
            self.in_channels,
            kernel_size=1,
            padding=0,
            bias=False,
            conv_cfg=conv_cfg,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)
        # Learnable gain parameter
        self.gamma = Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
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
