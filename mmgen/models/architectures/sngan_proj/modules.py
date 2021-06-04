from copy import deepcopy

import numpy as np
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_norm_layer,
                      build_upsample_layer, constant_init, xavier_init)

from mmgen.models.builder import MODULES
from mmgen.utils import check_dist_init


@MODULES.register_module()
class SNGANGenResBlock(nn.Module):

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)
    _default_shortcut_cfg = dict(
        kernel_size=1, stride=1, padding=0, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 num_classes=0,
                 use_cbn=True,
                 use_norm_affine=False,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='BN'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 upsample=True,
                 auto_sync_bn=True,
                 conv_cfg=None,
                 shortcut_cfg=None):

        super().__init__()
        self.learnable_sc = in_channels != out_channels or upsample
        self.with_upsample = upsample

        self.activateion = build_activation_layer(act_cfg)
        hidden_channels = out_channels if hidden_channels is None \
            else hidden_channels

        if self.with_upsample:
            self.upsample = build_upsample_layer(upsample_cfg)

        self.conv_cfg = self._default_conv_cfg
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)

        self.shortcut_cfg = self._default_shortcut_cfg
        if shortcut_cfg is not None:
            self.shortcut_cfg.update(shortcut_cfg)

        conv_blocks = [
            ConvModule(in_channels, hidden_channels, **self.conv_cfg),
            ConvModule(hidden_channels, out_channels, **self.conv_cfg)
        ]
        self.conv_blocks = nn.ModuleList(conv_blocks)

        norm_blocks = [
            SNcBatchNorm(in_channels, num_classes, use_cbn, norm_cfg,
                         use_norm_affine, auto_sync_bn),
            SNcBatchNorm(hidden_channels, num_classes, use_cbn, norm_cfg,
                         use_norm_affine, auto_sync_bn)
        ]
        self.norm_blocks = nn.ModuleList(norm_blocks)

        if self.learnable_sc:
            self.shortcut = ConvModule(in_channels, out_channels,
                                       **self.shortcut_cfg)
        self._init_weight()

    def forward(self, x, y=None):
        out = self.norm_blocks[0](x, y)
        out = self.activateion(out)
        if self.with_upsample:
            out = self.upsample(out)
        out = self.conv_blocks[0](out)

        out = self.norm_blocks[1](out, y)
        out = self.activateion(out)
        out = self.conv_blocks[1](out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x):
        out = x
        if self.learnable_sc:
            if self.upsample:
                out = self.upsample(out)
            out = self.shortcut(out)
        return out

    def _init_weight(self):
        """Initialize weights for the model."""
        xavier_init(self.conv_blocks, gain=np.sqrt(2), distribution='uniform')
        if self.learnable_sc:
            xavier_init(self.shortcut, gain=1, distribution='uniform')


@MODULES.register_module()
class SNGANDiscResBlock(nn.Module):

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)
    _default_shortcut_cfg = dict(
        kernel_size=1, stride=1, padding=0, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 with_spectral_norm=True,
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 shortcut_cfg=None):

        super().__init__()
        hidden_channels = in_channels if hidden_channels is None \
            else hidden_channels
        self.with_downsample = downsample

        self.conv_cfg = self._default_conv_cfg
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)
        self.conv_cfg['with_spectral_norm'] = with_spectral_norm

        self.shortcut_cfg = self._default_shortcut_cfg
        if shortcut_cfg is not None:
            self.shortcut_cfg.update(conv_cfg)
        self.shortcut_cfg['with_spectral_norm'] = with_spectral_norm

        self.activate = build_activation_layer(act_cfg)

        conv_blocks = [
            ConvModule(in_channels, hidden_channels, **self.conv_cfg),
            ConvModule(hidden_channels, out_channels, **self.conv_cfg)
        ]
        self.conv_blocks = nn.ModuleList(conv_blocks)

        if self.with_downsample:
            self.downsample = nn.AvgPool2d(2, 2)

        self.learnable_sc = in_channels != out_channels or downsample
        if self.learnable_sc:
            self.shortcut = ConvModule(in_channels, out_channels,
                                       **self.shortcut_cfg)

    def forward(self, x):
        out = self.activate(x)
        out = self.conv_blocks[0](out)
        out = self.activate(out)
        out = self.conv_blocks[1](out)
        if self.with_downsample:
            out = self.downsample(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x):
        out = x
        if self.learnable_sc:
            out = self.shortcut(out)
            if self.downsample:
                out = self.downsample(out)
        return out

    def _weight_init(self):
        xavier_init(self.conv, gain=np.sqrt(2), distribution='uniform')
        if self.learnable_sc:
            xavier_init(self.shortcut, gain=1, distribution='uniform')


@MODULES.register_module()
class SNGANDiscHeadResBlock(nn.Module):

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)
    _default_shortcut_cfg = dict(
        kernel_size=1, stride=1, padding=0, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 shortcut_cfg=None,
                 with_spectral_norm=True,
                 act_cfg=dict(type='ReLU')):

        super().__init__()

        self.conv_cfg = self._default_conv_cfg
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)
        self.conv_cfg['with_spectral_norm'] = with_spectral_norm

        self.shortcut_cfg = self._default_shortcut_cfg
        if shortcut_cfg is not None:
            self.shortcut_cfg.update(shortcut_cfg)
        self.shortcut_cfg['with_spectral_norm'] = with_spectral_norm

        self.activate = build_activation_layer(act_cfg)
        conv_blocks = [
            ConvModule(in_channels, out_channels, **self.conv_cfg),
            ConvModule(out_channels, out_channels, **self.conv_cfg)
        ]
        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.downsample = nn.AvgPool2d(2, 2)

        self.shortcut = ConvModule(in_channels, out_channels,
                                   **self.shortcut_cfg)

    def forward(self, x):
        out = self.conv_blocks[0](x)
        out = self.activate(out)
        out = self.conv_blocks[1](out)
        out = self.downsample(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x):
        out = self.downsample(x)
        out = self.shortcut(out)
        return out

    def _weight_init(self):
        xavier_init(self.conv, gain=np.sqrt(2), distribution='uniform')
        if self.learnable_sc:
            xavier_init(self.shortcut, gain=1, distribution='uniform')


@MODULES.register_module()
class SNcBatchNorm(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 use_cbn=True,
                 norm_cfg=dict(type='BN'),
                 use_norm_affine=False,
                 auto_sync_bn=True):
        super().__init__()
        norm_cfg = deepcopy(norm_cfg)
        norm_type = norm_cfg['type']

        if norm_type not in ['IN', 'BN', 'syncBN']:
            raise ValueError('Only support `IN` (InstanceNorm), '
                             '`BN` (BatcnNorm) and `syncBN` for '
                             'Class-conditional bn. '
                             'Receive norm_type: {}'.format(norm_type))
        norm_cfg['affine'] = use_norm_affine
        _, self.norm = build_norm_layer(norm_cfg, in_channels)

        if auto_sync_bn and check_dist_init():
            from torch.nn.modules.batchnorm import SyncBatchNorm
            self.norm = SyncBatchNorm.convert_sync_batchnorm(self.norm)

        self.use_cbn = use_cbn
        if use_cbn:
            if num_classes <= 0:
                raise ValueError('`num_classes` must be larger '
                                 'than 0 with `use_cbn=True`')
            self.weight_embedding = nn.Embedding(num_classes, in_channels)
            self.biase_embedding = nn.Embedding(num_classes, in_channels)
        self._init_weight()

    def forward(self, x, y=None):
        out = self.norm(x)
        if self.use_cbn:
            weight = self.weight_embedding(y).view(y.size(0), -1, 1, 1)
            bias = self.biase_embedding(y).view(y.size(0), -1, 1, 1)
            out = out * weight / bias
        return out

    def _init_weight(self):
        if self.use_cbn:
            constant_init(self.weight_embedding, 1)
            constant_init(self.biase_embedding, 0)
