from copy import deepcopy

import numpy as np
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, build_norm_layer,
                      build_upsample_layer, constant_init, xavier_init)

from mmgen.models.builder import MODULES
from mmgen.utils import check_dist_init


@MODULES.register_module()
class SNGANGenResBlock(nn.Module):
    """ResBlock used in Generator of SNGAN / Proj-GAN.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        hidden_channels (int, optional): Input channels of the second Conv
            layer of the block. If ``None`` is given, would be set as
            ``out_channels``. Defualt to None.
        num_classes (int, optional): Number of classes would like to generate.
            This argument would pass to norm layers and influence the structure
            and behavior of the normalization process. Default to 0.
        use_cbn (bool, optional): Whether use conditional normalization. This
            argument would pass to norm layers. Default to True.
        use_norm_affine (bool, optional): Whether use learnable affine
            parameters in norm operation when cbn is off. Default False.
        act_cfg (dict, optional): Config for activate function. Default
            to ``dict(type='ReLU')``.
        upsample_cfg (dict, optional): Config for the upsample method.
            Default to ``dict(type='nearest', scale_factor=2)``.
        upsample (bool, optional): Whether apply upsample operation in this
            module. Default to True.
        auto_sync_bn (bool, optional): Whether convert Batch Norm to
            Synchronized ones when Distributed training is on. Defualt to True.
        conv_cfg (dict | None): Config for conv blocks of this module. If pass
            ``None``, would use ``_default_conv_cfg``. Default to ``None``.
        with_spectral_norm (bool, optional): Whether use spectral norm for
            conv blocks and norm layers. Default to True.
        norm_eps (float, optional): ``eps`` used in norm layers.
            Default to 1e-4.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
    """

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)

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
                 with_spectral_norm=False,
                 norm_eps=1e-4,
                 init_cfg=dict(type='BigGAN')):

        super().__init__()
        self.learnable_sc = in_channels != out_channels or upsample
        self.with_upsample = upsample
        self.init_type = init_cfg.get('type', None)

        self.activate = build_activation_layer(act_cfg)
        hidden_channels = out_channels if hidden_channels is None \
            else hidden_channels

        if self.with_upsample:
            self.upsample = build_upsample_layer(upsample_cfg)

        self.conv_cfg = deepcopy(self._default_conv_cfg)
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)
        self.conv_cfg.setdefault('with_spectral_norm', with_spectral_norm)

        self.conv_1 = ConvModule(in_channels, hidden_channels, **self.conv_cfg)
        self.conv_2 = ConvModule(hidden_channels, out_channels,
                                 **self.conv_cfg)

        self.norm_1 = SNConditionNorm(in_channels, num_classes, use_cbn,
                                      norm_cfg, use_norm_affine, auto_sync_bn,
                                      norm_eps, init_cfg)
        self.norm_2 = SNConditionNorm(hidden_channels, num_classes, use_cbn,
                                      norm_cfg, use_norm_affine, auto_sync_bn,
                                      norm_eps, init_cfg)

        if self.learnable_sc:
            # use hyperparameters-fixed shortcut here
            self.shortcut = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm)
        self.init_weights()

    def forward(self, x, y=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            y (Tensor): Input label with shape (n, ).
                Default None.

        Returns:
            Tensor: Forward results.
        """

        out = self.norm_1(x, y)
        out = self.activate(out)
        if self.with_upsample:
            out = self.upsample(out)
        out = self.conv_1(out)

        out = self.norm_2(out, y)
        out = self.activate(out)
        out = self.conv_2(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x):
        out = x
        if self.learnable_sc:
            if self.with_upsample:
                out = self.upsample(out)
            out = self.shortcut(out)
        return out

    def init_weights(self):
        """Initialize weights for the model."""
        if self.init_type == 'BigGAN':
            nn.init.orthogonal_(self.conv_1.conv.weight)
            nn.init.orthogonal_(self.conv_2.conv.weight)
            if self.learnable_sc:
                nn.init.orthogonal_(self.shortcut.conv.weight)
        else:
            xavier_init(self.conv_1, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.conv_2, gain=np.sqrt(2), distribution='uniform')
            if self.learnable_sc:
                xavier_init(self.shortcut, gain=1, distribution='uniform')


@MODULES.register_module()
class SNGANDiscResBlock(nn.Module):
    """resblock used in discriminator of sngan / proj-gan.

    args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        hidden_channels (int, optional): input channels of the second conv
            layer of the block. if ``none`` is given, would be set as
            ``out_channels``. defualt to none.
        downsample (bool, optional): whether apply downsample operation in this
            module.  default to false.
        with_spectral_norm (bool, optional): whether use spectral norm for
            conv blocks and norm layers. default to true.
        act_cfg (dict, optional): config for activate function. default
            to ``dict(type='relu')``.
        conv_cfg (dict | none): config for conv blocks of this module. if pass
            ``none``, would use ``_default_conv_cfg``. default to ``none``.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
    """

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 with_spectral_norm=True,
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 init_cfg=dict(type='BigGAN')):

        super().__init__()
        hidden_channels = in_channels if hidden_channels is None \
            else hidden_channels
        self.with_downsample = downsample
        self.init_type = init_cfg.get('type', None)

        self.conv_cfg = deepcopy(self._default_conv_cfg)
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)
        self.conv_cfg.setdefault('with_spectral_norm', with_spectral_norm)

        self.activate = build_activation_layer(act_cfg)

        self.conv_1 = ConvModule(in_channels, hidden_channels, **self.conv_cfg)
        self.conv_2 = ConvModule(hidden_channels, out_channels,
                                 **self.conv_cfg)

        if self.with_downsample:
            self.downsample = nn.AvgPool2d(2, 2)

        self.learnable_sc = in_channels != out_channels or downsample
        if self.learnable_sc:
            # use hyperparameters-fixed shortcut here
            self.shortcut = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=None,
                with_spectral_norm=with_spectral_norm)
        self.init_weights()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        out = self.activate(x)
        out = self.conv_1(out)
        out = self.activate(out)
        out = self.conv_2(out)
        if self.with_downsample:
            out = self.downsample(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x):
        out = x
        if self.learnable_sc:
            out = self.shortcut(out)
            if self.with_downsample:
                out = self.downsample(out)
        return out

    def init_weights(self):
        if self.init_type == 'BigGAN':
            nn.init.orthogonal_(self.conv_1.conv.weight)
            nn.init.orthogonal_(self.conv_2.conv.weight)
            if self.learnable_sc:
                nn.init.orthogonal_(self.shortcut.conv.weight)
        else:
            xavier_init(self.conv_1, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.conv_2, gain=np.sqrt(2), distribution='uniform')
            if self.learnable_sc:
                xavier_init(self.shortcut, gain=1, distribution='uniform')


@MODULES.register_module()
class SNGANDiscHeadResBlock(nn.Module):
    """The first ResBlock used in discriminator of sngan / proj-gan. Compared
    to ``SNGANDisResBlock``, this module has a different forward order.

    args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        downsample (bool, optional): whether apply downsample operation in this
            module.  default to false.
        conv_cfg (dict | none): config for conv blocks of this module. if pass
            ``none``, would use ``_default_conv_cfg``. default to ``none``.
        with_spectral_norm (bool, optional): whether use spectral norm for
            conv blocks and norm layers. default to true.
        act_cfg (dict, optional): config for activate function. default
            to ``dict(type='relu')``.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
    """

    _default_conv_cfg = dict(kernel_size=3, stride=1, padding=1, act_cfg=None)

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 with_spectral_norm=True,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(type='BigGAN')):

        super().__init__()

        self.init_type = init_cfg.get('type', None)
        self.conv_cfg = deepcopy(self._default_conv_cfg)
        if conv_cfg is not None:
            self.conv_cfg.update(conv_cfg)
        self.conv_cfg.setdefault('with_spectral_norm', with_spectral_norm)

        self.activate = build_activation_layer(act_cfg)

        self.conv_1 = ConvModule(in_channels, out_channels, **self.conv_cfg)
        self.conv_2 = ConvModule(out_channels, out_channels, **self.conv_cfg)

        self.downsample = nn.AvgPool2d(2, 2)

        # use hyperparameters-fixed shortcut here
        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            with_spectral_norm=with_spectral_norm)
        self.init_weights()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        out = self.conv_1(x)
        out = self.activate(out)
        out = self.conv_2(out)
        out = self.downsample(out)

        shortcut = self.forward_shortcut(x)
        return out + shortcut

    def forward_shortcut(self, x):
        out = self.downsample(x)
        out = self.shortcut(out)
        return out

    def init_weights(self):
        if self.init_type == 'BigGAN':
            nn.init.orthogonal_(self.conv_1.conv.weight)
            nn.init.orthogonal_(self.conv_2.conv.weight)
            nn.init.orthogonal_(self.shortcut.conv.weight)
        else:
            xavier_init(self.conv_1, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.conv_2, gain=np.sqrt(2), distribution='uniform')
            xavier_init(self.shortcut, gain=1, distribution='uniform')


@MODULES.register_module()
class SNConditionNorm(nn.Module):
    """Conditional Normalization for SNGAN / Proj-GAN. The implementation
    refers to.

    https://github.com/pfnet-research/sngan_projection/blob/master/source/links/conditional_batch_normalization.py  # noda

    and

    https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/utils/model_ops.py  # noqa

    Args:
        in_channels (int): Number of the channels of the input feature map.
        num_classes (int): Number of the classes in the dataset. If ``use_cbn``
            is True, ``num_classes`` must larger than 0.
        use_cbn (bool, optional): Whether use conditional normalization. If
            ``use_cbn`` is True, two embedding layers would be used to mapping
            label to weight and bias used in normalization process.
        norm_cfg (dict, optional): Config for normalization method. Default
            to ``dict(type='BN')``.
        auto_sync_bn (bool, optional): Whether convert Batch Norm to
            Synchronized ones when Distributed training is on. Defualt to True.
        norm_eps (float, optional): eps for Normalization layers (both conditional
            and non-conditional ones). Default to `1e-4`.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 use_cbn=True,
                 norm_cfg=dict(type='BN'),
                 cbn_norm_affine=False,
                 auto_sync_bn=True,
                 norm_eps=1e-4,
                 init_cfg=dict(type='BigGAN')):
        super().__init__()
        self.use_cbn = use_cbn
        self.init_type = init_cfg.get('type', None)

        norm_cfg = deepcopy(norm_cfg)
        norm_type = norm_cfg['type']

        if norm_type not in ['IN', 'BN', 'SyncBN']:
            raise ValueError('Only support `IN` (InstanceNorm), '
                             '`BN` (BatcnNorm) and `SyncBN` for '
                             'Class-conditional bn. '
                             f'Receive norm_type: {norm_type}')

        if self.use_cbn:
            norm_cfg.setdefault('affine', cbn_norm_affine)
        norm_cfg.setdefault('eps', norm_eps)
        if check_dist_init() and auto_sync_bn and norm_type == 'BN':
            norm_cfg['type'] = 'SyncBN'

        _, self.norm = build_norm_layer(norm_cfg, in_channels)

        if self.use_cbn:
            if num_classes <= 0:
                raise ValueError('`num_classes` must be larger '
                                 'than 0 with `use_cbn=True`')
            self.weight_embedding = nn.Embedding(num_classes, in_channels)
            self.bias_embedding = nn.Embedding(num_classes, in_channels)
            self.reweight_embedding = self.init_type == 'BigGAN'

        self.init_weights()

    def forward(self, x, y=None):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            y (Tensor, optional): Input label with shape (n, ).
                Default None.

        Returns:
            Tensor: Forward results.
        """

        out = self.norm(x)
        if self.use_cbn:
            weight = self.weight_embedding(y)[:, :, None, None]
            bias = self.bias_embedding(y)[:, :, None, None]
            if self.reweight_embedding:
                weight = weight + 1.
            out = out * weight + bias
        return out

    def init_weights(self):
        if self.use_cbn:
            if self.init_type == 'BigGAN':
                nn.init.orthogonal_(self.weight_embedding.weight)
                nn.init.orthogonal_(self.bias_embedding.weight)
            else:
                constant_init(self.weight_embedding, 1)
                constant_init(self.bias_embedding, 0)
