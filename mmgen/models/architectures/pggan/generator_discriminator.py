# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.upsample import build_upsample_layer

from mmgen.models.builder import MODULES
from ..common import get_module_device
from .modules import (EqualizedLRConvDownModule, EqualizedLRConvModule,
                      EqualizedLRConvUpModule, MiniBatchStddevLayer,
                      PGGANDecisionHead, PGGANNoiseTo2DFeat)


@MODULES.register_module()
class PGGANGenerator(nn.Module):
    """Generator for PGGAN.

    Args:
        noise_size (int): Size of the input noise vector.
        out_scale (int): Output scale for the generated image.
        label_size (int, optional): Size of the label vector.
            Defaults to 0.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this
            number. Defaults to 8192.
        channel_decay (float, optional): Decay for channels of feature maps.
            Defaults to 1.0.
        max_channels (int, optional): Maximum channels for the feature
            maps in the generator block. Defaults to 512.
        fused_upconv (bool, optional): Whether use fused upconv.
            Defaults to True.
        conv_module_cfg (dict, optional): Config for the convolution
            module used in this generator. Defaults to None.
        fused_upconv_cfg (dict, optional): Config for the fused upconv
            module used in this generator. Defaults to None.
        upsample_cfg (dict, optional): Config for the upsampling operation.
            Defaults to None.
    """
    _default_fused_upconv_cfg = dict(
        conv_cfg=dict(type='deconv'),
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=dict(type='PixelNorm'),
        order=('conv', 'act', 'norm'))
    _default_conv_module_cfg = dict(
        conv_cfg=None,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=dict(type='PixelNorm'),
        order=('conv', 'act', 'norm'))

    _default_upsample_cfg = dict(type='nearest', scale_factor=2)

    def __init__(self,
                 noise_size,
                 out_scale,
                 label_size=0,
                 base_channels=8192,
                 channel_decay=1.,
                 max_channels=512,
                 fused_upconv=True,
                 conv_module_cfg=None,
                 fused_upconv_cfg=None,
                 upsample_cfg=None):
        super().__init__()
        self.noise_size = noise_size if noise_size else min(
            base_channels, max_channels)
        self.out_scale = out_scale
        self.out_log2_scale = int(np.log2(out_scale))
        # sanity check for the output scale
        assert out_scale == 2**self.out_log2_scale and out_scale >= 4
        self.label_size = label_size
        self.base_channels = base_channels
        self.channel_decay = channel_decay
        self.max_channels = max_channels
        self.fused_upconv = fused_upconv

        # set conv cfg
        self.conv_module_cfg = deepcopy(self._default_conv_module_cfg)
        # update with customized config
        if conv_module_cfg:
            self.conv_module_cfg.update(conv_module_cfg)

        if self.fused_upconv:
            self.fused_upconv_cfg = deepcopy(self._default_fused_upconv_cfg)
            # update with customized config
            if fused_upconv_cfg:
                self.fused_upconv_cfg.update(fused_upconv_cfg)

        self.upsample_cfg = deepcopy(self._default_upsample_cfg)
        if upsample_cfg is not None:
            self.upsample_cfg.update(upsample_cfg)

        self.noise2feat = PGGANNoiseTo2DFeat(noise_size + label_size,
                                             self._num_out_channels(1))

        self.torgb_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for s in range(2, self.out_log2_scale + 1):
            in_ch = self._num_out_channels(
                s - 1) if s == 2 else self._num_out_channels(s - 2)
            # setup torgb layers
            self.torgb_layers.append(
                self._get_torgb_layer(self._num_out_channels(s - 1)))
            # setup upconv or conv blocks
            self.conv_blocks.extend(self._get_upconv_block(in_ch, s))

        # build upsample layer for residual path
        self.upsample_layer = build_upsample_layer(self.upsample_cfg)

    def _get_torgb_layer(self, in_channels):
        return EqualizedLRConvModule(
            in_channels,
            3,
            kernel_size=1,
            stride=1,
            equalized_lr_cfg=dict(gain=1),
            bias=True,
            norm_cfg=None,
            act_cfg=None)

    def _num_out_channels(self, log_scale):
        return min(
            int(self.base_channels / (2.0**(log_scale * self.channel_decay))),
            self.max_channels)

    def _get_upconv_block(self, in_channels, log_scale):
        modules = []
        # start 4x4 scale
        if log_scale == 2:
            modules.append(
                EqualizedLRConvModule(in_channels,
                                      self._num_out_channels(log_scale - 1),
                                      **self.conv_module_cfg))
        # 8x8 --> 1024x1024 scales
        else:
            if self.fused_upconv:
                cfg_ = dict(upsample=dict(type='fused_nn'))
                cfg_.update(self.fused_upconv_cfg)
            else:
                cfg_ = dict(upsample=self.upsample_cfg)
                cfg_.update(self.conv_module_cfg)
            # up + conv
            modules.append(
                EqualizedLRConvUpModule(in_channels,
                                        self._num_out_channels(log_scale - 1),
                                        **cfg_))
            # refine conv
            modules.append(
                EqualizedLRConvModule(
                    self._num_out_channels(log_scale - 1),
                    self._num_out_channels(log_scale - 1),
                    **self.conv_module_cfg))

        return modules

    def forward(self,
                noise,
                label=None,
                num_batches=0,
                return_noise=False,
                transition_weight=1.,
                curr_scale=-1):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            label (Tensor, optional): Label vector with shape [N, C]. Defaults
                to None.
            num_batches (int, optional): The number of batch size. Defaults to
                0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            transition_weight (float, optional): The weight used in resolution
                transition. Defaults to 1.0.
            curr_scale (int, optional): The scale for the current inference or
                training. Defaults to -1.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img`` and
                ``noise_batch`` will be returned.
        """
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            assert noise.ndim == 2, ('The noise should be in shape of (n, c), '
                                     f'but got {noise.shape}')
            noise_batch = noise
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            # TODO: check pggan default noise type
            noise_batch = torch.randn((num_batches, self.noise_size))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))

        if label is not None:
            noise_batch = torch.cat(
                [noise_batch, label.to(noise_batch)], dim=1)

        # noise vector to 2D feature
        x = self.noise2feat(noise_batch)

        # build current computational graph
        curr_log2_scale = self.out_log2_scale if curr_scale < 0 else int(
            np.log2(curr_scale))

        # 4x4 scale
        x = self.conv_blocks[0](x)
        if curr_log2_scale <= 3:
            out_img = last_img = self.torgb_layers[0](x)

        # 8x8 and larger scales
        for s in range(3, curr_log2_scale + 1):
            x = self.conv_blocks[2 * s - 5](x)
            x = self.conv_blocks[2 * s - 4](x)
            if s + 1 == curr_log2_scale:
                last_img = self.torgb_layers[s - 2](x)
            elif s == curr_log2_scale:
                out_img = self.torgb_layers[s - 2](x)
                residual_img = self.upsample_layer(last_img)
                out_img = residual_img + transition_weight * (
                    out_img - residual_img)

        if return_noise:
            output = dict(
                fake_img=out_img, noise_batch=noise_batch, label=label)
            return output

        return out_img


@MODULES.register_module()
class PGGANDiscriminator(nn.Module):
    """Discriminator for PGGAN.

    Args:
        in_scale (int): The scale of the input image.
        label_size (int, optional): Size of the label vector. Defaults to
            0.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this
            number. Defaults to 8192.
        max_channels (int, optional): Maximum channels for the feature
            maps in the discriminator block. Defaults to 512.
        in_channels (int, optional): Number of channels in input images.
            Defaults to 3.
        channel_decay (float, optional): Decay for channels of feature
            maps. Defaults to 1.0.
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4).
        fused_convdown (bool, optional): Whether use fused downconv.
            Defaults to True.
        conv_module_cfg (dict, optional): Config for the convolution
            module used in this generator. Defaults to None.
        fused_convdown_cfg (dict, optional): Config for the fused downconv
            module used in this discriminator. Defaults to None.
        fromrgb_layer_cfg (dict, optional): Config for the fromrgb layer.
            Defaults to None.
        downsample_cfg (dict, optional): Config for the downsampling
            operation. Defaults to None.
    """
    _default_fromrgb_cfg = dict(
        conv_cfg=None,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=None,
        order=('conv', 'act', 'norm'))

    _default_conv_module_cfg = dict(
        kernel_size=3,
        padding=1,
        stride=1,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

    _default_convdown_cfg = dict(
        kernel_size=3,
        padding=1,
        stride=2,
        norm_cfg=None,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2))

    def __init__(self,
                 in_scale,
                 label_size=0,
                 base_channels=8192,
                 max_channels=512,
                 in_channels=3,
                 channel_decay=1.0,
                 mbstd_cfg=dict(group_size=4),
                 fused_convdown=True,
                 conv_module_cfg=None,
                 fused_convdown_cfg=None,
                 fromrgb_layer_cfg=None,
                 downsample_cfg=None):
        super().__init__()
        self.in_scale = in_scale
        self.in_log2_scale = int(np.log2(self.in_scale))
        self.label_size = label_size
        self.base_channels = base_channels
        self.max_channels = max_channels
        self.in_channels = in_channels
        self.channel_decay = channel_decay
        self.with_mbstd = mbstd_cfg is not None

        self.fused_convdown = fused_convdown

        self.conv_module_cfg = deepcopy(self._default_conv_module_cfg)
        if conv_module_cfg is not None:
            self.conv_module_cfg.update(conv_module_cfg)

        if self.fused_convdown:
            self.fused_convdown_cfg = deepcopy(self._default_convdown_cfg)
            if fused_convdown_cfg is not None:
                self.fused_convdown_cfg.update(fused_convdown_cfg)

        self.fromrgb_layer_cfg = deepcopy(self._default_fromrgb_cfg)
        if fromrgb_layer_cfg:
            self.fromrgb_layer_cfg.update(fromrgb_layer_cfg)

        # setup conv blocks
        self.conv_blocks = nn.ModuleList()
        self.fromrgb_layers = nn.ModuleList()

        for s in range(2, self.in_log2_scale + 1):
            self.fromrgb_layers.append(
                self._get_fromrgb_layer(self.in_channels, s))

            self.conv_blocks.extend(
                self._get_convdown_block(self._num_out_channels(s - 1), s))

        # setup downsample layer
        self.downsample_cfg = deepcopy(downsample_cfg)
        if self.downsample_cfg is None or self.downsample_cfg.get(
                'type', None) == 'avgpool':
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        elif self.downsample_cfg.get('type', None) in ['nearest', 'bilinear']:
            self.downsample = partial(
                F.interpolate,
                mode=self.downsample_cfg.pop('type'),
                **self.downsample_cfg)
        else:
            raise NotImplementedError(
                'We have not supported the downsampling with type'
                f' {downsample_cfg}.')

        # setup minibatch stddev layer
        if self.with_mbstd:
            self.mbstd_layer = MiniBatchStddevLayer(**mbstd_cfg)
            # minibatch stddev layer will concatenate an additional feature map
            # in channel dimension.
            decision_in_channels = self._num_out_channels(1) * 16 + 16
        else:
            decision_in_channels = self._num_out_channels(1) * 16

        # setup decision layer
        self.decision = PGGANDecisionHead(decision_in_channels,
                                          self._num_out_channels(0),
                                          1 + self.label_size)

    def _num_out_channels(self, log_scale):
        return min(
            int(self.base_channels / (2.0**(log_scale * self.channel_decay))),
            self.max_channels)

    def _get_fromrgb_layer(self, in_channels, log2_scale):
        return EqualizedLRConvModule(in_channels,
                                     self._num_out_channels(log2_scale - 1),
                                     **self.fromrgb_layer_cfg)

    def _get_convdown_block(self, in_channels, log2_scale):
        modules = []
        if log2_scale == 2:
            modules.append(
                EqualizedLRConvModule(in_channels,
                                      self._num_out_channels(log2_scale - 1),
                                      **self.conv_module_cfg))
        else:
            modules.append(
                EqualizedLRConvModule(in_channels,
                                      self._num_out_channels(log2_scale - 1),
                                      **self.conv_module_cfg))

            if self.fused_convdown:
                cfg_ = dict(downsample=dict(type='fused_pool'))
                cfg_.update(self.fused_convdown_cfg)
            else:
                cfg_ = dict(downsample=self.downsample)
                cfg_.update(self.conv_module_cfg)
            modules.append(
                EqualizedLRConvDownModule(
                    self._num_out_channels(log2_scale - 1),
                    self._num_out_channels(log2_scale - 2), **cfg_))
        return modules

    def forward(self, x, transition_weight=1., curr_scale=-1):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.
            transition_weight (float, optional): The weight used in resolution
                transition. Defaults to 1.0.
            curr_scale (int, optional): The scale for the current inference or
                training. Defaults to -1.

        Returns:
            Tensor: Predict score for the input image.
        """
        curr_log2_scale = self.in_log2_scale if curr_scale < 4 else int(
            np.log2(curr_scale))

        original_img = x

        x = self.fromrgb_layers[curr_log2_scale - 2](x)

        for s in range(curr_log2_scale, 2, -1):
            x = self.conv_blocks[2 * s - 5](x)
            x = self.conv_blocks[2 * s - 4](x)
            if s == curr_log2_scale:
                img_down = self.downsample(original_img)
                y = self.fromrgb_layers[curr_log2_scale - 3](img_down)
                x = y + transition_weight * (x - y)

        if self.with_mbstd:
            x = self.mbstd_layer(x)

        x = self.decision(x)

        if self.label_size > 0:
            return x[:, :1], x[:, 1:]

        return x
