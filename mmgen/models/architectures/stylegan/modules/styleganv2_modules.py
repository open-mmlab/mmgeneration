# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.activation import build_activation_layer
from mmcv.ops.fused_bias_leakyrelu import (FusedBiasLeakyReLU,
                                           fused_bias_leakyrelu)
from mmcv.ops.upfirdn2d import upfirdn2d
from mmcv.runner.dist_utils import get_dist_info

from mmgen.core.runners.fp16_utils import auto_fp16
from mmgen.models.architectures.pggan import (EqualizedLRConvModule,
                                              EqualizedLRLinearModule,
                                              equalized_lr)
from mmgen.models.common import AllGatherLayer
from mmgen.ops import conv2d, conv_transpose2d


class _FusedBiasLeakyReLU(FusedBiasLeakyReLU):
    """Wrap FusedBiasLeakyReLU to support FP16 training."""

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        return fused_bias_leakyrelu(x, self.bias.to(x.dtype),
                                    self.negative_slope, self.scale)


class EqualLinearActModule(nn.Module):
    """Equalized LR Linear Module with Activation Layer.

    This module is modified from ``EqualizedLRLinearModule`` defined in PGGAN.
    The major features updated in this module is adding support for activation
    layers used in StyleGAN2.

    Args:
        equalized_lr_cfg (dict | None, optional): Config for equalized lr.
            Defaults to dict(gain=1., lr_mul=1.).
        bias (bool, optional): Whether to use bias item. Defaults to True.
        bias_init (float, optional): The value for bias initialization.
            Defaults to ``0.``.
        act_cfg (dict | None, optional): Config for activation layer.
            Defaults to None.
    """

    def __init__(self,
                 *args,
                 equalized_lr_cfg=dict(gain=1., lr_mul=1.),
                 bias=True,
                 bias_init=0.,
                 act_cfg=None,
                 **kwargs):
        super().__init__()
        self.with_activation = act_cfg is not None
        # w/o bias in linear layer
        self.linear = EqualizedLRLinearModule(
            *args, bias=False, equalized_lr_cfg=equalized_lr_cfg, **kwargs)

        if equalized_lr_cfg is not None:
            self.lr_mul = equalized_lr_cfg.get('lr_mul', 1.)
        else:
            self.lr_mul = 1.

        # define bias outside linear layer
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.linear.out_features).fill_(bias_init))
        else:
            self.bias = None

        if self.with_activation:
            act_cfg = deepcopy(act_cfg)
            if act_cfg['type'] == 'fused_bias':
                self.act_type = act_cfg.pop('type')
                assert self.bias is not None
                self.activate = partial(fused_bias_leakyrelu, **act_cfg)
            else:
                self.act_type = 'normal'
                self.activate = build_activation_layer(act_cfg)
        else:
            self.act_type = None

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        if x.ndim >= 3:
            x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        if self.with_activation and self.act_type == 'fused_bias':
            x = self.activate(x, self.bias * self.lr_mul)
        elif self.bias is not None and self.with_activation:
            x = self.activate(x + self.bias * self.lr_mul)
        elif self.bias is not None:
            x = x + self.bias * self.lr_mul
        elif self.with_activation:
            x = self.activate(x)

        return x


def _make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class UpsampleUpFIRDn(nn.Module):
    """UpFIRDn for Upsampling.

    This module is used in the ``to_rgb`` layers in StyleGAN2 for upsampling
    the images.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        factor (int, optional): Upsampling factor. Defaults to 2.
    """

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = _make_kernel(kernel) * (factor**2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        out = upfirdn2d(
            x, self.kernel.to(x.dtype), up=self.factor, down=1, pad=self.pad)

        return out


class DownsampleUpFIRDn(nn.Module):
    """UpFIRDn for Downsampling.

    This module is mentioned in StyleGAN2 for dowampling the feature maps.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        factor (int, optional): Downsampling factor. Defaults to 2.
    """

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = _make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        """Forward function.

        Args:
            input (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        out = upfirdn2d(
            input,
            self.kernel.to(input.dtype),
            up=1,
            down=self.factor,
            pad=self.pad)

        return out


class Blur(nn.Module):
    """Blur module.

    This module is adopted rightly after upsampling operation in StyleGAN2.

    Args:
        kernel (Array): Blur kernel/filter used in UpFIRDn.
        pad (list[int]): Padding for features.
        upsample_factor (int, optional): Upsampling factor. Defaults to 1.
    """

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = _make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """

        # In Tero's implementation, he uses fp32
        return upfirdn2d(x, self.kernel.to(x.dtype), pad=self.pad)


class ModulatedConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2.

    This module implements the modulated convolution layers proposed in
    StyleGAN2. Details can be found in Analyzing and Improving the Image
    Quality of StyleGAN, CVPR2020.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 demodulate=True,
                 upsample=False,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 equalized_lr_cfg=dict(mode='fan_in', lr_mul=1., gain=1.),
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 eps=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        # sanity check for kernel size
        assert isinstance(self.kernel_size,
                          int) and (self.kernel_size >= 1
                                    and self.kernel_size % 2 == 1)
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps

        # build style modulation module
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg

        self.style_modulation = EqualLinearActModule(style_channels,
                                                     in_channels,
                                                     **style_mod_cfg)
        # set lr_mul for conv weight
        lr_mul_ = 1.
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get('lr_mul', 1.)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size,
                        kernel_size).div_(lr_mul_))

        # build blurry layer for upsampling
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)
        # build blurry layer for downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        # add equalized_lr hook for conv weight
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)

        self.padding = kernel_size // 2

    def forward(self, x, style):
        n, c, h, w = x.shape

        weight = self.weight
        # Pre-normalize inputs to avoid FP16 overflow.
        if x.dtype == torch.float16 and self.demodulate:
            weight = weight * (
                1 / np.sqrt(
                    self.in_channels * self.kernel_size * self.kernel_size) /
                weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True)
            )  # max_Ikk
            style = style / style.norm(
                float('inf'), dim=1, keepdim=True)  # max_I

        # process style code
        style = self.style_modulation(style).view(n, 1, c, 1,
                                                  1) + self.style_bias

        # combine weight and style
        weight = weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)

        weight = weight.view(n * self.out_channels, c, self.kernel_size,
                             self.kernel_size)

        weight = weight.to(x.dtype)
        if self.upsample:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size,
                                 self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.out_channels,
                                                    self.kernel_size,
                                                    self.kernel_size)
            x = conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)

        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])

        return x


class NoiseInjection(nn.Module):
    """Noise Injection Module.

    In StyleGAN2, they adopt this module to inject spatial random noise map in
    the generators.

    Args:
        noise_weight_init (float, optional): Initialization weight for noise
            injection. Defaults to ``0.``.
    """

    def __init__(self, noise_weight_init=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1).fill_(noise_weight_init))

    def forward(self, image, noise=None, return_noise=False):
        """Forward Function.

        Args:
            image (Tensor): Spatial features with a shape of (N, C, H, W).
            noise (Tensor, optional): Noises from the outside.
                Defaults to None.
            return_noise (bool, optional): Whether to return noise tensor.
                Defaults to False.

        Returns:
            Tensor: Output features.
        """
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        noise = noise.to(image.dtype)
        if return_noise:
            return image + self.weight.to(image.dtype) * noise, noise

        return image + self.weight.to(image.dtype) * noise


class ConstantInput(nn.Module):
    """Constant Input.

    In StyleGAN2, they substitute the original head noise input with such a
    constant input module.

    Args:
        channel (int): Channels for the constant input tensor.
        size (int, optional): Spatial size for the constant input.
            Defaults to 4.
    """

    def __init__(self, channel, size=4):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        elif mmcv.is_seq_of(size, int):
            assert len(
                size
            ) == 2, f'The length of size should be 2 but got {len(size)}'
        else:
            raise ValueError(f'Got invalid value in size, {size}')

        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, ...).

        Returns:
            Tensor: Output feature map.
        """
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ModulatedPEConv2d(nn.Module):
    r"""Modulated Conv2d in StyleGANv2 with Positional Encoding (PE).

    This module is modified from the ``ModulatedConv2d`` in StyleGAN2 to
    support the experiments in: Positional Encoding as Spatial Inductive Bias
    in GANs, CVPR'2021.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
        no_pad (bool, optional): Whether to removing the padding in
            convolution. Defaults to False.
        deconv2conv (bool, optional): Whether to substitute the transposed conv
            with (conv2d, upsampling). Defaults to False.
        interp_pad (int | None, optional): The padding number of interpolation
            pad. Defaults to None.
        up_config (dict, optional): Upsampling config.
            Defaults to dict(scale_factor=2, mode='nearest').
        up_after_conv (bool, optional): Whether to adopt upsampling after
            convolution. Defaults to False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 demodulate=True,
                 upsample=False,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 equalized_lr_cfg=dict(mode='fan_in', lr_mul=1., gain=1.),
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 eps=1e-8,
                 no_pad=False,
                 deconv2conv=False,
                 interp_pad=None,
                 up_config=dict(scale_factor=2, mode='nearest'),
                 up_after_conv=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_channels = style_channels
        self.demodulate = demodulate
        # sanity check for kernel size
        assert isinstance(self.kernel_size,
                          int) and (self.kernel_size >= 1
                                    and self.kernel_size % 2 == 1)
        self.upsample = upsample
        self.downsample = downsample
        self.style_bias = style_bias
        self.eps = eps
        self.no_pad = no_pad
        self.deconv2conv = deconv2conv
        self.interp_pad = interp_pad
        self.with_interp_pad = interp_pad is not None
        self.up_config = deepcopy(up_config)
        self.up_after_conv = up_after_conv

        # build style modulation module
        style_mod_cfg = dict() if style_mod_cfg is None else style_mod_cfg

        self.style_modulation = EqualLinearActModule(style_channels,
                                                     in_channels,
                                                     **style_mod_cfg)
        # set lr_mul for conv weight
        lr_mul_ = 1.
        if equalized_lr_cfg is not None:
            lr_mul_ = equalized_lr_cfg.get('lr_mul', 1.)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size,
                        kernel_size).div_(lr_mul_))

        # build blurry layer for upsampling
        if upsample and not self.deconv2conv:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, (pad0, pad1), upsample_factor=factor)

        # build blurry layer for downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        # add equalized_lr hook for conv weight
        if equalized_lr_cfg is not None:
            equalized_lr(self, **equalized_lr_cfg)

        # if `no_pad`, remove all of the padding in conv
        self.padding = kernel_size // 2 if not no_pad else 0

    def forward(self, x, style):
        """Forward function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).

        Returns:
            Tensor: Output feature with shape of (N, C, H, W).
        """
        n, c, h, w = x.shape
        # process style code
        style = self.style_modulation(style).view(n, 1, c, 1,
                                                  1) + self.style_bias

        # combine weight and style
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(n, self.out_channels, 1, 1, 1)

        weight = weight.view(n * self.out_channels, c, self.kernel_size,
                             self.kernel_size)

        if self.upsample and not self.deconv2conv:
            x = x.reshape(1, n * c, h, w)
            weight = weight.view(n, self.out_channels, c, self.kernel_size,
                                 self.kernel_size)
            weight = weight.transpose(1, 2).reshape(n * c, self.out_channels,
                                                    self.kernel_size,
                                                    self.kernel_size)
            x = conv_transpose2d(x, weight, padding=0, stride=2, groups=n)
            x = x.reshape(n, self.out_channels, *x.shape[-2:])
            x = self.blur(x)
        elif self.upsample and self.deconv2conv:
            if self.up_after_conv:
                x = x.reshape(1, n * c, h, w)
                x = conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])

            if self.with_interp_pad:
                h_, w_ = x.shape[-2:]
                up_cfg_ = deepcopy(self.up_config)
                up_scale = up_cfg_.pop('scale_factor')
                size_ = (h_ * up_scale + self.interp_pad,
                         w_ * up_scale + self.interp_pad)
                x = F.interpolate(x, size=size_, **up_cfg_)
            else:
                x = F.interpolate(x, **self.up_config)

            if not self.up_after_conv:
                h_, w_ = x.shape[-2:]
                x = x.view(1, n * c, h_, w_)
                x = conv2d(x, weight, padding=self.padding, groups=n)
                x = x.view(n, self.out_channels, *x.shape[2:4])

        elif self.downsample:
            x = self.blur(x)
            x = x.view(1, n * self.in_channels, *x.shape[-2:])
            x = conv2d(x, weight, stride=2, padding=0, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])
        else:
            x = x.view(1, n * c, h, w)
            x = conv2d(x, weight, stride=1, padding=self.padding, groups=n)
            x = x.view(n, self.out_channels, *x.shape[-2:])

        return x


class ModulatedStyleConv(nn.Module):
    """Modulated Style Convolution.

    In this module, we integrate the modulated conv2d, noise injector and
    activation layers into together.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to ``0.``.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 fp16_enabled=False,
                 conv_clamp=256):
        super().__init__()

        # add support for fp16
        self.fp16_enabled = fp16_enabled
        self.conv_clamp = float(conv_clamp)

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            upsample=upsample,
            blur_kernel=blur_kernel,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias)

        self.noise_injector = NoiseInjection()
        self.activate = _FusedBiasLeakyReLU(out_channels)

        # if self.fp16_enabled:
        #     self.half()

    @auto_fp16(apply_to=('x', 'noise'))
    def forward(self, x, style, noise=None, return_noise=False):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            noise (Tensor, optional): Noise for injection. Defaults to None.
            return_noise (bool, optional): Whether to return noise tensors.
                Defaults to False.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        out = self.conv(x, style)

        if return_noise:
            out, noise = self.noise_injector(
                out, noise=noise, return_noise=return_noise)
        else:
            out = self.noise_injector(
                out, noise=noise, return_noise=return_noise)

        # TODO: FP16 in activate layers
        out = self.activate(out)

        if self.fp16_enabled:
            out = torch.clamp(out, min=-self.conv_clamp, max=self.conv_clamp)

        if return_noise:
            return out, noise

        return out


class ModulatedPEStyleConv(nn.Module):
    """Modulated Style Convolution with Positional Encoding.

    This module is modified from the ``ModulatedStyleConv`` in StyleGAN2 to
    support the experiments in: Positional Encoding as Spatial Inductive Bias
    in GANs, CVPR'2021.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        style_channels (int): Channels for the style codes.
        demodulate (bool, optional): Whether to adopt demodulation.
            Defaults to True.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        equalized_lr_cfg (dict | None, optional): Configs for equalized lr.
            Defaults to dict(mode='fan_in', lr_mul=1., gain=1.).
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 upsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 demodulate=True,
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 **kwargs):
        super().__init__()

        self.conv = ModulatedPEConv2d(
            in_channels,
            out_channels,
            kernel_size,
            style_channels,
            demodulate=demodulate,
            upsample=upsample,
            blur_kernel=blur_kernel,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias,
            **kwargs)

        self.noise_injector = NoiseInjection()
        self.activate = _FusedBiasLeakyReLU(out_channels)

    def forward(self, x, style, noise=None, return_noise=False):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            noise (Tensor, optional): Noise for injection. Defaults to None.
            return_noise (bool, optional): Whether to return noise tensors.
                Defaults to False.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        out = self.conv(x, style)
        if return_noise:
            out, noise = self.noise_injector(
                out, noise=noise, return_noise=return_noise)
        else:
            out = self.noise_injector(
                out, noise=noise, return_noise=return_noise)

        out = self.activate(out)

        if return_noise:
            return out, noise

        return out


class ModulatedToRGB(nn.Module):
    """To RGB layer.

    This module is designed to output image tensor in StyleGAN2.

    Args:
        in_channels (int): Input channels.
        style_channels (int): Channels for the style codes.
        out_channels (int, optional): Output channels. Defaults to 3.
        upsample (bool, optional): Whether to adopt upsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        style_mod_cfg (dict, optional): Configs for style modulation module.
            Defaults to dict(bias_init=1.).
        style_bias (float, optional): Bias value for style code.
            Defaults to 0..
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
        out_fp32 (bool, optional): Whether to convert the output feature map to
            `torch.float32`. Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 style_channels,
                 out_channels=3,
                 upsample=True,
                 blur_kernel=[1, 3, 3, 1],
                 style_mod_cfg=dict(bias_init=1.),
                 style_bias=0.,
                 fp16_enabled=False,
                 conv_clamp=256,
                 out_fp32=True):
        super().__init__()

        if upsample:
            self.upsample = UpsampleUpFIRDn(blur_kernel)

        # add support for fp16
        self.fp16_enabled = fp16_enabled
        self.conv_clamp = float(conv_clamp)

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=1,
            style_channels=style_channels,
            demodulate=False,
            style_mod_cfg=style_mod_cfg,
            style_bias=style_bias)

        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        # enforece the output to be fp32 (follow Tero's implementation)
        self.out_fp32 = out_fp32

    @auto_fp16(apply_to=('x', 'style'))
    def forward(self, x, style, skip=None):
        """Forward Function.

        Args:
            x ([Tensor): Input features with shape of (N, C, H, W).
            style (Tensor): Style latent with shape of (N, C).
            skip (Tensor, optional): Tensor for skip link. Defaults to None.

        Returns:
            Tensor: Output features with shape of (N, C, H, W)
        """
        out = self.conv(x, style)
        out = out + self.bias.to(x.dtype)

        if self.fp16_enabled:
            out = torch.clamp(out, min=-self.conv_clamp, max=self.conv_clamp)

        # Here, Tero adopts FP16 at `skip`.
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class ConvDownLayer(nn.Sequential):
    """Convolution and Downsampling layer.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        downsample (bool, optional): Whether to adopt downsampling in features.
            Defaults to False.
        blur_kernel (list[int], optional): Blurry kernel.
            Defaults to [1, 3, 3, 1].
        bias (bool, optional): Whether to use bias parameter. Defaults to True.
        act_cfg (dict, optional): Activation configs.
            Defaults to dict(type='fused_bias').
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        conv_clamp (float, optional): Clamp the convolutional layer results to
            avoid gradient overflow. Defaults to `256.0`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 act_cfg=dict(type='fused_bias'),
                 fp16_enabled=False,
                 conv_clamp=256.):

        self.fp16_enabled = fp16_enabled
        self.conv_clamp = float(conv_clamp)
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        self.with_fused_bias = act_cfg is not None and act_cfg.get(
            'type') == 'fused_bias'
        if self.with_fused_bias:
            conv_act_cfg = None
        else:
            conv_act_cfg = act_cfg
        layers.append(
            EqualizedLRConvModule(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not self.with_fused_bias,
                norm_cfg=None,
                act_cfg=conv_act_cfg,
                equalized_lr_cfg=dict(mode='fan_in', gain=1.)))
        if self.with_fused_bias:
            layers.append(_FusedBiasLeakyReLU(out_channels))

        super(ConvDownLayer, self).__init__(*layers)

    @auto_fp16(apply_to=('x', ))
    def forward(self, x):
        x = super().forward(x)
        if self.fp16_enabled:
            x = torch.clamp(x, min=-self.conv_clamp, max=self.conv_clamp)
        return x


class ResBlock(nn.Module):
    """Residual block used in the discriminator of StyleGAN2.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size, same as :obj:`nn.Con2d`.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        convert_input_fp32 (bool, optional): Whether to convert input type to
            fp32 if not `fp16_enabled`. This argument is designed to deal with
            the cases where some modules are run in FP16 and others in FP32.
            Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 blur_kernel=[1, 3, 3, 1],
                 fp16_enabled=False,
                 convert_input_fp32=True):
        super().__init__()

        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32

        self.conv1 = ConvDownLayer(
            in_channels, in_channels, 3, blur_kernel=blur_kernel)
        self.conv2 = ConvDownLayer(
            in_channels,
            out_channels,
            3,
            downsample=True,
            blur_kernel=blur_kernel)

        self.skip = ConvDownLayer(
            in_channels,
            out_channels,
            1,
            downsample=True,
            act_cfg=None,
            bias=False,
            blur_kernel=blur_kernel)

    @auto_fp16()
    def forward(self, input):
        """Forward function.

        Args:
            input (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map.
        """
        # TODO: study whether this explicit datatype transfer will harm the
        # apex training speed
        if not self.fp16_enabled and self.convert_input_fp32:
            input = input.to(torch.float32)
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out


class ModMBStddevLayer(nn.Module):
    """Modified MiniBatch Stddev Layer.

    This layer is modified from ``MiniBatchStddevLayer`` used in PGGAN. In
    StyleGAN2, the authors add a new feature, `channel_groups`, into this
    layer.

    Note that to accelerate the training procedure, we also add a new feature
    of ``sync_std`` to achieve multi-nodes/machine training. This feature is
    still in beta version and we have tested it on 256 scales.

    Args:
        group_size (int, optional): The size of groups in batch dimension.
            Defaults to 4.
        channel_groups (int, optional): The size of groups in channel
            dimension. Defaults to 1.
        sync_std (bool, optional): Whether to use synchronized std feature.
            Defaults to False.
        sync_groups (int | None, optional): The size of groups in node
            dimension. Defaults to None.
        eps (float, optional): Epsilon value to avoid computation error.
            Defaults to 1e-8.
    """

    def __init__(self,
                 group_size=4,
                 channel_groups=1,
                 sync_std=False,
                 sync_groups=None,
                 eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps
        self.channel_groups = channel_groups
        self.sync_std = sync_std
        self.sync_groups = group_size if sync_groups is None else sync_groups

        if self.sync_std:
            assert torch.distributed.is_initialized(
            ), 'Only in distributed training can the sync_std be activated.'
            mmcv.print_log('Adopt synced minibatch stddev layer', 'mmgen')

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map with shape of (N, C, H, W).

        Returns:
            Tensor: Output feature map with shape of (N, C+1, H, W).
        """

        if self.sync_std:
            # concatenate all features
            all_features = torch.cat(AllGatherLayer.apply(x), dim=0)
            # get the exact features we need in calculating std-dev
            rank, ws = get_dist_info()
            local_bs = all_features.shape[0] // ws
            start_idx = local_bs * rank
            # avoid the case where start idx near the tail of features
            if start_idx + self.sync_groups > all_features.shape[0]:
                start_idx = all_features.shape[0] - self.sync_groups
            end_idx = min(local_bs * rank + self.sync_groups,
                          all_features.shape[0])

            x = all_features[start_idx:end_idx]

        # batch size should be smaller than or equal to group size. Otherwise,
        # batch size should be divisible by the group size.
        assert x.shape[
            0] <= self.group_size or x.shape[0] % self.group_size == 0, (
                'Batch size be smaller than or equal '
                'to group size. Otherwise,'
                ' batch size should be divisible by the group size.'
                f'But got batch size {x.shape[0]},'
                f' group size {self.group_size}')
        assert x.shape[1] % self.channel_groups == 0, (
            '"channel_groups" must be divided by the feature channels. '
            f'channel_groups: {self.channel_groups}, '
            f'feature channels: {x.shape[1]}')

        n, c, h, w = x.shape
        group_size = min(n, self.group_size)
        # [G, M, Gc, C', H, W]
        y = torch.reshape(x, (group_size, -1, self.channel_groups,
                              c // self.channel_groups, h, w))
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + self.eps)
        # [M, 1, 1, 1]
        y = y.mean(dim=(2, 3, 4), keepdim=True).squeeze(2)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)
