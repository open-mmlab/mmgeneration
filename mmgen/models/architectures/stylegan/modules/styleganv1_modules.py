# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmgen.models.architectures.pggan import (EqualizedLRConvModule,
                                              EqualizedLRConvUpModule,
                                              EqualizedLRLinearModule)
from mmgen.models.architectures.stylegan.modules import (Blur, ConstantInput,
                                                         NoiseInjection)


class AdaptiveInstanceNorm(nn.Module):
    r"""Adaptive Instance Normalization Module.

    Ref: https://github.com/rosinality/style-based-gan-pytorch/blob/master/model.py  # noqa

    Args:
        in_channel (int): The number of input's channel.
        style_dim (int): Style latent dimension.
    """

    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.affine = EqualizedLRLinearModule(style_dim, in_channel * 2)

        self.affine.bias.data[:in_channel] = 1
        self.affine.bias.data[in_channel:] = 0

    def forward(self, input, style):
        """Forward function.

        Args:
            input (Tensor): Input tensor with shape (n, c, h, w).
            style (Tensor): Input style tensor with shape (n, c).

        Returns:
            Tensor: Forward results.
        """
        style = self.affine(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class StyleConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 style_channels,
                 padding=1,
                 initial=False,
                 blur_kernel=[1, 2, 1],
                 upsample=False,
                 fused=False):
        """Convolutional style blocks composing of noise injector, AdaIN module
        and convolution layers.

        Args:
            in_channels (int): The channel number of the input tensor.
            out_channels (itn): The channel number of the output tensor.
            kernel_size (int): The kernel size of convolution layers.
            style_channels (int): The number of channels for style code.
            padding (int, optional): Padding of convolution layers.
                Defaults to 1.
            initial (bool, optional): Whether this is the first StyleConv of
                StyleGAN's generator. Defaults to False.
            blur_kernel (list, optional): The blurry kernel.
                Defaults to [1, 2, 1].
            upsample (bool, optional): Whether perform upsampling.
                Defaults to False.
            fused (bool, optional): Whether use fused upconv.
                Defaults to False.
        """
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channels)
        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        EqualizedLRConvUpModule(
                            in_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            act_cfg=dict(type='LeakyReLU',
                                         negative_slope=0.2)),
                        Blur(blur_kernel, pad=(1, 1)),
                    )
                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualizedLRConvModule(
                            in_channels,
                            out_channels,
                            kernel_size,
                            padding=padding,
                            act_cfg=None), Blur(blur_kernel, pad=(1, 1)))
            else:
                self.conv1 = EqualizedLRConvModule(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    act_cfg=None)

        self.noise_injector1 = NoiseInjection()
        self.activate1 = nn.LeakyReLU(0.2)
        self.adain1 = AdaptiveInstanceNorm(out_channels, style_channels)

        self.conv2 = EqualizedLRConvModule(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            act_cfg=None)
        self.noise_injector2 = NoiseInjection()
        self.activate2 = nn.LeakyReLU(0.2)
        self.adain2 = AdaptiveInstanceNorm(out_channels, style_channels)

    def forward(self,
                x,
                style1,
                style2,
                noise1=None,
                noise2=None,
                return_noise=False):
        """Forward function.

        Args:
            x (Tensor): Input tensor.
            style1 (Tensor): Input style tensor with shape (n, c).
            style2 (Tensor): Input style tensor with shape (n, c).
            noise1 (Tensor, optional): Noise tensor with shape (n, c, h, w).
                Defaults to None.
            noise2 (Tensor, optional): Noise tensor with shape (n, c, h, w).
                Defaults to None.
            return_noise (bool, optional): If True, ``noise1`` and ``noise2``
            will be returned with ``out``. Defaults to False.

        Returns:
            Tensor | tuple[Tensor]: Forward results.
        """
        out = self.conv1(x)
        if return_noise:
            out, noise1 = self.noise_injector1(
                out, noise=noise1, return_noise=return_noise)
        else:
            out = self.noise_injector1(
                out, noise=noise1, return_noise=return_noise)
        out = self.activate1(out)
        out = self.adain1(out, style1)

        out = self.conv2(out)
        if return_noise:
            out, noise2 = self.noise_injector2(
                out, noise=noise2, return_noise=return_noise)
        else:
            out = self.noise_injector2(
                out, noise=noise2, return_noise=return_noise)
        out = self.activate2(out)
        out = self.adain2(out, style2)

        if return_noise:
            return out, noise1, noise2

        return out
