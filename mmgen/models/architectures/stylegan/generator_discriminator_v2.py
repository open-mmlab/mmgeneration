# Copyright (c) OpenMMLab. All rights reserved.
import random

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

from mmgen.core.runners.fp16_utils import auto_fp16
from mmgen.models.architectures import PixelNorm
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from .ada.augment import AugmentPipe
from .ada.misc import constant
from .modules.styleganv2_modules import (ConstantInput, ConvDownLayer,
                                         EqualLinearActModule,
                                         ModMBStddevLayer, ModulatedStyleConv,
                                         ModulatedToRGB, ResBlock)
from .utils import get_mean_latent, style_mixing


@MODULES.register_module()
class StyleGANv2Generator(nn.Module):
    r"""StyleGAN2 Generator.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of convolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered official weights as
    follows:

    - stylegan2-ffhq-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - stylegan2-cat-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        generator = StyleGANv2Generator(1024, 512,
                                        pretrained=dict(
                                            ckpt_path=ckpt_http,
                                            prefix='generator_ema'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path. If you just want to load the original
    generator (not the ema model), please set the prefix with 'generator'.

    Note that our implementation allows to generate BGR image, while the
    original StyleGAN2 outputs RGB images by default. Thus, we provide
    ``bgr2rgb`` argument to convert the image space.

    Args:
        out_size (int): The output size of the StyleGAN2 generator.
        style_channels (int): The number of channels for style code.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to ``0.9``.
        num_fp16_scales (int, optional): The number of resolutions to use auto
            fp16 training. Different from ``fp16_enabled``, this argument
            allows users to adopt FP16 training only in several blocks.
            This behaviour is much more similar to the official implementation
            by Tero. Defaults to 0.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. If this flag is `True`, the whole module will be wrapped
            with ``auto_fp16``. Defaults to False.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
    """

    def __init__(self,
                 out_size,
                 style_channels,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 num_fp16_scales=0,
                 fp16_enabled=False,
                 pretrained=None):
        super().__init__()
        self.out_size = out_size
        self.style_channels = style_channels
        self.num_mlps = num_mlps
        self.channel_multiplier = channel_multiplier
        self.lr_mlp = lr_mlp
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode
        self.mix_prob = mix_prob
        self.num_fp16_scales = num_fp16_scales
        self.fp16_enabled = fp16_enabled

        # define style mapping layers
        mapping_layers = [PixelNorm()]

        for _ in range(num_mlps):
            mapping_layers.append(
                EqualLinearActModule(
                    style_channels,
                    style_channels,
                    equalized_lr_cfg=dict(lr_mul=lr_mlp, gain=1.),
                    act_cfg=dict(type='fused_bias')))

        self.style_mapping = nn.Sequential(*mapping_layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # constant input layer
        self.constant_input = ConstantInput(self.channels[4])
        # 4x4 stage
        self.conv1 = ModulatedStyleConv(
            self.channels[4],
            self.channels[4],
            kernel_size=3,
            style_channels=style_channels,
            blur_kernel=blur_kernel)
        self.to_rgb1 = ModulatedToRGB(
            self.channels[4],
            style_channels,
            upsample=False,
            fp16_enabled=fp16_enabled)

        # generator backbone (8x8 --> higher resolutions)
        self.log_size = int(np.log2(self.out_size))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channels_ = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channels_ = self.channels[2**i]

            # If `fp16_enabled` is True, all of layers will be run in auto
            # FP16. In the case of `num_fp16_sacles` > 0, only partial
            # layers will be run in fp16.
            _use_fp16 = (self.log_size - i) < num_fp16_scales or fp16_enabled

            self.convs.append(
                ModulatedStyleConv(
                    in_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    fp16_enabled=_use_fp16))
            self.convs.append(
                ModulatedStyleConv(
                    out_channels_,
                    out_channels_,
                    3,
                    style_channels,
                    upsample=False,
                    blur_kernel=blur_kernel,
                    fp16_enabled=_use_fp16))
            self.to_rgbs.append(
                ModulatedToRGB(
                    out_channels_,
                    style_channels,
                    upsample=True,
                    fp16_enabled=_use_fp16))  # set to global fp16

            in_channels_ = out_channels_

        self.num_latents = self.log_size * 2 - 2
        self.num_injected_noises = self.num_latents - 1

        # register buffer for injected noises
        for layer_idx in range(self.num_injected_noises):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.register_buffer(f'injected_noise_{layer_idx}',
                                 torch.randn(*shape))

        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmgen')

    def train(self, mode=True):
        if mode:
            if self.default_style_mode != self._default_style_mode:
                mmcv.print_log(
                    f'Switch to train style mode: {self._default_style_mode}',
                    'mmgen')
            self.default_style_mode = self._default_style_mode

        else:
            if self.default_style_mode != self.eval_style_mode:
                mmcv.print_log(
                    f'Switch to evaluation style mode: {self.eval_style_mode}',
                    'mmgen')
            self.default_style_mode = self.eval_style_mode

        return super(StyleGANv2Generator, self).train(mode)

    def make_injected_noise(self):
        """make noises that will be injected into feature maps.

        Returns:
            list[Tensor]: List of layer-wise noise tensor.
        """
        device = get_module_device(self)

        noises = [torch.randn(1, 1, 2**2, 2**2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        return get_mean_latent(self, num_samples, **kwargs)

    def style_mixing(self,
                     n_source,
                     n_target,
                     inject_index=1,
                     truncation_latent=None,
                     truncation=0.7):
        return style_mixing(
            self,
            n_source=n_source,
            n_target=n_target,
            inject_index=inject_index,
            truncation=truncation,
            truncation_latent=truncation_latent,
            style_channels=self.style_channels)

    @auto_fp16()
    def forward(self,
                styles,
                num_batches=-1,
                return_noise=False,
                return_latents=False,
                inject_index=None,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                injected_noise=None,
                randomize_noise=True):
        """Forward function.

        This function has been integrated with the truncation trick. Please
        refer to the usage of `truncation` and `truncation_latent`.

        Args:
            styles (torch.Tensor | list[torch.Tensor] | callable | None): In
                StyleGAN2, you can provide noise tensor or latent tensor. Given
                a list containing more than one noise or latent tensors, style
                mixing trick will be used in training. Of course, You can
                directly give a batch of noise through a ``torch.Tensor`` or
                offer a callable function to sample a batch of noise data.
                Otherwise, the ``None`` indicates to use the default noise
                sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            inject_index (int | None, optional): The index number for mixing
                style codes. Defaults to None.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            injected_noise (torch.Tensor | None, optional): Given a tensor, the
                random noise will be fixed as this input injected noise.
                Defaults to None.
            randomize_noise (bool, optional): If `False`, images are sampled
                with the buffered noise tensor injected to the style conv
                block. Defaults to True.

        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
        """
        # receive noise and conduct sanity check.
        if isinstance(styles, torch.Tensor):
            assert styles.shape[1] == self.style_channels
            styles = [styles]
        elif mmcv.is_seq_of(styles, torch.Tensor):
            for t in styles:
                assert t.shape[-1] == self.style_channels
        # receive a noise generator and sample noise.
        elif callable(styles):
            device = get_module_device(self)
            noise_generator = styles
            assert num_batches > 0
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    noise_generator((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [noise_generator((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]
        # otherwise, we will adopt default noise sampler.
        else:
            device = get_module_device(self)
            assert num_batches > 0 and not input_is_latent
            if self.default_style_mode == 'mix' and random.random(
            ) < self.mix_prob:
                styles = [
                    torch.randn((num_batches, self.style_channels))
                    for _ in range(2)
                ]
            else:
                styles = [torch.randn((num_batches, self.style_channels))]
            styles = [s.to(device) for s in styles]

        if not input_is_latent:
            noise_batch = styles
            styles = [self.style_mapping(s) for s in styles]
        else:
            noise_batch = None

        if injected_noise is None:
            if randomize_noise:
                injected_noise = [None] * self.num_injected_noises
            else:
                injected_noise = [
                    getattr(self, f'injected_noise_{i}')
                    for i in range(self.num_injected_noises)
                ]
        # use truncation trick
        if truncation < 1:
            style_t = []
            # calculate truncation latent on the fly
            if truncation_latent is None and not hasattr(
                    self, 'truncation_latent'):
                self.truncation_latent = self.get_mean_latent()
                truncation_latent = self.truncation_latent
            elif truncation_latent is None and hasattr(self,
                                                       'truncation_latent'):
                truncation_latent = self.truncation_latent

            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style - truncation_latent))

            styles = style_t
        # no style mixing
        if len(styles) < 2:
            inject_index = self.num_latents

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]
        # style mixing
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.num_latents - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.num_latents - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        # 4x4 stage
        out = self.constant_input(latent)
        out = self.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher resolutions
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], injected_noise[1::2],
                injected_noise[2::2], self.to_rgbs):
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)
            _index += 2

        # make sure the output image is torch.float32 to avoid RunTime Error
        # in other modules
        img = skip.to(torch.float32)

        if return_latents or return_noise:
            output_dict = dict(
                fake_img=img,
                latent=latent,
                inject_index=inject_index,
                noise_batch=noise_batch)
            return output_dict

        return img


@MODULES.register_module()
class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.

    The architecture of this discriminator is proposed in StyleGAN2. More
    details can be found in: Analyzing and Improving the Image Quality of
    StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered official weights as
    follows:

    - stylegan2-ffhq-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - stylegan2-cat-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: https://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        discriminator = StyleGAN2Discriminator(1024, 512,
                                               pretrained=dict(
                                                   ckpt_path=ckpt_http,
                                                   prefix='discriminator'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path.

    Note that our implementation adopts BGR image as input, while the
    original StyleGAN2 provides RGB images to the discriminator. Thus, we
    provide ``bgr2rgb`` argument to convert the image space. If your images
    follow the RGB order, please set it to ``True`` accordingly.

    Args:
        in_size (int): The input size of images.
        channel_multiplier (int, optional): The multiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        mbstd_cfg (dict, optional): Configs for minibatch-stddev layer.
            Defaults to dict(group_size=4, channel_groups=1).
        num_fp16_scales (int, optional): The number of resolutions to use auto
            fp16 training. Defaults to 0.
        fp16_enabled (bool, optional): Whether to use fp16 training in this
            module. Defaults to False.
        out_fp32 (bool, optional): Whether to convert the output feature map to
            `torch.float32`. Defaults to `True`.
        convert_input_fp32 (bool, optional): Whether to convert input type to
            fp32 if not `fp16_enabled`. This argument is designed to deal with
            the cases where some modules are run in FP16 and others in FP32.
            Defaults to True.
        input_bgr2rgb (bool, optional): Whether to reformat the input channels
            with order `rgb`. Since we provide several converted weights,
            whose input order is `rgb`. You can set this argument to True if
            you want to finetune on converted weights. Defaults to False.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
    """

    def __init__(self,
                 in_size,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 mbstd_cfg=dict(group_size=4, channel_groups=1),
                 num_fp16_scales=0,
                 fp16_enabled=False,
                 out_fp32=True,
                 convert_input_fp32=True,
                 input_bgr2rgb=False,
                 pretrained=None):
        super().__init__()
        self.num_fp16_scale = num_fp16_scales
        self.fp16_enabled = fp16_enabled
        self.convert_input_fp32 = convert_input_fp32
        self.out_fp32 = out_fp32

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        log_size = int(np.log2(in_size))

        in_channels = channels[in_size]

        _use_fp16 = num_fp16_scales > 0
        convs = [
            ConvDownLayer(3, channels[in_size], 1, fp16_enabled=_use_fp16)
        ]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]

            # add fp16 training for higher resolutions
            _use_fp16 = (log_size - i) < num_fp16_scales or fp16_enabled

            convs.append(
                ResBlock(
                    in_channels,
                    out_channel,
                    blur_kernel,
                    fp16_enabled=_use_fp16,
                    convert_input_fp32=convert_input_fp32))

            in_channels = out_channel

        self.convs = nn.Sequential(*convs)

        self.mbstd_layer = ModMBStddevLayer(**mbstd_cfg)

        self.final_conv = ConvDownLayer(in_channels + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinearActModule(
                channels[4] * 4 * 4,
                channels[4],
                act_cfg=dict(type='fused_bias')),
            EqualLinearActModule(channels[4], 1),
        )

        self.input_bgr2rgb = input_bgr2rgb
        if pretrained is not None:
            self._load_pretrained_model(**pretrained)

    def _load_pretrained_model(self,
                               ckpt_path,
                               prefix='',
                               map_location='cpu',
                               strict=True):
        state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                  map_location)
        self.load_state_dict(state_dict, strict=strict)
        mmcv.print_log(f'Load pretrained model from {ckpt_path}', 'mmgen')

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predict score for the input image.
        """
        # This setting was used to finetune on converted weights
        if self.input_bgr2rgb:
            x = x[:, [2, 1, 0], ...]

        x = self.convs(x)

        x = self.mbstd_layer(x)
        if not self.final_conv.fp16_enabled and self.convert_input_fp32:
            x = x.to(torch.float32)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x


@MODULES.register_module()
class ADAStyleGAN2Discriminator(StyleGAN2Discriminator):

    def __init__(self, in_size, *args, data_aug=None, **kwargs):
        """StyleGANv2 Discriminator with adaptive augmentation.

        Args:
            in_size (int): The input size of images.
            data_aug (dict, optional): Config for data
                augmentation. Defaults to None.
        """
        super().__init__(in_size, *args, **kwargs)
        self.with_ada = data_aug is not None
        if self.with_ada:
            self.ada_aug = build_module(data_aug)
            self.ada_aug.requires_grad = False
        self.log_size = int(np.log2(in_size))

    def forward(self, x):
        """Forward function."""
        if self.with_ada:
            x = self.ada_aug.aug_pipeline(x)
        return super().forward(x)


@MODULES.register_module()
class ADAAug(nn.Module):
    """Data Augmentation Module for Adaptive Discriminator augmentation.

    Args:
        aug_pipeline (dict, optional): Config for augmentation pipeline.
            Defaults to None.
        update_interval (int, optional): Interval for updating
            augmentation probability. Defaults to 4.
        augment_initial_p (float, optional): Initial augmentation
            probability. Defaults to 0..
        ada_target (float, optional): ADA target. Defaults to 0.6.
        ada_kimg (int, optional): ADA training duration. Defaults to 500.
    """

    def __init__(self,
                 aug_pipeline=None,
                 update_interval=4,
                 augment_initial_p=0.,
                 ada_target=0.6,
                 ada_kimg=500):
        super().__init__()
        self.aug_pipeline = AugmentPipe(**aug_pipeline)
        self.update_interval = update_interval
        self.ada_kimg = ada_kimg
        self.ada_target = ada_target

        self.aug_pipeline.p.copy_(torch.tensor(augment_initial_p))

        # this log buffer stores two numbers: num_scalars, sum_scalars.
        self.register_buffer('log_buffer', torch.zeros((2, )))

    def update(self, iteration=0, num_batches=0):
        """Update Augment probability.

        Args:
            iteration (int, optional): Training iteration.
                Defaults to 0.
            num_batches (int, optional): The number of reals batches.
                Defaults to 0.
        """

        if (iteration + 1) % self.update_interval == 0:

            adjust_step = float(num_batches * self.update_interval) / float(
                self.ada_kimg * 1000.)

            # get the mean value as the ada heuristic
            ada_heuristic = self.log_buffer[1] / self.log_buffer[0]
            adjust = np.sign(ada_heuristic.item() -
                             self.ada_target) * adjust_step
            # update the augment p
            # Note that p may be bigger than 1.0
            self.aug_pipeline.p.copy_((self.aug_pipeline.p + adjust).max(
                constant(0, device=self.log_buffer.device)))

            self.log_buffer = self.log_buffer * 0.
