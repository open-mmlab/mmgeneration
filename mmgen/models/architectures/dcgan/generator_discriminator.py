# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger
from ..common import get_module_device


@MODULES.register_module()
class DCGANGenerator(nn.Module):
    """Generator for DCGAN.

    Implementation Details for DCGAN architecture:

    #. Adopt transposed convolution in the generator;
    #. Use batchnorm in the generator except for the final output layer;
    #. Use ReLU in the generator in addition to the final output layer.

    More details can be found in the original paper:
    Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks
    http://arxiv.org/abs/1511.06434

    Args:
        output_scale (int | tuple[int]): Output scale for the generated
            image. If only a integer is provided, the output image will
            be a square shape. The tuple of two integers will set the
            height and width for the output image, respectively.
        out_channels (int, optional): The channel number of the output feature.
            Default to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Default to 1024.
        input_scale (int | tuple[int], optional): Output scale for the
            generated image. If only a integer is provided, the input feature
            ahead of the convolutional generator will be a square shape. The
            tuple of two integers will set the height and width for the input
            convolutional feature, respectively. Defaults to 4.
        noise_size (int, optional): Size of the input noise
            vector. Defaults to 100.
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='ReLU')``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
        pretrained (str, optional): Path for the pretrained model. Default to
            ``None``.
    """

    def __init__(self,
                 output_scale,
                 out_channels=3,
                 base_channels=1024,
                 input_scale=4,
                 noise_size=100,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='Tanh'),
                 pretrained=None):
        super().__init__()
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size

        # the number of times for upsampling
        self.num_upsamples = int(np.log2(output_scale // input_scale))

        # output 4x4 feature map
        self.noise2feat = ConvModule(
            noise_size,
            base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg)

        # build up upsampling backbone (excluding the output layer)
        upsampling = []
        curr_channel = base_channels
        for _ in range(self.num_upsamples - 1):
            upsampling.append(
                ConvModule(
                    curr_channel,
                    curr_channel // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))

            curr_channel //= 2

        self.upsampling = nn.Sequential(*upsampling)

        # output layer
        self.output_layer = ConvModule(
            curr_channel,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=None,
            act_cfg=out_act_cfg)

        self.init_weights(pretrained=pretrained)

    def forward(self, noise, num_batches=0, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img`` and
                ``noise_batch`` will be returned.
        """
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            if noise.ndim == 2:
                noise_batch = noise[:, :, None, None]
            elif noise.ndim == 4:
                noise_batch = noise
            else:
                raise ValueError('The noise should be in shape of (n, c) or '
                                 f'(n, c, 1, 1), but got {noise.shape}')
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size, 1, 1))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.noise_size, 1, 1))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))

        x = self.noise2feat(noise_batch)
        x = self.upsampling(x)
        x = self.output_layer(x)

        if return_noise:
            return dict(fake_img=x, noise_batch=noise_batch)

        return x

    def init_weights(self, pretrained=None):
        """Init weights for models.

        We just use the initialization method proposed in the original paper.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, _BatchNorm):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')


@MODULES.register_module()
class DCGANDiscriminator(nn.Module):
    """Discriminator for DCGAN.

    Implementation Details for DCGAN architecture:

    #. Adopt convolution in the discriminator;
    #. Use batchnorm in the discriminator except for the input and final \
       output layer;
    #. Use LeakyReLU in the discriminator in addition to the output layer.

    Args:
        input_scale (int): The scale of the input image.
        output_scale (int): The final scale of the convolutional feature.
        out_channels (int): The channel number of the final output layer.
        in_channels (int, optional): The channel number of the input image.
            Defaults to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Defaults to 128.
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='ReLU')``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
        pretrained (str, optional): Path for the pretrained model. Default to
            ``None``.
    """

    def __init__(self,
                 input_scale,
                 output_scale,
                 out_channels,
                 in_channels=3,
                 base_channels=128,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='LeakyReLU'),
                 out_act_cfg=None,
                 pretrained=None):
        super().__init__()
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.out_channels = out_channels
        self.base_channels = base_channels

        # the number of times for downsampling
        self.num_downsamples = int(np.log2(input_scale // output_scale))

        # build up downsampling backbone (excluding the output layer)
        downsamples = []
        for i in range(self.num_downsamples):
            # remove norm for the first conv
            norm_cfg_ = None if i == 0 else default_norm_cfg
            in_ch = in_channels if i == 0 else base_channels * 2**(i - 1)

            downsamples.append(
                ConvModule(
                    in_ch,
                    base_channels * 2**i,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg_,
                    act_cfg=default_act_cfg))
            curr_channels = base_channels * 2**i

        self.downsamples = nn.Sequential(*downsamples)

        # define output layer
        self.output_layer = ConvModule(
            curr_channels,
            out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=None,
            act_cfg=out_act_cfg)

        self.init_weights(pretrained=pretrained)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Fake or real image tensor.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """

        n = x.shape[0]
        x = self.downsamples(x)
        x = self.output_layer(x)

        # reshape to a flatten feature
        return x.view(n, -1)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        We just use the initialization method proposed in the original paper.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, _BatchNorm):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
