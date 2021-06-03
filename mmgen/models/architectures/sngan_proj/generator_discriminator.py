from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks.activation import build_activation_layer
from torch.nn.utils import spectral_norm

from mmgen.models.builder import MODULES
from mmgen.utils.check_dist_init import check_dist_init
from ..common import get_module_device
from .modules import (SNGAN_DisHeadResBlock, SNGAN_DisResBlock,
                      SNGAN_GenResBlock)


@MODULES.register_module()
class SNGANGenerator(nn.Module):

    # TODO: why wganpg use str as key?
    _defualt_channels_cfg = {
        32: [1, 1, 1],
        64: [16, 8, 4, 2],
        128: [16, 16, 8, 8, 4, 2]
    }

    # TODO: move this to _base_?
    _default_res_block_cfg = dict(
        upsample=True,
        use_cbn=True,
        use_norm_para=False,
        act_cfg=dict(type='ReLU'),
        norm_cfg=dict(type='BN'),
        upsample_cfg=dict(type='nearest', scale_factor=2),
        conv_cfg=dict(
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
        ),
        shortcut_cfg=dict(kernel_size=1, stride=1, padding=0, act_cfg=None))

    def __init__(self,
                 output_scale,
                 noise_size,
                 num_classes=0,
                 out_channels=3,
                 base_channels=1,
                 input_scale=4,
                 res_block_cfg=None,
                 channel_cfg=None,
                 auto_sync_bn=True):
        super().__init__()

        # TODO: shall we support image generation with not square shape?
        # if isinstance(input_scale, int):
        #     input_scale = (input_scale, input_scale)
        # if isinstance(output_scale, int):
        #     pass

        self.input_scale = input_scale
        self.output_scale = output_scale
        self.noise_size = noise_size

        self.res_block_cfg = deepcopy(self._default_res_block_cfg)
        if res_block_cfg is not None:
            self.res_block_cfg.update(res_block_cfg)
        if 'auto_sync_bn' not in self.res_block_cfg:
            self.res_block_cfg['auto_sync_bn'] = auto_sync_bn
        self.res_block_cfg['num_classes'] = num_classes

        channel_cfg = self._defualt_channels_cfg if channel_cfg is None \
            else channel_cfg
        if isinstance(channel_cfg, dict):
            if output_scale not in channel_cfg:
                raise KeyError(
                    '`output_scale={} is not found in `channel_cfg`, '
                    'only support configs for {}'.format(
                        output_scale, [chn for chn in channel_cfg.keys()]))
            self.channel_factor_list = channel_cfg[output_scale]
        elif isinstance(channel_cfg, list):
            self.channel_factor_list = channel_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             'receive {}'.format(type(channel_cfg)))

        self.noise2feat = nn.Linear(
            noise_size,
            input_scale**2 * base_channels * self.channel_factor_list[0])
        self.res_blocks = nn.ModuleList()
        for idx in range(len(self.channel_factor_list)):
            factor_input = self.channel_factor_list[idx]
            factor_output = self.channel_factor_list[idx+1] \
                if idx < len(self.channel_factor_list)-1 else 1

            self.res_blocks.append(
                SNGAN_GenResBlock(factor_input * base_channels,
                                  factor_output * base_channels,
                                  **self.res_block_cfg))

        to_rgb_norm_cfg = dict(type='BN')
        if check_dist_init() and auto_sync_bn:
            to_rgb_norm_cfg['type'] = 'SyncBN'

        self.to_rgb = ConvModule(
            factor_output * base_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=to_rgb_norm_cfg,
            act_cfg=dict(type='Tanh'),
            order=('norm', 'conv', 'act'))
        self.init_weight()

    def forward(self, noise, num_batches=0, y=None, return_noise=False):
        # TODO: add docs
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
            noise_batch = torch.randn((num_batches, self.noise_size))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))

        x = self.noise2feat(noise_batch)
        x = x.reshape(x.size(0), -1, self.input_scale, self.input_scale)

        for res_block in self.res_blocks:
            x = res_block(x, y)

        out_img = self.to_rgb(x)

        if return_noise:
            output = dict(fake_img=out_img, noise_batch=noise_batch)
            return output

        return out_img

    def init_weight(self, pretrained=False):
        if pretrained:  # TODO:
            pass
        else:
            xavier_init(self.noise2feat, gain=1, distribution='uniform')
            xavier_init(self.to_rgb.conv, gain=1, distribution='uniform')


@MODULES.register_module()
class SNGANDiscriminator(nn.Module):

    _defualt_channels_cfg = {
        32: [1, 1, 1],
        64: [2, 4, 8, 16],
        256: [2, 4, 8, 8, 16, 16]
    }
    _defualt_downsample_list_cfg = {
        32: [True, False, False],
        64: [True, True, True, True],
        256: [True, True, True, True, True, False]
    }

    _default_head_cfg = dict(
        conv_cfg=dict(
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=True),
        shortcut_cfg=dict(
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            with_spectral_norm=True))

    _default_res_block_cfg = dict(
        act_cfg=dict(type='ReLU'),
        conv_cfg=dict(
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=None,
            with_spectral_norm=True),
        shortcut_cfg=dict(
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None,
            with_spectral_norm=True))

    def __init__(self,
                 input_scale,
                 base_channels,
                 num_classes=0,
                 input_channels=3,
                 head_cfg=None,
                 res_block_cfg=None,
                 channel_cfg=None,
                 downsample_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.head_cfg = deepcopy(self._default_head_cfg)
        if head_cfg is not None:
            self.head_cfg.update(head_cfg)
        self.res_block_cfg = deepcopy(self._default_res_block_cfg)
        if res_block_cfg is not None:
            self.res_block_cfg.update(res_block_cfg)

        channel_cfg = self._defualt_channels_cfg if channel_cfg is None \
            else channel_cfg
        if isinstance(channel_cfg, dict):
            if input_scale not in channel_cfg:
                raise KeyError(
                    '`output_scale={} is not found in `channel_cfg`, '
                    'only support configs for {}'.format(
                        input_scale, [chn for chn in channel_cfg.keys()]))
            self.channel_factor_list = channel_cfg[input_scale]

        downsample_cfg = self._defualt_downsample_list_cfg \
            if downsample_cfg is None else downsample_cfg
        if isinstance(downsample_cfg, dict):
            if input_scale not in downsample_cfg:
                raise KeyError(
                    '`output_scale={} is not found in `downsample_cfg`, '
                    'only support configs for {}'.format(
                        input_scale, [chn for chn in downsample_cfg.keys()]))
            self.downsample_list = downsample_cfg[input_scale]

        if len(downsample_cfg) != len(channel_cfg):
            raise ValueError('TODO')

        self.from_rgb = SNGAN_DisHeadResBlock(input_channels, base_channels,
                                              **self.head_cfg)

        self.res_block = nn.ModuleList()
        for idx in range(len(downsample_cfg)):
            factor_input = 1 if idx == 0 else self.channel_factor_list[idx - 1]
            factor_output = self.channel_factor_list[idx]
            res_block_cfg_ = deepcopy(self.res_block_cfg)
            res_block_cfg_['downsample'] = self.downsample_list[idx]
            self.res_block.append(
                SNGAN_DisResBlock(factor_input * base_channels,
                                  factor_output * base_channels,
                                  **res_block_cfg_))

        self.decision = spectral_norm(
            nn.Linear(factor_output * base_channels, 1, bias=False))

        self.num_classes = num_classes
        if num_classes > 0:
            self.proj_y = nn.Embedding(num_classes,
                                       factor_output * base_channels)

        self.activate = build_activation_layer(act_cfg)
        self.init_weight()

    def forward(self, x, y=None):
        h = self.from_rgb(x)
        for res_block in self.res_block:
            h = res_block(h)
        h = self.activate(h)
        out = torch.sum(h, dim=[2, 3])

        if self.num_classes > 0:
            w_y = self.proj_y(y)
            # TODO: should check chainer's output shape
            # and the original paper
            out = out + torch.sum(
                w_y[..., None, None] * h, dim=(2, 3), keepdim=True)
        return out

    def init_weight(self, pretrained=False):
        if pretrained:  # TODO:
            pass
        else:
            xavier_init(self.decision, gain=1, distribution='uniform')
            xavier_init(self.proj_y, gain=1, distribution='uniform')
