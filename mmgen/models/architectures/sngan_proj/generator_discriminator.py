from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, xavier_init
from mmcv.runner import load_checkpoint
from mmcv.utils import build_from_cfg
from torch.nn.utils import spectral_norm

from mmgen.models.builder import MODULES
from mmgen.utils import check_dist_init
from mmgen.utils.logger import get_root_logger
from ..common import get_module_device


@MODULES.register_module()
class SNGANGenerator(nn.Module):

    # TODO: add doc here
    _defualt_channels_cfg = {
        32: [1, 1, 1],
        64: [16, 8, 4, 2],
        128: [16, 16, 8, 8, 4, 2]
    }

    _default_blocks_cfg = dict(type='SNGANGenResBlock')

    def __init__(self,
                 output_scale,
                 noise_size,
                 num_classes=0,
                 out_channels=3,
                 base_channels=128,
                 input_scale=4,
                 blocks_cfg=None,
                 channel_cfg=None,
                 auto_sync_bn=True,
                 act_cfg=dict(type='ReLU')):
        """this is a docstring
        TODO:
        """
        super().__init__()

        self.input_scale = input_scale
        self.output_scale = output_scale
        self.noise_size = noise_size

        self.blocks_cfg = deepcopy(self._default_blocks_cfg)
        if blocks_cfg is not None:
            self.blocks_cfg.update(blocks_cfg)
        if 'auto_sync_bn' not in self.blocks_cfg:
            self.blocks_cfg['auto_sync_bn'] = auto_sync_bn
        self.blocks_cfg['num_classes'] = num_classes
        self.blocks_cfg['act_cfg'] = act_cfg

        channel_cfg = self._defualt_channels_cfg if channel_cfg is None \
            else channel_cfg
        if isinstance(channel_cfg, dict):
            if output_scale not in channel_cfg:
                raise KeyError(f'`output_scale={output_scale} is not found in '
                               '`channel_cfg`, only support configs for '
                               f'{[chn for chn in channel_cfg.keys()]}')
            self.channel_factor_list = channel_cfg[output_scale]
        elif isinstance(channel_cfg, list):
            self.channel_factor_list = channel_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(channel_cfg)}')

        self.noise2feat = nn.Linear(
            noise_size,
            input_scale**2 * base_channels * self.channel_factor_list[0])

        self.conv_blocks = nn.ModuleList()
        for idx in range(len(self.channel_factor_list)):
            factor_input = self.channel_factor_list[idx]
            factor_output = self.channel_factor_list[idx+1] \
                if idx < len(self.channel_factor_list)-1 else 1

            # get block-specific config
            block_cfg_ = deepcopy(self.blocks_cfg)
            block_cfg_['in_channels'] = factor_input * base_channels
            block_cfg_['out_channels'] = factor_output * base_channels
            self.conv_blocks.append(build_from_cfg(block_cfg_, MODULES))

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

        self.init_weights()

    def forward(self,
                noise,
                num_batches=0,
                label=None,
                return_noise=False,
                return_label=True):
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

        if isinstance(label, torch.Tensor):
            assert label.ndim == 1, ('The label shoube be in shape of (n, )'
                                     f'but got {label.shape}.')
            label_batch = label
        elif callable(label):
            label_generator = label
            assert num_batches > 0
            label_batch = label_generator((num_batches))
        else:
            assert num_batches > 0
            label_batch = torch.randint(0, self.num_classes, (num_batches, ))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))

        x = self.noise2feat(noise_batch)
        x = x.reshape(x.size(0), -1, self.input_scale, self.input_scale)

        for conv_block in self.conv_blocks:
            x = conv_block(x, label_batch)

        out_img = self.to_rgb(x)

        out_dict = dict()
        if return_noise:
            out_dict['noise'] = noise_batch
        if return_label:
            out_dict['label'] = label_batch
        if not out_dict:
            return out_img
        else:
            out_dict['fake_img'] = out_img
            return out_dict

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            xavier_init(self.noise2feat, gain=1, distribution='uniform')
            xavier_init(self.to_rgb.conv, gain=1, distribution='uniform')
        else:
            raise TypeError("'pretrined' must be a str or None. "
                            f'But receive {type(pretrained)}.')


# @MODULES.register_module('SNGANDiscriminator')
# @MODULES.register_module('BigGANDiscriminator')
class ProjDiscriminator(nn.Module):

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

    _default_from_rgb_cfg = dict(type='SNGANDiscHeadResBlock')

    _default_blocks_cfg = dict(type='SNGANDiscResBlock')

    def __init__(self,
                 input_scale,
                 base_channels,
                 num_classes=0,
                 input_channels=3,
                 from_rgb_cfg=None,
                 blocks_cfg=None,
                 channel_cfg=None,
                 downsample_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=True):
        """"""
        super().__init__()

        # load and update from_rgb_cfg
        self.from_rgb_cfg = deepcopy(self._default_from_rgb_cfg)
        if from_rgb_cfg is not None:
            self.from_rgb_cfg.update(from_rgb_cfg)
        self.from_rgb_cfg['act_cfg'] = act_cfg
        self.from_rgb_cfg['with_spectral_norm'] = with_spectral_norm

        # load and update blocks_cfg
        self.blocks_cfg = deepcopy(self._default_blocks_cfg)
        if blocks_cfg is not None:
            self.blocks_cfg.update(blocks_cfg)
        self.blocks_cfg['act_cfg'] = act_cfg
        self.blocks_cfg['with_spectral_norm'] = with_spectral_norm

        # get config for channel factor list coresponding to the input_scale
        # channel_factor_list:
        channel_cfg = self._defualt_channels_cfg if channel_cfg is None \
            else channel_cfg
        if isinstance(channel_cfg, dict):
            if input_scale not in channel_cfg:
                raise KeyError(f'`input_scale={input_scale} is not found in '
                               '`channel_cfg`, only support configs for '
                               f'{[chn for chn in channel_cfg.keys()]}')
            self.channel_factor_list = channel_cfg[input_scale]

        # get config for downsample coresponding to the input_scale
        # downsample_cfg: whether apply downsample in this block
        downsample_cfg = self._defualt_downsample_list_cfg \
            if downsample_cfg is None else downsample_cfg
        if isinstance(downsample_cfg, dict):
            if input_scale not in downsample_cfg:
                raise KeyError(f'`output_scale={input_scale} is not found in '
                               '`downsample_cfg`, only support configs for '
                               f'{[chn for chn in downsample_cfg.keys()]}')
            self.downsample_list = downsample_cfg[input_scale]

        if len(downsample_cfg) != len(channel_cfg):
            raise ValueError(
                '`downsample_cfg` should have same length with `channel_cfg`, '
                f'but receive {len(downsample_cfg)} and {len(channel_cfg)}.')

        self.from_rgb = build_from_cfg(
            self.from_rgb_cfg, MODULES,
            dict(in_channels=input_channels, out_channels=base_channels))

        self.blocks = nn.ModuleList()
        for idx in range(len(downsample_cfg)):
            factor_input = 1 if idx == 0 else self.channel_factor_list[idx - 1]
            factor_output = self.channel_factor_list[idx]

            # get block-specific config
            block_cfg_ = deepcopy(self.blocks_cfg)
            block_cfg_['downsample'] = self.downsample_list[idx]
            block_cfg_['in_channels'] = factor_input * base_channels
            block_cfg_['out_channels'] = factor_output * base_channels
            self.blocks.append(build_from_cfg(block_cfg_, MODULES))

        self.decision = nn.Linear(factor_output * base_channels, 1, bias=False)
        if with_spectral_norm:
            self.decision = spectral_norm(self.decision)

        self.num_classes = num_classes
        if num_classes > 0:
            self.proj_y = nn.Embedding(num_classes,
                                       factor_output * base_channels)
            if with_spectral_norm:
                self.proj_y = spectral_norm(self.proj_y)

        self.activate = build_activation_layer(act_cfg)
        self.init_weights()

    def forward(self, x, label=None):
        h = self.from_rgb(x)
        for block in self.blocks:
            h = block(h)
        h = self.activate(h)
        h = torch.sum(h, dim=[2, 3])
        out = self.decision(h)

        if self.num_classes > 0:
            w_y = self.proj_y(label)
            out = out + torch.sum(w_y * h, dim=1, keepdim=True)
        return out

    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            xavier_init(self.decision, gain=1, distribution='uniform')
            xavier_init(self.proj_y, gain=1, distribution='uniform')
        else:
            raise TypeError("'pretrained' must by a str or None. "
                            f'But receive {type(pretrained)}.')
