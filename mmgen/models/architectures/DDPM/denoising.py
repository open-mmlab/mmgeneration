from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import load_checkpoint

from mmgen.models.builder import MODULES, build_module
from mmgen.utils import get_root_logger
from .modules import EmbedSequential


@MODULES.register_module()
class DenoisingUnet(nn.Module):
    """Denoising Unet."""

    _default_channels_cfg = {
        256: [1, 1, 2, 2, 4, 4],
        64: [1, 2, 3, 4],
        32: [1, 2, 2, 2]
    }

    def __init__(self,
                 image_size,
                 in_channels,
                 base_channels,
                 resblocks_per_downsample,
                 num_timesteps,
                 rescale_timesteps,
                 dropout=0,
                 embedding_channels=-1,
                 num_classes=0,
                 channels_cfg=None,
                 output_cfg=dict(mean='eps'),
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1,
                 use_scale_shift_norm=False,
                 num_heads=4,
                 resblock_cfg=dict(type='DenoisingResBlock'),
                 attention_cfg=dict(type='MultiHeadAttention'),
                 downsample_conv=True,
                 upsample_conv=True,
                 downsample_cfg=dict(type='DenoisingDownsample'),
                 upsample_cfg=dict(type='DenoisingUpsample'),
                 attention_res=[16, 8],
                 use_checkpoint=False):

        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.rescale_timesteps = rescale_timesteps

        self.output_cfg = deepcopy(output_cfg)
        self.mean_cfg = getattr(self.output_cfg, 'mean', 'eps')
        self.var_cfg = getattr(self.output_cfg, 'var', None)

        # double output_channels to output mean and var at same time
        out_channels = in_channels if self.var_cfg is None else 2 * in_channels
        self.out_channels = out_channels

        channels_cfg_ = deepcopy(self._default_channels_cfg)
        if channels_cfg is not None:
            channels_cfg_ = channels_cfg_.update(channels_cfg)
        if isinstance(channels_cfg_, dict):
            if image_size not in channels_cfg_:
                raise KeyError(f'`image_size={image_size} is not found in '
                               '`channel_cfg`, only support configs for '
                               f'{[chn for chn in channels_cfg_.keys()]}')
            self.channel_factor_list = channels_cfg_[image_size]
        elif isinstance(channels_cfg_, list):
            self.channel_factor_list = channels_cfg_
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(channels_cfg_)}')

        embedding_channels = base_channels * 4 \
            if embedding_channels == -1 else embedding_channels
        self.time_embedding = build_module(
            dict(type='TimeEmbedding'), {
                'in_channels': base_channels,
                'embedding_channels': embedding_channels
            })
        if self.num_classes != 0:
            self.label_embedding = nn.Embedding(self.num_classes,
                                                embedding_channels)

        # TODO: add set use_checkpoints later
        self.resblock_cfg = deepcopy(resblock_cfg)
        self.resblock_cfg.setdefault('dropout', dropout)
        self.resblock_cfg.setdefault('norm_cfg', norm_cfg)
        self.resblock_cfg.setdefault('act_cfg', act_cfg)
        self.resblock_cfg.setdefault('embedding_channels', embedding_channels)
        self.resblock_cfg.setdefault('use_scale_shift_norm',
                                     use_scale_shift_norm)
        self.resblock_cfg.setdefault('shortcut_kernel_size',
                                     shortcut_kernel_size)

        attention_ds = [image_size // int(res) for res in attention_res]
        self.attention_cfg = deepcopy(attention_cfg)
        self.attention_cfg.setdefault('num_heads', num_heads)
        self.attention_cfg.setdefault('norm_cfg', norm_cfg)

        self.downsample_cfg = deepcopy(downsample_cfg)
        self.downsample_cfg.setdefault('use_conv', downsample_conv)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.upsample_cfg.setdefault('use_conv', upsample_conv)

        ds = 1
        self.in_blocks = nn.ModuleList([
            EmbedSequential(
                nn.Conv2d(in_channels, base_channels, 3, 1, padding=1))
        ])
        self.in_channels_list = [base_channels]
        for level, factor in enumerate(self.channel_factor_list):
            in_channels_ = base_channels if level == 0 \
                else base_channels * self.channel_factor_list[level - 1]
            out_channels_ = base_channels * factor

            for _ in range(resblocks_per_downsample):
                layers = [
                    build_module(self.resblock_cfg, {
                        'in_channels': in_channels_,
                        'out_channels': out_channels_
                    })
                ]
                in_channels_ = out_channels_

                if ds in attention_ds:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))

                self.in_channels_list.append(in_channels_)
                self.in_blocks.append(EmbedSequential(*layers))

            if level != len(self.channel_factor_list) - 1:
                self.in_blocks.append(
                    EmbedSequential(
                        build_module(self.downsample_cfg,
                                     {'in_channels': in_channels_})))
                self.in_channels_list.append(in_channels_)
                ds *= 2

        self.mid_blocks = EmbedSequential(
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
            build_module(self.attention_cfg, {'in_channels': in_channels_}),
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
        )

        in_channels_list = deepcopy(self.in_channels_list)
        self.out_blocks = nn.ModuleList()
        for level, factor in enumerate(self.channel_factor_list[::-1]):
            for idx in range(resblocks_per_downsample + 1):
                layers = [
                    build_module(
                        self.resblock_cfg, {
                            'in_channels':
                            in_channels_ + in_channels_list.pop(),
                            'out_channels': base_channels * factor
                        })
                ]
                in_channels_ = base_channels * factor
                if ds in attention_ds:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))
                if (level != len(self.channel_factor_list) - 1
                        and idx == resblocks_per_downsample):
                    layers.append(
                        build_module(self.upsample_cfg,
                                     {'in_channels': in_channels_}))
                    ds //= 2
                self.out_blocks.append(EmbedSequential(*layers))

        self.out = ConvModule(
            in_channels=in_channels_,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            order=('norm', 'act', 'conv'))

    def forward(self, x_t, t, label=None, return_noise=False):
        """Forward function.
        Args:
            x_t (torch.Tensor):
            t (torch.Tensor):
            label (torch.Tensor, optional):
            return_noise (bool, optional):

        Returns:
            torch.Tensor | dict:
        """

        if self.rescale_timesteps:
            t = t.float() * (1000.0 / self.num_timesteps)
        embedding = self.time_embedding(t)

        if label is not None:
            assert hasattr(self, 'label_embedding')
            embedding = self.label_embedding(label) + embedding

        h, hs = x_t, []
        # forward downsample blocks
        for block in self.in_blocks:
            h = block(h, embedding)
            hs.append(h)

        # forward middle blocks
        h = self.mid_blocks(h, embedding)

        # forward upsample blocks
        for block in self.out_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), embedding)
        outputs = self.out(h)

        # split mean and learned from output
        output_dict = dict()
        if self.var_cfg is not None and 'FIXED' not in self.var_cfg.upper():
            mean, var = outputs.split(self.out_channels // 2, dim=1)
            if self.var_cfg.upper() == 'LEARNED_RANGE':
                # rescale [-1, 1] to [0, 1]
                output_dict['factor'] = (var + 1) / 2
            elif self.var_cfg.upper() == 'LEARNED':
                output_dict['log_var'] = var
            else:
                raise AttributeError()
        else:
            mean = outputs

        if self.mean_cfg.upper() == 'EPS':
            output_dict['eps_t_pred'] = mean
        elif self.mean_cfg.upper() == 'START_X':
            output_dict['x_0_pred'] = mean
        elif self.mean_cfg.upper() == 'PREVIOUS_X':
            output_dict['x_tm1_pred'] = mean

        if return_noise:
            output_dict['x_t'] = x_t
            output_dict['t'] = t

        return output_dict

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
            # TODO: official unet use serval zero-init, so werid
            pass
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
