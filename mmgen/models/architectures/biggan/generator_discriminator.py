from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn import normal_init, xavier_init
from mmcv.cnn.bricks import build_activation_layer
from mmcv.runner import load_checkpoint
from torch.nn.utils import spectral_norm

from mmgen.models.builder import MODULES, build_module
from mmgen.utils import get_root_logger
from ..common import get_module_device
from .modules import SelfAttentionBlock, SNConvModule


@MODULES.register_module()
class BigGANGenerator(nn.Module):
    # TODO:
    """[summary]

        Args:
            output_scale ([type]): [description]
            noise_size (int, optional): [description]. Defaults to 120.
            num_classes (int, optional): [description]. Defaults to 0.
            out_channels (int, optional): [description]. Defaults to 3.
            base_channels (int, optional): [description]. Defaults to 96.
            input_scale (int, optional): [description]. Defaults to 4.
            with_shared_embedding (bool, optional): [description]. Defaults to 
                True.
            shared_dim (int, optional): [description]. Defaults to 128.
            sn_eps ([type], optional): [description]. Defaults to 1e-6.
            init_type (str, optional): [description]. Defaults to 'ortho'.
            split_noise (bool, optional): [description]. Defaults to True.
            act_cfg ([type], optional): [description]. Defaults to 
                dict(type='ReLU').
            conv_cfg ([type], optional): [description]. Defaults to 
                dict(type='Conv2d').
            upsample_cfg ([type], optional): [description]. Defaults to 
                dict(type='nearest', scale_factor=2).
            with_spectral_norm (bool, optional): [description]. Defaults to 
                True.
            auto_sync_bn (bool, optional): [description]. Defaults to True.
            blocks_cfg ([type], optional): [description]. Defaults to 
                dict(type='BigGANGenResBlock').
            norm_cfg ([type], optional): [description]. Defaults to None.
            arch_cfg ([type], optional): [description]. Defaults to None.
            out_norm_cfg ([type], optional): [description]. Defaults to 
                dict(type='BN').
            pretrained ([type], optional): [description]. Defaults to None.
        """

    def __init__(self,
                 output_scale,
                 noise_size=120,
                 num_classes=0,
                 out_channels=3,
                 base_channels=96,
                 input_scale=4,
                 with_shared_embedding=True,
                 shared_dim=128,
                 sn_eps=1e-6,
                 init_type='ortho',
                 split_noise=True,
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=dict(type='Conv2d'),
                 upsample_cfg=dict(type='nearest', scale_factor=2),
                 with_spectral_norm=True,
                 auto_sync_bn=True,
                 blocks_cfg=dict(type='BigGANGenResBlock'),
                 norm_cfg=None,
                 arch_cfg=None,
                 out_norm_cfg=dict(type='BN'),
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.noise_size = noise_size
        self.num_classes = num_classes
        self.shared_dim = shared_dim
        self.with_shared_embedding = with_shared_embedding
        self.output_scale = output_scale
        self.arch = arch_cfg if arch_cfg else self._get_default_arch_cfg(
            self.output_scale, base_channels)
        self.input_scale = input_scale
        self.split_noise = split_noise
        self.blocks_cfg = deepcopy(blocks_cfg)
        self.upsample_cfg = deepcopy(upsample_cfg)

        # Validity Check
        # If 'num_classes' equals to zero, we shall set 'with_shared_embedding'
        # to False and
        if num_classes == 0:
            assert not self.with_shared_embedding
        else:
            if not self.with_shared_embedding:
                # If not `with_shared_embedding`, we will use `nn.Embedding` to
                # replace the original `Linear` layer in conditional BN.
                # Meanwhile, we do not adopt split noises.
                assert not self.split_noise

        # If using split latents, we may need to adjust noise_size
        if self.split_noise:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.noise_chunk_size = self.noise_size // self.num_slots
            # Recalculate latent dimensionality for even splitting into chunks
            self.noise_size = self.noise_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.noise_chunk_size = 0

        # First linear layer
        self.noise2feat = nn.Linear(
            self.noise_size // self.num_slots,
            self.arch['in_channels'][0] * (self.input_scale**2))
        if with_spectral_norm:
            self.noise2feat = spectral_norm(self.noise2feat, eps=sn_eps)

        # If using 'shared_embedding', we will get an unified embedding of
        # label for all blocks. If not, we just pass on the label to each
        # block.
        if with_shared_embedding:
            self.shared_embedding = nn.Embedding(num_classes, shared_dim)
        else:
            self.shared_embedding = nn.Identity()

        if num_classes > 0:
            self.dim_after_concat = (
                self.shared_dim + self.noise_chunk_size
                if self.with_shared_embedding else self.num_classes)
        else:
            self.dim_after_concat = self.noise_chunk_size

        self.blocks_cfg.update(
            dict(
                dim_after_concat=self.dim_after_concat,
                act_cfg=act_cfg,
                conv_cfg=conv_cfg,
                sn_eps=sn_eps,
                label_input=(num_classes > 0) and (not with_shared_embedding),
                with_spectral_norm=with_spectral_norm,
                auto_sync_bn=auto_sync_bn))

        self.conv_blocks = nn.ModuleList()
        for index in range(len(self.arch['out_channels'])):
            # change args to adapt to current block
            self.blocks_cfg.update(
                dict(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    upsample_cfg=self.upsample_cfg
                    if self.arch['upsample'][index] else None))
            self.conv_blocks.append(build_module(self.blocks_cfg))
            if self.arch['attention'][index]:
                self.conv_blocks.append(
                    SelfAttentionBlock(
                        self.arch['out_channels'][index],
                        with_spectral_norm=with_spectral_norm,
                        sn_eps=sn_eps))

        self.output_layer = SNConvModule(
            self.arch['out_channels'][-1],
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            with_spectral_norm=with_spectral_norm,
            spectral_norm_cfg=dict(eps=sn_eps),
            act_cfg=act_cfg,
            norm_cfg=out_norm_cfg,
            bias=True,
            order=('norm', 'act', 'conv'))

        self.init_weights(pretrained=pretrained, init_type=init_type)

    def _get_default_arch_cfg(self, output_scale, base_channels):
        assert output_scale in [32, 64, 128, 256, 512]
        _default_arch_cfgs = {
            '32': {
                'in_channels': [base_channels * item for item in [4, 4, 4]],
                'out_channels': [base_channels * item for item in [4, 4, 4]],
                'upsample': [True] * 3,
                'resolution': [8, 16, 32],
                'attention': [False, False, False]
            },
            '64': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 4]],
                'out_channels':
                [base_channels * item for item in [16, 8, 4, 2]],
                'upsample': [True] * 4,
                'resolution': [8, 16, 32, 64],
                'attention': [False, False, False, True]
            },
            '128': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 4, 2]],
                'out_channels':
                [base_channels * item for item in [16, 8, 4, 2, 1]],
                'upsample': [True] * 5,
                'resolution': [8, 16, 32, 64, 128],
                'attention': [False, False, False, True, False]
            },
            '256': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 8, 4, 2]],
                'out_channels':
                [base_channels * item for item in [16, 8, 8, 4, 2, 1]],
                'upsample': [True] * 6,
                'resolution': [8, 16, 32, 64, 128, 256],
                'attention': [False, False, False, True, False, False]
            },
            '512': {
                'in_channels':
                [base_channels * item for item in [16, 16, 8, 8, 4, 2, 1]],
                'out_channels':
                [base_channels * item for item in [16, 8, 8, 4, 2, 1, 1]],
                'upsample': [True] * 7,
                'resolution': [8, 16, 32, 64, 128, 256, 512],
                'attention': [False, False, False, True, False, False, False]
            }
        }

        return _default_arch_cfgs[str(output_scale)]

    def forward(self, noise, label, num_batches=0, return_noise=False):
        # TODO:
        """[summary]

        Args:
            noise ([type]): [description]
            label ([type]): [description]
            num_batches (int, optional): [description]. Defaults to 0.
            return_noise (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

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

        if self.num_classes == 0:
            label_batch = None

        elif isinstance(label, torch.Tensor):
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
        if label_batch is not None:
            label_batch = label_batch.to(get_module_device(self))
            class_vector = self.shared_embedding(label_batch)
        else:
            class_vector = None
        # If 'split noise', concat class vector and noise chunk
        if self.split_noise:
            zs = torch.split(noise_batch, self.noise_chunk_size, 1)
            z = zs[0]
            if class_vector is not None:
                ys = [torch.cat([class_vector, item], 1) for item in zs[1:]]
            else:
                ys = zs[1:]
        else:
            ys = [class_vector] * len(self.conv_blocks)
            z = noise_batch

        # First linear layer
        x = self.noise2feat(z)
        # Reshape
        x = x.view(x.size(0), -1, self.input_scale, self.input_scale)

        # Loop over blocks
        counter = 0
        for index, conv_block in enumerate(self.conv_blocks):
            # Second inner loop in case block has multiple layers
            if isinstance(conv_block, SelfAttentionBlock):
                x = conv_block(x)
            else:
                x = conv_block(x, ys[counter])
                counter += 1

        # Apply batchnorm-relu-conv-tanh at output
        out_img = torch.tanh(self.output_layer(x))

        if return_noise:
            output = dict(
                fake_img=out_img, noise_batch=noise_batch, label=label)
            return output

        return out_img

    def init_weights(self, pretrained=None, init_type='ortho'):
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
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                    if init_type == 'ortho':
                        nn.init.orthogonal_(m.weight)
                    elif init_type == 'N02':
                        normal_init(m, 0.0, 0.02)
                    elif init_type == 'xavier':
                        xavier_init(m)
                    else:
                        raise NotImplementedError(
                            f'{init_type} initialization \
                            not supported now.')
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')


@MODULES.register_module()
class BigGANDiscriminator(nn.Module):
    # TODO:
    """[summary]

        Args:
            input_scale ([type]): [description]
            num_classes (int, optional): [description]. Defaults to 0.
            in_channels (int, optional): [description]. Defaults to 3.
            out_channels (int, optional): [description]. Defaults to 1.
            base_channels (int, optional): [description]. Defaults to 96.
            sn_eps ([type], optional): [description]. Defaults to 1e-6.
            init_type (str, optional): [description]. Defaults to 'ortho'.
            act_cfg ([type], optional): [description]. Defaults to dict(type=
                'ReLU').
            conv_cfg ([type], optional): [description]. Defaults to dict(type=
                'Conv2d').
            with_spectral_norm (bool, optional): [description]. Defaults to 
                True.
            blocks_cfg ([type], optional): [description]. Defaults to dict(
                    type='BigGANDiscResBlock').
            arch_cfg ([type], optional): [description]. Defaults to None.
            pretrained ([type], optional): [description]. Defaults to None.
        """

    def __init__(self,
                 input_scale,
                 num_classes=0,
                 in_channels=3,
                 out_channels=1,
                 base_channels=96,
                 sn_eps=1e-6,
                 init_type='ortho',
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=dict(type='Conv2d'),
                 with_spectral_norm=True,
                 blocks_cfg=dict(type='BigGANDiscResBlock'),
                 arch_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.input_scale = input_scale
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.arch = arch_cfg if arch_cfg else self._get_default_arch_cfg(
            self.input_scale, self.in_channels, self.base_channels)
        self.blocks_cfg = deepcopy(blocks_cfg)
        self.blocks_cfg.update(
            dict(
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                sn_eps=sn_eps,
                with_spectral_norm=with_spectral_norm))

        self.conv_blocks = nn.ModuleList()
        for index in range(len(self.arch['out_channels'])):
            # change args to adapt to current block
            self.blocks_cfg.update(
                dict(
                    in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['out_channels'][index],
                    with_downsample=self.arch['downsample'][index],
                    head_block=(index == 0)))
            self.conv_blocks.append(build_module(self.blocks_cfg))
            if self.arch['attention'][index]:
                self.conv_blocks.append(
                    SelfAttentionBlock(
                        self.arch['out_channels'][index],
                        with_spectral_norm=with_spectral_norm,
                        sn_eps=sn_eps))

        self.activation = build_activation_layer(act_cfg)

        self.linear = nn.Linear(self.arch['out_channels'][-1], out_channels)
        if with_spectral_norm:
            self.linear = spectral_norm(self.linear, eps=sn_eps)

        self.proj_y = nn.Embedding(self.num_classes,
                                   self.arch['out_channels'][-1])
        if with_spectral_norm:
            self.proj_y = spectral_norm(self.proj_y, eps=sn_eps)

        self.init_weights(pretrained=pretrained, init_type=init_type)

    def _get_default_arch_cfg(self, input_scale, in_channels, base_channels):
        assert input_scale in [32, 64, 128, 256, 512]
        _default_arch_cfgs = {
            '32': {
                'in_channels':
                [in_channels] + [base_channels * item for item in [4, 4, 4]],
                'out_channels':
                [base_channels * item for item in [4, 4, 4, 4]],
                'downsample': [True, True, False, False],
                'resolution': [16, 8, 8, 8],
                'attention': [False, False, False, False]
            },
            '64': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 2, 4, 8]],
                'out_channels':
                [base_channels * item for item in [1, 2, 4, 8, 16]],
                'downsample': [True] * 4 + [False],
                'resolution': [32, 16, 8, 4, 4],
                'attention': [False, False, False, False, False]
            },
            '128': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 2, 4, 8, 16]],
                'out_channels':
                [base_channels * item for item in [1, 2, 4, 8, 16, 16]],
                'downsample': [True] * 5 + [False],
                'resolution': [64, 32, 16, 8, 4, 4],
                'attention': [True, False, False, False, False, False]
            },
            '256': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 2, 4, 8, 8, 16]],
                'out_channels':
                [base_channels * item for item in [1, 2, 4, 8, 8, 16, 16]],
                'downsample': [True] * 6 + [False],
                'resolution': [128, 64, 32, 16, 8, 4, 4],
                'attention': [False, True, False, False, False, False]
            },
            '512': {
                'in_channels': [in_channels] +
                [base_channels * item for item in [1, 1, 2, 4, 8, 8, 16]],
                'out_channels':
                [base_channels * item for item in [1, 1, 2, 4, 8, 8, 16, 16]],
                'downsample': [True] * 7 + [False],
                'resolution': [256, 128, 64, 32, 16, 8, 4, 4],
                'attention': [False, False, False, True, False, False, False]
            }
        }

        return _default_arch_cfgs[str(input_scale)]

    def forward(self, x, label=None):
        # TODO:
        """[summary]

        Args:
            x ([type]): [description]
            label ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        x0 = x
        for conv_block in self.conv_blocks:
            x0 = conv_block(x0)
        x0 = self.activation(x0)
        x0 = torch.sum(x0, dim=[2, 3])
        out = self.linear(x0)

        if self.num_classes > 0:
            w_y = self.proj_y(label)
            out = out + torch.sum(w_y * x0, dim=1, keepdim=True)
        return out

    def init_weights(self, pretrained=None, init_type='ortho'):
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
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                    if init_type == 'ortho':
                        nn.init.orthogonal_(m.weight)
                    elif init_type == 'N02':
                        normal_init(m, 0.0, 0.02)
                    elif init_type == 'xavier':
                        xavier_init(m)
                    else:
                        raise NotImplementedError(
                            f'{init_type} initialization \
                            not supported now.')
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
