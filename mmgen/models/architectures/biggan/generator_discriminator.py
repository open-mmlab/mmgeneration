import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import load_checkpoint
from torch.nn.utils import spectral_norm

from mmgen.models.builder import MODULES, build_module
from mmgen.utils import get_root_logger
from ..common import get_module_device
from .modules import SelfAttentionBlock


@MODULES.register_module()
class BigGANGenerator(nn.Module):

    def __init__(self,
                 output_scale,
                 noise_size=120,
                 num_classes=0,
                 out_channels=3,
                 base_channels=96,
                 input_scale=4,
                 with_shared_embedding=True,
                 shared_dim=128,
                 init_type='ortho',
                 split_noise=True,
                 act_cfg=dict(type='ReLU'),
                 conv_cfg=dict(type='Conv2d'),
                 with_spectral_norm=True,
                 auto_sync_bn=True,
                 blocks_cfg=dict(type='BigGANGenResBlock'),
                 norm_cfg=None,
                 arch_cfg=None,
                 out_norm_cfg=dict(type='BN'),
                 pretrained=None):
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

        # Validity Check
        if self.num_classes == 0:
            assert self.shared_dim == 0
        if self.with_shared_embedding is False:
            # If with_shared_embedding unused, we will use nn.Embedding replace
            # Linear layer for gain and bias in ccbn. In that situation, we
            # set split_noise unused
            assert self.split_noise is False

        # If using split latents, we may need to adjust noise_size
        if self.split_noise:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.noise_chunk_size = (self.noise_size // self.num_slots)
            # Recalculate latent dimensionality for even splitting into chunks
            self.noise_size = self.noise_chunk_size * self.num_slots
        else:
            self.num_slots = 1
            self.noise_chunk_size = 0

        self.linear = spectral_norm(
            nn.Linear(self.noise_size // self.num_slots,
                      self.arch['in_channels'][0] * (self.input_scale**2)),
            eps=1e-6)

        if with_shared_embedding:
            self.shared_embedding = nn.Embedding(num_classes, shared_dim)
        else:
            self.shared_embedding = nn.Identity()

        self.dim_after_concat = (
            self.shared_dim + self.noise_chunk_size
            if self.with_shared_embedding else self.num_classes)
        blocks_cfg.update(
            dict(
                dim_after_concat=self.dim_after_concat,
                act_cfg=act_cfg,
                shared_embedding=with_shared_embedding,
                with_spectral_norm=with_spectral_norm,
                auto_sync_bn=auto_sync_bn))

        self.conv_blocks = nn.ModuleList()
        for index in range(len(self.arch['out_channels'])):
            # change args to adapt to current block
            blocks_cfg.update(
                dict(in_channels=self.arch['in_channels'][index]))
            blocks_cfg.update(
                dict(out_channels=self.arch['out_channels'][index]))
            self.conv_blocks.append(build_module(blocks_cfg))
            if self.arch['attention'][index]:
                self.conv_blocks.append(
                    SelfAttentionBlock(self.arch['out_channels'][index]))

        self.output_layer = ConvModule(
            self.arch['out_channels'][-1],
            out_channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            with_spectral_norm=with_spectral_norm,
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
        # If hierarchical, concatenate zs and ys
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
        x = self.linear(z)
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
            # TODO:
            pass
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
