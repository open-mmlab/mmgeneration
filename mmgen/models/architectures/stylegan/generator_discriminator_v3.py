from copy import deepcopy

import mmcv
import torch
import torch.nn as nn
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from .utils import get_mean_latent


@MODULES.register_module()
class StyleGANv3Generator(nn.Module):
    """StyleGAN3 Generator.

    In StyleGAN3, we make several changes to StyleGANv2's generator which
    include transformed fourier features, filtered nonlinearities and
    non-critical sampling, etc. More details can be found in: Alias-Free
    Generative Adversarial Networks NeurIPS'2021.

    Ref: https://github.com/NVlabs/stylegan3

    Args:
        out_size (int): The output size of the StyleGAN3 generator.
        style_channels (int): The number of channels for style code.
        img_channels (int): The number of output's channels.
        noise_size (int, optional): Size of the input noise vector.
            Defaults to 512.
        c_dim (int, optional): Size of the input noise vector.
            Defaults to 0.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to 0.9.
        rgb2bgr (bool, optional): Whether to reformat the output channels
                with order `bgr`. We provide several pre-trained StyleGAN3
                weights whose output channels order is `rgb`. You can set
                this argument to True to use the weights.
        pretrained (str | dict, optional): Path for the pretrained model or
            dict containing information for pretained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict. Defaults to None.
        synthesis_cfg (dict, optional): Config for synthesis network. Defaults
            to dict(type='SynthesisNetwork').
        mapping_cfg (dict, optional): Config for mapping network. Defaults to
            dict(type='MappingNetwork').
    """

    def __init__(self,
                 out_size,
                 style_channels,
                 img_channels,
                 noise_size=512,
                 c_dim=0,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 rgb2bgr=False,
                 pretrained=None,
                 synthesis_cfg=dict(type='SynthesisNetwork'),
                 mapping_cfg=dict(type='MappingNetwork')):
        super().__init__()
        self.noise_size = noise_size
        self.c_dim = c_dim
        self.style_channels = style_channels
        self.out_size = out_size
        self.img_channels = img_channels
        self.rgb2bgr = rgb2bgr
        self._default_style_mode = default_style_mode
        self.default_style_mode = default_style_mode
        self.eval_style_mode = eval_style_mode

        self._synthesis_cfg = deepcopy(synthesis_cfg)
        self._synthesis_cfg.setdefault('style_channels', style_channels)
        self._synthesis_cfg.setdefault('out_size', out_size)
        self._synthesis_cfg.setdefault('img_channels', img_channels)
        self.synthesis = build_module(self._synthesis_cfg)

        self.num_ws = self.synthesis.num_ws
        self._mapping_cfg = deepcopy(mapping_cfg)
        self._mapping_cfg.setdefault('noise_size', noise_size)
        self._mapping_cfg.setdefault('c_dim', c_dim)
        self._mapping_cfg.setdefault('style_channels', style_channels)
        self._mapping_cfg.setdefault('num_ws', self.num_ws)
        self.style_mapping = build_module(self._mapping_cfg)

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

        return super(StyleGANv3Generator, self).train(mode)

    def forward(self,
                noise,
                label=None,
                num_batches=0,
                input_is_latent=False,
                truncation=1,
                num_truncation_layer=None,
                update_emas=False,
                return_noise=False,
                return_latent=False,
                **synthesis_kwargs):
        """Forward Function for stylegan3.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            label (torch.Tensor | callable | None): You can directly give a
                batch of label through a ``torch.Tensor`` or offer a callable
                function to sample a batch of label data. Otherwise, the
                ``None`` indicates to use the default label sampler.
                Defaults to None.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            input_is_latent (bool, optional): If `True`, the input tensor is
                the latent tensor. Defaults to False.
            truncation (float, optional): Truncation factor. Give value less
                than 1., the truncation trick will be adopted. Defaults to 1.
            num_truncation_layer (int, optional): Number of layers use
                truncated latent. Defaults to None.
            truncation_latent (torch.Tensor, optional): Mean truncation latent.
                Defaults to None.
            update_emas (bool, optional): Whether update moving average of
                average w and layer input. Defaults to False.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.
            return_latents (bool, optional): If True, ``latent`` will be
                returned in a dict with ``fake_img``. Defaults to False.
        Returns:
            torch.Tensor | dict: Generated image tensor or dictionary \
                containing more data.
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

        if self.c_dim == 0:
            label_batch = None

        elif isinstance(label, torch.Tensor):
            label_batch = label
        elif callable(label):
            label_generator = label
            assert num_batches > 0
            label_batch = label_generator((num_batches, ))
        else:
            assert num_batches > 0
            label_batch = torch.randint(0, self.c_dim, (num_batches, ))

        device = get_module_device(self)
        noise_batch = noise_batch.to(device)

        if label_batch:
            label_batch = label_batch.to(device)

        ws = self.style_mapping(
            noise_batch,
            label_batch,
            truncation=truncation,
            num_truncation_layer=num_truncation_layer,
            update_emas=update_emas)
        out_img = self.synthesis(
            ws, update_emas=update_emas, **synthesis_kwargs)

        if self.rgb2bgr:
            out_img = out_img[:, [2, 1, 0], ...]

        if return_noise or return_latent:
            output = dict(
                fake_img=out_img,
                noise_batch=noise_batch,
                label=label_batch,
                latent=ws)
            return output

        return out_img

    def get_mean_latent(self, num_samples=4096, **kwargs):
        """Get mean latent of W space in this generator.

        Args:
            num_samples (int, optional): Number of sample times. Defaults
                to 4096.

        Returns:
            Tensor: Mean latent of this generator.
        """
        if hasattr(self.style_mapping, 'w_avg'):
            return self.style_mapping.w_avg
        return get_mean_latent(self, num_samples, **kwargs)
