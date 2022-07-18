# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmgen.core import GenDataSample
from mmgen.models.architectures.common import get_module_device
from mmgen.registry import MODELS
from mmgen.typing import ForwardOutputs, ValTestStepInputs
from ..architectures.stylegan.utils import (apply_fractional_pseudo_rotation,
                                            apply_fractional_rotation,
                                            apply_fractional_translation,
                                            apply_integer_translation,
                                            rotation_matrix)
from ..common import get_valid_num_batches
from .stylegan2 import StyleGAN2


@MODELS.register_module()
class StyleGAN3(StyleGAN2):
    """Impelmentation of `Alias-Free Generative Adversarial Networks`. # noqa.

    Paper link: https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf # noqa

    Detailed architecture can be found in

    :class:~`mmgen.models.architectures.stylegan.generator_discriminator_v3.StyleGANv3Generator`  # noqa
    and
    :class:~`mmgen.models.architectures.stylegan.generator_discriminator_v2.StyleGAN2Discriminator`  # noqa
    """

    def test_step(self, data: ValTestStepInputs) -> ForwardOutputs:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (ValTestStepInputs): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            ForwardOutputs: Generated image or image dict.
        """
        inputs_dict, data_sample = self.data_preprocessor(data)
        # hard code to compute equivarience
        if 'mode' in inputs_dict and 'eq_cfg' in inputs_dict['mode']:
            batch_size = get_valid_num_batches(inputs_dict)
            outputs = self.sample_equivarience_pairs(
                batch_size,
                sample_mode=inputs_dict['mode']['sample_mode'],
                eq_cfg=inputs_dict['mode']['eq_cfg'],
                sample_kwargs=inputs_dict['mode']['sample_kwargs'])
        else:
            outputs = self(inputs_dict, data_sample)
        return outputs

    def val_step(self, data: ValTestStepInputs) -> ForwardOutputs:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (ValTestStepInputs): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            ForwardOutputs: Generated image or image dict.
        """
        inputs_dict, data_sample = self.data_preprocessor(data)
        # hard code to compute equivarience
        if 'mode' in inputs_dict and 'eq_cfg' in inputs_dict['mode']:
            batch_size = get_valid_num_batches(inputs_dict)
            outputs = self.sample_equivarience_pairs(
                batch_size,
                sample_mode=inputs_dict['mode']['sample_mode'],
                eq_cfg=inputs_dict['mode']['eq_cfg'],
                sample_kwargs=inputs_dict['mode']['sample_kwargs'])
        else:
            outputs = self(inputs_dict, data_sample)
        return outputs

    def sample_equivarience_pairs(self,
                                  batch_size,
                                  sample_mode='ema',
                                  eq_cfg=dict(
                                      compute_eqt_int=False,
                                      compute_eqt_frac=False,
                                      compute_eqr=False,
                                      translate_max=0.125,
                                      rotate_max=1),
                                  sample_kwargs=dict()):
        generator = self.generator if (sample_mode
                                       == 'orig') else self.generator_ema
        if hasattr(generator, 'module'):
            generator = generator.module

        device = get_module_device(generator)
        identity_matrix = torch.eye(3, device=device)

        s = []

        # Run mapping network.
        z = torch.randn([batch_size, self.noise_size], device=device)
        ws = generator.style_mapping(z=z)
        transform_matrix = getattr(
            getattr(getattr(generator, 'synthesis', None), 'input', None),
            'transform', None)

        # Generate reference image.
        transform_matrix[:] = identity_matrix
        orig = generator.synthesis(ws=ws, **sample_kwargs)

        batch_sample = [GenDataSample() for _ in range(batch_size)]
        # Integer translation (EQ-T).
        if eq_cfg['compute_eqt_int']:
            t = (torch.rand(2, device=device) * 2 -
                 1) * eq_cfg['translate_max']
            t = (t * generator.out_size).round() / generator.out_size
            transform_matrix[:] = identity_matrix
            transform_matrix[:2, 2] = -t
            img = generator.synthesis(ws=ws, **sample_kwargs)
            ref, mask = apply_integer_translation(orig, t[0], t[1])

            diff = (ref - img).square() * mask
            for idx in range(batch_size):
                data_sample = batch_sample[idx]
                setattr(data_sample, 'eqt_int',
                        GenDataSample(diff=diff, mask=mask))

            s += [(ref - img).square() * mask, mask]

        # Fractional translation (EQ-T_frac).
        if eq_cfg['compute_eqt_frac']:
            t = (torch.rand(2, device=device) * 2 -
                 1) * eq_cfg['translate_max']
            transform_matrix[:] = identity_matrix
            transform_matrix[:2, 2] = -t
            img = generator.synthesis(ws=ws, **sample_kwargs)
            ref, mask = apply_fractional_translation(orig, t[0], t[1])

            diff = (ref - img).square() * mask
            for idx in range(batch_size):
                data_sample = batch_sample[idx]
                setattr(data_sample, 'eqt_frac',
                        GenDataSample(diff=diff, mask=mask))

            s += [(ref - img).square() * mask, mask]

        # Rotation (EQ-R).
        if eq_cfg['compute_eqr']:
            angle = (torch.rand([], device=device) * 2 - 1) * (
                eq_cfg['rotate_max'] * np.pi)
            transform_matrix[:] = rotation_matrix(-angle)
            img = generator.synthesis(ws=ws, **sample_kwargs)
            ref, ref_mask = apply_fractional_rotation(orig, angle)
            pseudo, pseudo_mask = apply_fractional_pseudo_rotation(img, angle)
            mask = ref_mask * pseudo_mask

            diff = (ref - pseudo).square() * mask
            for idx in range(batch_size):
                data_sample = batch_sample[idx]
                setattr(data_sample, 'eqr',
                        GenDataSample(diff=diff, mask=mask))

            s += [(ref - pseudo).square() * mask, mask]

        return batch_sample
