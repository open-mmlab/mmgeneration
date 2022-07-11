# Copyright (c) OpenMMLab. All rights reserved.
from mmgen.registry import MODELS
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
