# Copyright (c) OpenMMLab. All rights reserved.
from .generator_discriminator_v1 import (StyleGAN1Discriminator,
                                         StyleGANv1Generator)
from .generator_discriminator_v2 import (StyleGAN2Discriminator,
                                         StyleGANv2Generator)
from .generator_discriminator_v3 import StyleGANv3Generator
from .mspie import MSStyleGAN2Discriminator, MSStyleGANv2Generator

__all__ = [
    'StyleGAN2Discriminator', 'StyleGANv2Generator', 'StyleGANv1Generator',
    'StyleGAN1Discriminator', 'MSStyleGAN2Discriminator',
    'MSStyleGANv2Generator', 'StyleGANv3Generator'
]
