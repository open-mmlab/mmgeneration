# Copyright (c) OpenMMLab. All rights reserved.
from .generator_discriminator import ProjDiscriminator, SNGANGenerator
from .modules import SNGANDiscHeadResBlock, SNGANDiscResBlock, SNGANGenResBlock

__all__ = [
    'ProjDiscriminator', 'SNGANGenerator', 'SNGANGenResBlock',
    'SNGANDiscResBlock', 'SNGANDiscHeadResBlock'
]
