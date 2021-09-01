# Copyright (c) OpenMMLab. All rights reserved.
from .generator_discriminator import BigGANDiscriminator, BigGANGenerator
from .generator_discriminator_deep import (BigGANDeepDiscriminator,
                                           BigGANDeepGenerator)
from .modules import (BigGANConditionBN, BigGANDeepDiscResBlock,
                      BigGANDeepGenResBlock, BigGANDiscResBlock,
                      BigGANGenResBlock, SelfAttentionBlock, SNConvModule)

__all__ = [
    'BigGANGenerator', 'BigGANGenResBlock', 'BigGANConditionBN',
    'BigGANDiscriminator', 'SelfAttentionBlock', 'BigGANDiscResBlock',
    'BigGANDeepDiscriminator', 'BigGANDeepGenerator', 'BigGANDeepDiscResBlock',
    'BigGANDeepGenResBlock', 'SNConvModule'
]
