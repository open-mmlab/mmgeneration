from .generator_discriminator import BigGANDiscriminator, BigGANGenerator
from .modules import (BigGANConditionBN, BigGANDiscResBlock, BigGANGenResBlock,
                      SelfAttentionBlock)

__all__ = [
    'BigGANGenerator', 'BigGANGenResBlock', 'BigGANConditionBN',
    'BigGANDiscriminator', 'SelfAttentionBlock', 'BigGANDiscResBlock'
]
