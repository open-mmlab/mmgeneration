from .generator_discriminator import BigGANDiscriminator, BigGANGenerator
from .modules import BigGANConditionBN, BigGANGenResBlock

__all__ = [
    'BigGANGenerator', 'BigGANGenResBlock', 'BigGANConditionBN',
    'BigGANDiscriminator'
]
