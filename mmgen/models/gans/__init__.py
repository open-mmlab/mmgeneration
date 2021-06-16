from .base_gan import BaseGAN
from .basic_conditional_gan import BasicConditionalGAN
from .cyclegan import CycleGAN
from .mspie_stylegan2 import MSPIEStyleGAN2
from .pix2pix import Pix2Pix
from .progressive_growing_unconditional_gan import ProgressiveGrowingGAN
from .singan import PESinGAN, SinGAN
from .static_unconditional_gan import StaticUnconditionalGAN

__all__ = [
    'BaseGAN', 'StaticUnconditionalGAN', 'ProgressiveGrowingGAN', 'SinGAN',
    'Pix2Pix', 'CycleGAN', 'MSPIEStyleGAN2', 'PESinGAN', 'BasicConditionalGAN'
]
