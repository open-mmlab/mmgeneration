# Copyright (c) OpenMMLab. All rights reserved.
from .base_gan import BaseConditionalGAN, BaseGAN
from .gan_data_processer import GANDataPreprocessor
from .lsgan import LSGAN
from .mspie_stylegan2 import MSPIEStyleGAN2
from .progressive_growing_unconditional_gan import ProgressiveGrowingGAN
from .sagan import SAGAN
from .singan import PESinGAN, SinGAN

__all__ = [
    'BaseGAN', 'BaseConditionalGAN', 'ProgressiveGrowingGAN', 'SinGAN',
    'MSPIEStyleGAN2', 'PESinGAN', 'SAGAN', 'GANDataPreprocessor', 'LSGAN'
]
