# Copyright (c) OpenMMLab. All rights reserved.
from .base_gan import BaseConditionalGAN, BaseGAN
from .biggan import BigGAN
from .dcgan import DCGAN
from .gan_data_processer import GANDataPreprocessor
from .ggan import GGAN
from .lsgan import LSGAN
from .mspie_stylegan2 import MSPIEStyleGAN2
from .progressive_growing_unconditional_gan import ProgressiveGrowingGAN
from .sagan import SAGAN
from .singan import PESinGAN, SinGAN
from .wgan_gp import WGANGP

__all__ = [
    'BaseGAN', 'BaseConditionalGAN', 'ProgressiveGrowingGAN', 'SinGAN',
    'MSPIEStyleGAN2', 'PESinGAN', 'SAGAN', 'GANDataPreprocessor', 'LSGAN',
    'DCGAN', 'WGANGP', 'GGAN', 'BigGAN'
]
