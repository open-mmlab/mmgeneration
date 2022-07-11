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
from .stylegan1 import StyleGANv1
from .stylegan2 import StyleGAN2
from .stylegan3 import StyleGAN3
from .wgan_gp import WGANGP

__all__ = [
    'BaseGAN', 'BaseConditionalGAN', 'ProgressiveGrowingGAN', 'SinGAN',
    'MSPIEStyleGAN2', 'PESinGAN', 'SAGAN', 'GANDataPreprocessor', 'LSGAN',
    'StyleGAN2', 'BigGAN', 'StyleGAN3', 'DCGAN', 'WGANGP', 'GGAN', 'BigGAN',
    'StyleGANv1'
]
