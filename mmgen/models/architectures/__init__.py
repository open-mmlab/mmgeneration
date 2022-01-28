# Copyright (c) OpenMMLab. All rights reserved.
from .arcface import IDLossModel
from .biggan import (BigGANDeepDiscriminator, BigGANDeepGenerator,
                     BigGANDiscriminator, BigGANGenerator, SNConvModule)
from .cyclegan import ResnetGenerator
from .dcgan import DCGANDiscriminator, DCGANGenerator
from .ddpm import DenoisingUnet
from .fid_inception import InceptionV3
from .lpips import PerceptualLoss
from .lsgan import LSGANDiscriminator, LSGANGenerator
from .pggan import (EqualizedLR, EqualizedLRConvDownModule,
                    EqualizedLRConvModule, EqualizedLRConvUpModule,
                    EqualizedLRLinearModule, MiniBatchStddevLayer,
                    PGGANDiscriminator, PGGANGenerator, PGGANNoiseTo2DFeat,
                    PixelNorm, equalized_lr)
from .pix2pix import PatchDiscriminator, generation_init_weights
from .positional_encoding import CatersianGrid, SinusoidalPositionalEmbedding
from .singan import SinGANMultiScaleDiscriminator, SinGANMultiScaleGenerator
from .sngan_proj import ProjDiscriminator, SNGANGenerator
from .stylegan import (MSStyleGAN2Discriminator, MSStyleGANv2Generator,
                       StyleGAN1Discriminator, StyleGAN2Discriminator,
                       StyleGANv1Generator, StyleGANv2Generator,
                       StyleGANv3Generator)
from .wgan_gp import WGANGPDiscriminator, WGANGPGenerator

__all__ = [
    'DCGANGenerator', 'DCGANDiscriminator', 'EqualizedLR',
    'EqualizedLRConvModule', 'equalized_lr', 'EqualizedLRLinearModule',
    'EqualizedLRConvUpModule', 'EqualizedLRConvDownModule', 'PixelNorm',
    'MiniBatchStddevLayer', 'PGGANNoiseTo2DFeat', 'PGGANGenerator',
    'PGGANDiscriminator', 'InceptionV3', 'SinGANMultiScaleDiscriminator',
    'SinGANMultiScaleGenerator', 'CatersianGrid',
    'SinusoidalPositionalEmbedding', 'StyleGAN2Discriminator',
    'StyleGANv2Generator', 'StyleGANv1Generator', 'StyleGAN1Discriminator',
    'MSStyleGAN2Discriminator', 'MSStyleGANv2Generator',
    'generation_init_weights', 'PatchDiscriminator', 'ResnetGenerator',
    'PerceptualLoss', 'WGANGPDiscriminator', 'WGANGPGenerator',
    'LSGANDiscriminator', 'LSGANGenerator', 'ProjDiscriminator',
    'SNGANGenerator', 'BigGANGenerator', 'SNConvModule', 'BigGANDiscriminator',
    'BigGANDeepGenerator', 'BigGANDeepDiscriminator', 'DenoisingUnet',
    'StyleGANv3Generator', 'IDLossModel'
]
