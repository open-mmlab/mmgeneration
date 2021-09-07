# Copyright (c) OpenMMLab. All rights reserved.
from .pix2pix import Pix2Pix
from .base_translation_model import BaseTranslationModel
from .cyclegan import CycleGAN
from .static_translation_gan import StaticTranslationGAN

__all__ = ['Pix2Pix', 'CycleGAN', 'BaseTranslationModel', 'StaticTranslationGAN']