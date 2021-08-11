from abc import ABCMeta
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from ..builder import MODELS, build_module

from .base_gan import BaseGAN

class BaseTranslationModel(BaseGAN):
    """Base translation Module."""
    def __init__(self, 
                 default_style, 
                 generator,
                 discriminator,
                 gan_loss,
                 reachable_styles=[],
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.fp16_enabled = False
        self._default_style = default_style.upper()
        self._reachable_styles = [style.upper() for style in reachable_styles]
        assert self._default_style in self._reachable_styles
        
        # TODO: Building G and D in following way can be used for 
        # Pix2Pix and CycleGAN
        
        self._gen_cfg = deepcopy(generator)
        # build style generators
        self.generators = dict()
        for style in self._reachable_styles:
            self.generators.update(style=build_module(generator))
        
        self._disc_cfg = deepcopy(discriminator)
        # build style-aware discriminators
        if discriminator is not None:
            self.discriminators = dict()
            for style in self._reachable_styles:
                self.discriminators.update(style=build_module(discriminator))
        # support no discriminator in testing
        else:
            self.discriminators = None
            
    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')
        
        # TODO:

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

    def forward_test(self, data, **kwargs):
        """Testing function for GANs.

        Args:
            data (torch.Tensor | dict | None): Input data. This data will be
                passed to different methods.
        """

        if kwargs.pop('mode', 'sampling') == 'sampling':
            return self.translation(data, **kwargs)

        raise NotImplementedError('Other specific testing functions should'
                                  ' be implemented by the sub-classes.')
        
    def is_style_reachable(self, style):
        return style.upper() in self._reachable_styles
        
    def _get_style_generator(self, style):
        assert self.is_style_reachable(style), f'{style} is not reachable'
        return self.generators.get(style.upper())
    
    def _get_style_discriminator(self, style):
        assert self.is_style_reachable(style), f'{style} is not reachable'
        return self.discriminators.get(style.upper())
        
    def translation(self, image, style=None, **kwargs):
        if style is None:
            style = self._default_style
        _model = self._get_style_generator(style)
        outputs = _model(image, **kwargs)
        return outputs
        
        

