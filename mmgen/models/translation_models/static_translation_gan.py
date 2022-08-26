# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch.nn as nn
from mmengine.model import (BaseModel, MMDistributedDataParallel,
                            is_model_wrapper)

from mmgen.registry import MODELS
from ..builder import build_module
from .base_translation_model import BaseTranslationModel


@MODELS.register_module()
class StaticTranslationGAN(BaseTranslationModel, BaseModel):
    """Basic translation model based on static unconditional GAN.

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator.
        gan_loss (dict): Config for the gan loss.
        pretrained (str | optional): Path for pretrained model.
            Defaults to None.
        disc_auxiliary_loss (dict | optional): Config for auxiliary loss to
            discriminator. Defaults to None.
        gen_auxiliary_loss (dict | optional): Config for auxiliary loss
            to generator. Defaults to None.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 *args,
                 pretrained=None,
                 data_preprocessor=dict(type='GANDataPreprocessor'),
                 **kwargs):
        BaseModel.__init__(self, data_preprocessor=data_preprocessor)
        BaseTranslationModel.__init__(self, *args, **kwargs)
        # Building generators and discriminators
        self._gen_cfg = deepcopy(generator)
        # build domain generators
        self.generators = nn.ModuleDict()
        for domain in self._reachable_domains:
            self.generators[domain] = build_module(generator)

        self._disc_cfg = deepcopy(discriminator)
        # build domain discriminators
        if discriminator is not None:
            self.discriminators = nn.ModuleDict()
            for domain in self._reachable_domains:
                self.discriminators[domain] = build_module(discriminator)
        # support no discriminator in testing
        else:
            self.discriminators = None

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        for domain in self._reachable_domains:
            if is_model_wrapper(self.generators):
                self.generators.module[domain].init_weights(
                    pretrained=pretrained)
            else:
                self.generators[domain].init_weights(pretrained=pretrained)
            if self.discriminators is not None:
                if is_model_wrapper(self.discriminators):
                    self.discriminators.module[domain].init_weights(
                        pretrained=pretrained)
                else:
                    self.discriminators[domain].init_weights(
                        pretrained=pretrained)

    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel`
        interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if isinstance(module, MMDistributedDataParallel):
            return module.module

        return module

    def _get_target_generator(self, domain):
        """get target generator."""
        assert self.is_domain_reachable(
            domain
        ), f'{domain} domain is not reachable, available domain list is\
            {self._reachable_domains}'

        return self.get_module(self.generators)[domain]

    def _get_target_discriminator(self, domain):
        """get target discriminator."""
        assert self.is_domain_reachable(
            domain
        ), f'{domain} domain is not reachable, available domain list is\
            {self._reachable_domains}'

        return self.get_module(self.discriminators)[domain]
