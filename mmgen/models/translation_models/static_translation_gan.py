# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch.nn as nn
from mmcv.parallel import MMDistributedDataParallel

from ..builder import MODELS, build_module
from ..gans import BaseGAN
from .base_translation_model import BaseTranslationModel


@MODELS.register_module()
class StaticTranslationGAN(BaseTranslationModel, BaseGAN):
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
                 gan_loss,
                 *args,
                 pretrained=None,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 **kwargs):
        BaseGAN.__init__(self)
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

        # support no gan_loss in testing
        if gan_loss is not None:
            self.gan_loss = build_module(gan_loss)
        else:
            self.gan_loss = None

        if disc_auxiliary_loss:
            self.disc_auxiliary_losses = build_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                self.gen_auxiliary_losses = nn.ModuleList(
                    [self.gen_auxiliary_losses])
        else:
            self.gen_auxiliary_losses = None

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        for domain in self._reachable_domains:
            self.generators[domain].init_weights(pretrained=pretrained)
            self.discriminators[domain].init_weights(pretrained=pretrained)

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

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
