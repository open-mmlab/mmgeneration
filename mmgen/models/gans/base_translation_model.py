from copy import deepcopy

import torch.nn as nn

from ..builder import MODELS, build_module
from .base_gan import BaseGAN


@MODELS.register_module()
class BaseTranslationModel(BaseGAN):
    """Base translation Module."""

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 default_style,
                 reachable_styles=[],
                 related_styles=[],
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.fp16_enabled = False
        self._default_style = default_style
        self._reachable_styles = reachable_styles
        self._related_styles = related_styles
        assert self._default_style in self._reachable_styles
        assert set(self._reachable_styles) <= set(self._related_styles)

        # TODO: Building G and D in following way can be used for
        # Pix2Pix and CycleGAN

        self._gen_cfg = deepcopy(generator)
        # build style generators
        self.generators = nn.ModuleDict()
        for style in self._reachable_styles:
            self.generators[style] = build_module(generator)

        self._disc_cfg = deepcopy(discriminator)
        # build style-aware discriminators
        if discriminator is not None:
            self.discriminators = nn.ModuleDict()
            for style in self._reachable_styles:
                self.discriminators[style] = build_module(discriminator)
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

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        # control the work flow in train step
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

        # TODO:

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

    def forward(self, img, test_mode=False, **kwargs):
        """Forward function.

        Args:
            img_a (Tensor): Input image from domain A.
            img_b (Tensor): Input image from domain B.
            meta (list[dict]): Input meta data.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """
        if not test_mode:
            return self.forward_train(img, **kwargs)

        return self.forward_test(img, **kwargs)

    def forward_train(self, img, style, **kwargs):
        styled_img = self.translation(img, style=style, **kwargs)
        results = dict(orig_img=img, styled_img=styled_img)
        return results

    def forward_test(self, img, style, **kwargs):
        # This is a trick for Pix2Pix and cyclegan
        self.train()
        styled_img = self.translation(img, style=style)
        results = dict(orig_img=img.cpu(), styled_img=styled_img.cpu())
        return results

    def is_style_reachable(self, style):
        return style in self._reachable_styles

    def _get_style_generator(self, style):
        assert self.is_style_reachable(
            style), f'{style} style is not reachable, available style list is\
            {self._reachable_styles}'

        return self.generators[style]

    def _get_style_discriminator(self, style):
        assert self.is_style_reachable(
            style), f'{style} style is not reachable, available style list is\
            {self._reachable_styles}'

        return self.discriminators[style]

    def translation(self, image, style=None, **kwargs):
        if style is None:
            style = self._default_style
        _model = self._get_style_generator(style)
        outputs = _model(image, **kwargs)
        return outputs
