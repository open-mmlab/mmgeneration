from functools import partial

import pytest
import torch

from mmgen.models.architectures import DCGANDiscriminator
from mmgen.models.architectures.pggan.generator_discriminator import \
    PGGANDiscriminator
from mmgen.models.losses import (DiscShiftLoss, GradientPenaltyLoss,
                                 disc_shift_loss, gradient_penalty_loss)
from mmgen.models.losses.disc_auxiliary_loss import (R1GradientPenalty,
                                                     r1_gradient_penalty_loss)


class TestDiscShiftLoss(object):

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((2, 10))
        cls.default_cfg = dict(
            loss_weight=0.1, data_info=dict(pred='disc_pred'))
        cls.default_input_dict = dict(disc_pred=cls.input_tensor)

    def test_disc_shift_loss(self):
        loss = disc_shift_loss(self.input_tensor)
        assert loss.ndim == 0
        assert loss >= 0

        loss = disc_shift_loss(self.input_tensor, weight=-0.1)
        assert loss.ndim == 0
        assert loss <= 0

        loss = disc_shift_loss(self.input_tensor, reduction='none')
        assert loss.ndim == 2
        assert (loss >= 0).all()

        loss_sum = disc_shift_loss(self.input_tensor, reduction='sum')
        loss_avg = disc_shift_loss(self.input_tensor, avg_factor=1000)
        assert loss_avg.ndim == 0 and loss_sum.ndim == 0
        assert loss_sum > loss_avg

        with pytest.raises(ValueError):
            _ = disc_shift_loss(
                self.input_tensor, reduction='sum', avg_factor=100)

    def test_module_wrapper(self):
        # test with default config
        loss_module = DiscShiftLoss(**self.default_cfg)
        loss = loss_module(self.default_input_dict)
        assert loss.ndim == 0

        with pytest.raises(NotImplementedError):
            _ = loss_module(self.default_input_dict, 1)

        with pytest.raises(AssertionError):
            _ = loss_module(1, outputs_dict=self.default_input_dict)
        input_ = dict(outputs_dict=self.default_input_dict)
        loss = loss_module(**input_)
        assert loss.ndim == 0

        with pytest.raises(AssertionError):
            _ = loss_module(self.input_tensor)

        # test without data_info
        loss_module = DiscShiftLoss(data_info=None)
        loss = loss_module(self.input_tensor)
        assert loss.ndim == 0


class TestGradientPenalty:

    @classmethod
    def setup_class(cls):
        cls.input_img = torch.randn((2, 3, 8, 8))
        cls.disc = DCGANDiscriminator(
            input_scale=8, output_scale=4, out_channels=5)
        cls.pggan_disc = PGGANDiscriminator(
            in_scale=8, base_channels=32, max_channels=32)
        cls.data_info = dict(
            discriminator='disc', real_data='real_imgs', fake_data='fake_imgs')

    def test_gp_loss(self):
        loss = gradient_penalty_loss(self.disc, self.input_img, self.input_img)
        assert loss > 0

        loss = gradient_penalty_loss(
            self.disc, self.input_img, self.input_img, norm_mode='HWC')
        assert loss > 0

        with pytest.raises(NotImplementedError):
            _ = gradient_penalty_loss(
                self.disc, self.input_img, self.input_img, norm_mode='xxx')

        loss = gradient_penalty_loss(
            self.disc,
            self.input_img,
            self.input_img,
            norm_mode='HWC',
            weight=10)
        assert loss > 0

        loss = gradient_penalty_loss(
            self.disc,
            self.input_img,
            self.input_img,
            norm_mode='HWC',
            mask=torch.ones_like(self.input_img),
            weight=10)
        assert loss > 0

        data_dict = dict(
            real_imgs=self.input_img,
            fake_imgs=self.input_img,
            disc=partial(self.pggan_disc, transition_weight=0.5, curr_scale=8))
        gp_loss = GradientPenaltyLoss(
            loss_weight=10, norm_mode='pixel', data_info=self.data_info)

        loss = gp_loss(data_dict)
        assert loss > 0
        loss = gp_loss(outputs_dict=data_dict)
        assert loss > 0
        with pytest.raises(NotImplementedError):
            _ = gp_loss(asdf=1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1., 2, outputs_dict=data_dict)


class TestR1GradientPenalty:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(discriminator='disc', real_data='real_imgs')
        cls.disc = DCGANDiscriminator(
            input_scale=8, output_scale=4, out_channels=5)
        cls.pggan_disc = PGGANDiscriminator(
            in_scale=8, base_channels=32, max_channels=32)
        cls.input_img = torch.randn((2, 3, 8, 8))

    def test_r1_regularizer(self):
        loss = r1_gradient_penalty_loss(self.disc, self.input_img)
        assert loss > 0

        loss = r1_gradient_penalty_loss(
            self.disc, self.input_img, norm_mode='HWC')
        assert loss > 0

        with pytest.raises(NotImplementedError):
            _ = r1_gradient_penalty_loss(
                self.disc, self.input_img, norm_mode='xxx')

        loss = r1_gradient_penalty_loss(
            self.disc, self.input_img, norm_mode='HWC', weight=10)
        assert loss > 0

        loss = r1_gradient_penalty_loss(
            self.disc,
            self.input_img,
            norm_mode='HWC',
            mask=torch.ones_like(self.input_img),
            weight=10)
        assert loss > 0

        data_dict = dict(
            real_imgs=self.input_img,
            disc=partial(self.pggan_disc, transition_weight=0.5, curr_scale=8))
        gp_loss = R1GradientPenalty(
            loss_weight=10, norm_mode='pixel', data_info=self.data_info)

        loss = gp_loss(data_dict)
        assert loss > 0
        loss = gp_loss(outputs_dict=data_dict)
        assert loss > 0
        with pytest.raises(NotImplementedError):
            _ = gp_loss(asdf=1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1.)

        with pytest.raises(AssertionError):
            _ = gp_loss(1., 2, outputs_dict=data_dict)
