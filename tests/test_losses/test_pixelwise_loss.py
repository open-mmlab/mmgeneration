# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.distributions.normal import Normal

from mmgen.models.architectures.pix2pix import UnetGenerator
from mmgen.models.losses import L1Loss, MSELoss
from mmgen.models.losses.pixelwise_loss import (
    DiscretizedGaussianLogLikelihoodLoss, GaussianKLDLoss, approx_gaussian_cdf)


class TestPixelwiseLosses:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(pred='fake_imgs', target='real_imgs')
        cls.gen = UnetGenerator(3, 3)

    def test_pixelwise_losses(self):
        with pytest.raises(ValueError):
            # only 'none', 'mean' and 'sum' are supported
            L1Loss(reduction='InvalidValue')

        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        # test l1 loss
        l1_loss = L1Loss(
            loss_weight=1.0, reduction='mean', data_info=self.data_info)
        loss = l1_loss(outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss.shape == ()
        assert loss.item() == 0.25

        l1_loss = L1Loss(
            loss_weight=1.0, reduction='mean', data_info=self.data_info)
        loss = l1_loss(dict(fake_imgs=pred, real_imgs=target))
        assert loss.shape == ()
        assert loss.item() == 0.25

        l1_loss = L1Loss(loss_weight=0.5, reduction='none')
        loss = l1_loss(pred, target)
        assert loss.shape == (1, 1, 64, 64)
        assert (loss == torch.ones(1, 1, 64, 64) * weight * 0.5).all()

        l1_loss = L1Loss(loss_weight=0.5, reduction='sum')
        loss = l1_loss(pred, target)
        assert loss.shape == ()
        assert loss.item() == 512

        # test MSE loss
        mse_loss = MSELoss(loss_weight=1.0, data_info=self.data_info)
        loss = mse_loss(outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss.shape == ()
        assert loss.item() == 0.25

        mse_loss = MSELoss(loss_weight=1.0, data_info=self.data_info)
        loss = mse_loss(dict(fake_imgs=pred, real_imgs=target))
        assert loss.shape == ()
        assert loss.item() == 0.25

        mse_loss = MSELoss(loss_weight=0.5)
        loss = mse_loss(pred, target)
        assert loss.shape == ()
        assert loss.item() == 0.1250


class TestGaussianKLDLoss:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(
            mean_pred='mean_pred',
            mean_target='mean_target',
            logvar_pred='logvar_pred',
            logvar_target='logvar_target')
        cls.tar_shape = [2, 2, 4, 4]

        cls.mean_pred = torch.zeros(cls.tar_shape)
        cls.mean_target = torch.ones(cls.tar_shape)
        cls.logvar_pred = torch.zeros(cls.tar_shape)
        cls.logvar_target = torch.ones(cls.tar_shape)
        cls.output_dict = dict(
            mean_pred=cls.mean_pred,
            mean_target=cls.mean_target,
            logvar_pred=cls.logvar_pred,
            logvar_target=cls.logvar_target)
        cls.gt_loss = ((torch.exp(torch.ones(1)) - 1) / 2).item()

    def test_gaussian_kld_loss(self):

        # test reduction --> mean
        gaussian_kld_loss = GaussianKLDLoss(
            data_info=self.data_info, reduction='mean')
        loss = gaussian_kld_loss(self.output_dict)
        assert (loss == self.gt_loss).all()

        # test reduction --> batchmean
        gaussian_kld_loss = GaussianKLDLoss(
            data_info=self.data_info, reduction='batchmean')
        loss = gaussian_kld_loss(self.output_dict)
        num_elements = self.tar_shape[1] * self.tar_shape[2] * \
            self.tar_shape[3]
        assert (loss == (self.gt_loss * num_elements)).all()

        # test weight --> int
        gaussian_kld_loss = GaussianKLDLoss(
            loss_weight=2, data_info=self.data_info, reduction='mean')
        loss = gaussian_kld_loss(self.output_dict)
        assert (loss == self.gt_loss * 2).all()

        # test weight --> tensor
        weight = torch.randn(*self.tar_shape)
        gaussian_kld_loss = GaussianKLDLoss(
            loss_weight=weight, data_info=self.data_info, reduction='mean')
        loss = gaussian_kld_loss(self.output_dict)
        assert torch.allclose(loss, weight.mean() * self.gt_loss, atol=1e-6)

        # test weight --> tensor & batchmean
        weight = torch.randn(*self.tar_shape)
        gaussian_kld_loss = GaussianKLDLoss(
            loss_weight=weight,
            data_info=self.data_info,
            reduction='batchmean')
        loss = gaussian_kld_loss(self.output_dict)
        assert torch.allclose(
            loss, weight.sum([1, 2, 3]).mean() * self.gt_loss, atol=1e-6)


def test_approx_gaussian_cdf():
    pos = torch.rand(2, 2)
    gaussian_dist = Normal(0, 1)
    assert torch.allclose(
        approx_gaussian_cdf(pos), gaussian_dist.cdf(pos), atol=1e-3)


class TestDistLoss():

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(
            mean='mean_pred', logvar='logvar_pred', x='real_imgs')
        cls.tar_shape = [2, 2, 4, 4]

        cls.mean_pred = torch.zeros(cls.tar_shape)
        cls.logvar_pred = torch.zeros(cls.tar_shape)
        cls.real_imgs = torch.zeros(cls.tar_shape)

        cls.output_dict = dict(
            mean_pred=cls.mean_pred,
            logvar_pred=cls.logvar_pred,
            real_imgs=cls.real_imgs)
        norm_dist = Normal(0, 1)
        cls.gt_loss = torch.log(
            norm_dist.cdf(torch.FloatTensor([1 / 255])) -
            norm_dist.cdf(torch.FloatTensor([-1 / 255])))

    def test_disc_gaussian_log_likelihood_loss(self):
        # test reduction --> mean
        disc_gaussian_loss = DiscretizedGaussianLogLikelihoodLoss(
            data_info=self.data_info, reduction='mean')
        loss = disc_gaussian_loss(self.output_dict)
        assert (loss == self.gt_loss).all()

        # test reduction --> batchmean
        disc_gaussian_loss = DiscretizedGaussianLogLikelihoodLoss(
            data_info=self.data_info, reduction='batchmean')
        loss = disc_gaussian_loss(self.output_dict)
        num_elements = self.tar_shape[1] * self.tar_shape[2] * \
            self.tar_shape[3]
        assert (loss == (self.gt_loss * num_elements)).all()

        # test weight --> int
        disc_gaussian_loss = DiscretizedGaussianLogLikelihoodLoss(
            loss_weight=2, data_info=self.data_info, reduction='mean')
        loss = disc_gaussian_loss(self.output_dict)
        assert (loss == self.gt_loss * 2).all()

        # # test weight --> tensor
        weight = torch.randn(*self.tar_shape)
        disc_gaussian_loss = DiscretizedGaussianLogLikelihoodLoss(
            loss_weight=weight, data_info=self.data_info, reduction='mean')
        loss = disc_gaussian_loss(self.output_dict)
        assert torch.allclose(loss, weight.mean() * self.gt_loss, atol=1e-6)

        # test weight --> tensor & batchmean
        weight = torch.randn(*self.tar_shape)
        disc_gaussian_loss = DiscretizedGaussianLogLikelihoodLoss(
            loss_weight=weight,
            data_info=self.data_info,
            reduction='batchmean')
        loss = disc_gaussian_loss(self.output_dict)
        assert torch.allclose(
            loss, weight.sum([1, 2, 3]).mean() * self.gt_loss, atol=1e-6)
