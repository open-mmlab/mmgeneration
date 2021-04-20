import pytest
import torch

from mmgen.models.architectures.pix2pix import UnetGenerator
from mmgen.models.losses import L1Loss, MSELoss


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
