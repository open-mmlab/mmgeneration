# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmgen.models.architectures.pix2pix import UnetGenerator
from mmgen.models.architectures.stylegan import StyleGANv2Generator
from mmgen.models.losses import GeneratorPathRegularizer, PerceptualLoss
from mmgen.models.losses.pixelwise_loss import l1_loss, mse_loss


class TestPathRegularizer:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(generator='generator', num_batches='num_batches')
        cls.gen = StyleGANv2Generator(32, 10, num_mlps=2)

    def test_path_regularizer_cpu(self):
        gen = self.gen

        output_dict = dict(generator=gen, num_batches=2)
        pl = GeneratorPathRegularizer(data_info=self.data_info)
        pl_loss = pl(output_dict)
        assert pl_loss > 0

        output_dict = dict(generator=gen, num_batches=2, iteration=3)
        pl = GeneratorPathRegularizer(data_info=self.data_info, interval=2)
        pl_loss = pl(outputs_dict=output_dict)
        assert pl_loss is None

        with pytest.raises(NotImplementedError):
            _ = pl(asdf=1.)

        with pytest.raises(AssertionError):
            _ = pl(1.)

        with pytest.raises(AssertionError):
            _ = pl(1., 2, outputs_dict=output_dict)

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not hasattr(torch.backends.cudnn, 'allow_tf32'),
        reason='requires cuda')
    def test_path_regularizer_cuda(self):
        gen = self.gen.cuda()

        output_dict = dict(generator=gen, num_batches=2)
        pl = GeneratorPathRegularizer(data_info=self.data_info).cuda()
        pl_loss = pl(output_dict)
        assert pl_loss > 0

        output_dict = dict(generator=gen, num_batches=2, iteration=3)
        pl = GeneratorPathRegularizer(
            data_info=self.data_info, interval=2).cuda()
        pl_loss = pl(outputs_dict=output_dict)
        assert pl_loss is None

        with pytest.raises(NotImplementedError):
            _ = pl(asdf=1.)

        with pytest.raises(AssertionError):
            _ = pl(1.)

        with pytest.raises(AssertionError):
            _ = pl(1., 2, outputs_dict=output_dict)


class TestPerceptualLoss:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(pred='fake_imgs', target='real_imgs')
        cls.gen = UnetGenerator(3, 3)

    def test_perceptual_loss(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        # test perceptual and style loss
        perceptual_loss = PerceptualLoss(data_info=self.data_info)
        loss_percep, loss_style = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == () and loss_style.shape == ()
        assert id(perceptual_loss.criterion) == id(l1_loss)

        # test only perceptual loss
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, style_weight=0)
        loss_percep, loss_style = perceptual_loss(
            dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == ()
        assert loss_style is None

        # test only style loss
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, perceptual_weight=0)
        loss_percep, loss_style = perceptual_loss(
            dict(fake_imgs=pred, real_imgs=target))
        assert loss_style.shape == ()
        assert loss_percep is None

        # test with different layer weights
        layer_weights = {'1': 1., '2': 2., '3': 3.}
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, layer_weights=layer_weights)
        loss_percep, loss_style = perceptual_loss(
            dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == () and loss_style.shape == ()
        assert perceptual_loss.layer_weights == layer_weights and \
            perceptual_loss.layer_weights_style == layer_weights

        # test with different perceptual and style layers
        layer_weights = {'1': 1., '2': 2., '3': 3.}
        layer_weights_style = {'4': 4., '5': 5., '6': 6.}
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info,
            layer_weights=layer_weights,
            layer_weights_style=layer_weights_style)
        loss_percep, loss_style = perceptual_loss(
            dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == () and loss_style.shape == ()
        assert perceptual_loss.layer_weights == layer_weights and \
            perceptual_loss.layer_weights_style == layer_weights_style

        # test MSE critierion
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, criterion='mse')
        loss_percep, loss_style = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == () and loss_style.shape == ()
        assert id(perceptual_loss.criterion) == id(mse_loss)

        # test VGG 16
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info,
            vgg_type='vgg16',
            pretrained='torchvision://vgg16')
        loss_percep, loss_style = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == () and loss_style.shape == ()
        # TODO need to check whether vgg16 is loaded
        # assert perceptual_loss.vgg

        # test cuda
        device = 'cuda:0'
        perceptual_loss = PerceptualLoss(data_info=self.data_info).to(device)
        loss_percep, loss_style = perceptual_loss(
            outputs_dict=dict(
                fake_imgs=pred.to(device), real_imgs=target.to(device)))
        assert str(loss_percep.device) == device and \
            str(loss_style.device) == device
