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

    def test_perceptual_loss_cpu(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        perceptual_loss = PerceptualLoss(data_info=self.data_info)
        loss_perceptual = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_perceptual.shape == ()
        assert id(perceptual_loss.criterion) == id(l1_loss)

    def test_only_perceptual_loss(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, style_weight=0)
        loss_percep = perceptual_loss(dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == ()
        assert perceptual_loss.style_weight == 0

    def test_only_style_loss(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, perceptual_weight=0)
        loss_style = perceptual_loss(dict(fake_imgs=pred, real_imgs=target))
        assert loss_style.shape == ()
        assert perceptual_loss.perceptual_weight == 0

    def test_with_different_layer_weights(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        layer_weights = {'1': 1., '2': 2., '3': 3.}
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, layer_weights=layer_weights)
        loss_perceptual = perceptual_loss(
            dict(fake_imgs=pred, real_imgs=target))
        assert loss_perceptual.shape == ()
        assert perceptual_loss.layer_weights == layer_weights and \
            perceptual_loss.layer_weights_style == layer_weights

    def test_with_different_perceptual_and_style_layers(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        layer_weights = {'1': 1., '2': 2., '3': 3.}
        layer_weights_style = {'4': 4., '5': 5., '6': 6.}
        perceptual_loss = PerceptualLoss(
            data_info=self.data_info,
            layer_weights=layer_weights,
            layer_weights_style=layer_weights_style)
        loss_perceptual = perceptual_loss(
            dict(fake_imgs=pred, real_imgs=target))
        assert loss_perceptual.shape == ()
        assert perceptual_loss.layer_weights == layer_weights and \
            perceptual_loss.layer_weights_style == layer_weights_style

    def test_MSE_critierion(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, criterion='mse')
        loss_perceptual = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_perceptual.shape == ()
        assert id(perceptual_loss.criterion) == id(mse_loss)

    def test_VGG_16(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        perceptual_loss = PerceptualLoss(
            data_info=self.data_info,
            vgg_type='vgg16',
            pretrained='torchvision://vgg16')
        loss_perceptual = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_perceptual.shape == ()
        # TODO need to check whether vgg16 is loaded
        # assert perceptual_loss.vgg

    def test_split_style_loss(self):
        unknown_h, unknown_w = (32, 32)
        weight = torch.zeros(1, 1, 64, 64)
        weight[0, 0, :unknown_h, :unknown_w] = 1
        pred = weight.clone()
        target = weight.clone() * 2

        perceptual_loss = PerceptualLoss(
            data_info=self.data_info, split_style_loss=True)
        loss_percep, loss_style = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_percep.shape == () and loss_style.shape == ()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_perceptual_loss_cuda(self):
        pred = torch.rand([2, 3, 256, 256]).cuda()
        target = torch.rand_like(pred).cuda()
        perceptual_loss = PerceptualLoss(data_info=self.data_info).cuda()
        loss_perceptual = perceptual_loss(
            outputs_dict=dict(fake_imgs=pred, real_imgs=target))
        assert loss_perceptual.shape == ()
