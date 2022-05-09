# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from mmgen.models import BasicConditionalGAN, build_model


class TestBasicConditionalGAN(object):

    @classmethod
    def setup_class(cls):
        cls.default_config = dict(
            type='BasicConditionalGAN',
            generator=dict(
                type='SNGANGenerator',
                output_scale=32,
                base_channels=256,
                num_classes=10),
            discriminator=dict(
                type='ProjDiscriminator',
                input_scale=32,
                base_channels=128,
                num_classes=10),
            gan_loss=dict(type='GANLoss', gan_type='hinge'),
            disc_auxiliary_loss=None,
            gen_auxiliary_loss=None,
            train_cfg=None,
            test_cfg=None)

        cls.generator_cfg = dict(
            type='SAGANGenerator',
            output_scale=32,
            num_classes=10,
            base_channels=256,
            attention_after_nth_block=2,
            with_spectral_norm=True)
        cls.disc_cfg = dict(
            type='SAGANDiscriminator',
            input_scale=32,
            num_classes=10,
            base_channels=128,
            attention_after_nth_block=1)
        cls.gan_loss = dict(type='GANLoss', gan_type='hinge')
        cls.disc_auxiliary_loss = [
            dict(
                type='DiscShiftLoss',
                loss_weight=0.5,
                data_info=dict(pred='disc_pred_fake')),
            dict(
                type='DiscShiftLoss',
                loss_weight=0.5,
                data_info=dict(pred='disc_pred_real'))
        ]

    def test_default_dcgan_model_cpu(self):
        sngan = build_model(self.default_config)
        assert isinstance(sngan, BasicConditionalGAN)
        assert not sngan.with_disc_auxiliary_loss
        assert sngan.with_disc

        # test forward train
        with pytest.raises(NotImplementedError):
            _ = sngan(None, return_loss=True)
        # test forward test
        imgs = sngan(None, return_loss=False, mode='sampling', num_batches=2)
        assert imgs.shape == (2, 3, 32, 32)

        # test train step
        data = torch.randn((2, 3, 32, 32))
        label = torch.randint(0, 10, (2, ))
        data_input = dict(img=data, gt_label=label)
        optimizer_g = torch.optim.SGD(sngan.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            sngan.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = sngan.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

        # more tests for different configs with heavy computation
        # test disc_steps
        config_ = deepcopy(self.default_config)
        config_['train_cfg'] = dict(disc_steps=2)
        sngan = build_model(config_)
        model_outputs = sngan.train_step(data_input, optim_dict)
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' not in model_outputs['log_vars']
        assert sngan.disc_steps == 2

        model_outputs = sngan.train_step(
            data_input, optim_dict, running_status=dict(iteration=1))
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' in model_outputs['log_vars']

        # test customized config
        sagan = BasicConditionalGAN(
            self.generator_cfg,
            self.disc_cfg,
            self.gan_loss,
            self.disc_auxiliary_loss,
        )
        # test train step
        data = torch.randn((2, 3, 32, 32))
        data_input = dict(img=data, gt_label=label)
        optimizer_g = torch.optim.SGD(sngan.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            sngan.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = sagan.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

        sagan = BasicConditionalGAN(
            self.generator_cfg, self.disc_cfg, self.gan_loss,
            dict(
                type='DiscShiftLoss',
                loss_weight=0.5,
                data_info=dict(pred='disc_pred_fake')),
            dict(type='GeneratorPathRegularizer'))
        assert isinstance(sagan.disc_auxiliary_losses, nn.ModuleList)
        assert isinstance(sagan.gen_auxiliary_losses, nn.ModuleList)

        sagan = BasicConditionalGAN(
            self.generator_cfg, self.disc_cfg, self.gan_loss,
            dict(
                type='DiscShiftLoss',
                loss_weight=0.5,
                data_info=dict(pred='disc_pred_fake')),
            [dict(type='GeneratorPathRegularizer')])
        assert isinstance(sagan.disc_auxiliary_losses, nn.ModuleList)
        assert isinstance(sagan.gen_auxiliary_losses, nn.ModuleList)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_default_dcgan_model_cuda(self):
        sngan = build_model(self.default_config).cuda()
        assert isinstance(sngan, BasicConditionalGAN)
        assert not sngan.with_disc_auxiliary_loss
        assert sngan.with_disc

        # test forward train
        with pytest.raises(NotImplementedError):
            _ = sngan(None, return_loss=True)
        # test forward test
        imgs = sngan(None, return_loss=False, mode='sampling', num_batches=2)
        assert imgs.shape == (2, 3, 32, 32)

        # test train step
        data = torch.randn((2, 3, 32, 32)).cuda()
        label = torch.randint(0, 10, (2, )).cuda()
        data_input = dict(img=data, gt_label=label)
        optimizer_g = torch.optim.SGD(sngan.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            sngan.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = sngan.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

        # more tests for different configs with heavy computation
        # test disc_steps
        config_ = deepcopy(self.default_config)
        config_['train_cfg'] = dict(disc_steps=2)
        sngan = build_model(config_).cuda()
        model_outputs = sngan.train_step(data_input, optim_dict)
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' not in model_outputs['log_vars']
        assert sngan.disc_steps == 2

        model_outputs = sngan.train_step(
            data_input, optim_dict, running_status=dict(iteration=1))
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' in model_outputs['log_vars']
