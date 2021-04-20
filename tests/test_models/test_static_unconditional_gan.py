from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from mmgen.models import StaticUnconditionalGAN, build_model


class TestStaticUnconditionalGAN(object):

    @classmethod
    def setup_class(cls):
        cls.default_config = dict(
            type='StaticUnconditionalGAN',
            generator=dict(
                type='DCGANGenerator', output_scale=16, base_channels=32),
            discriminator=dict(
                type='DCGANDiscriminator',
                input_scale=16,
                output_scale=4,
                out_channels=5),
            gan_loss=dict(type='GANLoss', gan_type='vanilla'),
            disc_auxiliary_loss=None,
            gen_auxiliary_loss=None,
            train_cfg=None,
            test_cfg=None)

        cls.generator_cfg = dict(
            type='DCGANGenerator', output_scale=16, base_channels=32)
        cls.disc_cfg = dict(
            type='DCGANDiscriminator',
            input_scale=16,
            output_scale=4,
            out_channels=5)
        cls.gan_loss = dict(type='GANLoss', gan_type='vanilla')
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
        dcgan = build_model(self.default_config)
        assert isinstance(dcgan, StaticUnconditionalGAN)
        assert not dcgan.with_disc_auxiliary_loss
        assert dcgan.with_disc

        # test forward train
        with pytest.raises(NotImplementedError):
            _ = dcgan(None, return_loss=True)
        # test forward test
        imgs = dcgan(None, return_loss=False, mode='sampling', num_batches=2)
        assert imgs.shape == (2, 3, 16, 16)

        # test train step
        data = torch.randn((2, 3, 16, 16))
        data_input = dict(real_img=data)
        optimizer_g = torch.optim.SGD(dcgan.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            dcgan.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = dcgan.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

        # more tests for different configs with heavy computation
        # test disc_steps
        config_ = deepcopy(self.default_config)
        config_['train_cfg'] = dict(disc_steps=2)
        dcgan = build_model(config_)
        model_outputs = dcgan.train_step(data_input, optim_dict)
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' not in model_outputs['log_vars']
        assert dcgan.disc_steps == 2

        model_outputs = dcgan.train_step(
            data_input, optim_dict, running_status=dict(iteration=1))
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' in model_outputs['log_vars']

        # test customized config
        dcgan = StaticUnconditionalGAN(
            self.generator_cfg,
            self.disc_cfg,
            self.gan_loss,
            self.disc_auxiliary_loss,
        )
        # test train step
        data = torch.randn((2, 3, 16, 16))
        data_input = dict(real_img=data)
        optimizer_g = torch.optim.SGD(dcgan.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            dcgan.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = dcgan.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

        dcgan = StaticUnconditionalGAN(
            self.generator_cfg, self.disc_cfg, self.gan_loss,
            dict(
                type='DiscShiftLoss',
                loss_weight=0.5,
                data_info=dict(pred='disc_pred_fake')),
            dict(type='GeneratorPathRegularizer'))
        assert isinstance(dcgan.disc_auxiliary_losses, nn.ModuleList)
        assert isinstance(dcgan.gen_auxiliary_losses, nn.ModuleList)

        dcgan = StaticUnconditionalGAN(
            self.generator_cfg, self.disc_cfg, self.gan_loss,
            dict(
                type='DiscShiftLoss',
                loss_weight=0.5,
                data_info=dict(pred='disc_pred_fake')),
            [dict(type='GeneratorPathRegularizer')])
        assert isinstance(dcgan.disc_auxiliary_losses, nn.ModuleList)
        assert isinstance(dcgan.gen_auxiliary_losses, nn.ModuleList)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_default_dcgan_model_cuda(self):
        dcgan = build_model(self.default_config).cuda()
        assert isinstance(dcgan, StaticUnconditionalGAN)
        assert not dcgan.with_disc_auxiliary_loss
        assert dcgan.with_disc

        # test forward train
        with pytest.raises(NotImplementedError):
            _ = dcgan(None, return_loss=True)
        # test forward test
        imgs = dcgan(None, return_loss=False, mode='sampling', num_batches=2)
        assert imgs.shape == (2, 3, 16, 16)

        # test train step
        data = torch.randn((2, 3, 16, 16)).cuda()
        data_input = dict(real_img=data)
        optimizer_g = torch.optim.SGD(dcgan.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            dcgan.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = dcgan.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

        # more tests for different configs with heavy computation in GPU
        # test disc_steps
        config_ = deepcopy(self.default_config)
        config_['train_cfg'] = dict(disc_steps=2)
        dcgan = build_model(config_).cuda()
        model_outputs = dcgan.train_step(data_input, optim_dict)
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' not in model_outputs['log_vars']
        assert dcgan.disc_steps == 2

        model_outputs = dcgan.train_step(
            data_input, optim_dict, running_status=dict(iteration=1))
        assert 'loss_disc_fake' in model_outputs['log_vars']
        assert 'loss_disc_fake_g' in model_outputs['log_vars']
