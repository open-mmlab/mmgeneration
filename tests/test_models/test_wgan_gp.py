import pytest
import torch

from mmgen.models.gans import StaticUnconditionalGAN


class TestWGANGP:

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(
            type='WGANGPGenerator', noise_size=128, out_scale=128)

        cls.discriminator_cfg = dict(
            type='WGANGPDiscriminator',
            in_channel=3,
            in_scale=128,
            conv_module_cfg=dict(
                conv_cfg=None,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
                norm_cfg=dict(type='GN'),
                order=('conv', 'norm', 'act')))

        cls.gan_loss = dict(type='GANLoss', gan_type='wgan')
        cls.disc_auxiliary_loss = dict(
            type='GradientPenaltyLoss',
            loss_weight=10,
            norm_mode='pixel',
            data_info=dict(
                discriminator='disc',
                real_data='real_imgs',
                fake_data='fake_imgs'))
        cls.train_cfg = None

    def test_wgangp_cpu(self):
        # test default config
        wgangp = StaticUnconditionalGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            self.gan_loss,
            disc_auxiliary_loss=self.disc_auxiliary_loss,
            train_cfg=self.train_cfg)

        # test sample from noise
        outputs = wgangp.sample_from_noise(None, num_batches=2)
        assert outputs.shape == (2, 3, 128, 128)

        outputs = wgangp.sample_from_noise(
            None, num_batches=2, return_noise=True, sample_model='orig')
        assert outputs['fake_img'].shape == (2, 3, 128, 128)

        # test train step
        data = torch.randn((2, 3, 128, 128))
        data_input = dict(real_img=data)
        optimizer_g = torch.optim.SGD(wgangp.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            wgangp.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = wgangp.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_wgangp_cuda(self):
        # test default config
        wgangp = StaticUnconditionalGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            self.gan_loss,
            disc_auxiliary_loss=self.disc_auxiliary_loss,
            train_cfg=self.train_cfg).cuda()

        # test sample from noise
        outputs = wgangp.sample_from_noise(None, num_batches=2)
        assert outputs.shape == (2, 3, 128, 128)

        outputs = wgangp.sample_from_noise(
            None, num_batches=2, return_noise=True, sample_model='orig')
        assert outputs['fake_img'].shape == (2, 3, 128, 128)

        # test train step
        data = torch.randn((2, 3, 128, 128)).cuda()
        data_input = dict(real_img=data)
        optimizer_g = torch.optim.SGD(wgangp.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            wgangp.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = wgangp.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2
