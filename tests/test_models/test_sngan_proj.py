import pytest
import torch

from mmgen.models.gans import BasicConditionalGAN


class TestSNGAN_PROJ:

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(
            type='SNGANGenerator',
            output_scale=32,
            base_channels=256,
            num_classes=10)

        cls.discriminator_cfg = dict(
            type='ProjDiscriminator',
            input_scale=32,
            base_channels=128,
            num_classes=10)

        cls.disc_auxiliary_loss = None
        cls.gan_loss = dict(type='GANLoss', gan_type='hinge')
        cls.train_cfg = None

    def test_sngan_proj_cpu(self):
        # test default config
        snganproj = BasicConditionalGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            self.gan_loss,
            disc_auxiliary_loss=None,
            train_cfg=self.train_cfg)

        # test sample from noise
        outputs = snganproj.sample_from_noise(None, num_batches=2)
        assert outputs.shape == (2, 3, 32, 32)

        outputs = snganproj.sample_from_noise(
            None, num_batches=2, return_noise=True, sample_model='orig')
        assert outputs['fake_img'].shape == (2, 3, 32, 32)

        # test train step
        img = torch.randn((2, 3, 32, 32))
        lab = torch.randint(0, 10, (2, ))
        data_input = dict(img=img, gt_label=lab)
        optimizer_g = torch.optim.SGD(
            snganproj.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            snganproj.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = snganproj.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_sngan_proj_cuda(self):
        # test default config
        snganproj = BasicConditionalGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            self.gan_loss,
            disc_auxiliary_loss=self.disc_auxiliary_loss,
            train_cfg=self.train_cfg).cuda()

        # test sample from noise
        outputs = snganproj.sample_from_noise(None, num_batches=2)
        assert outputs.shape == (2, 3, 32, 32)

        outputs = snganproj.sample_from_noise(
            None, num_batches=2, return_noise=True, sample_model='orig')
        assert outputs['fake_img'].shape == (2, 3, 32, 32)

        # test train step
        img = torch.randn((2, 3, 32, 32)).cuda()
        lab = torch.randint(0, 10, (2, )).cuda()
        data_input = dict(img=img, gt_label=lab)
        optimizer_g = torch.optim.SGD(
            snganproj.generator.parameters(), lr=0.01)
        optimizer_d = torch.optim.SGD(
            snganproj.discriminator.parameters(), lr=0.01)
        optim_dict = dict(generator=optimizer_g, discriminator=optimizer_d)

        model_outputs = snganproj.train_step(data_input, optim_dict)
        assert 'results' in model_outputs
        assert 'log_vars' in model_outputs
        assert model_outputs['num_samples'] == 2
