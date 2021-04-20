import torch

from mmgen.models.gans.singan import PESinGAN, SinGAN


class TestSinGAN:

    @classmethod
    def setup_class(cls):
        cls.generator = dict(
            type='SinGANMultiScaleGenerator',
            in_channels=3,
            out_channels=3,
            num_scales=3)

        cls.disc = dict(
            type='SinGANMultiScaleDiscriminator', in_channels=3, num_scales=3)

        cls.gan_loss = dict(type='GANLoss', gan_type='wgan', loss_weight=1)
        cls.disc_auxiliary_loss = [
            dict(
                type='GradientPenaltyLoss',
                loss_weight=0.1,
                norm_mode='pixel',
                data_info=dict(
                    discriminator='disc_partial',
                    real_data='real_imgs',
                    fake_data='fake_imgs'))
        ]
        cls.gen_auxiliary_loss = dict(
            type='MSELoss',
            loss_weight=10,
            data_info=dict(pred='recon_imgs', target='real_imgs'),
        )

        cls.train_cfg = dict(
            noise_weight_init=0.1,
            iters_per_scale=2,
            curr_scale=-1,
            disc_steps=3,
            generator_steps=3,
            lr_d=0.0005,
            lr_g=0.0005,
            lr_scheduler_args=dict(milestones=[1600], gamma=0.1))

        cls.data_batch = dict(
            real_scale0=torch.randn(1, 3, 25, 25),
            real_scale1=torch.randn(1, 3, 30, 30),
            real_scale2=torch.randn(1, 3, 32, 32),
        )
        cls.data_batch['input_sample'] = torch.zeros_like(
            cls.data_batch['real_scale0'])

    def test_singan_cpu(self):
        singan = SinGAN(self.generator, self.disc, self.gan_loss,
                        self.disc_auxiliary_loss, self.gen_auxiliary_loss,
                        self.train_cfg, None)

        for i in range(6):
            output = singan.train_step(self.data_batch, None)
            if i == 0:
                assert output['results']['fake_imgs'].shape[-2:] == (25, 25)
            elif i == 2:
                assert output['results']['fake_imgs'].shape[-2:] == (30, 30)
            elif i == 5:
                assert output['results']['fake_imgs'].shape[-2:] == (32, 32)

        singan = SinGAN(self.generator, self.disc, self.gan_loss, None, None,
                        self.train_cfg, None)

        for i in range(6):
            output = singan.train_step(self.data_batch, None)
            if i == 0:
                assert output['results']['fake_imgs'].shape[-2:] == (25, 25)
            elif i == 2:
                assert output['results']['fake_imgs'].shape[-2:] == (30, 30)
            elif i == 5:
                assert output['results']['fake_imgs'].shape[-2:] == (32, 32)

        # test sample from noise
        img = singan.sample_from_noise(None, num_batches=1)
        assert img.shape == (1, 3, 32, 32)


class TestPESinGAN:

    @classmethod
    def setup_class(cls):
        cls.generator = dict(
            type='SinGANMSGeneratorPE',
            in_channels=3,
            out_channels=3,
            num_scales=3,
            interp_pad=True,
            noise_with_pad=True)

        cls.disc = dict(
            type='SinGANMultiScaleDiscriminator', in_channels=3, num_scales=3)

        cls.gan_loss = dict(type='GANLoss', gan_type='wgan', loss_weight=1)
        cls.disc_auxiliary_loss = [
            dict(
                type='GradientPenaltyLoss',
                loss_weight=0.1,
                norm_mode='pixel',
                data_info=dict(
                    discriminator='disc_partial',
                    real_data='real_imgs',
                    fake_data='fake_imgs'))
        ]
        cls.gen_auxiliary_loss = dict(
            type='MSELoss',
            loss_weight=10,
            data_info=dict(pred='recon_imgs', target='real_imgs'),
        )

        cls.train_cfg = dict(
            noise_weight_init=0.1,
            iters_per_scale=2,
            curr_scale=-1,
            disc_steps=3,
            generator_steps=3,
            lr_d=0.0005,
            lr_g=0.0005,
            lr_scheduler_args=dict(milestones=[1600], gamma=0.1),
            fixed_noise_with_pad=True)

        cls.data_batch = dict(
            real_scale0=torch.randn(1, 3, 25, 25),
            real_scale1=torch.randn(1, 3, 30, 30),
            real_scale2=torch.randn(1, 3, 32, 32),
        )
        cls.data_batch['input_sample'] = torch.zeros_like(
            cls.data_batch['real_scale0'])

    def test_pesingan_cpu(self):
        singan = PESinGAN(self.generator, self.disc, self.gan_loss,
                          self.disc_auxiliary_loss, self.gen_auxiliary_loss,
                          self.train_cfg, None)

        for i in range(6):
            output = singan.train_step(self.data_batch, None)
            if i == 0:
                assert singan.fixed_noises[0].shape[-2] == 35
                assert output['results']['fake_imgs'].shape[-2:] == (25, 25)
            elif i == 2:
                assert output['results']['fake_imgs'].shape[-2:] == (30, 30)
            elif i == 5:
                assert output['results']['fake_imgs'].shape[-2:] == (32, 32)

        singan = PESinGAN(
            dict(
                type='SinGANMSGeneratorPE',
                in_channels=3,
                out_channels=3,
                num_scales=3,
                interp_pad=True,
                noise_with_pad=False), self.disc, self.gan_loss, None, None,
            dict(
                noise_weight_init=0.1,
                iters_per_scale=2,
                curr_scale=-1,
                disc_steps=3,
                generator_steps=3,
                lr_d=0.0005,
                lr_g=0.0005,
                lr_scheduler_args=dict(milestones=[1600], gamma=0.1),
                fixed_noise_with_pad=False), None)

        for i in range(6):
            output = singan.train_step(self.data_batch, None)
            if i == 0:
                assert output['results']['fake_imgs'].shape[-2:] == (25, 25)
            elif i == 2:
                assert output['results']['fake_imgs'].shape[-2:] == (30, 30)
            elif i == 5:
                assert output['results']['fake_imgs'].shape[-2:] == (32, 32)

        # test sample from noise
        img = singan.sample_from_noise(None, num_batches=1)
        assert img.shape == (1, 3, 32, 32)
