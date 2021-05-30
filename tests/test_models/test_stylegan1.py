import numpy as np
import pytest
import torch

from mmgen.models import build_model

# from mmgen.models.gans import StyleGANV1


class TestStyleGANV1:

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(
            type='StyleGANv1Generator', out_size=32, style_channels=512)
        cls.discriminator_cfg = dict(type='StyleGAN1Discriminator', in_size=32)
        cls.gan_loss = dict(type='GANLoss', gan_type='wgan')
        cls.disc_auxiliary_loss = [
            dict(
                type='R1GradientPenalty',
                loss_weight=10,
                norm_mode='HWC',
                data_info=dict(
                    discriminator='disc_partial', real_data='real_imgs'))
        ]
        cls.train_cfg = dict(
            use_ema=True,
            nkimgs_per_scale={
                '8': 0.006,
                '16': 0.006,
                '32': 0.012
            },
            optimizer_cfg=dict(
                generator=dict(type='Adam', lr=0.003, betas=(0.0, 0.99)),
                discriminator=dict(type='Adam', lr=0.003, betas=(0.0, 0.99))),
            g_lr_base=0.003,
            d_lr_base=0.003)
        cls.stylegan_cfg = dict(
            type='ProgressiveGrowingGAN',
            generator=cls.generator_cfg,
            discriminator=cls.discriminator_cfg,
            gan_loss=cls.gan_loss,
            disc_auxiliary_loss=cls.disc_auxiliary_loss,
            train_cfg=cls.train_cfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_stylegan1_cuda(self):
        # test default config
        stylegan = build_model(self.stylegan_cfg).cuda()
        data_batch = dict(real_img=torch.randn(3, 3, 32, 32).cuda())

        for iter_num in range(5):
            outputs = stylegan.train_step(
                data_batch,
                None,
                running_status=dict(iteration=iter_num, batch_size=3))
            results = outputs['results']
            if iter_num == 1:
                assert results['fake_imgs'].shape == (3, 3, 8, 8)
            elif iter_num == 2:
                assert results['fake_imgs'].shape == (3, 3, 16, 16)
                assert np.isclose(stylegan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert results['fake_imgs'].shape == (3, 3, 16, 16)
            elif iter_num == 4:
                assert results['fake_imgs'].shape == (3, 3, 32, 32)
                assert np.isclose(stylegan._actual_nkimgs[1], 0.012, atol=1e-8)

    def test_stylegan1_cpu(self):
        # test default config
        stylegan = build_model(self.stylegan_cfg)

        data_batch = dict(real_img=torch.randn(3, 3, 32, 32))

        for iter_num in range(5):
            outputs = stylegan.train_step(
                data_batch,
                None,
                running_status=dict(iteration=iter_num, batch_size=3))
            results = outputs['results']
            if iter_num == 1:
                assert results['fake_imgs'].shape == (3, 3, 8, 8)
            elif iter_num == 2:
                assert results['fake_imgs'].shape == (3, 3, 16, 16)
                assert np.isclose(stylegan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert results['fake_imgs'].shape == (3, 3, 16, 16)
            elif iter_num == 4:
                assert results['fake_imgs'].shape == (3, 3, 32, 32)
                assert np.isclose(stylegan._actual_nkimgs[1], 0.012, atol=1e-8)
