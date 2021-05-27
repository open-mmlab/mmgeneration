from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmgen.models.gans import ProgressiveGrowingGAN


class TestPGGAN:

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(
            type='PGGANGenerator',
            noise_size=8,
            out_scale=16,
            base_channels=32,
            max_channels=32)

        cls.discriminator_cfg = dict(
            type='PGGANDiscriminator', in_scale=16, label_size=0)

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
        cls.train_cfg = dict(
            use_ema=True,
            nkimgs_per_scale={
                '4': 0.004,
                '8': 0.008,
                '16': 0.016
            },
            optimizer_cfg=dict(
                generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
                discriminator=dict(type='Adam', lr=0.0002,
                                   betas=(0.5, 0.999))),
            g_lr_base=0.0001,
            d_lr_base=0.0001,
            g_lr_schedule={'16': 0.00005})

    def test_pggan_cpu(self):
        # test default config
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            self.gan_loss,
            disc_auxiliary_loss=self.disc_auxiliary_loss,
            train_cfg=self.train_cfg)

        data_batch = dict(real_img=torch.randn(3, 3, 16, 16))

        for iter_num in range(6):
            outputs = pggan.train_step(
                data_batch,
                None,
                running_status=dict(iteration=iter_num, batch_size=3))
            results = outputs['results']
            if iter_num == 1:
                assert results['fake_imgs'].shape == (3, 3, 4, 4)
            elif iter_num == 2:
                assert results['fake_imgs'].shape == (3, 3, 8, 8)
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert results['fake_imgs'].shape == (3, 3, 8, 8)
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
                assert np.isclose(
                    pggan.optimizer['generator'].defaults['lr'],
                    0.0001,
                    atol=1e-8)
            elif iter_num == 5:
                assert results['fake_imgs'].shape == (3, 3, 16, 16)
                assert np.isclose(pggan._actual_nkimgs[-1], 0.012, atol=1e-8)
                assert np.isclose(
                    pggan.optimizer['generator'].defaults['lr'],
                    0.00005,
                    atol=1e-8)

        # test sample from noise
        outputs = pggan.sample_from_noise(None, num_batches=2)
        assert outputs.shape == (4, 3, 16, 16)

        outputs = pggan.sample_from_noise(
            None,
            num_batches=2,
            return_noise=True,
            transition_weight=0.2,
            sample_model='ema')
        assert outputs['fake_img'].shape == (2, 3, 16, 16)

        outputs = pggan.sample_from_noise(
            None, num_batches=2, return_noise=True, sample_model='orig')
        assert outputs['fake_img'].shape == (2, 3, 16, 16)

        with pytest.raises(RuntimeError):
            data_batch = dict(real_img=torch.randn(3, 3, 4, 32))
            _ = pggan.train_step(
                data_batch, None, running_status=dict(iteration=5))

        # test customized config
        train_cfg_ = deepcopy(self.train_cfg)
        train_cfg_['use_ema'] = False
        train_cfg_['interp_real_cfg'] = dict(
            mode='bilinear', align_corners=False)
        train_cfg_['interp_real_cfg'] = dict(
            mode='bilinear', align_corners=False)
        train_cfg_['reset_optim_for_new_scale'] = False
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            self.gan_loss,
            disc_auxiliary_loss=None,
            train_cfg=train_cfg_)

        data_batch = dict(real_img=torch.randn(3, 3, 16, 16))

        outputs = pggan.train_step(
            data_batch, None, running_status=dict(iteration=0, batch_size=3))
        results = outputs['results']
        assert results['fake_imgs'].shape == (3, 3, 4, 4)
        assert not pggan.with_gen_auxiliary_loss
        assert not pggan.with_disc_auxiliary_loss
        assert not pggan.use_ema

        data_batch = dict(real_img=torch.randn(3, 3, 16, 16))

        for iter_num in range(1, 3):
            outputs = pggan.train_step(
                data_batch,
                None,
                running_status=dict(iteration=iter_num, batch_size=3))
            results = outputs['results']
            if iter_num == 1:
                assert results['fake_imgs'].shape == (3, 3, 4, 4)
            elif iter_num == 2:
                assert results['fake_imgs'].shape == (3, 3, 8, 8)
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)

        train_cfg_ = deepcopy(self.train_cfg)
        train_cfg_['optimizer_cfg'] = dict(
            generator=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            None,
            None,
            disc_auxiliary_loss=dict(
                type='DiscShiftLoss',
                loss_weight=0.5,
                data_info=dict(pred='disc_pred_fake')),
            gen_auxiliary_loss=dict(type='GeneratorPathRegularizer'),
            train_cfg=train_cfg_)
        assert pggan.with_gen_auxiliary_loss
        assert isinstance(pggan.disc_auxiliary_losses, nn.ModuleList)
        assert pggan.gan_loss is None
        assert pggan.discriminator is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pggan_cuda(self):
        # test default config
        pggan = ProgressiveGrowingGAN(
            self.generator_cfg,
            self.discriminator_cfg,
            self.gan_loss,
            disc_auxiliary_loss=self.disc_auxiliary_loss,
            train_cfg=self.train_cfg).cuda()

        data_batch = dict(real_img=torch.randn(3, 3, 32, 32).cuda())

        for iter_num in range(6):
            outputs = pggan.train_step(
                data_batch,
                None,
                running_status=dict(iteration=iter_num, batch_size=3))
            results = outputs['results']
            if iter_num == 1:
                assert results['fake_imgs'].shape == (3, 3, 4, 4)
            elif iter_num == 2:
                assert results['fake_imgs'].shape == (3, 3, 8, 8)
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 3:
                assert results['fake_imgs'].shape == (3, 3, 8, 8)
                assert np.isclose(pggan._actual_nkimgs[0], 0.006, atol=1e-8)
            elif iter_num == 5:
                assert results['fake_imgs'].shape == (3, 3, 16, 16)
                assert np.isclose(pggan._actual_nkimgs[-1], 0.012, atol=1e-8)
