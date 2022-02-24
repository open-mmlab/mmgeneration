# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import pytest
import torch

from mmgen.apis import (init_model, sample_ddpm_model, sample_img2img_model,
                        sample_unconditional_model)


class TestSampleUnconditionalModel:

    @classmethod
    def setup_class(cls):
        project_dir = os.path.abspath(os.path.join(__file__, '../../..'))
        config = mmcv.Config.fromfile(
            os.path.join(
                project_dir,
                'configs/dcgan/dcgan_celeba-cropped_64_b128x1_300k.py'))
        cls.model = init_model(config, checkpoint=None, device='cpu')

    def test_sample_unconditional_model_cpu(self):
        res = sample_unconditional_model(
            self.model, 5, num_batches=2, sample_model='orig')
        assert res.shape == (5, 3, 64, 64)

        res = sample_unconditional_model(
            self.model, 4, num_batches=2, sample_model='orig')
        assert res.shape == (4, 3, 64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_sample_unconditional_model_cuda(self):
        model = self.model.cuda()
        res = sample_unconditional_model(
            model, 5, num_batches=2, sample_model='orig')
        assert res.shape == (5, 3, 64, 64)

        res = sample_unconditional_model(
            model, 4, num_batches=2, sample_model='orig')
        assert res.shape == (4, 3, 64, 64)


class TestSampleTranslationModel:

    @classmethod
    def setup_class(cls):
        project_dir = os.path.abspath(os.path.join(__file__, '../../..'))
        pix2pix_config = mmcv.Config.fromfile(
            os.path.join(
                project_dir,
                'configs/pix2pix/pix2pix_vanilla_unet_bn_facades_b1x1_80k.py'))
        cls.pix2pix = init_model(pix2pix_config, checkpoint=None, device='cpu')
        cyclegan_config = mmcv.Config.fromfile(
            os.path.join(
                project_dir,
                'configs/cyclegan/cyclegan_lsgan_resnet_in_facades_b1x1_80k.py'
            ))
        cls.cyclegan = init_model(
            cyclegan_config, checkpoint=None, device='cpu')
        cls.img_path = os.path.join(
            os.path.dirname(__file__), '..', 'data/unpaired/testA/5.jpg')

    def test_translation_model_cpu(self):
        res = sample_img2img_model(
            self.pix2pix, self.img_path, target_domain='photo')
        assert res.shape == (1, 3, 256, 256)

        res = sample_img2img_model(
            self.cyclegan, self.img_path, target_domain='photo')
        assert res.shape == (1, 3, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_translation_model_cuda(self):
        res = sample_img2img_model(
            self.pix2pix.cuda(), self.img_path, target_domain='photo')
        assert res.shape == (1, 3, 256, 256)

        res = sample_img2img_model(
            self.cyclegan.cuda(), self.img_path, target_domain='photo')
        assert res.shape == (1, 3, 256, 256)


class TestDiffusionModel:

    @classmethod
    def setup_class(cls):
        project_dir = os.path.abspath(os.path.join(__file__, '../../..'))
        ddpm_config = mmcv.Config.fromfile(
            os.path.join(
                project_dir, 'configs/improved_ddpm/'
                'ddpm_cosine_hybird_timestep-4k_drop0.3_'
                'cifar10_32x32_b8x16_500k.py'))
        # change timesteps to speed up test process
        ddpm_config.model['num_timesteps'] = 10
        cls.model = init_model(ddpm_config, checkpoint=None, device='cpu')

    def test_diffusion_model_cpu(self):
        # save_intermedia is False
        res = sample_ddpm_model(
            self.model, num_samples=3, num_batches=2, same_noise=True)
        assert res.shape == (3, 3, 32, 32)

        # save_intermedia is True
        res = sample_ddpm_model(
            self.model,
            num_samples=2,
            num_batches=2,
            same_noise=True,
            save_intermedia=True)
        assert isinstance(res, dict)
        assert all([i in res for i in range(10)])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_diffusion_model_cuda(self):
        model = self.model.cuda()
        # save_intermedia is False
        res = sample_ddpm_model(
            model, num_samples=3, num_batches=2, same_noise=True)
        assert res.shape == (3, 3, 32, 32)

        # save_intermedia is True
        res = sample_ddpm_model(
            model,
            num_samples=2,
            num_batches=2,
            same_noise=True,
            save_intermedia=True)
        assert isinstance(res, dict)
        assert all([i in res for i in range(10)])
