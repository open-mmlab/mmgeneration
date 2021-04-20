import os

import mmcv
import pytest
import torch

from mmgen.apis import (init_model, sample_img2img_model,
                        sample_uncoditional_model)


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
        res = sample_uncoditional_model(
            self.model, 5, num_batches=2, sample_model='orig')
        assert res.shape == (5, 3, 64, 64)

        res = sample_uncoditional_model(
            self.model, 4, num_batches=2, sample_model='orig')
        assert res.shape == (4, 3, 64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_sample_unconditional_model_cuda(self):
        model = self.model.cuda()
        res = sample_uncoditional_model(
            model, 5, num_batches=2, sample_model='orig')
        assert res.shape == (5, 3, 64, 64)

        res = sample_uncoditional_model(
            model, 4, num_batches=2, sample_model='orig')
        assert res.shape == (4, 3, 64, 64)


class TestSampleTranslationModel:

    @classmethod
    def setup_class(cls):
        project_dir = os.path.abspath(os.path.join(__file__, '../../..'))
        pix2pix_config = mmcv.Config.fromfile(
            os.path.join(
                project_dir,
                'configs/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py'))
        cls.pix2pix = init_model(pix2pix_config, checkpoint=None, device='cpu')
        cyclegan_config = mmcv.Config.fromfile(
            os.path.join(
                project_dir,
                'configs/cyclegan/cyclegan_lsgan_resnet_in_1x1_80k_facades.py')
        )
        cls.cyclegan = init_model(
            cyclegan_config, checkpoint=None, device='cpu')
        cls.img_path = os.path.join(
            os.path.dirname(__file__), '..', 'data/unpaired/testA/5.jpg')

    def test_translation_model_cpu(self):
        res = sample_img2img_model(self.pix2pix, self.img_path)
        assert res.shape == (1, 3, 256, 256)

        res = sample_img2img_model(self.cyclegan, self.img_path)
        assert res.shape == (1, 3, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_translation_model_cuda(self):
        res = sample_img2img_model(self.pix2pix.cuda(), self.img_path)
        assert res.shape == (1, 3, 256, 256)

        res = sample_img2img_model(self.cyclegan.cuda(), self.img_path)
        assert res.shape == (1, 3, 256, 256)
