import os
import os.path as osp
import pickle
from typing import Optional, Sequence
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmgen.core import (FrechetInceptionDistance, InceptionScore,
                        MultiScaleStructureSimilarity, PrecisionAndRecall,
                        SlicedWassersteinDistance)
from mmgen.core.evaluation.metrics import GenMetric
from mmgen.datasets import PackGenInputs, UnconditionalImageDataset
from mmgen.models import LSGAN, DCGANGenerator, GANDataPreprocessor

logger = MMLogger(name='mmgen')


def construct_inception_pkl(inception_path):
    data_root = osp.dirname(inception_path)
    os.makedirs(data_root, exist_ok=True)
    with open(inception_path, 'wb') as file:
        feat = np.random.rand(10, 2048)
        mean = np.mean(feat, 0)
        cov = np.cov(feat, rowvar=False)
        inception_feat = dict(raw_feature=feat, real_mean=mean, real_cov=cov)
        pickle.dump(inception_feat, file)


class inception_mock(nn.Module):

    def __init__(self, style):
        super().__init__()
        self.style = style

    def forward(self, x, *args, **kwargs):
        mock_feat = torch.randn(x.shape[0], 2048)
        if self.style.upper() in ['STYLEGAN', 'IS']:
            return mock_feat
        else:
            return [mock_feat]


class vgg_pytorch_classifier(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.randn(x.shape[0], 4096)


class vgg_mock(nn.Module):

    def __init__(self, style):
        super().__init__()
        self.classifier = nn.Sequential(nn.Identity(), nn.Identity(),
                                        nn.Identity(),
                                        vgg_pytorch_classifier())
        self.style = style

    def forward(self, x, *args, **kwargs):
        if self.style.upper() == 'STYLEGAN':
            return torch.randn(x.shape[0], 4096)
        else:  # torch
            return torch.randn(x.shape[0], 7 * 7 * 512)


class ToyMetric(GenMetric):

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(fake_nums, real_nums, fake_key, real_key,
                         sample_model, collect_device, prefix)

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        pass

    def compute_metrics(self, results_fake, results_real) -> dict:
        pass


class TestBaseMetric(TestCase):

    def test_init(self):
        toy_metric = ToyMetric(fake_nums=10)
        self.assertEquals(toy_metric.SAMPLER_MODE, 'normal')
        self.assertEquals(toy_metric._color_order, 'bgr')

        model = MagicMock()
        dataloader = MagicMock()
        dataset = MagicMock()
        dataloader.dataset = dataset
        dataloader.batch_size = 4

        sampler = toy_metric.get_metric_sampler(model, dataloader,
                                                [toy_metric])
        self.assertEqual(sampler._dataloader.dataset, dataset)


class TestFID(TestCase):

    inception_pkl = osp.join(
        osp.dirname(__file__), '..', '..',
        'data/inception_pkl/inception_feat.pkl')

    mock_inception_stylegan = MagicMock(
        return_value=(inception_mock('StyleGAN'), 'StyleGAN'))
    mock_inception_pytorch = MagicMock(
        return_value=(inception_mock('PyTorch'), 'PyTorch'))

    def test_init(self):
        construct_inception_pkl(self.inception_pkl)

        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_stylegan):

            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_key='real',
                fake_key='fake',
                inception_pkl=self.inception_pkl)

            self.assertIsNone(fid.real_mean)
            self.assertIsNone(fid.real_cov)

        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        fid.prepare(module, dataloader)

        self.assertIsNotNone(fid.real_mean)
        self.assertIsNotNone(fid.real_cov)

    def test_prepare(self):

        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()

        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_stylegan):
            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_nums=2,
                real_key='real',
                fake_key='fake',
                inception_pkl=self.inception_pkl)
        fid.prepare(module, dataloader)

    # def test_load_inception(self):
    #     fid = FrechetInceptionDistance(
    #         fake_nums=2,
    #         real_nums=2,
    #         real_key='real',
    #         fake_key='fake',
    #         inception_style='PyTorch',
    #         inception_pkl=self.inception_pkl)
    #     self.assertEqual(fid.inception_style.upper(), 'PYTORCH')

    def test_process_and_compute(self):
        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_stylegan):
            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_nums=2,
                real_key='real',
                fake_key='fake',
                inception_pkl=self.inception_pkl)
        gen_images = torch.randn(4, 3, 2, 2)
        fid.process(None, gen_images)
        fid.process(None, gen_images)

        fid.fake_results.clear()
        gen_images = {'ema': {'fake': torch.randn(1, 3, 2, 2)}}
        fid.process(None, gen_images)
        gen_images = {'ema': torch.randn(1, 3, 2, 2)}
        fid.process(None, gen_images)

        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_pytorch):
            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_nums=2,
                real_key='real',
                fake_key='fake',
                inception_style='PyTorch',
                inception_pkl=self.inception_pkl)
        fid.set_color_order('rgb')
        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        fid.prepare(module, dataloader)
        fid.process(None, torch.randn(4, 3, 2, 2))

        metric = fid.evaluate()
        self.assertIsInstance(metric, dict)
        self.assertTrue('fid' in metric)
        self.assertTrue('mean' in metric)
        self.assertTrue('cov' in metric)


class TestIS(TestCase):

    mock_inception_stylegan = MagicMock(
        return_value=(inception_mock('IS'), 'StyleGAN'))
    mock_inception_pytorch = MagicMock(
        return_value=(inception_mock('IS'), 'PyTorch'))

    def test_init(self):
        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(fake_nums=2, fake_key='fake')

        self.assertEqual(IS.resize, True)
        self.assertEqual(IS.splits, 10)
        self.assertEqual(IS.resize_method, 'bicubic')

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', use_pillow_resize=False)
        self.assertEqual(IS.use_pillow_resize, False)

        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        IS.prepare(module, dataloader)

    # def test_load_inception(self):
    #     IS = InceptionScore(fake_nums=2, inception_style='PyTorch')
    #     self.assertEqual(IS.inception_style.upper(), 'PYTORCH')

    def test_process_and_compute(self):
        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(fake_nums=2, fake_key='fake')
        gen_images = torch.randn(4, 3, 2, 2)
        IS.process(None, gen_images)
        IS.process(None, gen_images)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_pytorch):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', inception_style='PyTorch')
        gen_images = torch.randn(4, 3, 2, 2)
        IS.process(None, gen_images)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', sample_model='orig')
        gen_images = {
            'orig': torch.randn(1, 3, 2, 2),
            'ema': torch.randn(1, 3, 2, 2)
        }
        IS.process(None, gen_images)
        gen_images = {'orig': {'fake': torch.randn(1, 3, 2, 2)}}
        IS.process(None, gen_images)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', sample_model='orig')
        IS.set_color_order('rgb')
        IS.process(None, torch.randn(4, 3, 2, 2))

        metric = IS.evaluate()
        self.assertIsInstance(metric, dict)
        self.assertTrue('is' in metric)
        self.assertTrue('is_std' in metric)


class TestPR:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='mmgen.LoadImageFromFile', key='img', io_backend='disk'),
            dict(type='mmgen.Resize', scale=(128, 128)),
            PackGenInputs(meta_keys=[])
        ]
        dataset = UnconditionalImageDataset(
            data_root='tests/data/image', pipeline=pipeline, test_mode=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = GANDataPreprocessor()
        generator = DCGANGenerator(128, noise_size=10, base_channels=20)
        cls.module = LSGAN(generator, data_preprocessor=gan_data_preprocessor)

        cls.mock_vgg_pytorch = MagicMock(
            return_value=(vgg_mock('PyTorch'), 'False'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pr_cuda(self):
        pr = PrecisionAndRecall(10, sample_model='orig', auto_save=False)
        self.module.cuda()
        sampler = pr.get_metric_sampler(self.module, self.dataloader, [pr])
        pr.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            pr.process(data_batch, predictions)
        pr_score = pr.compute_metrics(pr.fake_results)
        print(pr_score)
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0

    def test_pr_cpu(self):
        with patch.object(PrecisionAndRecall, '_load_vgg',
                          self.mock_vgg_pytorch):
            pr = PrecisionAndRecall(10, sample_model='orig', auto_save=False)
        sampler = pr.get_metric_sampler(self.module, self.dataloader, [pr])
        pr.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            pr.process(data_batch, predictions)
        pr_score = pr.evaluate()
        print(pr_score)
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0


class TestMS_SSIM(TestCase):

    def test_init(self):
        MS_SSIM = MultiScaleStructureSimilarity(
            fake_nums=10, fake_key='fake', sample_model='ema')
        self.assertEqual(MS_SSIM.num_pairs, 5)

        with self.assertRaises(AssertionError):
            MultiScaleStructureSimilarity(fake_nums=9)

    def test_process_and_evaluate(self):
        MS_SSIM = MultiScaleStructureSimilarity(
            fake_nums=4, fake_key='fake', sample_model='ema')

        input_batch_size = 6
        input_pairs = 6 // 2
        gen_images = torch.randn(input_batch_size, 3, 32, 32)

        MS_SSIM.process(None, gen_images)
        MS_SSIM.process(None, gen_images)
        self.assertEqual(len(MS_SSIM.fake_results), input_pairs)
        metric_1 = MS_SSIM.evaluate()
        self.assertTrue('avg' in metric_1)

        MS_SSIM.fake_results.clear()
        MS_SSIM.process(None, gen_images[:4])
        self.assertEqual(len(MS_SSIM.fake_results), 4 // 2)
        metric_2 = MS_SSIM.evaluate()
        self.assertTrue('avg' in metric_2)

        MS_SSIM.fake_results.clear()
        gen_images = {
            'orig': torch.randn(2, 3, 32, 32),
            'ema': torch.randn(2, 3, 32, 32)
        }
        MS_SSIM.process(None, gen_images)

        gen_images = {'ema': {'fake': torch.randn(2, 3, 32, 32)}}
        MS_SSIM.process(None, gen_images)

        # test prefix
        MS_SSIM = MultiScaleStructureSimilarity(
            fake_nums=4, fake_key='fake', sample_model='ema', prefix='ms-ssim')


class TestSWD(TestCase):

    def test_init(self):
        swd = SlicedWassersteinDistance(fake_nums=10, image_shape=(3, 32, 32))
        self.assertEqual(len(swd.real_results), 2)

    def test_prosess(self):
        swd = SlicedWassersteinDistance(fake_nums=4, image_shape=(3, 32, 32))
        torch.random.manual_seed(42)
        real_img = torch.rand(100, 3, 32, 32)
        gen_img = torch.rand(100, 3, 32, 32)

        swd.process(real_img, gen_img)

        output = swd.evaluate()
        result = [16.495922580361366, 24.15413036942482, 20.325026474893093]
        output = [item / 100 for item in output.values()]
        result = [item / 100 for item in result]
        np.testing.assert_almost_equal(output, result, decimal=1)

        swd = SlicedWassersteinDistance(
            fake_nums=4,
            fake_key='fake',
            real_key='img',
            sample_model='orig',
            image_shape=(3, 32, 32))

        random_tensor = torch.randn(2, 3, 32, 32)
        real_img = dict(inputs=dict(img=random_tensor, img_1=random_tensor))
        gen_img = dict(orig=random_tensor)
        swd.process(real_img, gen_img)

        gen_img = dict(orig=dict(fake=random_tensor))
        swd.process(real_img, gen_img)
