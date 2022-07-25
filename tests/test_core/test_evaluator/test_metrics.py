import math
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
from mmengine import BaseDataElement
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from mmengine.utils import TORCH_VERSION, digit_version
from torch.utils.data.dataloader import DataLoader

from mmgen.core import (FrechetInceptionDistance, GenDataSample,
                        InceptionScore, MultiScaleStructureSimilarity,
                        PixelData, PrecisionAndRecall,
                        SlicedWassersteinDistance)
from mmgen.core.evaluation.metrics import (Equivariance, GenMetric,
                                           PerceptualPathLength, TransFID,
                                           TransIS)
from mmgen.datasets import (PackGenInputs, PairedImageDataset,
                            UnconditionalImageDataset)
from mmgen.models import (LSGAN, DCGANGenerator, GANDataPreprocessor, Pix2Pix,
                          StyleGAN3, StyleGANv2Generator, StyleGANv3Generator)

logger = MMLogger(name='mmgen')


def process_fn(data_batch, predictions):

    # data_batch is loaded from normal dataloader
    if isinstance(data_batch, list):
        _data_batch = []
        for data in data_batch:
            if isinstance(data['data_sample'], BaseDataElement):
                _data_batch.append(
                    dict(
                        inputs=data['inputs'],
                        data_sample=data['data_sample'].to_dict()))
            else:
                _data_batch.append(data)
    else:
        _data_batch = data_batch

    _predictions = []
    for pred in predictions:
        _predictions.append(pred.to_dict())
    return _data_batch, _predictions


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
        toy_metric = ToyMetric(fake_nums=10, real_nums=10)
        self.assertEquals(toy_metric.SAMPLER_MODE, 'normal')

        model = MagicMock()
        dataset = MagicMock()
        dataset.__len__ = 20
        dataloader = DataLoader(dataset=dataset, batch_size=4)

        sampler = toy_metric.get_metric_sampler(model, dataloader,
                                                [toy_metric])
        self.assertEqual(sampler.dataset, dataset)
        self.assertEqual(len(sampler), math.ceil(10 / 4))

        toy_metric = ToyMetric(fake_nums=10, real_nums=20)


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

    # NOTE: do not test load inception network to save time
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
        gen_samples = [
            GenDataSample(fake=PixelData(data=gen_images[i])).to_dict()
            for i in range(4)
        ]
        fid.process(None, gen_samples)
        fid.process(None, gen_samples)

        fid.fake_results.clear()
        gen_sample = [
            GenDataSample(
                orig=GenDataSample(fake=PixelData(
                    data=torch.randn(3, 2, 2)))).to_dict()
        ]
        fid.process(None, gen_sample)
        gen_sample = [
            GenDataSample(
                orig=GenDataSample(
                    fake_img=PixelData(data=torch.randn(3, 2, 2)))).to_dict()
        ]
        fid.process(None, gen_sample)

        with patch.object(FrechetInceptionDistance, '_load_inception',
                          self.mock_inception_pytorch):
            fid = FrechetInceptionDistance(
                fake_nums=2,
                real_nums=2,
                real_key='real',
                fake_key='fake',
                inception_style='PyTorch',
                inception_pkl=self.inception_pkl)
        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.device = 'cpu'
        dataloader = MagicMock()
        fid.prepare(module, dataloader)
        gen_samples = [
            GenDataSample(fake_img=PixelData(
                data=torch.randn(3, 2, 2))).to_dict() for _ in range(4)
        ]
        fid.process(None, gen_samples)

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

    # NOTE: do not test load inception network to save time
    # def test_load_inception(self):
    #     IS = InceptionScore(fake_nums=2, inception_style='PyTorch')
    #     self.assertEqual(IS.inception_style.upper(), 'PYTORCH')

    def test_process_and_compute(self):
        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(fake_nums=2, fake_key='fake')
        gen_images = torch.randn(4, 3, 2, 2)
        gen_samples = [
            GenDataSample(fake_img=PixelData(data=img)).to_dict()
            for img in gen_images
        ]
        IS.process(None, gen_samples)
        IS.process(None, gen_samples)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_pytorch):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', inception_style='PyTorch')
        gen_images = torch.randn(4, 3, 2, 2)
        gen_samples = [
            GenDataSample(fake_img=PixelData(data=img)).to_dict()
            for img in gen_images
        ]
        IS.process(None, gen_samples)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', sample_model='orig')
        gen_samples = [
            GenDataSample(
                ema=GenDataSample(
                    fake_img=PixelData(data=torch.randn(3, 2, 2))),
                orig=GenDataSample(
                    fake_img=PixelData(data=torch.randn(3, 2, 2)))).to_dict()
        ]
        IS.process(None, gen_samples)
        gen_samples = [
            GenDataSample(
                orig=GenDataSample(fake=PixelData(
                    data=torch.randn(3, 2, 2)))).to_dict()
        ]
        IS.process(None, gen_samples)

        with patch.object(InceptionScore, '_load_inception',
                          self.mock_inception_stylegan):
            IS = InceptionScore(
                fake_nums=2, fake_key='fake', sample_model='orig')
        gen_samples = [
            GenDataSample(fake_img=PixelData(
                data=torch.randn(3, 2, 2))).to_dict() for _ in range(4)
        ]
        IS.process(None, gen_samples)

        metric = IS.evaluate()
        self.assertIsInstance(metric, dict)
        self.assertTrue('is' in metric)
        self.assertTrue('is_std' in metric)


class TestPR:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='mmgen.LoadImageFromFile', key='img',
                 io_backend='disk'),  # noqa
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
        cls.module = LSGAN(
            generator, data_preprocessor=gan_data_preprocessor)  # noqa

        cls.mock_vgg_pytorch = MagicMock(
            return_value=(vgg_mock('PyTorch'), 'False'))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires cuda')  # noqa
    def test_pr_cuda(self):
        pr = PrecisionAndRecall(10, sample_model='orig', auto_save=False)
        self.module.cuda()
        sampler = pr.get_metric_sampler(self.module, self.dataloader, [pr])
        pr.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            predictions = [pred.to_dict() for pred in predictions]
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
            predictions = [pred.to_dict() for pred in predictions]
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
        gen_samples = [
            GenDataSample(fake_img=PixelData(data=img)).to_dict()
            for img in gen_images
        ]

        MS_SSIM.process(None, gen_samples)
        MS_SSIM.process(None, gen_samples)
        self.assertEqual(len(MS_SSIM.fake_results), input_pairs)
        metric_1 = MS_SSIM.evaluate()
        self.assertTrue('avg' in metric_1)

        MS_SSIM.fake_results.clear()
        MS_SSIM.process(None, gen_samples[:4])
        self.assertEqual(len(MS_SSIM.fake_results), 4 // 2)
        metric_2 = MS_SSIM.evaluate()
        self.assertTrue('avg' in metric_2)

        MS_SSIM.fake_results.clear()
        gen_samples = [
            GenDataSample(
                ema=GenDataSample(
                    fake_img=PixelData(data=torch.randn(3, 32, 32))),
                orig=GenDataSample(
                    fake_img=PixelData(
                        data=torch.randn(3, 32, 32)))).to_dict()
        ] * 2
        MS_SSIM.process(None, gen_samples)

        gen_samples = [
            GenDataSample(
                ema=GenDataSample(fake=PixelData(
                    data=torch.randn(3, 32, 32)))).to_dict()
        ] * 2
        MS_SSIM.process(None, gen_samples)

        # test prefix
        MS_SSIM = MultiScaleStructureSimilarity(
            fake_nums=4, fake_key='fake', sample_model='ema', prefix='ms-ssim')


class TestSWD(TestCase):

    def test_init(self):
        swd = SlicedWassersteinDistance(
            fake_nums=10, image_shape=(3, 32, 32))  # noqa
        self.assertEqual(len(swd.real_results), 2)

    def test_prosess(self):
        model = MagicMock()
        model.data_preprocessor = GANDataPreprocessor()
        swd = SlicedWassersteinDistance(fake_nums=100, image_shape=(3, 32, 32))
        swd.prepare(model, None)

        torch.random.manual_seed(42)
        real_samples = [
            dict(inputs=torch.rand(3, 32, 32) * 255.) for _ in range(100)
        ]
        fake_samples = [
            GenDataSample(
                fake_img=PixelData(data=torch.rand(3, 32, 32) * 2 -
                                   1)).to_dict() for _ in range(100)
        ]

        swd.process(real_samples, fake_samples)
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
        swd.prepare(model, None)

        real_samples = [dict(inputs=torch.rand(3, 32, 32)) for _ in range(2)]
        fake_samples = [
            GenDataSample(
                orig=GenDataSample(
                    fake_img=PixelData(data=torch.rand(3, 32, 32)))).to_dict()
            for _ in range(2)
        ]


@pytest.mark.skipif(
    digit_version(TORCH_VERSION) <= digit_version('1.6.0'),
    reason='version limitation')
class TestEquivariance:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='mmgen.LoadImageFromFile', key='img', io_backend='disk'),
            dict(type='mmgen.Resize', scale=(64, 64)),
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
        generator = StyleGANv3Generator(64, 8, 3, noise_size=8)
        cls.module = StyleGAN3(
            generator, data_preprocessor=gan_data_preprocessor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    @torch.no_grad()
    def test_eq_cuda(self):
        eq = Equivariance(
            2,
            eq_cfg=dict(
                compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True),
            sample_mode='orig')
        self.module.cuda()
        sampler = eq.get_metric_sampler(self.module, self.dataloader, [eq])
        eq.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            eq.process(_data_batch, _predictions)
        eq_res = eq.compute_metrics(eq.fake_results)
        isinstance(eq_res['eqt_int'], float) and isinstance(
            eq_res['eqt_frac'], float) and isinstance(eq_res['eqr'], float)

    @torch.no_grad()
    def test_eq_cpu(self):
        eq = Equivariance(
            2,
            eq_cfg=dict(
                compute_eqt_int=True, compute_eqt_frac=True, compute_eqr=True),
            sample_mode='orig')
        sampler = eq.get_metric_sampler(self.module, self.dataloader, [eq])
        eq.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            eq.process(_data_batch, _predictions)
        eq_res = eq.compute_metrics(eq.fake_results)
        isinstance(eq_res['eqt_int'], float) and isinstance(
            eq_res['eqt_frac'], float) and isinstance(eq_res['eqr'], float)


class TestPPL:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='mmgen.LoadImageFromFile', key='img', io_backend='disk'),
            dict(type='mmgen.Resize', scale=(64, 64)),
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
        generator = StyleGANv2Generator(64, 8)
        cls.module = LSGAN(generator, data_preprocessor=gan_data_preprocessor)

        cls.mock_vgg_pytorch = MagicMock(
            return_value=(vgg_mock('PyTorch'), 'False'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_ppl_cuda(self):
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-z',
            space='Z',
            sample_model='orig',
            latent_dim=8)
        self.module.cuda()
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-w',
            space='W',
            sample_model='orig',
            latent_dim=8)
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0

    def test_ppl_cpu(self):
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-z',
            space='Z',
            sample_model='orig',
            latent_dim=8)
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0
        ppl = PerceptualPathLength(
            fake_nums=2,
            prefix='ppl-w',
            space='W',
            sample_model='orig',
            latent_dim=8)
        sampler = ppl.get_metric_sampler(self.module, self.dataloader, [ppl])
        ppl.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            ppl.process(_data_batch, _predictions)
        ppl_res = ppl.compute_metrics(ppl.fake_results)
        assert ppl_res['ppl_score'] >= 0


class TestTransFID:
    inception_pkl = osp.join(
        osp.dirname(__file__), '..', '..',
        'data/inception_pkl/inception_feat.pkl')

    mock_inception_stylegan = MagicMock(
        return_value=(inception_mock('StyleGAN'), 'StyleGAN'))
    mock_inception_pytorch = MagicMock(
        return_value=(inception_mock('PyTorch'), 'PyTorch'))

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(
                type='mmgen.LoadPairedImageFromFile',
                io_backend='disk',
                key='pair',
                domain_a='edge',
                domain_b='shoe',
                flag='color'),
            dict(
                type='TransformBroadcaster',
                mapping={'img': ['img_edge', 'img_shoe']},
                auto_remap=True,
                share_random_params=True,
                transforms=[
                    dict(
                        type='mmgen.Resize',
                        scale=(286, 286),
                        interpolation='bicubic'),
                    dict(type='mmgen.FixedCrop', crop_size=(256, 256))
                ]),
            dict(
                type='mmgen.PackGenInputs',
                keys=['img_edge', 'img_shoe', 'pair'])
        ]
        dataset = PairedImageDataset(
            data_root='tests/data/paired', pipeline=pipeline, test_mode=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = GANDataPreprocessor()
        generator = dict(
            type='UnetGenerator',
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type='BN'),
            use_dropout=True,
            init_cfg=dict(type='normal', gain=0.02))
        discriminator = dict(
            type='PatchDiscriminator',
            in_channels=6,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='BN'),
            init_cfg=dict(type='normal', gain=0.02))
        cls.module = Pix2Pix(
            generator,
            discriminator,
            data_preprocessor=gan_data_preprocessor,
            default_domain='shoe',
            reachable_domains=['shoe'],
            related_domains=['shoe', 'edge'])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_trans_fid_cuda(self):
        with patch.object(TransFID, '_load_inception',
                          self.mock_inception_stylegan):
            fid = TransFID(
                prefix='FID-Full',
                fake_nums=2,
                real_key='img_shoe',
                fake_key='fake_shoe',
                inception_style='PyTorch')
        self.module.cuda()
        sampler = fid.get_metric_sampler(self.module, self.dataloader, [fid])
        fid.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            fid.process(_data_batch, _predictions)
        fid_res = fid.compute_metrics(fid.fake_results)
        assert fid_res['fid'] >= 0 and fid_res['mean'] >= 0 and fid_res[
            'cov'] >= 0

    def test_trans_fid_cpu(self):
        with patch.object(TransFID, '_load_inception',
                          self.mock_inception_stylegan):
            fid = TransFID(
                prefix='FID-Full',
                fake_nums=2,
                real_key='img_shoe',
                fake_key='fake_shoe',
                inception_style='PyTorch')
        sampler = fid.get_metric_sampler(self.module, self.dataloader, [fid])
        fid.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            fid.process(_data_batch, _predictions)
        fid_res = fid.compute_metrics(fid.fake_results)
        assert fid_res['fid'] >= 0 and fid_res['mean'] >= 0 and fid_res[
            'cov'] >= 0


class TestTransIS:
    inception_pkl = osp.join(
        osp.dirname(__file__), '..', '..',
        'data/inception_pkl/inception_feat.pkl')

    mock_inception_stylegan = MagicMock(
        return_value=(inception_mock('StyleGAN'), 'StyleGAN'))
    mock_inception_pytorch = MagicMock(
        return_value=(inception_mock('PyTorch'), 'PyTorch'))

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(
                type='mmgen.LoadPairedImageFromFile',
                io_backend='disk',
                key='pair',
                domain_a='edge',
                domain_b='shoe',
                flag='color'),
            dict(
                type='TransformBroadcaster',
                mapping={'img': ['img_edge', 'img_shoe']},
                auto_remap=True,
                share_random_params=True,
                transforms=[
                    dict(
                        type='mmgen.Resize',
                        scale=(286, 286),
                        interpolation='bicubic'),
                    dict(type='mmgen.FixedCrop', crop_size=(256, 256))
                ]),
            dict(
                type='mmgen.PackGenInputs',
                keys=['img_edge', 'img_shoe', 'pair'])
        ]
        dataset = PairedImageDataset(
            data_root='tests/data/paired', pipeline=pipeline, test_mode=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = GANDataPreprocessor()
        generator = dict(
            type='UnetGenerator',
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type='BN'),
            use_dropout=True,
            init_cfg=dict(type='normal', gain=0.02))
        discriminator = dict(
            type='PatchDiscriminator',
            in_channels=6,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='BN'),
            init_cfg=dict(type='normal', gain=0.02))
        cls.module = Pix2Pix(
            generator,
            discriminator,
            data_preprocessor=gan_data_preprocessor,
            default_domain='shoe',
            reachable_domains=['shoe'],
            related_domains=['shoe', 'edge'])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_trans_is_cuda(self):
        with patch.object(TransIS, '_load_inception',
                          self.mock_inception_stylegan):
            IS = TransIS(
                prefix='IS-Full',
                fake_nums=2,
                inception_style='PyTorch',
                fake_key='fake_shoe',
                sample_model='orig')
        self.module.cuda()
        sampler = IS.get_metric_sampler(self.module, self.dataloader, [IS])
        IS.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            IS.process(_data_batch, _predictions)
        IS_res = IS.compute_metrics(IS.fake_results)
        assert 'is' in IS_res and 'is_std' in IS_res

    def test_trans_is_cpu(self):
        with patch.object(TransIS, '_load_inception',
                          self.mock_inception_stylegan):
            IS = TransIS(
                prefix='IS-Full',
                fake_nums=2,
                inception_style='PyTorch',
                fake_key='fake_shoe',
                sample_model='orig')
        sampler = IS.get_metric_sampler(self.module, self.dataloader, [IS])
        IS.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            _data_batch, _predictions = process_fn(data_batch, predictions)
            IS.process(_data_batch, _predictions)
        IS_res = IS.compute_metrics(IS.fake_results)
        assert 'is' in IS_res and 'is_std' in IS_res
