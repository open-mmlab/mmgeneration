# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
import torch

from mmgen.core.evaluation.metric_utils import extract_inception_features
from mmgen.core.evaluation.metrics import (FID, IS, MS_SSIM, PPL, PR, SWD,
                                           GaussianKLD)
from mmgen.datasets import UnconditionalImageDataset, build_dataloader
from mmgen.models import build_model
from mmgen.models.architectures import InceptionV3
from mmgen.utils import download_from_url

# def test_inception_download():
#     from mmgen.core.evaluation.metrics import load_inception
#     from mmgen.utils import MMGEN_CACHE_DIR

#     args_FID_pytorch = dict(type='pytorch', normalize_input=False)
#     args_FID_tero = dict(type='StyleGAN', inception_path='')
#     args_IS_pytorch = dict(type='pytorch')
#     args_IS_tero = dict(
#         type='StyleGAN',
#         inception_path=osp.join(MMGEN_CACHE_DIR, 'inception-2015-12-05.pt'))

#     tar_style_list = ['pytorch', 'StyleGAN', 'pytorch', 'StyleGAN']

#     for inception_args, metric, tar_style in zip(
#         [args_FID_pytorch, args_FID_tero, args_IS_pytorch, args_IS_tero],
#         ['FID', 'FID', 'IS', 'IS'], tar_style_list):
#         model, style = load_inception(inception_args, metric)
#         assert style == tar_style

#     args_empty = ''
#     with pytest.raises(TypeError) as exc_info:
#         load_inception(args_empty, 'FID')

#     args_error_path = dict(type='StyleGAN', inception_path='error-path')
#     with pytest.raises(RuntimeError) as exc_info:
#         load_inception(args_error_path, 'FID')


def test_swd_metric():
    img_nchw_1 = torch.rand((100, 3, 32, 32))
    img_nchw_2 = torch.rand((100, 3, 32, 32))

    metric = SWD(100, (3, 32, 32))
    metric.prepare()
    metric.feed(img_nchw_1, 'reals')
    metric.feed(img_nchw_2, 'fakes')
    result = [16.495922580361366, 24.15413036942482, 20.325026474893093]
    output = metric.summary()
    result = [item / 100 for item in result]
    output = [item / 100 for item in output]
    np.testing.assert_almost_equal(output, result, decimal=1)


def test_ms_ssim():
    img_nhwc_1 = torch.rand((100, 3, 32, 32))
    img_nhwc_2 = torch.rand((100, 3, 32, 32))

    metric = MS_SSIM(100)
    metric.prepare()
    metric.feed(img_nhwc_1, 'reals')
    metric.feed(img_nhwc_2, 'fakes')
    ssim_result = metric.summary()
    assert ssim_result < 1


class TestExtractInceptionFeat:

    @classmethod
    def setup_class(cls):
        cls.inception = InceptionV3(
            load_fid_inception=False, resize_input=True)
        pipeline = [
            dict(type='LoadImageFromFile', key='real_img'),
            dict(
                type='Resize',
                keys=['real_img'],
                scale=(299, 299),
                keep_ratio=False,
            ),
            dict(
                type='Normalize',
                keys=['real_img'],
                mean=[127.5] * 3,
                std=[127.5] * 3,
                to_rgb=False),
            dict(type='Collect', keys=['real_img'], meta_keys=[]),
            dict(type='ImageToTensor', keys=['real_img'])
        ]
        dataset = UnconditionalImageDataset(
            osp.join(osp.dirname(__file__), '..', 'data'), pipeline)
        cls.data_loader = build_dataloader(dataset, 3, 0, dist=False)

    def test_extr_inception_feat(self):
        feat = extract_inception_features(self.data_loader, self.inception, 5)
        assert feat.shape[0] == 5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_extr_inception_feat_cuda(self):
        inception = torch.nn.DataParallel(self.inception)
        feat = extract_inception_features(self.data_loader, inception, 5)
        assert feat.shape[0] == 5

    @torch.no_grad()
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_with_tero_implement(self):
        self.inception = InceptionV3(
            load_fid_inception=True, resize_input=False)
        img = torch.randn((2, 3, 1024, 1024))
        feature_ours = self.inception(img)[0].view(img.shape[0], -1)

        # Tero implementation
        download_from_url(
            'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',  # noqa
            dest_dir='./work_dirs/cache')
        net = torch.jit.load(
            './work_dirs/cache/inception-2015-12-05.pt').eval().cuda()
        net = torch.nn.DataParallel(net)
        feature_tero = net(img, return_features=True)

        print(feature_ours.shape)

        print((feature_tero.cpu() - feature_ours).abs().mean())


class TestFID:

    @classmethod
    def setup_class(cls):
        cls.reals = [torch.randn(2, 3, 128, 128) for _ in range(5)]
        cls.fakes = [torch.randn(2, 3, 128, 128) for _ in range(5)]

    def test_fid(self):
        fid = FID(
            3,
            inception_args=dict(
                normalize_input=False, load_fid_inception=False))
        for b in self.reals:
            fid.feed(b, 'reals')

        for b in self.fakes:
            fid.feed(b, 'fakes')

        fid_score, mean, cov = fid.summary()
        assert fid_score > 0 and mean > 0 and cov > 0

        # To reduce the size of git repo, we remove the following test
        # fid = FID(
        #     3,
        #     inception_args=dict(
        #         normalize_input=False, load_fid_inception=False),
        #     inception_pkl=osp.join(
        #         osp.dirname(__file__), '..', 'data', 'test_dirty.pkl'))
        # assert fid.num_real_feeded == 3
        # for b in self.reals:
        #     fid.feed(b, 'reals')

        # for b in self.fakes:
        #     fid.feed(b, 'fakes')

        # fid_score, mean, cov = fid.summary()
        # assert fid_score > 0 and mean > 0 and cov > 0


class TestPR:

    @classmethod
    def setup_class(cls):
        cls.reals = [torch.rand(2, 3, 128, 128) for _ in range(5)]
        cls.fakes = [torch.rand(2, 3, 128, 128) for _ in range(5)]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pr_cuda(self):
        pr = PR(10)
        pr.prepare()
        for b in self.fakes:
            pr.feed(b.cuda(), 'fakes')
        for b in self.reals:
            pr.feed(b.cuda(), 'reals')
        pr_score = pr.summary()
        print(pr_score)
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0

    def test_pr_cpu(self):
        pr = PR(10)
        pr.prepare()
        for b in self.fakes:
            pr.feed(b, 'fakes')
        for b in self.reals:
            pr.feed(b, 'reals')
        pr_score = pr.summary()
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0


class TestIS:

    @classmethod
    def setup_class(cls):
        cls.reals = [torch.randn(2, 3, 128, 128) for _ in range(5)]
        cls.fakes = [torch.randn(2, 3, 128, 128) for _ in range(5)]

    def test_is_cpu(self):
        inception_score = IS(10, resize=True, splits=10)
        inception_score.prepare()
        for b in self.reals:
            inception_score.feed(b, 'reals')

        for b in self.fakes:
            inception_score.feed(b, 'fakes')

        score, std = inception_score.summary()
        assert score > 0 and std >= 0

    @torch.no_grad()
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_is_cuda(self):
        inception_score = IS(10, resize=True, splits=10)
        inception_score.prepare()
        for b in self.reals:
            inception_score.feed(b.cuda(), 'reals')

        for b in self.fakes:
            inception_score.feed(b.cuda(), 'fakes')

        score, std = inception_score.summary()
        assert score > 0 and std >= 0


class TestPPL:

    @classmethod
    def setup_class(cls):
        cls.model_cfg = dict(
            type='StaticUnconditionalGAN',
            generator=dict(
                type='StyleGANv2Generator',
                out_size=256,
                style_channels=512,
            ),
            discriminator=dict(
                type='StyleGAN2Discriminator',
                in_size=256,
            ),
            gan_loss=dict(type='GANLoss', gan_type='wgan-logistic-ns'),
            train_cfg=dict(use_ema=True))

    def test_ppl_cpu(self):
        self.model = build_model(self.model_cfg)
        ppl = PPL(10)
        ppl_iterator = iter(ppl.get_sampler(self.model, 2, 'ema'))
        ppl.prepare()
        for b in ppl_iterator:
            ppl.feed(b, 'fakes')
        score = ppl.summary()
        assert score >= 0

    @torch.no_grad()
    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_ppl_cuda(self):
        self.model = build_model(self.model_cfg).cuda()
        ppl = PPL(10)
        ppl_iterator = iter(ppl.get_sampler(self.model, 2, 'ema'))
        ppl.prepare()
        for b in ppl_iterator:
            ppl.feed(b, 'fakes')
        score = ppl.summary()
        assert score >= 0


def test_kld_gaussian():
    # we only test at bz = 1 to test the numerical accuracy
    # due to the time and memory cost
    tar_shape = [2, 3, 4, 4]
    mean1, mean2 = torch.rand(*tar_shape, 1), torch.rand(*tar_shape, 1)
    # var1, var2 = torch.rand(2, 3, 4, 4, 1), torch.rand(2, 3, 4, 4, 1)
    var1 = torch.randint(1, 3, (*tar_shape, 1)).float()
    var2 = torch.randint(1, 3, (*tar_shape, 1)).float()

    def pdf(x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var) * torch.exp(-(x - mean)**2 /
                                                         (2 * var)))

    delta = 0.0001
    indice = torch.arange(-5, 5, delta).repeat(*mean1.shape)
    p = pdf(indice, mean1, var1)  # pdf of target distribution
    q = pdf(indice, mean2, var2)  # pdf of predicted distribution

    kld_manually = (p * torch.log(p / q) * delta).sum(dim=(1, 2, 3, 4)).mean()

    data = dict(
        mean_pred=mean2,
        mean_target=mean1,
        logvar_pred=torch.log(var2),
        logvar_target=torch.log(var1))

    metric = GaussianKLD(2)
    metric.prepare()
    metric.feed(data, 'reals')
    kld = metric.summary()
    # this is a quite loose limitation for we cannot choose delta which is
    # small enough for precise kld calculation
    np.testing.assert_almost_equal(kld, kld_manually, decimal=1)
    # assert (kld - kld_manually < 1e-1).all()

    metric_base_2 = GaussianKLD(2, base='2')
    metric_base_2.prepare()
    metric_base_2.feed(data, 'reals')
    kld_base_2 = metric_base_2.summary()
    np.testing.assert_almost_equal(kld_base_2, kld / np.log(2), decimal=4)
    # assert kld_base_2 == kld / np.log(2)

    # test wrong log_base
    with pytest.raises(AssertionError):
        GaussianKLD(2, base='10')

    # test other reduction --> mean
    metric = GaussianKLD(2, reduction='mean')
    metric.prepare()
    metric.feed(data, 'reals')
    kld = metric.summary()

    # test other reduction --> sum
    metric = GaussianKLD(2, reduction='sum')
    metric.prepare()
    metric.feed(data, 'reals')
    kld = metric.summary()

    # test other reduction --> error
    with pytest.raises(AssertionError):
        metric = GaussianKLD(2, reduction='none')
