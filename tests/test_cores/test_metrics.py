import os.path as osp

import numpy as np
import pytest
import torch

from mmgen.core.evaluation.metric_utils import extract_inception_features
from mmgen.core.evaluation.metrics import FID, IS, MS_SSIM, PPL, PR, SWD
from mmgen.datasets import UnconditionalImageDataset, build_dataloader
from mmgen.models import build_model
from mmgen.models.architectures import InceptionV3

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
        cls.data_loader = build_dataloader(dataset, 3, 4, dist=False)

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
