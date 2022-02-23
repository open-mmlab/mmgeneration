# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial

import pytest
import torch

from mmgen.models import build_module
# yapf:disable
from mmgen.models.architectures.biggan import (BigGANDeepDiscResBlock,
                                               BigGANDeepDiscriminator,
                                               BigGANDeepGenerator,
                                               BigGANDeepGenResBlock)

# yapf:enable


class TestBigGANDeepGenResBlock:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='BigGANDeepGenResBlock',
            in_channels=32,
            out_channels=16,
            dim_after_concat=100,
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(type='nearest', scale_factor=2),
            sn_eps=1e-6,
            bn_eps=1e-5,
            with_spectral_norm=True,
            input_is_label=False,
            auto_sync_bn=True,
            channel_ratio=4)
        cls.x = torch.randn(2, 32, 8, 8)
        cls.y = torch.randn(2, 100)
        cls.label = torch.randint(0, 100, (2, ))

    def test_biggan_deep_gen_res_block(self):
        # test default setting
        module = build_module(self.default_cfg)
        assert isinstance(module, BigGANDeepGenResBlock)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 16, 16)

        # test without upsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(upsample_cfg=None))
        module = build_module(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 8, 8)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = build_module(cfg)
        out = module(self.x, self.label)
        assert out.shape == (2, 16, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = build_module(cfg)
        out = module(self.x, self.y)
        assert out.shape == (2, 16, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_deep_gen_res_block_cuda(self):
        # test default setting
        module = build_module(self.default_cfg).cuda()
        assert isinstance(module, BigGANDeepGenResBlock)
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 16, 16)

        # test without upsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(upsample_cfg=None))
        module = build_module(cfg).cuda()
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 8, 8)

        # test input_is_label
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(input_is_label=True))
        module = build_module(cfg).cuda()
        out = module(self.x.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 16, 16)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = build_module(cfg).cuda()
        out = module(self.x.cuda(), self.y.cuda())
        assert out.shape == (2, 16, 16, 16)


class TestBigGANDeepDiscResBlock:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type='BigGANDeepDiscResBlock',
            in_channels=32,
            out_channels=64,
            channel_ratio=4,
            act_cfg=dict(type='ReLU', inplace=False),
            sn_eps=1e-6,
            with_downsample=True,
            with_spectral_norm=True)
        cls.x = torch.randn(2, 32, 16, 16)

    def test_biggan_deep_disc_res_block(self):
        # test default setting
        module = build_module(self.default_cfg)
        assert isinstance(module, BigGANDeepDiscResBlock)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

        # test with_downsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(with_downsample=False))
        module = build_module(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 16, 16)

        # test different channel_ratio
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(channel_ratio=8))
        module = build_module(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = build_module(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_deep_disc_res_block_cuda(self):
        # test default setting
        module = build_module(self.default_cfg).cuda()
        assert isinstance(module, BigGANDeepDiscResBlock)
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 8, 8)

        # test with_downsample
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(with_downsample=False))
        module = build_module(cfg).cuda()
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 16, 16)

        # test different channel_ratio
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(channel_ratio=8))
        module = build_module(cfg)
        out = module(self.x)
        assert out.shape == (2, 64, 8, 8)

        # test torch-sn
        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(sn_style='torch'))
        module = build_module(cfg).cuda()
        out = module(self.x.cuda())
        assert out.shape == (2, 64, 8, 8)


class TestBigGANDeepGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((3, 120))
        num_classes = 1000
        cls.label = torch.randint(0, num_classes, (3, ))
        cls.default_config = dict(
            type='BigGANDeepGenerator',
            output_scale=128,
            num_classes=num_classes,
            base_channels=4)

    def test_biggan_deep_generator(self):

        # test default setting with builder
        g = build_module(self.default_config)
        assert isinstance(g, BigGANDeepGenerator)
        res = g(self.noise, self.label)
        assert res.shape == (3, 3, 128, 128)

        # test 'return_noise'
        res = g(self.noise, self.label, return_noise=True)
        assert res['fake_img'].shape == (3, 3, 128, 128)
        assert res['noise_batch'].shape == (3, 120)
        assert res['label'].shape == (3, )

        res = g(None, None, num_batches=3, return_noise=True)
        assert res['fake_img'].shape == (3, 3, 128, 128)
        assert res['noise_batch'].shape == (3, 120)
        assert res['label'].shape == (3, )

        # test callable
        noise = torch.randn
        label = partial(torch.randint, 0, 1000)
        res = g(noise, label, num_batches=2)
        assert res.shape == (2, 3, 128, 128)

        # test different output scale
        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=256))
        g = build_module(cfg)
        noise = torch.randn((3, 120))
        res = g(noise, self.label)
        assert res.shape == (3, 3, 256, 256)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 256, 256)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=512))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 512, 512)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=64))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 64, 64)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=32))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 32, 32)

        # test with `concat_noise=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(concat_noise=False))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test with `with_spectral_norm=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_spectral_norm=False))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test different num_classes
        cfg = deepcopy(self.default_config)
        cfg.update(
            dict(
                num_classes=0, with_shared_embedding=False,
                concat_noise=False))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test no shared embedding
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_shared_embedding=False, concat_noise=False))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test torch-sn
        cfg = deepcopy(self.default_config)
        cfg.update(dict(sn_style='torch'))
        g = build_module(cfg)
        res = g(self.noise, self.label)
        assert res.shape == (3, 3, 128, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_deep_generator_cuda(self):

        # test default setting with builder
        g = build_module(self.default_config).cuda()
        assert isinstance(g, BigGANDeepGenerator)
        res = g(self.noise.cuda(), self.label.cuda())
        assert res.shape == (3, 3, 128, 128)

        # test 'return_noise'
        res = g(self.noise.cuda(), self.label.cuda(), return_noise=True)
        assert res['fake_img'].shape == (3, 3, 128, 128)
        assert res['noise_batch'].shape == (3, 120)
        assert res['label'].shape == (3, )

        res = g(None, None, num_batches=3, return_noise=True)
        assert res['fake_img'].shape == (3, 3, 128, 128)
        assert res['noise_batch'].shape == (3, 120)
        assert res['label'].shape == (3, )

        # test callable
        noise = torch.randn
        label = partial(torch.randint, 0, 1000)
        res = g(noise, label, num_batches=2)
        assert res.shape == (2, 3, 128, 128)

        # test different output scale
        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=256))
        g = build_module(cfg).cuda()
        noise = torch.randn((3, 120))
        res = g(noise.cuda(), self.label.cuda())
        assert res.shape == (3, 3, 256, 256)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 256, 256)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=512))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 512, 512)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=64))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 64, 64)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(output_scale=32))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 32, 32)

        # test with `concat_noise=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(concat_noise=False))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test with `with_spectral_norm=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_spectral_norm=False))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test different num_classes
        cfg = deepcopy(self.default_config)
        cfg.update(
            dict(
                num_classes=0, with_shared_embedding=False,
                concat_noise=False))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test no shared embedding
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_shared_embedding=False, concat_noise=False))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test torch-sn
        cfg = deepcopy(self.default_config)
        cfg.update(dict(sn_style='torch'))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)


class TestBigGANDeepDiscriminator(object):

    @classmethod
    def setup_class(cls):
        num_classes = 1000
        cls.default_config = dict(
            type='BigGANDeepDiscriminator',
            input_scale=128,
            num_classes=num_classes,
            base_channels=8)
        cls.x = torch.randn((2, 3, 128, 128))
        cls.label = torch.randint(0, num_classes, (2, ))

    def test_biggan_deep_discriminator(self):
        # test default settings
        d = build_module(self.default_config)
        assert isinstance(d, BigGANDeepDiscriminator)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        # test different init types
        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_type='N02'))
        d = build_module(cfg)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_type='xavier'))
        d = build_module(cfg)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        # test different num_classes
        cfg = deepcopy(self.default_config)
        cfg.update(dict(num_classes=0))
        d = build_module(cfg)
        y = d(self.x, None)
        assert y.shape == (2, 1)

        # test with `with_spectral_norm=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_spectral_norm=False))
        d = build_module(cfg)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

        # test torch-sn
        cfg = deepcopy(self.default_config)
        cfg.update(dict(sn_style='torch'))
        d = build_module(cfg)
        y = d(self.x, self.label)
        assert y.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_deep_discriminator_cuda(self):
        # test default settings
        d = build_module(self.default_config).cuda()
        assert isinstance(d, BigGANDeepDiscriminator)
        y = d(self.x.cuda(), self.label.cuda())
        assert y.shape == (2, 1)

        # test different init types
        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_type='N02'))
        d = build_module(cfg).cuda()
        y = d(self.x.cuda(), self.label.cuda())
        assert y.shape == (2, 1)

        cfg = deepcopy(self.default_config)
        cfg.update(dict(init_type='xavier'))
        d = build_module(cfg).cuda()
        y = d(self.x.cuda(), self.label.cuda())
        assert y.shape == (2, 1)

        # test different num_classes
        cfg = deepcopy(self.default_config)
        cfg.update(dict(num_classes=0))
        d = build_module(cfg).cuda()
        y = d(self.x.cuda(), None)
        assert y.shape == (2, 1)

        # test with `with_spectral_norm=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_spectral_norm=False))
        d = build_module(cfg).cuda()
        y = d(self.x.cuda(), self.label.cuda())
        assert y.shape == (2, 1)

        # test torch-sn
        cfg = deepcopy(self.default_config)
        cfg.update(dict(sn_style='torch'))
        d = build_module(cfg).cuda()
        y = d(self.x.cuda(), self.label.cuda())
        assert y.shape == (2, 1)
