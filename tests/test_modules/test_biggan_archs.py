from copy import deepcopy
from functools import partial

import pytest
import torch

from mmgen.models import build_module
from mmgen.models.architectures.biggan import BigGANGenerator


class TestBigGANGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((3, 120))
        num_classes = 1000
        cls.label = torch.randint(0, num_classes, (3, ))
        cls.default_config = dict(
            type='BigGANGenerator',
            output_scale=128,
            num_classes=num_classes,
            base_channels=4)

    def test_biggan_generator(self):

        # test default setting with builder
        g = build_module(self.default_config)
        assert isinstance(g, BigGANGenerator)
        res = g(self.noise, self.label)
        assert res.shape == (3, 3, 128, 128)

        # test 'return_noise'
        res = g(self.noise, self.label, return_noise=True)
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
        noise = torch.randn((3, 119))
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

        # test with `split_noise=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(split_noise=False))
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
        cfg.update(dict(num_classes=0, with_shared_embedding=False))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test no shared embedding
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_shared_embedding=False, split_noise=False))
        g = build_module(cfg)
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_generator_cuda(self):

        # test default setting with builder
        g = build_module(self.default_config).cuda()
        assert isinstance(g, BigGANGenerator)
        res = g(self.noise.cuda(), self.label.cuda())
        assert res.shape == (3, 3, 128, 128)

        # test 'return_noise'
        res = g(self.noise.cuda(), self.label.cuda(), return_noise=True)
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
        noise = torch.randn((3, 119)).cuda()
        res = g(noise, self.label.cuda())
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

        # test with `split_noise=False`
        cfg = deepcopy(self.default_config)
        cfg.update(dict(split_noise=False))
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
        cfg.update(dict(num_classes=0, with_shared_embedding=False))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)

        # test no shared embedding
        cfg = deepcopy(self.default_config)
        cfg.update(dict(with_shared_embedding=False, split_noise=False))
        g = build_module(cfg).cuda()
        res = g(None, None, num_batches=3)
        assert res.shape == (3, 3, 128, 128)


class TestBigGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        num_classes = 1000
        cls.default_config = dict(
            type='BigGANDiscriminator',
            input_scale=128,
            num_classes=num_classes,
            base_channels=8)
        cls.x = torch.randn((2, 3, 128, 128))
        cls.label = torch.randint(0, num_classes, (2, ))

    def test_biggan_discriminator(self):
        # test default settings
        d = build_module(self.default_config)
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_biggan_discriminator_cuda(self):
        # test default settings
        d = build_module(self.default_config).cuda()
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
