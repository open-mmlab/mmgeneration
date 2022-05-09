# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmgen.models.architectures.stylegan import StyleGANv3Generator
from mmgen.models.architectures.stylegan.modules import (MappingNetwork,
                                                         SynthesisInput,
                                                         SynthesisLayer,
                                                         SynthesisNetwork)


class TestMappingNetwork:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            noise_size=4,
            c_dim=0,
            style_channels=4,
            num_ws=2,
            num_layers=2,
            lr_multiplier=0.01,
            w_avg_beta=0.998)

    def test_cpu(self):
        module = MappingNetwork(**self.default_cfg)
        z = torch.randn([1, 4])
        c = None
        y = module(z, c)
        assert y.shape == (1, 2, 4)

        # test update_emas
        y = module(z, c, update_emas=True)
        assert y.shape == (1, 2, 4)

        # test truncation
        y = module(z, c, truncation=2)
        assert y.shape == (1, 2, 4)

        # test with c_dim>0
        cfg = deepcopy(self.default_cfg)
        cfg.update(c_dim=2)
        module = MappingNetwork(**cfg)
        z = torch.randn([2, 4])
        c = torch.eye(2)
        y = module(z, c)
        assert y.shape == (2, 2, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = MappingNetwork(**self.default_cfg).cuda()
        z = torch.randn([1, 4]).cuda()
        c = None
        y = module(z, c)
        assert y.shape == (1, 2, 4)

        # test update_emas
        y = module(z, c, update_emas=True).cuda()
        assert y.shape == (1, 2, 4)

        # test truncation
        y = module(z, c, truncation=2).cuda()
        assert y.shape == (1, 2, 4)

        # test with c_dim>0
        cfg = deepcopy(self.default_cfg)
        cfg.update(c_dim=2)
        module = MappingNetwork(**cfg).cuda()
        z = torch.randn([2, 4]).cuda()
        c = torch.eye(2).cuda()
        y = module(z, c)
        assert y.shape == (2, 2, 4)


class TestSynthesisInput:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            style_channels=6,
            channels=4,
            size=8,
            sampling_rate=16,
            bandwidth=2)

    def test_cpu(self):
        module = SynthesisInput(**self.default_cfg)
        x = torch.randn((2, 6))
        y = module(x)
        assert y.shape == (2, 4, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = SynthesisInput(**self.default_cfg).cuda()
        x = torch.randn((2, 6)).cuda()
        y = module(x)
        assert y.shape == (2, 4, 8, 8)


class TestSynthesisLayer:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            style_channels=6,
            is_torgb=False,
            is_critically_sampled=False,
            use_fp16=False,
            conv_kernel=3,
            in_channels=3,
            out_channels=3,
            in_size=16,
            out_size=16,
            in_sampling_rate=16,
            out_sampling_rate=16,
            in_cutoff=2,
            out_cutoff=2,
            in_half_width=6,
            out_half_width=6)

    def test_cpu(self):
        module = SynthesisLayer(**self.default_cfg)
        x = torch.randn((2, 3, 16, 16))
        w = torch.randn((2, 6))
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)

        # test update_emas
        y = module(x, w, update_emas=True)
        assert y.shape == (2, 3, 16, 16)

        # test force_fp32
        cfg = deepcopy(self.default_cfg)
        cfg.update(use_fp16=True)
        module = SynthesisLayer(**cfg)
        x = torch.randn((2, 3, 16, 16))
        w = torch.randn((2, 6))
        y = module(x, w, force_fp32=False)
        assert y.shape == (2, 3, 16, 16)
        assert y.dtype == torch.float32

        # test critically_sampled
        cfg = deepcopy(self.default_cfg)
        cfg.update(is_critically_sampled=True)
        module = SynthesisLayer(**cfg)
        x = torch.randn((2, 3, 16, 16))
        w = torch.randn((2, 6))
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = SynthesisLayer(**self.default_cfg).cuda()
        x = torch.randn((2, 3, 16, 16)).cuda()
        w = torch.randn((2, 6)).cuda()
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)

        # test update_emas
        y = module(x, w, update_emas=True).cuda()
        assert y.shape == (2, 3, 16, 16)

        # test critically_sampled
        cfg = deepcopy(self.default_cfg)
        cfg.update(is_critically_sampled=True)
        module = SynthesisLayer(**cfg).cuda()
        x = torch.randn((2, 3, 16, 16)).cuda()
        w = torch.randn((2, 6)).cuda()
        y = module(x, w)
        assert y.shape == (2, 3, 16, 16)


class TestSynthesisNetwork:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            style_channels=8, out_size=16, img_channels=3, num_layers=4)

    def test_cpu(self):
        module = SynthesisNetwork(**self.default_cfg)
        ws = torch.randn((2, 6, 8))
        y = module(ws)
        assert y.shape == (2, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = SynthesisNetwork(**self.default_cfg).cuda()
        ws = torch.randn((2, 6, 8)).cuda()
        y = module(ws)
        assert y.shape == (2, 3, 16, 16)


class TestStyleGAN3Generator:

    @classmethod
    def setup_class(cls):
        synthesis_cfg = {
            'type': 'SynthesisNetwork',
            'channel_base': 1024,
            'channel_max': 16,
            'magnitude_ema_beta': 0.999
        }
        cls.default_cfg = dict(
            noise_size=6,
            style_channels=8,
            out_size=16,
            img_channels=3,
            synthesis_cfg=synthesis_cfg)
        synthesis_r_cfg = {
            'type': 'SynthesisNetwork',
            'channel_base': 1024,
            'channel_max': 16,
            'magnitude_ema_beta': 0.999,
            'conv_kernel': 1,
            'use_radial_filters': True
        }
        cls.s3_r_cfg = dict(
            noise_size=6,
            style_channels=8,
            out_size=16,
            img_channels=3,
            synthesis_cfg=synthesis_r_cfg)

    def test_cpu(self):
        generator = StyleGANv3Generator(**self.default_cfg)
        z = torch.randn((2, 6))
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)

        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(rgb2bgr=True))
        generator = StyleGANv3Generator(**cfg)
        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        # test return latents
        result = generator(None, num_batches=2, return_latents=True)
        assert isinstance(result, dict)
        assert result['fake_img'].shape == (2, 3, 16, 16)
        assert result['noise_batch'].shape == (2, 6)
        assert result['latent'].shape == (2, 16, 8)

        # test input_is_latent
        result = generator(
            None, num_batches=2, input_is_latent=True, return_latents=True)
        assert isinstance(result, dict)
        assert result['fake_img'].shape == (2, 3, 16, 16)
        assert result['noise_batch'].shape == (2, 8)
        assert result['latent'].shape == (2, 16, 8)

        generator = StyleGANv3Generator(**self.s3_r_cfg)
        z = torch.randn((2, 6))
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        generator = StyleGANv3Generator(**self.default_cfg).cuda()
        z = torch.randn((2, 6)).cuda()
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)

        cfg = deepcopy(self.default_cfg)
        cfg.update(dict(rgb2bgr=True))
        generator = StyleGANv3Generator(**cfg).cuda()
        y = generator(None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

        generator = StyleGANv3Generator(**self.s3_r_cfg).cuda()
        z = torch.randn((2, 6)).cuda()
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        res = generator(torch.randn, num_batches=1)
        assert res.shape == (1, 3, 16, 16)
