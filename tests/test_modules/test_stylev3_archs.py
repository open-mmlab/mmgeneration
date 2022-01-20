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
            z_dim=4,
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        module = SynthesisLayer(**self.default_cfg).cuda()
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
        synthesis_kwargs = dict(num_layers=4)
        cls.default_cfg = dict(
            z_dim=6,
            c_dim=0,
            style_channels=8,
            out_size=16,
            img_channels=3,
            **synthesis_kwargs)

    def test_cpu(self):
        generator = StyleGANv3Generator(**self.default_cfg)
        z = torch.randn((2, 6))
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)

        y = generator(None, None, num_batches=2)
        assert y.shape == (2, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cuda(self):
        generator = StyleGANv3Generator(**self.default_cfg).cuda()
        z = torch.randn((2, 6)).cuda()
        c = None
        y = generator(z, c)
        assert y.shape == (2, 3, 16, 16)
