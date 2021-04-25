from copy import deepcopy

import pytest
import torch

from mmgen.models.architectures.stylegan.generator_discriminator_v2 import (
    StyleGAN2Discriminator, StyleGANv2Generator)
from mmgen.models.architectures.stylegan.modules import (Blur,
                                                         ModulatedStyleConv,
                                                         ModulatedToRGB)
from mmgen.models.architectures.stylegan.mspie import (
    MSStyleGAN2Discriminator, MSStyleGANv2Generator)


class TestBlur:

    @classmethod
    def setup_class(cls):
        cls.kernel = [1, 3, 3, 1]
        cls.pad = (1, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_blur_cuda(self):
        blur = Blur(self.kernel, self.pad)
        x = torch.randn((2, 3, 8, 8))
        res = blur(x)

        assert res.shape == (2, 3, 7, 7)


class TestModStyleConv:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            style_channels=5,
            upsample=True)

    def test_mod_styleconv_cpu(self):
        conv = ModulatedStyleConv(**self.default_cfg)
        input_x = torch.randn((2, 3, 4, 4))
        input_style = torch.randn((2, 5))

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 8, 8)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        conv = ModulatedStyleConv(**_cfg)
        input_x = torch.randn((2, 3, 4, 4))
        input_style = torch.randn((2, 5))

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_mod_styleconv_cuda(self):
        conv = ModulatedStyleConv(**self.default_cfg).cuda()
        input_x = torch.randn((2, 3, 4, 4)).cuda()
        input_style = torch.randn((2, 5)).cuda()

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 8, 8)

        _cfg = deepcopy(self.default_cfg)
        _cfg['upsample'] = False
        conv = ModulatedStyleConv(**_cfg).cuda()
        input_x = torch.randn((2, 3, 4, 4)).cuda()
        input_style = torch.randn((2, 5)).cuda()

        res = conv(input_x, input_style)
        assert res.shape == (2, 1, 4, 4)


class TestToRGB:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_channels=5, style_channels=5, out_channels=3)

    def test_torgb_cpu(self):
        model = ModulatedToRGB(**self.default_cfg)
        input_x = torch.randn((2, 5, 4, 4))
        style = torch.randn((2, 5))

        res = model(input_x, style)
        assert res.shape == (2, 3, 4, 4)

        input_x = torch.randn((2, 5, 8, 8))
        style = torch.randn((2, 5))
        skip = torch.randn(2, 3, 4, 4)
        res = model(input_x, style, skip)
        assert res.shape == (2, 3, 8, 8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_torgb_cuda(self):
        model = ModulatedToRGB(**self.default_cfg).cuda()
        input_x = torch.randn((2, 5, 4, 4)).cuda()
        style = torch.randn((2, 5)).cuda()

        res = model(input_x, style)
        assert res.shape == (2, 3, 4, 4)

        input_x = torch.randn((2, 5, 8, 8)).cuda()
        style = torch.randn((2, 5)).cuda()
        skip = torch.randn(2, 3, 4, 4).cuda()
        res = model(input_x, style, skip)
        assert res.shape == (2, 3, 8, 8)


class TestStyleGAN2Generator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            out_size=64, style_channels=16, num_mlps=4, channel_multiplier=1)

    def test_stylegan2_g_cpu(self):
        # test default config
        g = StyleGANv2Generator(**self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        truncation_mean = g.get_mean_latent()
        res = g(
            None,
            num_batches=2,
            randomize_noise=False,
            truncation=0.7,
            truncation_latent=truncation_mean)
        assert res.shape == (2, 3, 64, 64)

        res = g.style_mixing(2, 2, truncation_latent=truncation_mean)
        assert res.shape[2] == 64

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        styles = [torch.randn((1, 16)) for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)) for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 256
        g = StyleGANv2Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_fp16_stylegan2_G_cuda(self):

        g = StyleGANv2Generator(**self.default_cfg, num_fp16_scales=2).cuda()
        res = g(None, num_batches=2)
        assert res.dtype == torch.float32

        g = StyleGANv2Generator(**self.default_cfg, fp16_enabled=True).cuda()
        res = g(None, num_batches=2)
        assert res.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_g_cuda(self):
        # test default config
        g = StyleGANv2Generator(**self.default_cfg).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        styles = [torch.randn((1, 16)).cuda() for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)).cuda() for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 256
        g = StyleGANv2Generator(**cfg_).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)


class TestMSStyleGAN2Generator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            out_size=64, style_channels=16, num_mlps=4, channel_multiplier=1)

    def test_msstylegan2_g_cpu(self):
        # test default config
        g = MSStyleGANv2Generator(**self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        styles = [torch.randn((1, 16)) for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)) for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 256
        g = MSStyleGANv2Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        g = MSStyleGANv2Generator(deconv2conv=True, **self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        g = MSStyleGANv2Generator(deconv2conv=True, **self.default_cfg)
        truncation_mean = g.get_mean_latent()
        res = g(
            None,
            num_batches=2,
            randomize_noise=False,
            chosen_scale=2,
            truncation=0.7,
            truncation_latent=truncation_mean)
        assert res.shape == (2, 3, 96, 96)

        res = g.style_mixing(2, 2, truncation_latent=truncation_mean)
        assert res.shape[2] == 64

        g = MSStyleGANv2Generator(
            no_pad=True, deconv2conv=True, interp_pad=4, **self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        g = MSStyleGANv2Generator(
            deconv2conv=True, up_after_conv=True, **self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        g = MSStyleGANv2Generator(
            deconv2conv=True, up_after_conv=True, **self.default_cfg)
        res = g(None, num_batches=2, chosen_scale=4)
        assert res.shape == (2, 3, 128, 128)

        g = MSStyleGANv2Generator(
            deconv2conv=True, up_after_conv=True, **self.default_cfg)
        res = g(None, num_batches=2, chosen_scale=(4, 4))
        assert res.shape == (2, 3, 128, 128)

        g = MSStyleGANv2Generator(
            head_pos_encoding=dict(
                type='SPE', embedding_dim=256, padding_idx=0, init_size=128),
            **self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        g = MSStyleGANv2Generator(
            head_pos_encoding=dict(
                type='SPE', embedding_dim=256, padding_idx=0, init_size=128),
            interp_head=True,
            **self.default_cfg)
        res = g(None, num_batches=1, chosen_scale=(4, 4))
        assert res.shape == (1, 3, 128, 128)

        g = MSStyleGANv2Generator(
            head_pos_encoding=dict(type='CatersianGrid'), **self.default_cfg)
        res = g(None, num_batches=2, chosen_scale=(4, 4))
        assert res.shape == (2, 3, 128, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_ms_g_cuda(self):
        # test default config
        g = MSStyleGANv2Generator(**self.default_cfg).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        random_noise = g.make_injected_noise()
        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        styles = [torch.randn((1, 16)).cuda() for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 64, 64)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)).cuda() for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 256
        g = MSStyleGANv2Generator(**cfg_).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)


class TestStyleGANv2Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64, channel_multiplier=1)

    def test_stylegan2_disc_cpu(self):
        d = StyleGAN2Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_stylegan2_disc_cuda(self):
        d = StyleGAN2Discriminator(**self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_fp16_stylegan2_disc_cuda(self):
        d = StyleGAN2Discriminator(
            **self.default_cfg, num_fp16_scales=2).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)
        assert score.dtype == torch.float32


class TestMSStyleGANv2Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64, channel_multiplier=1)

    def test_msstylegan2_disc_cpu(self):
        d = MSStyleGAN2Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

        d = MSStyleGAN2Discriminator(
            with_adaptive_pool=True, **self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_msstylegan2_disc_cuda(self):
        d = MSStyleGAN2Discriminator(**self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)

        d = MSStyleGAN2Discriminator(
            with_adaptive_pool=True, **self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)
