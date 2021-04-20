from copy import deepcopy

import pytest
import torch

from mmgen.models.architectures.stylegan import (StyleGAN1Discriminator,
                                                 StyleGANv1Generator)
from mmgen.models.architectures.stylegan.modules.styleganv1_modules import (
    AdaptiveInstanceNorm, StyleConv)


class TestAdaptiveInstanceNorm:

    @classmethod
    def setup_class(cls):
        cls.in_channel = 512
        cls.style_dim = 512

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_adain_cuda(self):
        adain = AdaptiveInstanceNorm(self.in_channel, self.style_dim).cuda()
        x = torch.randn((2, 512, 8, 8)).cuda()
        style = torch.randn((2, 512)).cuda()
        res = adain(x, style)

        assert res.shape == (2, 512, 8, 8)


class TestStyleConv:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=512,
            out_channels=256,
            kernel_size=3,
            style_channels=512,
            padding=1,
            initial=False,
            blur_kernel=[1, 2, 1],
            upsample=True,
            fused=False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_styleconv_cuda(self):
        conv = StyleConv(**self.default_cfg).cuda()
        input_x = torch.randn((2, 512, 32, 32)).cuda()
        input_style1 = torch.randn((2, 512)).cuda()
        input_style2 = torch.randn((2, 512)).cuda()

        res = conv(input_x, input_style1, input_style2)
        assert res.shape == (2, 256, 64, 64)


class TestStyleGAN1Generator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            out_size=256,
            style_channels=512,
            num_mlps=8,
            blur_kernel=[1, 2, 1],
            lr_mlp=0.01,
            default_style_mode='mix',
            eval_style_mode='single',
            mix_prob=0.9)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_g_cuda(self):
        # test default config
        g = StyleGANv1Generator(**self.default_cfg).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        styles = [torch.randn((1, 512)).cuda() for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)).cuda() for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 128
        g = StyleGANv1Generator(**cfg_).cuda()
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 128, 128)

        # test generate function
        truncation_latent = g.get_mean_latent()
        assert truncation_latent.shape == (1, 512)
        style_mixing_images = g.style_mixing(
            curr_scale=32,
            truncation_latent=truncation_latent,
            n_source=4,
            n_target=4)
        assert style_mixing_images.shape == (25, 3, 32, 32)

    def test_g_cpu(self):
        # test default config
        g = StyleGANv1Generator(**self.default_cfg)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 256, 256)

        random_noise = g.make_injected_noise()
        res = g(
            None,
            num_batches=1,
            injected_noise=random_noise,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            None, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        styles = [torch.randn((1, 512)) for _ in range(2)]
        res = g(
            styles, num_batches=1, injected_noise=None, randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        res = g(
            torch.randn,
            num_batches=1,
            injected_noise=None,
            randomize_noise=False)
        assert res.shape == (1, 3, 256, 256)

        g.eval()
        assert g.default_style_mode == 'single'

        g.train()
        assert g.default_style_mode == 'mix'

        with pytest.raises(AssertionError):
            styles = [torch.randn((1, 6)) for _ in range(2)]
            _ = g(styles, injected_noise=None, randomize_noise=False)

        cfg_ = deepcopy(self.default_cfg)
        cfg_['out_size'] = 128
        g = StyleGANv1Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 128, 128)

        # test generate function
        truncation_latent = g.get_mean_latent()
        assert truncation_latent.shape == (1, 512)
        style_mixing_images = g.style_mixing(
            curr_scale=32,
            truncation_latent=truncation_latent,
            n_source=4,
            n_target=4)
        assert style_mixing_images.shape == (25, 3, 32, 32)


class TestStyleGANv1Disc:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_stylegan1_disc_cuda(self):
        d = StyleGAN1Discriminator(**self.default_cfg).cuda()
        img = torch.randn((2, 3, 64, 64)).cuda()
        score = d(img)
        assert score.shape == (2, 1)

    def test_stylegan1_disc_cpu(self):
        d = StyleGAN1Discriminator(**self.default_cfg)
        img = torch.randn((2, 3, 64, 64))
        score = d(img)
        assert score.shape == (2, 1)
