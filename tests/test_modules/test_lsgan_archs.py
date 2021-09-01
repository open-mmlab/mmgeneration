import pytest
import torch

from mmgen.models import LSGANDiscriminator, LSGANGenerator, build_module


class TestLSGANGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((3, 128))
        cls.default_config = dict(
            type='LSGANGenerator', noise_size=128, output_scale=128)

    def test_lsgan_generator(self):

        # test default setting with builder
        g = build_module(self.default_config)
        assert isinstance(g, LSGANGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 128, 128)
        x = g(None, num_batches=3, return_noise=True)
        assert x['noise_batch'].shape == (3, 128)
        x = g(self.noise, return_noise=True)
        assert x['noise_batch'].shape == (3, 128)
        x = g(torch.randn, num_batches=3, return_noise=True)
        assert x['noise_batch'].shape == (3, 128)

        # test different output_scale
        config = dict(type='LSGANGenerator', noise_size=128, output_scale=64)
        g = build_module(config)
        assert isinstance(g, LSGANGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_lsgan_generator_cuda(self):

        # test default setting with builder
        g = build_module(self.default_config).cuda()
        assert isinstance(g, LSGANGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 128, 128)
        x = g(None, num_batches=3, return_noise=True)
        assert x['noise_batch'].shape == (3, 128)
        x = g(self.noise.cuda(), return_noise=True)
        assert x['noise_batch'].shape == (3, 128)
        x = g(torch.randn, num_batches=3, return_noise=True)
        assert x['noise_batch'].shape == (3, 128)

        # test different output_scale
        config = dict(type='LSGANGenerator', noise_size=128, output_scale=64)
        g = build_module(config).cuda()
        assert isinstance(g, LSGANGenerator)
        x = g(None, num_batches=3)
        assert x.shape == (3, 3, 64, 64)


class TestLSGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.x = torch.randn((2, 3, 128, 128))
        cls.default_config = dict(
            type='LSGANDiscriminator', in_channels=3, input_scale=128)

    def test_lsgan_discriminator(self):

        # test default setting with builder
        d = build_module(self.default_config)
        assert isinstance(d, LSGANDiscriminator)
        score = d(self.x)
        assert score.shape == (2, 1)

        # test different input_scale
        config = dict(type='LSGANDiscriminator', in_channels=3, input_scale=64)
        d = build_module(config)
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x)
        assert score.shape == (2, 1)

        # test different config
        config = dict(
            type='LSGANDiscriminator',
            in_channels=3,
            input_scale=64,
            out_act_cfg=dict(type='Sigmoid'))
        d = build_module(config)
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_lsgan_discriminator_cuda(self):

        # test default setting with builder
        d = build_module(self.default_config).cuda()
        assert isinstance(d, LSGANDiscriminator)
        score = d(self.x.cuda())
        assert score.shape == (2, 1)

        # test different input_scale
        config = dict(type='LSGANDiscriminator', in_channels=3, input_scale=64)
        d = build_module(config).cuda()
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x.cuda())
        assert score.shape == (2, 1)

        # test different config
        config = dict(
            type='LSGANDiscriminator',
            in_channels=3,
            input_scale=64,
            out_act_cfg=dict(type='Sigmoid'))
        d = build_module(config).cuda()
        assert isinstance(d, LSGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x.cuda())
        assert score.shape == (2, 1)
