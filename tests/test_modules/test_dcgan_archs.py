import pytest
import torch

from mmgen.models import DCGANDiscriminator, DCGANGenerator, build_module


class TestDCGANGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((2, 100))
        cls.default_config = dict(
            type='DCGANGenerator', output_scale=16, base_channels=32)

    def test_dcgan_generator(self):

        # test default setting with builder
        g = build_module(self.default_config)
        assert isinstance(g, DCGANGenerator)
        assert g.num_upsamples == 2
        assert not g.output_layer.with_norm
        assert len(g.upsampling) == 1
        assert g.upsampling[0].with_norm
        assert g.noise2feat.with_norm
        assert isinstance(g.output_layer.activate, torch.nn.Tanh)
        # check forward function
        img = g(self.noise)
        assert img.shape == (2, 3, 16, 16)
        img = g(self.noise[:, :, None, None])
        assert img.shape == (2, 3, 16, 16)
        img = g(torch.randn, num_batches=2)
        assert img.shape == (2, 3, 16, 16)
        img = g(None, num_batches=2)
        assert img.shape == (2, 3, 16, 16)
        with pytest.raises(ValueError):
            _ = g(torch.randn((1, 100, 3)))
        with pytest.raises(AssertionError):
            _ = g(torch.randn)
        with pytest.raises(AssertionError):
            _ = g(None)
        with pytest.raises(AssertionError):
            _ = g(torch.randn(2, 10))
        results = g(self.noise, return_noise=True)
        assert results['noise_batch'].shape == (2, 100, 1, 1)

        # sanity check for args with cpu model
        g = DCGANGenerator(32, base_channels=64)
        img = g(self.noise)
        assert img.shape == (2, 3, 32, 32)
        assert g.base_channels == 64
        g = DCGANGenerator(16, out_channels=1, base_channels=32)
        img = g(self.noise)
        assert img.shape == (2, 1, 16, 16)
        g = DCGANGenerator(16, noise_size=10, base_channels=32)
        with pytest.raises(AssertionError):
            _ = g(self.noise)
        img = g(torch.randn(2, 10))
        assert img.shape == (2, 3, 16, 16)
        g = DCGANGenerator(
            16, default_act_cfg=dict(type='LeakyReLU'), base_channels=32)
        assert isinstance(g.noise2feat.activate, torch.nn.LeakyReLU)
        assert isinstance(g.upsampling[0].activate, torch.nn.LeakyReLU)
        assert isinstance(g.output_layer.activate, torch.nn.Tanh)

        with pytest.raises(TypeError):
            _ = DCGANGenerator(
                16, noise_size=10, base_channels=32, pretrained=dict())

        # check for cuda
        if not torch.cuda.is_available():
            return

        g = build_module(self.default_config).cuda()
        assert isinstance(g, DCGANGenerator)
        assert g.num_upsamples == 2
        assert not g.output_layer.with_norm
        assert len(g.upsampling) == 1
        assert g.upsampling[0].with_norm
        # check forward function
        img = g(self.noise)
        assert img.shape == (2, 3, 16, 16)
        img = g(self.noise[:, :, None, None])
        assert img.shape == (2, 3, 16, 16)
        img = g(torch.randn, num_batches=2)
        assert img.shape == (2, 3, 16, 16)
        img = g(None, num_batches=2)
        assert img.shape == (2, 3, 16, 16)
        with pytest.raises(ValueError):
            _ = g(torch.randn((1, 100, 3)))
        with pytest.raises(AssertionError):
            _ = g(torch.randn)
        with pytest.raises(AssertionError):
            _ = g(None)
        results = g(self.noise, return_noise=True)
        assert results['noise_batch'].shape == (2, 100, 1, 1)


class TestDCGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.input_tensor = torch.randn((2, 3, 32, 32))
        cls.default_config = dict(
            type='DCGANDiscriminator',
            input_scale=32,
            output_scale=4,
            out_channels=5)

    def test_dcgan_discriminator(self):
        # test default setting with builder
        d = build_module(self.default_config)
        pred = d(self.input_tensor)
        assert pred.shape == (2, 5)
        assert d.num_downsamples == 3
        assert len(d.downsamples) == 3
        assert not d.downsamples[0].with_norm
        assert not d.output_layer.with_norm
        assert not d.output_layer.with_activation
        assert isinstance(d.downsamples[1].activate, torch.nn.LeakyReLU)
        assert isinstance(d.downsamples[1].norm, torch.nn.BatchNorm2d)

        # sanity check for args with cpu model
        d = DCGANDiscriminator(input_scale=64, output_scale=8, out_channels=2)
        assert d.input_scale == 64 and d.output_scale == 8
        assert d.num_downsamples == 3
        assert d.out_channels == 2
        pred = d(torch.randn((1, 3, 64, 64)))
        assert pred.shape == (1, 50)

        with pytest.raises(TypeError):
            _ = DCGANDiscriminator(32, 4, 2, pretrained=dict())

        # check for cuda
        if not torch.cuda.is_available():
            return

        # test default setting with builder on GPU
        d = build_module(self.default_config).cuda()
        pred = d(self.input_tensor.cuda())
        assert pred.shape == (2, 5)
        assert d.num_downsamples == 3
        assert len(d.downsamples) == 3
        assert not d.downsamples[0].with_norm
        assert not d.output_layer.with_norm
        assert not d.output_layer.with_activation
        assert isinstance(d.downsamples[1].activate, torch.nn.LeakyReLU)
