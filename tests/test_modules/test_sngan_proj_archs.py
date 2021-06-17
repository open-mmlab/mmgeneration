from copy import deepcopy

import pytest
import torch

from mmgen.models import ProjDiscriminator, SNGANGenerator, build_module


class TestSNGANPROJGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((2, 128))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SNGANGenerator',
            noise_size=128,
            output_scale=32,
            num_classes=10,
            base_channels=32)

    def test_sngan_proj_generator(self):

        # test default setting with builder
        g = build_module(self.default_config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test return noise
        x = g(None, num_batches=2, return_noise=True)
        assert x['fake_img'].shape == (2, 3, 32, 32)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(self.noise, label=self.label, return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(torch.randn, num_batches=2, return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        # test different output_scale
        config = deepcopy(self.default_config)
        config['output_scale'] = 64
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 64, 64)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 64
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [16, 8, 4, 2]}
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> error (key not find)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {64: [16, 8, 4, 2]}
        with pytest.raises(KeyError):
            g = build_module(config)

        # test different channels_cfg --> error (type not match)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = '1234'
        with pytest.raises(ValueError):
            g = build_module(config)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test norm_eps
        config = deepcopy(self.default_config)
        config['norm_eps'] = 1e-9
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test pretrained --> raise error
        config = deepcopy(self.default_config)
        config['pretrained'] = 42
        with pytest.raises(TypeError):
            g = build_module(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_sngan_proj_generator_cuda(self):

        # test default setting with builder
        g = build_module(self.default_config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test return noise
        x = g(None, num_batches=2, return_noise=True)
        assert x['fake_img'].shape == (2, 3, 32, 32)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(self.noise.cuda(), label=self.label.cuda(), return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        x = g(torch.randn, num_batches=2, return_noise=True)
        assert x['noise_batch'].shape == (2, 128)
        assert x['label'].shape == (2, )

        # test different output_scale
        config = deepcopy(self.default_config)
        config['output_scale'] = 64
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 64, 64)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 64
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [16, 8, 4, 2]}
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = True
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test norm_eps
        config = deepcopy(self.default_config)
        config['norm_eps'] = 1e-9
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)


class TestLSGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.x = torch.randn((2, 3, 32, 32))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='ProjDiscriminator',
            input_scale=32,
            num_classes=10,
            input_channels=3)

    def test_sngan_proj_discriminator(self):

        # test default setting with builder
        d = build_module(self.default_config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different input_scale
        config = deepcopy(self.default_config)
        config['input_scale'] = 64
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x, self.label)
        assert score.shape == (2, 1)

        # test num_classes == 0 (w/o proj_y)
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 128
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [2, 4, 8, 16]}
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different channels_cfg --> error (key not find)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {64: [2, 4, 8, 16]}
        with pytest.raises(KeyError):
            d = build_module(config)

        # test different channels_cfg --> error (type not match)
        config = deepcopy(self.default_config)
        config['channels_cfg'] = '1234'
        with pytest.raises(ValueError):
            d = build_module(config)

        # test different downsample_cfg --> list
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = [True, False, False]
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different downsample_cfg --> dict
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = {
            32: [True, False, False],
            64: [True, True, True, True]
        }
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different downsample_cfg --> error (key not find)
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = {64: [True, True, True, True]}
        with pytest.raises(KeyError):
            d = build_module(config)

        # test different downsample_cfg --> error (type not match)
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = '1234'
        with pytest.raises(ValueError):
            d = build_module(config)

        # test downsample_cfg and channels_cfg not match
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = [True, False, False]
        config['channels_cfg'] = [1, 1, 1, 1]
        with pytest.raises(ValueError):
            d = build_module(config)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = False
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test pretrained --> raise error
        config = deepcopy(self.default_config)
        config['pretrained'] = 42
        with pytest.raises(TypeError):
            d = build_module(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_lsgan_discriminator_cuda(self):

        # test default setting with builder
        d = build_module(self.default_config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different input_scale
        config = deepcopy(self.default_config)
        config['input_scale'] = 64
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        x = torch.randn((2, 3, 64, 64)).cuda()
        score = d(x, self.label.cuda())
        assert score.shape == (2, 1)

        # test num_classes == 0 (w/o proj_y)
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 128
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different channels_cfg --> list
        config = deepcopy(self.default_config)
        config['channels_cfg'] = [1, 1, 1]
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different channels_cfg --> dict
        config = deepcopy(self.default_config)
        config['channels_cfg'] = {32: [1, 1, 1], 64: [2, 4, 8, 16]}
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different downsample_cfg --> list
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = [True, False, False]
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different downsample_cfg --> dict
        config = deepcopy(self.default_config)
        config['downsample_cfg'] = {
            32: [True, False, False],
            64: [True, True, True, True]
        }
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different act_cfg
        config = deepcopy(self.default_config)
        config['act_cfg'] = dict(type='Sigmoid')
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different with_spectral_norm
        config = deepcopy(self.default_config)
        config['with_spectral_norm'] = False
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)


class TestSNGANGenResBlock(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn(2, 16, 5, 5)
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SNGANGenResBlock',
            num_classes=10,
            in_channels=16,
            out_channels=16,
            use_cbn=True,
            use_norm_affine=False,
            norm_cfg=dict(type='BN'),
            upsample_cfg=dict(type='nearest', scale_factor=2),
            upsample=True,
            init_cfg=dict(type='BigGAN'))

    def test_snganGenResBlock(self):

        # test default config
        block = build_module(self.default_config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 10, 10)

        # test no upsample config and no learnable sc
        config = deepcopy(self.default_config)
        config['upsample'] = False
        block = build_module(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test learnable shortcut + w/o upsample
        config = deepcopy(self.default_config)
        config['out_channels'] = 32
        config['upsample'] = False
        block = build_module(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 32, 5, 5)

        # test init_cfg + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        config['upsample'] = False
        block = build_module(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = build_module(config)
        out = block(self.input, self.label)
        assert out.shape == (2, 16, 10, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_snganGenResBlock_cuda(self):

        # test default config
        block = build_module(self.default_config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 10, 10)

        # test no upsample config and no learnable sc
        config = deepcopy(self.default_config)
        config['upsample'] = False
        block = build_module(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test learnable shortcut + w/o upsample
        config = deepcopy(self.default_config)
        config['out_channels'] = 32
        config['upsample'] = False
        block = build_module(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 32, 5, 5)

        # test init_cfg + w/o learnable shortcut
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        config['upsample'] = False
        block = build_module(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = build_module(config).cuda()
        out = block(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 16, 10, 10)


class TestSNDiscResBlock(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn(2, 16, 10, 10)
        cls.default_config = dict(
            type='SNGANDiscResBlock',
            in_channels=16,
            out_channels=16,
            downsample=True,
            init_cfg=dict(type='BigGAN'))

    def test_snganDiscResBlock(self):
        # test default config
        block = build_module(self.default_config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = build_module(config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

        # test w/o learnabel shortcut + w/o downsample
        config = deepcopy(self.default_config)
        config['downsample'] = False
        config['out_channels'] = 8
        block = build_module(config)
        out = block(self.input)
        assert out.shape == (2, 8, 10, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_snganDiscResBlock_cuda(self):
        # test default config
        block = build_module(self.default_config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = build_module(config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test w/o learnabel shortcut + w/o downsample
        config = deepcopy(self.default_config)
        config['downsample'] = False
        config['out_channels'] = 8
        block = build_module(config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 8, 10, 10)


class TestSNDiscHeadResBlock(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn(2, 16, 10, 10)
        cls.default_config = dict(
            type='SNGANDiscHeadResBlock',
            in_channels=16,
            out_channels=16,
            init_cfg=dict(type='BigGAN'))

    def test_snganDiscHeadResBlock(self):
        # test default config
        block = build_module(self.default_config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = build_module(config)
        out = block(self.input)
        assert out.shape == (2, 16, 5, 5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_snganDiscHeadResBlock_cuda(self):
        # test default config
        block = build_module(self.default_config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)

        # test conv_cfg
        config = deepcopy(self.default_config)
        config['conv_cfg'] = dict(
            kernel_size=1, stride=1, padding=0, act_cfg=None)
        block = build_module(config).cuda()
        out = block(self.input.cuda())
        assert out.shape == (2, 16, 5, 5)


class TestSNConditionalNorm(object):

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn((2, 4, 4, 4))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SNConditionNorm',
            in_channels=4,
            num_classes=10,
            use_cbn=True,
            cbn_norm_affine=False,
            init_cfg=dict(type='BigGAN'))

    def test_conditionalNorm(self):
        # test build from default config
        norm = build_module(self.default_config)
        out = norm(self.input, self.label)
        assert out.shape == (2, 4, 4, 4)

        # test w/o use_cbn
        config = deepcopy(self.default_config)
        config['use_cbn'] = False
        norm = build_module(config)
        out = norm(self.input)
        assert out.shape == (2, 4, 4, 4)

        # test num_class < 0 and cbn = False
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        config['use_cbn'] = False
        norm = build_module(config)
        out = norm(self.input)
        assert out.shape == (2, 4, 4, 4)

        # test num_classes <= 0 and cbn = True
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        with pytest.raises(ValueError):
            norm = build_module(config)

        # test IN
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='IN')
        norm = build_module(config)
        out = norm(self.input, self.label)
        assert out.shape == (2, 4, 4, 4)

        # test SyncBN
        # config = deepcopy(self.default_config)
        # config['norm_cfg'] = dict(type='SyncBN')
        # norm = build_module(config)
        # out = norm(self.input, self.label)
        # assert out.shape == (2, 4, 4, 4)

        # test unknown norm type
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='GN')
        with pytest.raises(ValueError):
            norm = build_module(config)

        # test init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        norm = build_module(config)
        out = norm(self.input, self.label)
        assert out.shape == (2, 4, 4, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_conditionalNorm_cuda(self):
        # test build from default config
        norm = build_module(self.default_config).cuda()
        out = norm(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test w/o use_cbn
        config = deepcopy(self.default_config)
        config['use_cbn'] = False
        norm = build_module(config).cuda()
        out = norm(self.input.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test num_class < 0 and cbn = False
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        config['use_cbn'] = False
        norm = build_module(config).cuda()
        out = norm(self.input.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test num_classes <= 0 and cbn = True
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        with pytest.raises(ValueError):
            norm = build_module(config)

        # test IN
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='IN')
        norm = build_module(config).cuda()
        out = norm(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 4, 4, 4)

        # test SyncBN
        # config = deepcopy(self.default_config)
        # config['norm_cfg'] = dict(type='SyncBN')
        # norm = build_module(config)
        # out = norm(self.input, self.label)
        # assert out.shape == (2, 4, 4, 4)

        # test unknown norm type
        config = deepcopy(self.default_config)
        config['norm_cfg'] = dict(type='GN')
        with pytest.raises(ValueError):
            norm = build_module(config)

        # test init_cfg
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='chainer')
        norm = build_module(config).cuda()
        out = norm(self.input.cuda(), self.label.cuda())
        assert out.shape == (2, 4, 4, 4)
