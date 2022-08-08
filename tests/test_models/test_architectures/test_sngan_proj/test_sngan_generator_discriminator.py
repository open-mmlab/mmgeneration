# Copyright (c) OpenMMLab. All rights reserved.
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

        # test num_classes == 0 and `use_cbn = True`
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        with pytest.raises(ValueError):
            g = build_module(config)

        # test num_classes == 0 and `use_cbn = False`
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        config['use_cbn'] = False
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

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

        # test with_embedding_spectral_norm
        config = deepcopy(self.default_config)
        config['with_embedding_spectral_norm'] = True
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

        # test sn_eps
        config = deepcopy(self.default_config)
        config['sn_eps'] = 1e-12
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> Studio
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='studio')
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> SNGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        g = build_module(config)
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            g = build_module(config)

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

        # test with_embedding_spectral_norm
        config = deepcopy(self.default_config)
        config['with_embedding_spectral_norm'] = True
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

        # test sn_eps
        config = deepcopy(self.default_config)
        config['sn_eps'] = 1e-12
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2).cuda()
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test different init_cfg --> SNGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan')
        g = build_module(config).cuda()
        assert isinstance(g, SNGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)


class TestSNGANPROJDiscriminator(object):

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

        # test different init_cfg --> studio
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='studio')
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg --> sngan
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan-proj')
        d = build_module(config)
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different init_cfg --> raise error
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='wgan-gp')
        with pytest.raises(NotImplementedError):
            d = build_module(config)

        # test pretrained --> raise error
        config = deepcopy(self.default_config)
        config['pretrained'] = 42
        with pytest.raises(TypeError):
            d = build_module(config)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_sngan_proj_discriminator_cuda(self):

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

        # test different init_cfg --> BigGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='biggan')
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different init_cfg --> sngan
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sngan-proj')
        d = build_module(config).cuda()
        assert isinstance(d, ProjDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)
