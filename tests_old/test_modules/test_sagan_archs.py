# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmgen.models import ProjDiscriminator as SAGANDiscriminator
from mmgen.models import SNGANGenerator as SAGANGenerator
from mmgen.models import build_module


class TestSAGANGenerator(object):

    @classmethod
    def setup_class(cls):
        cls.noise = torch.randn((2, 128))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SAGANGenerator',
            base_channels=32,
            noise_size=128,
            output_scale=32,
            attention_cfg=dict(type='SelfAttentionBlock'),
            attention_after_nth_block=2,
            num_classes=10)

    def test_sagan_generator(self):

        # test default setting with builder
        g = build_module(self.default_config)
        assert isinstance(g, SAGANGenerator)
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
        assert isinstance(g, SAGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 64, 64)

        # test different attention_after_nth_block
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = [1, 2]
        g = build_module(config)
        assert isinstance(g, SAGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # wrong type of attention_after_nth_block --> wrong type
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = '1'
        with pytest.raises(ValueError):
            g = build_module(config)

        # wrong type of attention_after_nth_block --> wrong type of list
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = ['1', '2']
        with pytest.raises(ValueError):
            g = build_module(config)

        # test init_cfg --> SAGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        g = build_module(config)
        assert isinstance(g, SAGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_sagan_generator_cuda(self):

        # test default setting with builder
        g = build_module(self.default_config).cuda()
        assert isinstance(g, SAGANGenerator)
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
        assert isinstance(g, SAGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 64, 64)

        # test different attention_after_nth_block
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = [1, 2]
        g = build_module(config).cuda()
        assert isinstance(g, SAGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)

        # test init_cfg --> SAGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        g = build_module(config).cuda()
        assert isinstance(g, SAGANGenerator)
        x = g(None, num_batches=2)
        assert x.shape == (2, 3, 32, 32)


class TestSAGANDiscriminator(object):

    @classmethod
    def setup_class(cls):
        cls.x = torch.randn((2, 3, 32, 32))
        cls.label = torch.randint(0, 10, (2, ))
        cls.default_config = dict(
            type='SAGANDiscriminator',
            input_scale=32,
            num_classes=10,
            input_channels=3)

    def test_sngan_proj_discriminator(self):

        # test default setting with builder
        d = build_module(self.default_config)
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different input_scale
        config = deepcopy(self.default_config)
        config['input_scale'] = 64
        d = build_module(config)
        assert isinstance(d, SAGANDiscriminator)
        x = torch.randn((2, 3, 64, 64))
        score = d(x, self.label)
        assert score.shape == (2, 1)

        # test num_classes == 0 (w/o proj_y)
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        d = build_module(config)
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 128
        d = build_module(config)
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # test different attention_after_nth_block
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = [1, 2]
        d = build_module(config)
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

        # wrong type of attention_after_nth_block --> wrong type
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = '1'
        with pytest.raises(ValueError):
            d = build_module(config)

        # wrong type of attention_after_nth_block --> wrong type of list
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = ['1', '2']
        with pytest.raises(ValueError):
            d = build_module(config)

        # test init_cfg --> SAGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        d = build_module(config)
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x, self.label)
        assert score.shape == (2, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_lsgan_discriminator_cuda(self):

        # test default setting with builder
        d = build_module(self.default_config).cuda()
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different input_scale
        config = deepcopy(self.default_config)
        config['input_scale'] = 64
        d = build_module(config).cuda()
        assert isinstance(d, SAGANDiscriminator)
        x = torch.randn((2, 3, 64, 64)).cuda()
        score = d(x, self.label.cuda())
        assert score.shape == (2, 1)

        # test num_classes == 0 (w/o proj_y)
        config = deepcopy(self.default_config)
        config['num_classes'] = 0
        d = build_module(config).cuda()
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different base_channels
        config = deepcopy(self.default_config)
        config['base_channels'] = 128
        d = build_module(config).cuda()
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test different attention_after_nth_block
        config = deepcopy(self.default_config)
        config['attention_after_nth_block'] = [1, 2]
        d = build_module(config).cuda()
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)

        # test init_cfg --> SAGAN
        config = deepcopy(self.default_config)
        config['init_cfg'] = dict(type='sagan')
        d = build_module(config).cuda()
        assert isinstance(d, SAGANDiscriminator)
        score = d(self.x.cuda(), self.label.cuda())
        assert score.shape == (2, 1)
