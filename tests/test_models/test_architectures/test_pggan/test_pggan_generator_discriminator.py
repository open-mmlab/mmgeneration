# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import pytest
import torch

from mmgen.models.architectures.pggan import PGGANDiscriminator, PGGANGenerator


class TestPGGANGenerator:

    @classmethod
    def setup_class(cls):
        cls.default_noise = torch.randn((2, 8))
        cls.default_cfg = dict(
            noise_size=8, out_scale=16, base_channels=32, max_channels=32)

    def test_pggan_generator(self):
        # test with default cfg
        gen = PGGANGenerator(**self.default_cfg)
        res = gen(None, num_batches=2, transition_weight=0.1)
        assert res.shape == (2, 3, 16, 16)

        res = gen(self.default_noise, transition_weight=0.2)
        assert res.shape == (2, 3, 16, 16)
        with pytest.raises(AssertionError):
            _ = gen(self.default_noise[:, :, None], transition_weight=0.2)

        with pytest.raises(AssertionError):
            _ = gen(torch.randn((2, 1)), transition_weight=0.2)

        res = gen(torch.randn, num_batches=2, transition_weight=0.2)
        assert res.shape == (2, 3, 16, 16)

        # test with input scale
        res = gen(None, num_batches=2, curr_scale=4)
        assert res.shape == (2, 3, 4, 4)
        res = gen(None, num_batches=2, curr_scale=8)
        assert res.shape == (2, 3, 8, 8)

        # test return noise
        res = gen(None, num_batches=2, curr_scale=8, return_noise=True)
        assert res['fake_img'].shape == (2, 3, 8, 8)
        assert res['label'] is None
        assert isinstance(res['noise_batch'], torch.Tensor)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['out_scale'] = 32
        gen = PGGANGenerator(**cfg)
        res = gen(None, num_batches=2, transition_weight=0.1)
        assert res.shape == (2, 3, 32, 32)

        cfg = deepcopy(self.default_cfg)
        cfg['out_scale'] = 4
        gen = PGGANGenerator(**cfg)
        res = gen(None, num_batches=2, transition_weight=0.1)
        assert res.shape == (2, 3, 4, 4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pggan_generator_cuda(self):
        # test with default cfg
        gen = PGGANGenerator(**self.default_cfg).cuda()
        res = gen(None, num_batches=2, transition_weight=0.1)
        assert res.shape == (2, 3, 16, 16)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['out_scale'] = 32
        gen = PGGANGenerator(**cfg).cuda()
        res = gen(None, num_batches=2, transition_weight=0.1)
        assert res.shape == (2, 3, 32, 32)


class TestPGGANDiscriminator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_scale=16, label_size=2)
        cls.default_inputx16 = torch.randn((2, 3, 16, 16))
        cls.default_inputx4 = torch.randn((2, 3, 4, 4))
        cls.default_inputx8 = torch.randn((2, 3, 8, 8))

    def test_pggan_discriminator(self):
        # test with default cfg
        disc = PGGANDiscriminator(**self.default_cfg)

        score, label = disc(self.default_inputx16, transition_weight=0.1)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx8, transition_weight=0.1, curr_scale=8)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx4, transition_weight=0.1, curr_scale=4)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)

        disc = PGGANDiscriminator(
            in_scale=16,
            mbstd_cfg=None,
            downsample_cfg=dict(type='nearest', scale_factor=0.5))

        score = disc(self.default_inputx16, transition_weight=0.1)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score = disc(self.default_inputx8, transition_weight=0.1, curr_scale=8)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score = disc(self.default_inputx4, transition_weight=0.1, curr_scale=4)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        assert not disc.with_mbstd

        with pytest.raises(NotImplementedError):
            _ = PGGANDiscriminator(
                in_scale=16, mbstd_cfg=None, downsample_cfg=dict(type='xx'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pggan_discriminator_cuda(self):
        # test with default cfg
        disc = PGGANDiscriminator(**self.default_cfg).cuda()

        score, label = disc(
            self.default_inputx16.cuda(), transition_weight=0.1)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx8.cuda(), transition_weight=0.1, curr_scale=8)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx4.cuda(), transition_weight=0.1, curr_scale=4)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
