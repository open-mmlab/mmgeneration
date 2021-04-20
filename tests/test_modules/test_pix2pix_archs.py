from copy import deepcopy

import pytest
import torch

from mmgen.models.architectures.pix2pix import (PatchDiscriminator,
                                                UnetGenerator)


class TestUnetGenerator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type='BN'),
            use_dropout=True,
            init_cfg=dict(type='normal', gain=0.02))

    def test_pix2pix_generator_cpu(self):
        # test with default cfg
        real_a = torch.randn((2, 3, 256, 256))
        gen = UnetGenerator(**self.default_cfg)
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_down'] = 7
        gen = UnetGenerator(**cfg)
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pix2pix_generator_cuda(self):
        # test with default cfg
        real_a = torch.randn((2, 3, 256, 256)).cuda()
        gen = UnetGenerator(**self.default_cfg).cuda()
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_down'] = 7
        gen = UnetGenerator(**cfg).cuda()
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)


class TestPatchDiscriminator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=6,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='BN'),
            init_cfg=dict(type='normal', gain=0.02))
        cls.default_inputx256 = torch.randn((2, 6, 256, 256))

    def test_pix2pix_discriminator_cpu(self):
        # test with default cfg
        disc = PatchDiscriminator(**self.default_cfg)
        score = disc(self.default_inputx256)
        assert score.shape == (2, 1, 30, 30)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_conv'] = 4
        disc = PatchDiscriminator(**cfg)
        score = disc(self.default_inputx256)
        assert score.shape == (2, 1, 14, 14)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pix2pix_discriminator_cuda(self):
        # test with default cfg
        disc = PatchDiscriminator(**self.default_cfg).cuda()
        score = disc(self.default_inputx256.cuda())
        assert score.shape == (2, 1, 30, 30)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_conv'] = 4
        disc = PatchDiscriminator(**cfg).cuda()
        score = disc(self.default_inputx256.cuda())
        assert score.shape == (2, 1, 14, 14)
