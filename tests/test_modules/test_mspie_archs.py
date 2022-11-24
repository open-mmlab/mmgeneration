# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch

from mmgen.models.architectures.stylegan import MSStyleGANv2Generator


class TestMSStyleGAN2:

    @classmethod
    def setup_class(cls):
        cls.generator_cfg = dict(out_size=32, style_channels=16)
        cls.disc_cfg = dict(in_size=32, with_adaptive_pool=True)

    def test_msstylegan2_cpu(self):

        # test normal forward
        cfg_ = deepcopy(self.generator_cfg)
        g = MSStyleGANv2Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        # set mix_prob as 1.0 and 0 to force cover lines
        cfg_ = deepcopy(self.generator_cfg)
        cfg_['mix_prob'] = 1
        g = MSStyleGANv2Generator(**cfg_)
        res = g(torch.randn, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        cfg_ = deepcopy(self.generator_cfg)
        cfg_['mix_prob'] = 0
        g = MSStyleGANv2Generator(**cfg_)
        res = g(torch.randn, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        cfg_ = deepcopy(self.generator_cfg)
        cfg_['mix_prob'] = 1
        g = MSStyleGANv2Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)

        cfg_ = deepcopy(self.generator_cfg)
        cfg_['mix_prob'] = 0
        g = MSStyleGANv2Generator(**cfg_)
        res = g(None, num_batches=2)
        assert res.shape == (2, 3, 32, 32)
