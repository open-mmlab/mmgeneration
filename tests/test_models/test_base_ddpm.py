# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmgen.models.diffusions import UniformTimeStepSampler


def test_uniform_sampler():
    sampler = UniformTimeStepSampler(10)
    timesteps = sampler(2)
    assert timesteps.shape == torch.Size([
        2,
    ])
    assert timesteps.max() < 10 and timesteps.min() >= 0

    timesteps = sampler.__call__(2)
    assert timesteps.shape == torch.Size([
        2,
    ])
    assert timesteps.max() < 10 and timesteps.min() >= 0
