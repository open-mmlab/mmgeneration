# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmgen.ops.stylegan3.ops import bias_act, upfirdn2d


class TestStyleGAN3Ops:

    @classmethod
    def setup_class(cls):
        cls.input = torch.randn((1, 3, 16, 16))
        cls.bias = torch.randn(3)
        cls.kernel = nn.Parameter(torch.randn(3, 3), requires_grad=False)

    def test_s3_ops_cpu(self):
        out = upfirdn2d.upfirdn2d(self.input, self.kernel)
        assert out.shape == (1, 3, 14, 14)

        out = upfirdn2d.upfirdn2d(
            self.input, self.kernel, up=2, down=1, padding=1)
        assert out.shape == (1, 3, 32, 32)

        out = upfirdn2d.upfirdn2d(
            self.input, self.kernel, up=1, down=2, padding=1)
        assert out.shape == (1, 3, 8, 8)

        out = bias_act.bias_act(self.input)
        assert out.shape == (1, 3, 16, 16)

        # test bias
        out = bias_act.bias_act(self.input, self.bias)
        assert out.shape == (1, 3, 16, 16)

        # test gain
        out = bias_act.bias_act(self.input, gain=0.5)
        assert out.shape == (1, 3, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_s3_ops_cuda(self):
        out = upfirdn2d.upfirdn2d(self.input.cuda(), self.kernel.cuda())
        assert out.shape == (1, 3, 14, 14)

        out = upfirdn2d.upfirdn2d(
            self.input.cuda(), self.kernel.cuda(), up=2, down=1, padding=1)
        assert out.shape == (1, 3, 32, 32)

        out = upfirdn2d.upfirdn2d(
            self.input.cuda(), self.kernel.cuda(), up=1, down=2, padding=1)
        assert out.shape == (1, 3, 8, 8)

        out = bias_act.bias_act(self.input.cuda())
        assert out.shape == (1, 3, 16, 16)

        # test bias
        out = bias_act.bias_act(self.input.cuda(), self.bias.cuda())
        assert out.shape == (1, 3, 16, 16)

        # test gain
        out = bias_act.bias_act(self.input.cuda(), gain=0.5)
        assert out.shape == (1, 3, 16, 16)
