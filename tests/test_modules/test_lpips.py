import pytest
import torch

from mmgen.models.architectures import PerceptualLoss


class TestLpips:

    @classmethod
    def setup_class(cls):
        cls.pretrained = False

    def test_lpips(self):
        percept = PerceptualLoss(use_gpu=False, pretrained=self.pretrained)
        img_a = torch.randn((2, 3, 256, 256))
        img_b = torch.randn((2, 3, 256, 256))
        perceptual_loss = percept(img_a, img_b)
        assert perceptual_loss.shape == (2, 1, 1, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_lpips_cuda(self):
        percept = PerceptualLoss(use_gpu=True, pretrained=self.pretrained)
        img_a = torch.randn((2, 3, 256, 256)).cuda()
        img_b = torch.randn((2, 3, 256, 256)).cuda()
        perceptual_loss = percept(img_a, img_b)
        assert perceptual_loss.shape == (2, 1, 1, 1)
