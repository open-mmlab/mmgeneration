import pytest
import torch

from mmgen.models.architectures.stylegan import StyleGANv2Generator
from mmgen.models.losses import GeneratorPathRegularizer


class TestPathRegularizer:

    @classmethod
    def setup_class(cls):
        cls.data_info = dict(generator='generator', num_batches='num_batches')
        cls.gen = StyleGANv2Generator(32, 10, num_mlps=2)

    def test_path_regularizer_cpu(self):
        gen = self.gen

        output_dict = dict(generator=gen, num_batches=2)
        pl = GeneratorPathRegularizer(data_info=self.data_info)
        pl_loss = pl(output_dict)
        assert pl_loss > 0

        output_dict = dict(generator=gen, num_batches=2, iteration=3)
        pl = GeneratorPathRegularizer(data_info=self.data_info, interval=2)
        pl_loss = pl(outputs_dict=output_dict)
        assert pl_loss is None

        with pytest.raises(NotImplementedError):
            _ = pl(asdf=1.)

        with pytest.raises(AssertionError):
            _ = pl(1.)

        with pytest.raises(AssertionError):
            _ = pl(1., 2, outputs_dict=output_dict)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_path_regularizer_cuda(self):
        gen = self.gen.cuda()

        output_dict = dict(generator=gen, num_batches=2)
        pl = GeneratorPathRegularizer(data_info=self.data_info).cuda()
        pl_loss = pl(output_dict)
        assert pl_loss > 0

        output_dict = dict(generator=gen, num_batches=2, iteration=3)
        pl = GeneratorPathRegularizer(
            data_info=self.data_info, interval=2).cuda()
        pl_loss = pl(outputs_dict=output_dict)
        assert pl_loss is None

        with pytest.raises(NotImplementedError):
            _ = pl(asdf=1.)

        with pytest.raises(AssertionError):
            _ = pl(1.)

        with pytest.raises(AssertionError):
            _ = pl(1., 2, outputs_dict=output_dict)
