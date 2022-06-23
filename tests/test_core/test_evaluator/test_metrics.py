import pytest
import torch
from mmengine.runner import Runner

from mmgen.core import PrecisionAndRecall
from mmgen.datasets import PackGenInputs, UnconditionalImageDataset
from mmgen.models import LSGAN, DCGANGenerator, GANDataPreprocessor


class TestPR:

    @classmethod
    def setup_class(cls):
        pipeline = [
            dict(type='mmgen.LoadImageFromFile', key='img', io_backend='disk'),
            dict(type='mmgen.Resize', scale=(128, 128)),
            PackGenInputs(meta_keys=[])
        ]
        dataset = UnconditionalImageDataset(
            data_root='tests/data/image', pipeline=pipeline, test_mode=True)
        cls.dataloader = Runner.build_dataloader(
            dict(
                batch_size=2,
                dataset=dataset,
                sampler=dict(type='DefaultSampler')))
        gan_data_preprocessor = GANDataPreprocessor()
        generator = DCGANGenerator(128, noise_size=10, base_channels=20)
        cls.module = LSGAN(generator, data_preprocessor=gan_data_preprocessor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pr_cuda(self):
        pr = PrecisionAndRecall(10, sample_mode='orig', auto_save=False)
        self.module.cuda()
        sampler = pr.get_metric_sampler(self.module, self.dataloader, [pr])
        pr.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            pr.process(data_batch, predictions)
        pr_score = pr.compute_metrics(pr.fake_results)
        print(pr_score)
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0

    def test_pr_cpu(self):
        pr = PrecisionAndRecall(10, sample_mode='orig', auto_save=False)
        sampler = pr.get_metric_sampler(self.module, self.dataloader, [pr])
        pr.prepare(self.module, self.dataloader)
        for data_batch in sampler:
            predictions = self.module.test_step(data_batch)
            pr.process(data_batch, predictions)
        pr_score = pr.evaluate()
        print(pr_score)
        assert pr_score['precision'] >= 0 and pr_score['recall'] >= 0
