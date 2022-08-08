from typing import Optional, Sequence
from unittest import TestCase
from unittest.mock import MagicMock

from mmengine.registry import METRICS

from mmgen.core import GenEvaluator
from mmgen.core.evaluation.metrics import GenerativeMetric, GenMetric


@METRICS.register_module()
class ToyGenMetric_normal(GenMetric):
    name = 'toy_normal'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(fake_nums, real_nums, fake_key, real_key, sample_mode,
                         collect_device, prefix)

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        pass

    def compute_metrics(self, results_fake, results_real) -> dict:
        return dict(normal=1)


@METRICS.register_module()
class ToyGenMetric_gen(GenerativeMetric):
    name = 'toy_full'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(fake_nums, real_nums, fake_key, real_key, sample_mode,
                         collect_device, prefix)

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        pass

    def compute_metrics(self, results_fake, results_real) -> dict:
        return dict(full=1)


@METRICS.register_module()
class MockMetric(GenMetric):
    name = 'mock'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(fake_nums, real_nums, fake_key, real_key, sample_mode,
                         collect_device, prefix)

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        pass

    def compute_metrics(self, results_fake, results_real) -> dict:
        return dict(mock=3)


class TestEvaluator(TestCase):

    def test_prepare(self):
        evaluator = GenEvaluator(metrics=[
            dict(type='ToyGenMetric_normal', fake_nums=10),
            dict(type='ToyGenMetric_gen', fake_nums=10),
        ])
        self.assertEqual(len(evaluator.metrics), 2)
        self.assertFalse(evaluator.is_ready)
        module = MagicMock()
        module.data_preprocessor = MagicMock()
        module.data_preprocessor.bgr_to_rgb = False
        module.data_preprocessor.rgb_to_bgr = False

        dataloader = MagicMock()
        dataloader.batch_size = 10
        dataloader.dataset = MagicMock()
        evaluator.prepare_metrics(module, dataloader)
        self.assertTrue(evaluator.is_ready)

        # test if `is_ready` work
        evaluator.prepare_metrics(module, dataloader)

        metrics_sampler_list = evaluator.prepare_samplers(module, dataloader)
        self.assertEqual(len(metrics_sampler_list), 2)
        for metrics, sampler in metrics_sampler_list:
            metric = metrics[0]
            if metric.name == 'toy_normal':
                self.assertEqual(sampler.dataset, dataloader.dataset)
