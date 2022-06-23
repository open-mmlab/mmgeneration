# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from abc import ABCMeta
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import is_list_of, print_log
from mmengine.dist import (all_gather, all_reduce, broadcast_object_list,
                           collect_results, get_world_size, is_main_process)
from mmengine.evaluator import BaseMetric
from PIL import Image
from scipy import linalg
from scipy.stats import entropy
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from mmgen.models.architectures.common import get_module_device
from mmgen.models.architectures.lpips import PerceptualLoss
from mmgen.registry import METRICS
from mmgen.typing import ForwardInputs, ForwardOutputs, ValTestStepInputs
from .inception_utils import (disable_gpu_fuser_on_pt19, load_inception,
                              prepare_inception_feat)
from .metric_utils import (finalize_descriptors, get_descriptors_for_minibatch,
                           get_gaussian_kernel, laplacian_pyramid, ms_ssim,
                           slerp, sliced_wasserstein)


class GenMetric(BaseMetric):
    """Metric for MMGeneration.

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        real_nums (int): Numbers of the real image need for the metric. If `-1`
            is passed means all images from the dataset is need. Defaults to 0.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_mode (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    SAMPLER_MODE = 'normal'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.sample_mode = sample_mode

        self.fake_nums = fake_nums
        self.real_nums = real_nums
        self.real_key = real_key
        self.fake_key = fake_key
        self.real_results: List[Any] = []
        self.fake_results: List[Any] = []

        self._color_order = 'bgr'

    @property
    def real_nums_per_device(self):
        """Number of real images need for current device."""
        return math.ceil(self.real_nums / get_world_size())

    @property
    def fake_nums_per_device(self):
        """Number of fake images need for current device."""
        return math.ceil(self.fake_nums / get_world_size())

    def set_color_order(self, color_order: str) -> None:
        """Set color order for the input image.

        Args:
            color_order (str): The color order of input image.
        """
        self._color_order = color_order

    def _collect_target_results(self, target: str) -> Optional[list]:
        """Collected results in distributed environments.

        Args:
            target (str): Target results to collect.

        Returns:
            Optional[list]: The collected results.
        """
        assert target in [
            'fake', 'real'
        ], ('Only support to collect \'fake\' or \'real\' results.')
        results = getattr(self, f'{target}_results')
        size = getattr(self, f'{target}_nums')
        size = len(results) * get_world_size() if size == -1 else size

        if len(results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.{target}_results`.'
                ' Please ensure that the processed results are properly added '
                f'into `self.{target}_results` in `process` method.')

        if is_list_of(results, Tensor):
            # apply all_gather for tensor results
            results = torch.cat(results, dim=0)
            results = torch.cat(all_gather(results), dim=0)[:size]
            results = torch.tensor_split(results, size)
        else:
            # apply collect_results (all_gather_object) for non-tensor results
            results = collect_results(results, size, self.collect_device)

        # on non-main process, results should be `None`
        if is_main_process() and len(results) != size:
            raise ValueError(f'Length of results is \'{len(results)}\', not '
                             f'equals to target size \'{size}\'.')
        return results

    def evaluate(self) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches. Different like :class:`~mmengine.evaluator.BaseMetric`,
        this function evaluate the metric with paired results (`results_fake`
        and `results_real`).

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
                names of the metrics, and the values are corresponding results.
        """

        results_fake = self._collect_target_results(target='fake')
        results_real = self._collect_target_results(target='real')

        if is_main_process():
            # pack to list, align with BaseMetrics
            _metrics = self.compute_metrics(results_fake, results_real)
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.real_results.clear()
        self.fake_results.clear()

        return metrics[0]

    @classmethod
    def get_metric_sampler(cls, model: nn.Module, dataloader: DataLoader,
                           metrics: List['GenMetric']) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader

    def compute_metrics(self, results_fake, results_real) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Prepare for the pre-calculating items of the metric. Defaults to do
        nothing.

        Args:
            module (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for the real images.
        """
        return


@METRICS.register_module('SWD')
@METRICS.register_module()
class SlicedWassersteinDistance(GenMetric):
    """SWD (Sliced Wasserstein distance) metric. We calculate the SWD of two
    sets of images in the following way. In every 'feed', we obtain the
    Laplacian pyramids of every images and extract patches from the Laplacian
    pyramids as descriptors. In 'summary', we normalize these descriptors along
    channel, and reshape them so that we can use these descriptors to represent
    the distribution of real/fake images. And we can calculate the sliced
    Wasserstein distance of the real and fake descriptors as the SWD of the
    real and fake images.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/sliced_wasserstein.py # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        image_shape (tuple): Image shape in order "CHW".
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_mode (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    name = 'SWD'

    def __init__(self,
                 fake_nums: int,
                 image_shape: tuple,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        assert collect_device == 'cpu', (
            'SWD only support \'cpu\' as collect_device, but receive '
            f'\'{collect_device}\'.')
        super().__init__(fake_nums, 0, fake_key, real_key, sample_mode,
                         collect_device, prefix)

        self.nhood_size = 7  # height and width of the extracted patches
        self.nhoods_per_image = 128  # number of extracted patches per image
        self.dir_repeats = 4  # times of sampling directions
        self.dirs_per_repeat = 128  # number of directions per sampling
        self.resolutions = []
        res = image_shape[1]
        while res >= 16 and len(self.resolutions) < 4:
            self.resolutions.append(res)
            res //= 2
        self.n_pyramids = len(self.resolutions)

        self.gaussian_k = get_gaussian_kernel()
        self.real_results = [[] for res in self.resolutions]
        self.fake_results = [[] for res in self.resolutions]

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """_summary_

        Args:
            data_batch (Sequence[dict]): _description_
            predictions (Sequence[dict]): _description_
        """
        # lod: layer_of_descriptors

        # parse real images
        real_img = data_batch[self.real_key]

        # parse fake images
        if isinstance(predictions, dict):
            fake_img = predictions[self.sample_mode]
            # get target image from the dict
            if isinstance(fake_img, dict):
                fake_img = fake_img[self.fake_key]
        else:
            fake_img = predictions

        assert real_img.shape[1:] == self.image_shape
        real_pyramid = laplacian_pyramid(real_img, self.n_pyramids - 1,
                                         self.gaussian_k)
        # real images
        for lod, level in enumerate(real_pyramid):
            desc = get_descriptors_for_minibatch(level, self.nhood_size,
                                                 self.nhoods_per_image)
            self.real_results[lod].append(desc)

        # fake images
        assert fake_img.shape[1:] == self.image_shape
        fake_pyramid = laplacian_pyramid(fake_img, self.n_pyramids - 1,
                                         self.gaussian_k)
        for lod, level in enumerate(fake_pyramid):
            desc = get_descriptors_for_minibatch(level, self.nhood_size,
                                                 self.nhoods_per_image)
            self.fake_results[lod].append(desc)

    def compute_metrics(self, results_fake, results_real) -> dict:
        fake_descs = [finalize_descriptors(d) for d in results_fake]
        real_descs = [finalize_descriptors(d) for d in results_real]
        distance = [
            sliced_wasserstein(dreal, dfake, self.dir_repeats,
                               self.dirs_per_repeat)
            for dreal, dfake in zip(real_descs, fake_descs)
        ]
        del real_descs
        del fake_descs

        distance = [d * 1e3 for d in distance]  # multiply by 10^3
        result = distance + [np.mean(distance)]

        return {'score': ', '.join([str(round(d, 2)) for d in result])}


class GenerativeMetric(GenMetric, metaclass=ABCMeta):
    """Metric for generative metrics. Except for the preparation phase
    (:meth:`prepare`), generative metrics do not need extra real images.

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        real_nums (int): Numbers of the real image need for the metric. If `-1`
            is passed means all images from the dataset is need. Defaults to 0.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_mode (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    SAMPLER_MODE = 'Generative'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, real_nums, fake_key, real_key, sample_mode,
                         collect_device, prefix)

    @classmethod
    def get_metric_sampler(cls, model: nn.Module, dataloader: DataLoader,
                           metrics: GenMetric):
        """Get sampler for generative metrics. Returns a dummy iterator, whose
        return value of each iteration is a dict containing batch size and
        sample mode to generate images.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images. Used to get
                batch size during generate fake images.
            metrics (List['GenMetric']): Metrics with the same sampler mode.

        Returns:
            :class:`dummy_iterator`: Sampler for generative metrics.
        """

        batch_size = dataloader.batch_size

        sample_mode = metrics[0].sample_mode
        assert all([metric.sample_mode == sample_mode for metric in metrics
                    ]), ('\'sample_mode\' between metrics is inconsistency.')

        class dummy_iterator:

            def __init__(self, batch_size, max_length, sample_mode) -> None:
                self.batch_size = batch_size
                self.max_length = max_length
                self.sample_mode = sample_mode

            def __iter__(self) -> Iterator:
                self.idx = 0
                return self

            def __next__(self) -> ForwardInputs:
                if self.idx > self.max_length:
                    raise StopIteration
                self.idx += batch_size
                return dict(mode=self.sample_mode, num_batches=self.batch_size)

        return dummy_iterator(
            batch_size=batch_size,
            max_length=max([metric.fake_nums_per_device
                            for metric in metrics]),
            sample_mode=sample_mode)

    def evaluate(self) -> dict():
        """Evaluate generative metric. In this function we only collect
        :attr:`fake_results` because generative metrics do not need real
        images.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
                names of the metrics, and the values are corresponding results.
        """
        results_fake = self._collect_target_results(target='fake')

        if is_main_process():
            # pack to list, align with BaseMetrics
            _metrics = self.compute_metrics(results_fake)
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.fake_results.clear()

        return metrics[0]

    def compute_metrics(self, results) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """


@METRICS.register_module('FID-Full')
@METRICS.register_module('FID')
@METRICS.register_module()
class FrechetInceptionDistance(GenerativeMetric):
    """FID metric. In this metric, we calculate the distance between real
    distributions and fake distributions. The distributions are modeled by the
    real samples and fake samples, respectively. `Inception_v3` is adopted as
    the feature extractor, which is widely used in StyleGAN and BigGAN.

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        real_nums (int): Numbers of the real images need for the metric. If -1
            is passed, means all real images in the dataset will be used.
            Defaults to -1.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        inception_pkl (str, optional): Path to reference inception pickle file.
            If `None`, the statistical value of real distribution will be
            calculated at running time. Defaults to None.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_mode (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'FID'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = -1,
                 inception_style='StyleGAN',
                 inception_path: Optional[str] = None,
                 inception_pkl: Optional[str] = None,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, real_nums, fake_key, real_key, sample_mode,
                         collect_device, prefix)
        self.real_mean = None
        self.real_cov = None
        self.device = 'cpu'
        self.inception, self.inception_style = self._load_inception(
            inception_style, inception_path)
        self.inception_pkl = inception_pkl
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.inception.cuda()
            self.collect_device = 'gpu'

    def prepare(self, module: nn.Module, dataloader: DataLoader) -> None:
        """Preparing inception feature for the real images.

        Args:
            module (nn.Module): The model to evaluate.
            dataloader (DataLoader): The dataloader for real images.
        """
        inception_feat = prepare_inception_feat(dataloader, self,
                                                module.data_preprocessor)
        if self.real_nums != -1:
            assert self.real_nums <= inception_feat.shape[0], (
                f'Need \'{self.real_nums}\' of real nums, but only '
                f'\'{inception_feat.shape[0]}\' images be found in the '
                'inception feature.')
            inception_feat = inception_feat[np.random.choice(
                inception_feat.shape[0], size=self.real_nums, replace=True)]
        if is_main_process():
            self.real_mean = np.mean(inception_feat, 0)
            self.real_cov = np.cov(inception_feat, rowvar=False)

    def _load_inception(
            self, inception_style: str,
            inception_path: Optional[str]) -> Tuple[nn.Module, str]:
        """Load inception and return the successful loaded style.

        Args:
            inception_style (str): Target style of Inception network want to
                load.
            inception_path (Optional[str]): The path to the inception.

        Returns:
            Tuple[nn.Module, str]: The actually loaded inception network and
                corresponding style.
        """
        if inception_style == 'StyleGAN':
            args = dict(type='StyleGAN', path=inception_path)
        else:
            args = dict(type='Pytorch', normalize_input=False)
        inception, style = load_inception(args, 'FID')
        inception.eval()
        return inception, style

    def forward_inception(self, image: Tensor) -> Tensor:
        """Feed image to inception network and get the output feature.

        Args:
            image (Tensor): Image tensor fed to the Inception network.

        Returns:
            Tensor: Image feature extracted from inception.
        """

        if self._color_order == 'bgr':
            image = image[:, [2, 1, 0]]

        image = image.to(self.device)
        if self.inception_style == 'StyleGAN':
            image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(image, return_features=True)
        else:
            feat = self.inception(image)[0].view(image.shape[0], -1)
        return feat

    def process(self, data_batch: ValTestStepInputs,
                predictions: ForwardOutputs) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        # real images should be preprocessed. Ignore data_batch
        if isinstance(predictions, dict):
            fake_img = predictions[self.sample_mode]
            if isinstance(fake_img, dict):
                fake_img = fake_img[self.fake_key]
        else:
            fake_img = predictions
        feat = self.forward_inception(fake_img)
        feat_list = list(torch.tensor_split(feat, feat.shape[0]))
        self.fake_results += feat_list

    @staticmethod
    def _calc_fid(sample_mean: np.ndarray,
                  sample_cov: np.ndarray,
                  real_mean: np.ndarray,
                  real_cov: np.ndarray,
                  eps: float = 1e-6) -> Tuple[float]:
        """Refer to the implementation from:

        https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
        """
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm(
                (sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(
            real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return float(fid), float(mean_norm), float(trace)

    def compute_metrics(self, fake_results: list) -> dict:
        """Compulate the result of FID metric.

        Args:
            fake_results (list): List of image feature of fake images.

        Returns:
            dict: A dict of the computed FID metric and its mean and
                covariance.
        """
        fake_feats = torch.cat(fake_results, dim=0)
        fake_feats_np = fake_feats.cpu().numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)

        return {'fid': fid, 'mean': mean, 'cov': cov}


@METRICS.register_module('IS')
@METRICS.register_module()
class InceptionScore(GenerativeMetric):
    """IS (Inception Score) metric. The images are split into groups, and the
    inception score is calculated on each group of images, then the mean and
    standard deviation of the score is reported. The calculation of the
    inception score on a group of images involves first using the inception v3
    model to calculate the conditional probability for each image (p(y|x)). The
    marginal probability is then calculated as the average of the conditional
    probabilities for the images in the group (p(y)). The KL divergence is then
    calculated for each image as the conditional probability multiplied by the
    log of the conditional probability minus the log of the marginal
    probability. The KL divergence is then summed over all images and averaged
    over all classes and the exponent of the result is calculated to give the
    final score.

    Ref: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py  # noqa

    Note that we highly recommend that users should download the Inception V3
    script module from the following address. Then, the `inception_pkl` can
    be set with user's local path. If not given, we will use the Inception V3
    from pytorch model zoo. However, this may bring significant different in
    the final results.

    Tero's Inception V3: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt  # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        resize (bool, optional): Whether resize image to 299x299. Defaults to
            True.
        splits (int, optional): The number of groups. Defaults to 10.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        resize_method (str): Resize method. If `resize` is False, this will be
            ignored. Defaults to 'bicubic'.
        use_pil_resize (bool): Whether use Bicubic interpolation with
            Pillow's backend. If set as True, the evaluation process may be a
            little bit slow, but achieve a more accurate IS result. Defaults
            to False.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_mode (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'IS'

    def __init__(self,
                 fake_nums: int = 5e4,
                 resize: bool = True,
                 splits: int = 10,
                 inception_style: str = 'StyleGAN',
                 inception_path: Optional[str] = None,
                 resize_method='bicubic',
                 use_pillow_resize: bool = True,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode='ema',
                 collect_device: str = 'cpu',
                 prefix: str = None):
        super().__init__(fake_nums, 0, fake_key, real_key, sample_mode,
                         collect_device, prefix)

        self.resize = resize
        self.resize_method = resize_method
        self.splits = splits
        self.device = 'cpu'

        if not use_pillow_resize:
            print_log(
                'We strongly recommend to use the bicubic resize with '
                'Pillow backend. Otherwise, the results maybe '
                'unreliable', 'current')
        self.use_pillow_resize = use_pillow_resize

        self.inception, self.inception_style = self._load_inception(
            inception_style, inception_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.inception = self.inception.cuda()
            self.collect_device = 'gpu'

    def _load_inception(
            self, inception_style: str,
            inception_path: Optional[str]) -> Tuple[nn.Module, str]:
        """Load pretrain model of inception network.
        Args:
            inception_style (str): Target style of Inception network want to
                load.
            inception_path (Optional[str]): The path to the inception.

        Returns:
            Tuple[nn.Module, str]: The actually loaded inception network and
                corresponding style.
        """
        inception, style = load_inception(
            dict(type=inception_style, path='inception_path'), 'IS')
        inception.eval()
        return inception, style

    def _preprocess(self, image: Tensor) -> Tensor:
        """Preprocess image before pass to the Inception. Preprocess operations
        contain channel conversion and resize.

        Args:
            image (Tensor): Image tensor before preprocess.

        Returns:
            Tensor: Image tensor after resize and channel conversion
                (if need.)
        """
        if self._color_order == 'bgr':
            image = image[:, [2, 1, 0]]
        if not self.resize:
            return image
        if self.use_pillow_resize:
            image = (image.clone() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            x_np = [x_.permute(1, 2, 0).detach().cpu().numpy() for x_ in image]

            # use bicubic resize as default
            x_pil = [Image.fromarray(x_).resize((299, 299)) for x_ in x_np]
            x_ten = torch.cat(
                [torch.FloatTensor(np.array(x_)[None, ...]) for x_ in x_pil])
            x_ten = (x_ten / 127.5 - 1).to(torch.float)
            return x_ten.permute(0, 3, 1, 2)
        else:
            return F.interpolate(
                image, size=(299, 299), mode=self.resize_method)

    def process(self, data_batch: Optional[Sequence[dict]],
                predictions: Union[Sequence[dict], Tensor]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
            return

        if isinstance(predictions, dict):
            fake_img = predictions[self.sample_mode]
            # get target image from the dict
            if isinstance(fake_img, dict):
                fake_img = fake_img[self.fake_key]
        else:
            fake_img = predictions

        fake_img = self._preprocess(fake_img).to(self.device)
        if self.inception_style == 'StyleGAN':
            fake_img = (fake_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(fake_img, no_output_bias=True)
        else:
            feat = F.softmax(self.inception(fake_img), dim=1)

        # NOTE: feat is shape like (bz, 1000), convert to a list
        self.fake_results += list(torch.tensor_split(feat, feat.shape[0]))

    def compute_metrics(self, fake_results: list) -> dict:
        """Compute the results of Inception Score metric.

        Args:
            fake_results (list): List of image feature of fake images.

        Returns:
            dict: A dict of the computed IS metric and its standard error
        """
        split_scores = []
        preds = torch.cat(fake_results, dim=0).cpu().numpy()
        # check for the size
        assert preds.shape[0] >= self.fake_nums
        preds = preds[:self.fake_nums]
        for k in range(self.splits):
            part = preds[k * (self.fake_nums // self.splits):(k + 1) *
                         (self.fake_nums // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        mean, std = np.mean(split_scores), np.std(split_scores)

        return {'is': float(mean), 'is_std': float(std)}


@METRICS.register_module('MS_SSIM')
@METRICS.register_module()
class MultiScaleStructureSimilarity(GenerativeMetric):
    """MS-SSIM (Multi-Scale Structure Similarity) metric.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_mode (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    name = 'MS-SSIM'

    def __init__(self,
                 fake_nums: int,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(fake_nums, 0, fake_key, real_key, sample_mode,
                         collect_device, prefix)

        assert fake_nums % 2 == 0
        self.num_pairs = fake_nums // 2

        self.sum = 0.0
        # self.device = 'cpu'

    def process(self, data_batch: Optional[Sequence[dict]],
                predictions: Union[Sequence[dict], Tensor]) -> None:
        """Feed data to the metric.

        Args:
            data_batch (Tensor): Real images from dataloader. Do not be used
                in this metric.
            predictions (Union[Sequence[dict], Tensor]): Generated images.
        """
        if isinstance(predictions, dict):
            minibatch = predictions[self.sample_mode]
            if isinstance(minibatch, dict):
                minibatch = minibatch[self.fake_key]
        else:
            minibatch = predictions

        minibatch = ((minibatch + 1) / 2)
        minibatch = minibatch.clamp_(0, 1)
        half1 = minibatch[0::2].cpu().data.numpy().transpose((0, 2, 3, 1))
        half1 = (half1 * 255).astype('uint8')
        half2 = minibatch[1::2].cpu().data.numpy().transpose((0, 2, 3, 1))
        half2 = (half2 * 255).astype('uint8')
        score = ms_ssim(half1, half2)
        self.sum += score * (minibatch.shape[0] // 2)

    def evaluate(self):
        """Collect and evaluate MS-SSIM.

        Different like other metrics, MS-SSIM
        only save score for each minibatch in the memory and do not save
        images to `self.fake_results`. Therefore we only collect
        :attr:`self.sum` across the device (if need).
        """
        self.sum = torch.Tensor(self.sum)
        results = all_reduce(self.sum, op='mean')
        # all_reduce sync `results` across device, all process can run the
        # following code
        metrics = self.compute_metrics(results)

        # reset the `sum`
        self.sum = 0.0
        return metrics

    def compute_metrics(self, results):
        """Computed the result of MS-SSIM.

        Returns:
            dict: Calculated MS-SSIM result.
        """
        avg = results / self.num_pairs
        return {'avg': str(round(avg.item(), 4))}


@METRICS.register_module('PPL')
@METRICS.register_module()
class PerceptualPathLength(GenerativeMetric):
    r"""Perceptual path length.

        Measure the difference between consecutive images (their VGG16
        embeddings) when interpolating between two random inputs. Drastic
        changes mean that multiple features have changed together and that
        they might be entangled.

        Ref: https://github.com/rosinality/stylegan2-pytorch/blob/master/ppl.py # noqa

        Args:
            num_images (int): The number of evaluated generated samples.
            image_shape (tuple, optional): Image shape in order "CHW". Defaults
                to None.
            crop (bool, optional): Whether crop images. Defaults to True.
            epsilon (float, optional): Epsilon parameter for path sampling.
                Defaults to 1e-4.
            space (str, optional): Latent space. Defaults to 'W'.
            sampling (str, optional): Sampling mode, whether sampling in full
                path or endpoints. Defaults to 'end'.
            latent_dim (int, optional): Latent dimension of input noise.
                Defaults to 512.
    """
    SAMPLER_MODE = 'path'

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = 0,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 crop=True,
                 epsilon=1e-4,
                 space='W',
                 sampling='end',
                 latent_dim=512):
        super().__init__(fake_nums, real_nums, fake_key, real_key, sample_mode,
                         collect_device, prefix)
        self.crop = crop

        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.latent_dim = latent_dim

    @torch.no_grad()
    def process(self, data_batch: ValTestStepInputs,
                predictions: ForwardOutputs) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        # real images should be preprocessed. Ignore data_batch
        if isinstance(predictions, dict):
            fake_img = predictions[self.sample_mode]
            if isinstance(fake_img, dict):
                fake_img = fake_img[self.fake_key]
        else:
            fake_img = predictions
        feat = self._compute_distance(fake_img)
        feat_list = list(torch.tensor_split(feat, feat.shape[0]))
        self.fake_results += feat_list

    @torch.no_grad()
    def _compute_distance(self, images):
        """Feed data to the metric.

        Args:
            images (Tensor): Input tensor.
        """
        # use minibatch's device type to initialize a lpips calculator
        if not hasattr(self, 'percept'):
            self.percept = PerceptualLoss(
                use_gpu=images.device.type.startswith('cuda'))
        # crop and resize images
        if self.crop:
            c = images.shape[2] // 8
            minibatch = images[:, :, c * 3:c * 7, c * 2:c * 6]

        factor = minibatch.shape[2] // 256
        if factor > 1:
            minibatch = F.interpolate(
                minibatch,
                size=(256, 256),
                mode='bilinear',
                align_corners=False)
        # calculator and store lpips score
        distance = self.percept(minibatch[::2], minibatch[1::2]).view(
            minibatch.shape[0] // 2) / (
                self.epsilon**2)
        return distance.to('cpu')

    @torch.no_grad()
    def compute_metrics(self, fake_results: list) -> dict:
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        distances = torch.cat(self.fake_results, dim=0).numpy()
        lo = np.percentile(distances, 1, interpolation='lower')
        hi = np.percentile(distances, 99, interpolation='higher')
        filtered_dist = np.extract(
            np.logical_and(lo <= distances, distances <= hi), distances)
        ppl_score = float(filtered_dist.mean())
        return {'ppl_score': ppl_score}

    def get_metric_sampler(self, model: nn.Module, dataloader: DataLoader,
                           metrics: GenMetric):
        """Get sampler for generative metrics. Returns a dummy iterator, whose
        return value of each iteration is a dict containing batch size and
        sample mode to generate images.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images. Used to get
                batch size during generate fake images.
            metrics (List['GenMetric']): Metrics with the same sampler mode.

        Returns:
            :class:`dummy_iterator`: Sampler for generative metrics.
        """

        batch_size = dataloader.batch_size

        sample_mode = metrics[0].sample_mode
        assert all([metric.sample_mode == sample_mode for metric in metrics
                    ]), ('\'sample_model\' between metrics is inconsistency.')

        class PPLSampler:
            """StyleGAN series generator's sampling iterator for PPL metric.

            Args:
                generator (nn.Module): StyleGAN series' generator.
                num_images (int): The number of evaluated generated samples.
                batch_size (int): Batch size of generated images.
                space (str, optional): Latent space. Defaults to 'W'.
                sampling (str, optional): Sampling mode, whether sampling in
                    full path or endpoints. Defaults to 'end'.
                epsilon (float, optional): Epsilon parameter for path sampling.
                    Defaults to 1e-4.
                latent_dim (int, optional): Latent dimension of input noise.
                    Defaults to 512.
            """

            def __init__(self,
                         generator,
                         num_images,
                         batch_size,
                         space='W',
                         sampling='end',
                         epsilon=1e-4,
                         latent_dim=512):
                assert space in ['Z', 'W']
                assert sampling in ['full', 'end']
                n_batch = num_images // batch_size

                resid = num_images - (n_batch * batch_size)
                self.batch_sizes = [batch_size] * n_batch + ([resid] if
                                                             resid > 0 else [])
                self.device = get_module_device(generator)
                self.generator = generator
                self.latent_dim = latent_dim
                self.space = space
                self.sampling = sampling
                self.epsilon = epsilon

            def __iter__(self):
                self.idx = 0
                return self

            @torch.no_grad()
            def __next__(self):
                if self.idx >= len(self.batch_sizes):
                    raise StopIteration
                batch = self.batch_sizes[self.idx]
                inputs = torch.randn([batch * 2, self.latent_dim],
                                     device=self.device)
                if self.sampling == 'full':
                    lerp_t = torch.rand(batch, device=self.device)
                else:
                    lerp_t = torch.zeros(batch, device=self.device)

                if self.space == 'W':
                    assert hasattr(self.generator, 'style_mapping')
                    latent = self.generator.style_mapping(inputs)
                    latent_t0, latent_t1 = latent[::2], latent[1::2]
                    latent_e0 = torch.lerp(latent_t0, latent_t1, lerp_t[:,
                                                                        None])
                    latent_e1 = torch.lerp(latent_t0, latent_t1,
                                           lerp_t[:, None] + self.epsilon)
                    latent_e = torch.stack([latent_e0, latent_e1],
                                           1).view(*latent.shape)
                else:
                    latent_t0, latent_t1 = inputs[::2], inputs[1::2]
                    latent_e0 = slerp(latent_t0, latent_t1, lerp_t[:, None])
                    latent_e1 = slerp(latent_t0, latent_t1,
                                      lerp_t[:, None] + self.epsilon)
                    latent_e = torch.stack([latent_e0, latent_e1],
                                           1).view(*inputs.shape)

                self.idx += 1
                return dict(noise=latent_e)

        ppl_sampler = PPLSampler(
            model.generator_ema
            if self.sample_mode == 'ema' else model.generator,
            num_images=max([metric.fake_nums_per_device
                            for metric in metrics]),
            batch_size=batch_size,
            space=self.space,
            sampling=self.sampling,
            epsilon=self.epsilon,
            latent_dim=self.latent_dim)
        return ppl_sampler


@METRICS.register_module()
class TransFID(FrechetInceptionDistance):

    def __init__(self,
                 fake_nums: int,
                 real_nums: int = -1,
                 inception_style='StyleGAN',
                 inception_path: Optional[str] = None,
                 inception_pkl: Optional[str] = None,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_model: str = 'ema',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(fake_nums, real_nums, inception_style, inception_path,
                         inception_pkl, fake_key, real_key, sample_model,
                         collect_device, prefix)

        self.SAMPLER_MODE = 'normal'

    @classmethod
    def get_metric_sampler(cls, model: nn.Module, dataloader: DataLoader,
                           metrics: List['GenMetric']) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader

    def process(self, data_batch: ValTestStepInputs,
                predictions: ForwardOutputs) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        # real images should be preprocessed. Ignore data_batch
        if isinstance(predictions, dict):
            fake_img = predictions[self.fake_key]
        else:
            fake_img = predictions
        feat = self.forward_inception(fake_img)
        feat_list = list(torch.tensor_split(feat, feat.shape[0]))
        self.fake_results += feat_list


@METRICS.register_module()
class TransIS(InceptionScore):
    """IS (Inception Score) metric. The images are split into groups, and the
    inception score is calculated on each group of images, then the mean and
    standard deviation of the score is reported. The calculation of the
    inception score on a group of images involves first using the inception v3
    model to calculate the conditional probability for each image (p(y|x)). The
    marginal probability is then calculated as the average of the conditional
    probabilities for the images in the group (p(y)). The KL divergence is then
    calculated for each image as the conditional probability multiplied by the
    log of the conditional probability minus the log of the marginal
    probability. The KL divergence is then summed over all images and averaged
    over all classes and the exponent of the result is calculated to give the
    final score.

    Ref: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py  # noqa

    Note that we highly recommend that users should download the Inception V3
    script module from the following address. Then, the `inception_pkl` can
    be set with user's local path. If not given, we will use the Inception V3
    from pytorch model zoo. However, this may bring significant different in
    the final results.

    Tero's Inception V3: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt  # noqa

    Args:
        fake_nums (int): Numbers of the generated image need for the metric.
        resize (bool, optional): Whether resize image to 299x299. Defaults to
            True.
        splits (int, optional): The number of groups. Defaults to 10.
        inception_style (str): The target inception style want to load. If the
            given style cannot be loaded successful, will attempt to load a
            valid one. Defaults to 'StyleGAN'.
        inception_path (str, optional): Path the the pretrain Inception
            network. Defaults to None.
        resize_method (str): Resize method. If `resize` is False, this will be
            ignored. Defaults to 'bicubic'.
        use_pil_resize (bool): Whether use Bicubic interpolation with
            Pillow's backend. If set as True, the evaluation process may be a
            little bit slow, but achieve a more accurate IS result. Defaults
            to False.
        fake_key (Optional[str]): Key for get fake images of the output dict.
            Defaults to None.
        real_key (Optional[str]): Key for get real images from the input dict.
            Defaults to 'img'.
        sample_mode (str): Sampling mode for the generative model. Support
            'orig' and 'ema'. Defaults to 'ema'.
        collect_device (str, optional): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 fake_nums: int = 50000,
                 resize: bool = True,
                 splits: int = 10,
                 inception_style: str = 'StyleGAN',
                 inception_path: Optional[str] = None,
                 resize_method='bicubic',
                 use_pillow_resize: bool = True,
                 fake_key: Optional[str] = None,
                 real_key: Optional[str] = 'img',
                 sample_mode='ema',
                 collect_device: str = 'cpu',
                 prefix: str = None):
        super().__init__(fake_nums, resize, splits, inception_style,
                         inception_path, resize_method, use_pillow_resize,
                         fake_key, real_key, sample_mode, collect_device,
                         prefix)
        self.SAMPLER_MODE = 'normal'

    def process(self, data_batch: Optional[Sequence[dict]],
                predictions: Union[Sequence[dict], Tensor]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.fake_results``, which will be used
        to compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from the model.
        """
        if len(self.fake_results) >= self.fake_nums_per_device:
            return

        if isinstance(predictions, dict):
            fake_img = predictions[self.fake_key]
        else:
            fake_img = predictions

        fake_img = self._preprocess(fake_img).to(self.device)
        if self.inception_style == 'StyleGAN':
            fake_img = (fake_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            with disable_gpu_fuser_on_pt19():
                feat = self.inception(fake_img, no_output_bias=True)
        else:
            feat = F.softmax(self.inception(fake_img), dim=1)

        # NOTE: feat is shape like (bz, 1000), convert to a list
        self.fake_results += list(torch.tensor_split(feat, feat.shape[0]))

    @classmethod
    def get_metric_sampler(cls, model: nn.Module, dataloader: DataLoader,
                           metrics: List['GenMetric']) -> DataLoader:
        """Get sampler for normal metrics. Directly returns the dataloader.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (DataLoader): Dataloader for real images.
            metrics (List['GenMetric']): Metrics with the same sample mode.

        Returns:
            DataLoader: Default sampler for normal metrics.
        """
        return dataloader
