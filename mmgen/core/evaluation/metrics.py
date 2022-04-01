# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info
from scipy import linalg, signal
from scipy.stats import entropy
from torchvision import models
from torchvision.models.inception import inception_v3

from mmgen.models.architectures import InceptionV3
from mmgen.models.architectures.common import get_module_device
from mmgen.models.architectures.lpips import PerceptualLoss
from mmgen.models.losses import gaussian_kld
from mmgen.utils import MMGEN_CACHE_DIR
from mmgen.utils.io_utils import download_from_url
from ..registry import METRICS
from .metric_utils import (_f_special_gauss, _hox_downsample,
                           compute_pr_distances, finalize_descriptors,
                           get_descriptors_for_minibatch, get_gaussian_kernel,
                           laplacian_pyramid, slerp)

TERO_INCEPTION_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'  # noqa


def load_inception(inception_args, metric):
    """Load Inception Model from given ``inception_args`` and ``metric``. This
    function would try to load Inception under the guidance of 'type' given in
    `inception_args`, if not given, we would try best to load Tero's ones. In
    detailly, we would first try to load the model from disk with the given
    'inception_path', and then try to download the checkpoint from
    'inception_url'. If both method are failed, pytorch version of Inception
    would be loaded.

    Args:
        inception_args (dict): Keyword args for inception net.
        metric (string): Metric to use the Inception. This argument would
            influence the pytorch's Inception loading.
    Returns:
        model (torch.nn.Module): Loaded Inception model.
        style (string): The version of the loaded Inception.
    """

    if not isinstance(inception_args, dict):
        raise TypeError('Receive invalid \'inception_args\': '
                        f'\'{inception_args}\'')

    _inception_args = deepcopy(inception_args)
    inceptoin_type = _inception_args.pop('type', None)

    if torch.__version__ < '1.6.0':
        mmcv.print_log(
            'Current Pytorch Version not support script module, load '
            'Inception Model from torch model zoo. If you want to use '
            'Tero\' script model, please update your Pytorch higher '
            f'than \'1.6\' (now is {torch.__version__})', 'mmgen')
        return _load_inception_torch(_inception_args, metric), 'pytorch'

    # load pytorch version is specific
    if inceptoin_type != 'StyleGAN':
        return _load_inception_torch(_inception_args, metric), 'pytorch'

    # try to load Tero's version
    path = _inception_args.get('inception_path', TERO_INCEPTION_URL)

    # try to parse `path` as web url and download
    if 'http' not in path:
        model = _load_inception_from_path(path)
        if isinstance(model, torch.nn.Module):
            return model, 'StyleGAN'

    # try to parse `path` as path on disk
    model = _load_inception_from_url(path)
    if isinstance(model, torch.nn.Module):
        return model, 'StyleGAN'

    raise RuntimeError('Cannot Load Inception Model, please check the input '
                       f'`inception_args`: {inception_args}')


def _load_inception_from_path(inception_path):
    mmcv.print_log(
        'Try to load Tero\'s Inception Model from '
        f'\'{inception_path}\'.', 'mmgen')
    try:
        model = torch.jit.load(inception_path)
        mmcv.print_log('Load Tero\'s Inception Model successfully.', 'mmgen')
    except Exception as e:
        model = None
        mmcv.print_log(
            'Load Tero\'s Inception Model failed. '
            f'\'{e}\' occurs.', 'mmgen')
    return model


def _load_inception_from_url(inception_url):
    inception_url = inception_url if inception_url else TERO_INCEPTION_URL
    mmcv.print_log(f'Try to download Inception Model from {inception_url}...',
                   'mmgen')
    try:
        path = download_from_url(inception_url, dest_dir=MMGEN_CACHE_DIR)
        mmcv.print_log('Download Finished.')
        return _load_inception_from_path(path)
    except Exception as e:
        mmcv.print_log(f'Download Failed. {e} occurs.')
        return None


def _load_inception_torch(inception_args, metric):
    assert metric in ['FID', 'IS']
    if metric == 'FID':
        inception_model = InceptionV3([3], **inception_args)
    elif metric == 'IS':
        inception_model = inception_v3(pretrained=True, transform_input=False)
        mmcv.print_log(
            'Load Inception V3 Network from Pytorch Model Zoo '
            'for IS calculation. The results can only used '
            'for monitoring purposes. To get more accuracy IS, '
            'please use Tero\'s Inception V3 checkpoints '
            'and use Bicubic Interpolation with Pillow backend '
            'for image resizing. More details may refer to '
            'https://github.com/open-mmlab/mmgeneration/blob/master/docs/en/quick_run.md#is.',  # noqa
            'mmgen')
    return inception_model


def _ssim_for_multi_scale(img1,
                          img2,
                          max_val=255,
                          filter_size=11,
                          filter_sigma=1.5,
                          k1=0.01,
                          k2=0.03):
    """Calculate SSIM (structural similarity) and contrast sensitivity.

    Ref:
    Image quality assessment: From error visibility to structural similarity.

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.

    Returns:
        tuple: Pair containing the mean SSIM and contrast sensitivity between
        `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_f_special_gauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)),
                   axis=(1, 2, 3))  # Return for each image individually.
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs


def ms_ssim(img1,
            img2,
            max_val=255,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03,
            weights=None):
    """Calculate MS-SSIM (multi-scale structural similarity).

    Ref:
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    PGGAN's implementation:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py

    Args:
        img1 (ndarray): Images with range [0, 255] and order "NHWC".
        img2 (ndarray): Images with range [0, 255] and order "NHWC".
        max_val (int): the dynamic range of the images (i.e., the difference
            between the maximum the and minimum allowed values).
            Default to 255.
        filter_size (int): Size of blur kernel to use (will be reduced for
            small images). Default to 11.
        filter_sigma (float): Standard deviation for Gaussian blur kernel (will
            be reduced for small images). Default to 1.5.
        k1 (float): Constant used to maintain stability in the SSIM calculation
            (0.01 in the original paper). Default to 0.01.
        k2 (float): Constant used to maintain stability in the SSIM calculation
            (0.03 in the original paper). Default to 0.03.
        weights (list): List of weights for each level; if none, use five
            levels and the weights from the original paper. Default to None.

    Returns:
        float: MS-SSIM score between `img1` and `img2`.
    """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).' %
            (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' %
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab
    # code.
    weights = np.array(
        weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    im1, im2 = [x.astype(np.float32) for x in [img1, img2]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim_for_multi_scale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim.append(ssim)
        mcs.append(cs)
        im1, im2 = [_hox_downsample(x) for x in [im1, im2]]

    # Clip to zero. Otherwise we get NaNs.
    mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
    mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

    # Average over images only at the end.
    return np.mean(
        np.prod(mcs[:-1, :]**weights[:-1, np.newaxis], axis=0) *
        (mssim[-1, :]**weights[-1]))


def sliced_wasserstein(distribution_a,
                       distribution_b,
                       dir_repeats=4,
                       dirs_per_repeat=128):
    r"""sliced Wasserstein distance of two sets of patches.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py  # noqa

    Args:
        distribution_a (Tensor): Descriptors of first distribution.
        distribution_b (Tensor): Descriptors of second distribution.
        dir_repeats (int): The number of projection times. Default to 4.
        dirs_per_repeat (int): The number of directions per projection.
            Default to 128.

    Returns:
        float: sliced Wasserstein distance.
    """
    if torch.cuda.is_available():
        distribution_b = distribution_b.cuda()
    assert distribution_a.ndim == 2
    assert distribution_a.shape == distribution_b.shape
    assert dir_repeats > 0 and dirs_per_repeat > 0
    distribution_a = distribution_a.to(distribution_b.device)
    results = []
    for _ in range(dir_repeats):
        dirs = torch.randn(distribution_a.shape[1], dirs_per_repeat)
        dirs /= torch.sqrt(torch.sum((dirs**2), dim=0, keepdim=True))
        dirs = dirs.to(distribution_b.device)
        proj_a = torch.matmul(distribution_a, dirs)
        proj_b = torch.matmul(distribution_b, dirs)
        # To save cuda memory, we perform sort in cpu
        proj_a, _ = torch.sort(proj_a.cpu(), dim=0)
        proj_b, _ = torch.sort(proj_b.cpu(), dim=0)
        dists = torch.abs(proj_a - proj_b)
        results.append(torch.mean(dists).item())
    torch.cuda.empty_cache()
    return sum(results) / dir_repeats


class Metric(ABC):
    """The abstract base class of metrics. Basically, we split calculation into
    three steps. First, we initialize the metric object and do some
    preparation. Second, we will feed the real and fake images into metric
    object batch by batch, and we calculate intermediate results of these
    batches. Finally, We use these intermediate results to summarize the final
    result. And the result as a string can be obtained by property
    'result_str'.

    Args:
        num_images (int): The number of real/fake images needed to calculate
            metric.
        image_shape (tuple): Shape of the real/fake images with order "CHW".
    """

    def __init__(self, num_images, image_shape=None):
        self.num_images = num_images
        self.image_shape = image_shape
        self.num_real_need = num_images
        self.num_fake_need = num_images
        self.num_real_feeded = 0  # record of the fed real images
        self.num_fake_feeded = 0  # record of the fed fake images
        self._result_str = None  # string of metric result

    @property
    def result_str(self):
        """Get results in string format.

        Returns:
            str: results in string format
        """
        if not self._result_str:
            self.summary()
            return self._result_str

        return self._result_str

    def feed(self, batch, mode):
        """Feed a image batch into metric calculator and perform intermediate
        operation in 'feed_op' function.

        Args:
            batch (Tensor | dict): Images or dict to be fed into
                metric object. If ``Tensor`` is passed, the order of ``Tensor``
                should be "NCHW". If ``dict`` is passed, each term in the
                ``dict`` are ``Tensor`` with order "NCHW".
            mode (str): Mark the batch as real or fake images. Value can be
                'reals' or 'fakes',
        """
        _, ws = get_dist_info()
        if mode == 'reals':
            if self.num_real_feeded == self.num_real_need:
                return 0

            if isinstance(batch, dict):
                batch_size = [v for v in batch.values()][0].shape[0]
                end = min(batch_size,
                          self.num_real_need - self.num_real_feeded)
                batch_to_feed = {k: v[:end, ...] for k, v in batch.items()}
            else:
                batch_size = batch.shape[0]
                end = min(batch_size,
                          self.num_real_need - self.num_real_feeded)
                batch_to_feed = batch[:end, ...]

            global_end = min(batch_size * ws,
                             self.num_real_need - self.num_real_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_real_feeded += global_end
            return end

        elif mode == 'fakes':
            if self.num_fake_feeded == self.num_fake_need:
                return 0

            batch_size = batch.shape[0]
            end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
            if isinstance(batch, dict):
                batch_to_feed = {k: v[:end, ...] for k, v in batch.items()}
            else:
                batch_to_feed = batch[:end, ...]

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end
        else:
            raise ValueError(
                'The expected mode should be set to \'reals\' or \'fakes\','
                f'but got \'{mode}\'')

    def check(self):
        """Check the numbers of image."""
        assert self.num_real_feeded == self.num_fake_feeded == self.num_images

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """please implement in subclass."""

    @abstractmethod
    def feed_op(self, batch, mode):
        """please implement in subclass."""

    @abstractmethod
    def summary(self):
        """please implement in subclass."""


@METRICS.register_module()
class FID(Metric):
    """FID metric.

    In this metric, we calculate the distance between real distributions and
    fake distributions. The distributions are modeled by the real samples and
    fake samples, respectively.

    `Inception_v3` is adopted as the feature extractor, which is widely used in
    StyleGAN and BigGAN.

    Args:
        num_images (int): The number of images to be tested.
        image_shape (tuple[int], optional): Image shape. Defaults to None.
        inception_pkl (str, optional): Path to reference inception pickle file.
            If `None`, the statistical value of real distribution will be
            calculated at running time. Defaults to None.
        bgr2rgb (bool, optional): If True, reformat the BGR image to RGB
            format. Defaults to True.
        inception_args (dict, optional): Keyword args for inception net.
            Defaults to `dict(normalize_input=False)`.
    """
    name = 'FID'

    def __init__(self,
                 num_images,
                 image_shape=None,
                 inception_pkl=None,
                 bgr2rgb=True,
                 inception_args=dict(normalize_input=False)):
        super().__init__(num_images, image_shape=image_shape)
        self.inception_pkl = inception_pkl
        self.real_feats = []
        self.fake_feats = []
        self.real_mean = None
        self.real_cov = None
        self.bgr2rgb = bgr2rgb
        self.device = 'cpu'

        self.inception_net, self.inception_style = load_inception(
            inception_args, 'FID')

        if torch.cuda.is_available():
            self.inception_net = self.inception_net.cuda()
            self.device = 'cuda'
        self.inception_net.eval()

        mmcv.print_log(f'FID: Adopt Inception in {self.inception_style} style',
                       'mmgen')

    def prepare(self):
        """Prepare for evaluating models with this metric."""
        # if `inception_pkl` is provided, read mean and cov stat
        if self.inception_pkl is not None and mmcv.is_filepath(
                self.inception_pkl):
            with open(self.inception_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.inception_pkl}',
                    'mmgen')
            self.num_real_feeded = self.num_images

    @torch.no_grad()
    def feed_op(self, batch, mode):
        """Feed data to the metric.

        Args:
            batch (Tensor): Input tensor.
            mode (str): The mode of current data batch. 'reals' or 'fakes'.
        """
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0]]
        batch = batch.to(self.device)

        if self.inception_style == 'StyleGAN':
            batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            feat = self.inception_net(batch, return_features=True)
        else:
            feat = self.inception_net(batch)[0].view(batch.shape[0], -1)

        # gather all of images if using distributed training
        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(feat) for _ in range(ws)]
            dist.all_gather(placeholder, feat)
            feat = torch.cat(placeholder, dim=0)

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            if mode == 'reals':
                self.real_feats.append(feat.cpu())
            elif mode == 'fakes':
                self.fake_feats.append(feat.cpu())
            else:
                raise ValueError(
                    f"The expected mode should be set to 'reals' or 'fakes,\
                    but got '{mode}'")

    @staticmethod
    def _calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
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

        return fid, mean_norm, trace

    @torch.no_grad()
    def summary(self):
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        # calculate reference inception stat
        if self.real_mean is None:
            feats = torch.cat(self.real_feats, dim=0)
            assert feats.shape[0] >= self.num_images
            feats = feats[:self.num_images]
            feats_np = feats.numpy()
            self.real_mean = np.mean(feats_np, 0)
            self.real_cov = np.cov(feats_np, rowvar=False)

        # calculate fake inception stat
        fake_feats = torch.cat(self.fake_feats, dim=0)
        assert fake_feats.shape[0] >= self.num_images
        fake_feats = fake_feats[:self.num_images]
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        # calculate distance between real and fake statistics
        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)

        # results for print/table
        self._result_str = (f'{fid:.4f} ({mean:.5f}/{cov:.5f})')
        # results for log_buffer
        self._result_dict = dict(fid=fid, fid_mean=mean, fid_cov=cov)

        return fid, mean, cov

    def clear_fake_data(self):
        """Clear fake data."""
        self.fake_feats = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        """Clear data buffers.

        Args:
            clear_reals (bool, optional): Whether to clear real data.
                Defaults to False.
        """
        self.clear_fake_data()
        if clear_reals:
            self.real_feats = []
            self.num_real_feeded = 0


@METRICS.register_module()
class MS_SSIM(Metric):
    """MS-SSIM (Multi-Scale Structure Similarity) metric.

    Ref: https://github.com/tkarras/progressive_growing_of_gans/blob/master/metrics/ms_ssim.py # noqa

    Args:
        num_images (int): The number of evaluated generated samples.
        image_shape (tuple, optional): Image shape in order "CHW". Defaults to
            None.
    """
    name = 'MS-SSIM'

    def __init__(self, num_images, image_shape=None):
        super().__init__(num_images, image_shape)
        assert num_images % 2 == 0
        self.num_pairs = num_images // 2

    def prepare(self):
        """Prepare for evaluating models with this metric."""
        self.sum = 0.0

    @torch.no_grad()
    def feed_op(self, minibatch, mode):
        """Feed data to the metric.

        Args:
            batch (Tensor): Input tensor.
            mode (str): The mode of current data batch. 'reals' or 'fakes'.
        """
        if mode == 'reals':
            return
        minibatch = ((minibatch + 1) / 2)
        minibatch = minibatch.clamp_(0, 1)
        half1 = minibatch[0::2].cpu().data.numpy().transpose((0, 2, 3, 1))
        half1 = (half1 * 255).astype('uint8')
        half2 = minibatch[1::2].cpu().data.numpy().transpose((0, 2, 3, 1))
        half2 = (half2 * 255).astype('uint8')
        score = ms_ssim(half1, half2)
        self.sum += score * (minibatch.shape[0] // 2)

    @torch.no_grad()
    def summary(self):
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        self.check()
        avg = self.sum / self.num_pairs
        self._result_str = str(round(avg.item(), 4))
        return avg


@METRICS.register_module()
class SWD(Metric):
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
        num_images (int): The number of evaluated generated samples.
        image_shape (tuple): Image shape in order "CHW".
    """
    name = 'SWD'

    def __init__(self, num_images, image_shape):
        super().__init__(num_images, image_shape)

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

    def prepare(self):
        """Prepare for evaluating models with this metric."""
        self.real_descs = [[] for res in self.resolutions]
        self.fake_descs = [[] for res in self.resolutions]
        self.gaussian_k = get_gaussian_kernel()

    @torch.no_grad()
    def feed_op(self, minibatch, mode):
        """Feed data to the metric.

        Args:
            batch (Tensor): Input tensor.
            mode (str): The mode of current data batch. 'reals' or 'fakes'.
        """
        assert minibatch.shape[1:] == self.image_shape
        if mode == 'reals':
            real_pyramid = laplacian_pyramid(minibatch, self.n_pyramids - 1,
                                             self.gaussian_k)
            # lod: layer_of_descriptors
            for lod, level in enumerate(real_pyramid):
                desc = get_descriptors_for_minibatch(level, self.nhood_size,
                                                     self.nhoods_per_image)
                self.real_descs[lod].append(desc)
        elif mode == 'fakes':
            fake_pyramid = laplacian_pyramid(minibatch, self.n_pyramids - 1,
                                             self.gaussian_k)
            for lod, level in enumerate(fake_pyramid):
                desc = get_descriptors_for_minibatch(level, self.nhood_size,
                                                     self.nhoods_per_image)
                self.fake_descs[lod].append(desc)
        else:
            raise ValueError(f'{mode} is not a implemented feed mode.')

    @torch.no_grad()
    def summary(self):
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        self.check()
        real_descs = [finalize_descriptors(d) for d in self.real_descs]
        fake_descs = [finalize_descriptors(d) for d in self.fake_descs]
        del self.real_descs
        del self.fake_descs
        distance = [
            sliced_wasserstein(dreal, dfake, self.dir_repeats,
                               self.dirs_per_repeat)
            for dreal, dfake in zip(real_descs, fake_descs)
        ]
        del real_descs
        del fake_descs
        distance = [d * 1e3 for d in distance]  # multiply by 10^3
        result = distance + [np.mean(distance)]
        self._result_str = ', '.join([str(round(d, 2)) for d in result])
        return result


@METRICS.register_module()
class PR(Metric):
    r"""Improved Precision and recall metric.

        In this metric, we draw real and generated samples respectively, and
        embed them into a high-dimensional feature space using a pre-trained
        classifier network. We use these features to estimate the corresponding
        manifold. We obtain the estimation by calculating pairwise Euclidean
        distances between all feature vectors in the set and, for each feature
        vector, construct a hypersphere with radius equal to the distance to its
        kth nearest neighbor. Together, these hyperspheres define a volume in
        the feature space that serves as an estimate of the true manifold.
        Precision is quantified by querying for each generated image whether
        the image is within the estimated manifold of real images.
        Symmetrically, recall is calculated by querying for each real image
        whether the image is within estimated manifold of generated image.

        Ref: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/precision_recall.py  # noqa

        Note that we highly recommend that users should download the vgg16
        script module from the following address. Then, the `vgg16_script` can
        be set with user's local path. If not given, we will use the vgg16 from
        pytorch model zoo. However, this may bring significant different in the
        final results.

        Tero's vgg16: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt

        Args:
            num_images (int): The number of evaluated generated samples.
            image_shape (tuple): Image shape in order "CHW". Defaults to None.
            num_real_need (int | None, optional): The number of real images.
                Defaults to None.
            full_dataset (bool, optional): Whether to use full dataset for
                evaluation. Defaults to False.
            k (int, optional): Kth nearest parameter. Defaults to 3.
            bgr2rgb (bool, optional): Whether to change the order of image
                channel. Defaults to True.
            vgg16_script (str, optional): Path for the Tero's vgg16 module.
                Defaults to 'work_dirs/cache/vgg16.pt'.
            row_batch_size (int, optional): The batch size of row data.
                Defaults to 10000.
            col_batch_size (int, optional): The batch size of col data.
                Defaults to 10000.
        """
    name = 'PR'

    def __init__(self,
                 num_images,
                 image_shape=None,
                 num_real_need=None,
                 full_dataset=False,
                 k=3,
                 bgr2rgb=True,
                 vgg16_script='work_dirs/cache/vgg16.pt',
                 row_batch_size=10000,
                 col_batch_size=10000):
        super().__init__(num_images, image_shape)
        mmcv.print_log('loading vgg16 for improved precision and recall...',
                       'mmgen')
        if os.path.isfile(vgg16_script):
            self.vgg16 = torch.jit.load('work_dirs/cache/vgg16.pt').eval()
            self.use_tero_scirpt = True
        else:
            mmcv.print_log(
                'Cannot load Tero\'s script module. Use official '
                'vgg16 instead', 'mmgen')
            self.vgg16 = models.vgg16(pretrained=True).eval()
            self.use_tero_scirpt = False
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.vgg16 = self.vgg16.cuda()
            self.device = 'cuda'
        self.k = k

        self.bgr2rgb = bgr2rgb
        self.full_dataset = full_dataset
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        if num_real_need:
            self.num_real_need = num_real_need

        if self.full_dataset:
            self.num_real_need = 10000000

    def prepare(self, *args, **kwargs):
        """Prepare for evaluating models with this metric."""
        self.features_of_reals = []
        self.features_of_fakes = []

    @torch.no_grad()
    def feed_op(self, batch, mode):
        """Feed data to the metric.

        Args:
            batch (Tensor): Input tensor.
            mode (str): The mode of current data batch. 'reals' or 'fakes'.
        """
        batch = batch.to(self.device)
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0], ...]
        if self.use_tero_scirpt:
            batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if mode == 'reals':
            self.features_of_reals.append(self.extract_features(batch))
        elif mode == 'fakes':
            self.features_of_fakes.append(self.extract_features(batch))
        else:
            raise ValueError(f'{mode} is not a implemented feed mode.')

    def check(self):
        if not self.full_dataset:
            assert (self.num_real_feeded == self.num_real_need
                    and self.num_fake_feeded == self.num_fake_need)
        else:
            assert self.num_fake_feeded == self.num_fake_need
            mmcv.print_log(
                f'Test for the full dataset with {self.num_real_feeded}'
                ' real images', 'mmgen')

    @torch.no_grad()
    def summary(self):
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        self.check()

        real_features = torch.cat(self.features_of_reals)
        gen_features = torch.cat(self.features_of_fakes)

        self._result_dict = {}
        rank, ws = get_dist_info()

        for name, manifold, probes in [
            ('precision', real_features, gen_features),
            ('recall', gen_features, real_features)
        ]:
            kth = []
            for manifold_batch in manifold.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=manifold_batch,
                    col_features=manifold,
                    num_gpus=ws,
                    rank=rank,
                    col_batch_size=self.col_batch_size)
                kth.append(
                    distance.to(torch.float32).kthvalue(self.k + 1).values.
                    to(torch.float16) if rank == 0 else None)
            kth = torch.cat(kth) if rank == 0 else None
            pred = []
            for probes_batch in probes.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=probes_batch,
                    col_features=manifold,
                    num_gpus=ws,
                    rank=rank,
                    col_batch_size=self.col_batch_size)
                pred.append((distance <= kth).any(
                    dim=1) if rank == 0 else None)
            self._result_dict[name] = float(
                torch.cat(pred).to(torch.float32).mean() if rank ==
                0 else 'nan')

        precision = self._result_dict['precision']
        recall = self._result_dict['recall']
        self._result_str = f'precision: {precision}, recall:{recall}'
        return self._result_dict

    def extract_features(self, images):
        """Extracting image features.

        Args:
            images (torch.Tensor): Images tensor.
        Returns:
            torch.Tensor: Vgg16 features of input images.
        """
        if self.use_tero_scirpt:
            feature = self.vgg16(images, return_features=True)
        else:
            batch = F.interpolate(images, size=(224, 224))
            before_fc = self.vgg16.features(batch)
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)

        return feature


@METRICS.register_module()
class IS(Metric):
    """IS (Inception Score) metric.

    The images are split into groups, and the inception score is calculated
    on each group of images, then the mean and standard deviation of the score
    is reported. The calculation of the inception score on a group of images
    involves first using the inception v3 model to calculate the conditional
    probability for each image (p(y|x)). The marginal probability is then
    calculated as the average of the conditional probabilities for the images
    in the group (p(y)). The KL divergence is then calculated for each image as
    the conditional probability multiplied by the log of the conditional
    probability minus the log of the marginal probability. The KL divergence is
    then summed over all images and averaged over all classes and the exponent
    of the result is calculated to give the final score.

    Ref: https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py  # noqa

    Note that we highly recommend that users should download the Inception V3
    script module from the following address. Then, the `inception_pkl` can
    be set with user's local path. If not given, we will use the Inception V3
    from pytorch model zoo. However, this may bring significant different in
    the final results.

    Tero's Inception V3: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt  # noqa

    Args:
        num_images (int): The number of evaluated generated samples.
        image_shape (tuple, optional): Image shape in order "CHW". Defaults to
            None.
        bgr2rgb (bool, optional): If True, reformat the BGR image to RGB
            format. In default, our model generate images in the BGR order.
            Thus, we use `True` as the default behavior. Please switch to
            `False`, if the input is in the `RGB` order. Defaults to True.
        resize (bool, optional): Whether resize image to 299x299. Defaults to
            True.
        splits (int, optional): The number of groups. Defaults to 10.
        use_pil_resize (bool, optional): Whether use Bicubic interpolation with
            Pillow's backend. If set as True, the evaluation process may be a
            little bit slow, but achieve a more accurate IS result. Defaults
            to False.
        inception_args (dict, optional): Keyword args for inception net.
            Defaults to ``dict(type='StyleGAN', inception_path=INCEPTION_URL)``.
    """
    name = 'IS'

    def __init__(self,
                 num_images,
                 image_shape=None,
                 bgr2rgb=True,
                 resize=True,
                 splits=10,
                 use_pil_resize=True,
                 inception_args=dict(
                     type='StyleGAN', inception_path=TERO_INCEPTION_URL)):
        super().__init__(num_images, image_shape)
        mmcv.print_log('Loading Inception V3 for IS...', 'mmgen')

        model, style = load_inception(inception_args, 'IS')

        self.inception_model = model
        self.use_tero_script = style == 'StyleGAN'

        self.num_real_feeded = self.num_images
        self.resize = resize
        self.splits = splits
        self.bgr2rgb = bgr2rgb
        self.use_pil_resize = use_pil_resize
        self._pil_resize_warned = False

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.inception_model = self.inception_model.cuda()
            self.device = 'cuda'
        self.inception_model.eval()

    def pil_resize(self, x):
        """Apply Bicubic interpolation with Pillow backend. Before and after
        interpolation operation, we have to perform a type conversion between
        torch.tensor and PIL.Image, and these operations make resize process a
        bit slow.

        Args:
            x (Tensor): Input tensor, should have four dimension and
                        range in [-1, 1].

        Returns:
            torch.FloatTensor: Resized tensor.
        """
        if not self._pil_resize_warned:
            mmcv.print_log(
                '`use_pil_resize` is set as True, apply Bicubic '
                'interpolation with Pillow backend. We perform '
                'type conversion between torch.tensor and '
                'PIL.Image in this function and make this process '
                'a little bit slow.', 'mmgen')
            self._pil_resize_warned = True

        from PIL import Image
        if x.ndim != 4:
            raise ValueError('Input images should have 4 dimensions, '
                             'here receive input with {} '
                             'dimensions.'.format(x.ndim))

        x = (x.clone() * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        x_np = [x_.permute(1, 2, 0).detach().cpu().numpy() for x_ in x]
        x_pil = [Image.fromarray(x_).resize((299, 299)) for x_ in x_np]
        x_ten = torch.cat(
            [torch.FloatTensor(np.array(x_)[None, ...]) for x_ in x_pil])
        x_ten = (x_ten / 127.5 - 1).to(torch.float)
        return x_ten.permute(0, 3, 1, 2)

    def get_pred(self, x):
        """Get prediction from inception model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            np.array: Inception score.
        """
        if self.use_tero_script:
            x = self.inception_model(x, no_output_bias=True)
        else:
            # specify the dimension to avoid warning
            x = F.softmax(self.inception_model(x), dim=1)
        return x

    def prepare(self):
        """Prepare for evaluating models with this metric."""
        self.preds = []

    @torch.no_grad()
    def feed_op(self, batch, mode):
        """Feed data to the metric.

        Args:
            batch (Tensor): Input tensor.
            mode (str): The mode of current data batch. 'reals' or 'fakes'.
        """
        if mode == 'reals':
            pass
        elif mode == 'fakes':
            if self.bgr2rgb:
                batch = batch[:, [2, 1, 0], ...]
            if self.resize:
                if self.use_pil_resize:
                    batch = self.pil_resize(batch)
                else:
                    batch = F.interpolate(
                        batch, size=(299, 299), mode='bilinear')
            if self.use_tero_script:
                batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            batch = batch.to(self.device)

            # get prediction
            pred = self.get_pred(batch)

            if dist.is_initialized():
                ws = dist.get_world_size()
                placeholder = [torch.zeros_like(pred) for _ in range(ws)]
                dist.all_gather(placeholder, pred)
                pred = torch.cat(placeholder, dim=0)

            # in distributed training, we only collect features at rank-0.
            if (dist.is_initialized()
                    and dist.get_rank() == 0) or not dist.is_initialized():
                self.preds.append(pred.cpu().numpy())
        else:
            raise ValueError(f'{mode} is not a implemented feed mode.')

    @torch.no_grad()
    def summary(self):
        """Summarize the results.

        TODO: support `master_only`

        Returns:
            dict | list: Summarized results.
        """
        split_scores = []
        self.preds = np.concatenate(self.preds, axis=0)
        # check for the size
        assert self.preds.shape[0] >= self.num_images
        self.preds = self.preds[:self.num_images]
        for k in range(self.splits):
            part = self.preds[k * (self.num_images // self.splits):(k + 1) *
                              (self.num_images // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        mean, std = np.mean(split_scores), np.std(split_scores)
        # results for print/table
        self._result_str = f'mean: {mean:.3f}, std: {std:.3f}'
        # results for log_buffer
        self._result_dict = {'is': mean, 'is_std': std}
        return mean, std

    def clear_fake_data(self):
        """Clear fake data."""
        self.preds = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        """Clear data buffers.

        Args:
            clear_reals (bool, optional): Whether to clear real data.
                Defaults to False.
        """
        self.clear_fake_data()


@METRICS.register_module()
class PPL(Metric):
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
    name = 'PPL'

    def __init__(self,
                 num_images,
                 image_shape=None,
                 crop=True,
                 epsilon=1e-4,
                 space='W',
                 sampling='end',
                 latent_dim=512):
        super().__init__(num_images, image_shape=image_shape)
        self.crop = crop
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.latent_dim = latent_dim
        self.num_images = num_images * 2
        self.num_real_feeded = self.num_images

    def prepare(self):
        """Prepare for evaluating models with this metric."""
        self.dist_list = []

    @torch.no_grad()
    def feed_op(self, minibatch, mode):
        """Feed data to the metric.

        Args:
            batch (Tensor): Input tensor.
            mode (str): The mode of current data batch. 'reals' or 'fakes'.
        """
        if mode == 'reals':
            return
        # use minibatch's device type to initialize a lpips calculator
        if not hasattr(self, 'percept'):
            self.percept = PerceptualLoss(
                use_gpu=minibatch.device.type.startswith('cuda'))
        # crop and resize images
        if self.crop:
            c = minibatch.shape[2] // 8
            minibatch = minibatch[:, :, c * 3:c * 7, c * 2:c * 6]

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
        self.dist_list.append(distance.to('cpu').numpy())

    @torch.no_grad()
    def summary(self):
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        distances = np.concatenate(self.dist_list, 0)
        lo = np.percentile(distances, 1, interpolation='lower')
        hi = np.percentile(distances, 99, interpolation='higher')
        filtered_dist = np.extract(
            np.logical_and(lo <= distances, distances <= hi), distances)
        ppl_score = filtered_dist.mean()
        self._result_str = f'{ppl_score:.1f}'
        return ppl_score

    def get_sampler(self, model, batch_size, sample_model):
        """Get sampler for sampling along the path.

        Args:
            model (nn.Module): Generative model.
            batch_size (int): Sampling batch size.
            sample_model (str): Which model you want to use. ['ema',
                'orig']. Defaults to 'ema'.

        Returns:
            Object: A sampler for calculating path length regularization.
        """
        if sample_model == 'ema':
            generator = model.generator_ema
        else:
            generator = model.generator
        ppl_sampler = PPLSampler(generator, self.num_images, batch_size,
                                 self.space, self.sampling, self.epsilon,
                                 self.latent_dim)
        return ppl_sampler


class PPLSampler:
    """StyleGAN series generator's sampling iterator for PPL metric.

    Args:
        generator (nn.Module): StyleGAN series' generator.
        num_images (int): The number of evaluated generated samples.
        batch_size (int): Batch size of generated images.
        space (str, optional): Latent space. Defaults to 'W'.
        sampling (str, optional): Sampling mode, whether sampling in full
            path or endpoints. Defaults to 'end'.
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
        self.batch_sizes = [batch_size] * n_batch + ([resid]
                                                     if resid > 0 else [])
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
        injected_noise = self.generator.make_injected_noise()
        inputs = torch.randn([batch * 2, self.latent_dim], device=self.device)
        if self.sampling == 'full':
            lerp_t = torch.rand(batch, device=self.device)
        else:
            lerp_t = torch.zeros(batch, device=self.device)

        if self.space == 'W':
            assert hasattr(self.generator, 'style_mapping')
            latent = self.generator.style_mapping(inputs)
            latent_t0, latent_t1 = latent[::2], latent[1::2]
            latent_e0 = torch.lerp(latent_t0, latent_t1, lerp_t[:, None])
            latent_e1 = torch.lerp(latent_t0, latent_t1,
                                   lerp_t[:, None] + self.epsilon)
            latent_e = torch.stack([latent_e0, latent_e1],
                                   1).view(*latent.shape)
            image = self.generator([latent_e],
                                   input_is_latent=True,
                                   injected_noise=injected_noise)
        else:
            latent_t0, latent_t1 = inputs[::2], inputs[1::2]
            latent_e0 = slerp(latent_t0, latent_t1, lerp_t[:, None])
            latent_e1 = slerp(latent_t0, latent_t1,
                              lerp_t[:, None] + self.epsilon)
            latent_e = torch.stack([latent_e0, latent_e1],
                                   1).view(*inputs.shape)
            image = self.generator([latent_e],
                                   input_is_latent=False,
                                   injected_noise=injected_noise)

        self.idx += 1
        return image


@METRICS.register_module()
class GaussianKLD(Metric):
    r"""Gaussian KLD (Kullback-Leibler divergence) metric. We calculate the
    KLD between two gaussian distribution via `mean` and `log_variance`.
    The passed batch should be a dict instance and contain ``mean_pred``,
    ``mean_target``, ``logvar_pred``, ``logvar_target``.
    When call ``feed`` operation, only ``reals`` mode is needed,

    The calculation of KLD can be formulated as:

    .. math::
        :nowrap:

        \begin{align}
        KLD(p||q) &= -\int{p(x)\log{q(x)} dx} + \int{p(x)\log{p(x)} dx} \\
            &= \frac{1}{2}\log{(2\pi \sigma_2^2)} +
            \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} -
            \frac{1}{2}(1 + \log{2\pi \sigma_1^2}) \\
            &= \log{\frac{\sigma_2}{\sigma_1}} +
            \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
        \end{align}

    where `p` and `q` denote target and predicted distribution respectively.

    Args:
        num_images (int): The number of samples to be tested.
        base (str, optional): The log base of calculated KLD. Support
            ``'e'`` and ``'2'``. Defaults to ``'e'``.
        reduction (string, optional): Specifies the reduction to apply to the
            output. Support ``'batchmean'``, ``'sum'`` and ``'mean'``. If
            ``reduction == 'batchmean'``, the sum of the output will be divided
            by batchsize. If ``reduction == 'sum'``, the output will be summed.
            If ``reduction == 'mean'``, the output will be divided by the
            number of elements in the output. Defaults to ``'batchmean'``.

    """
    name = 'GaussianKLD'

    def __init__(self, num_images, base='e', reduction='batchmean'):
        super().__init__(num_images, image_shape=None)
        assert reduction in [
            'sum', 'batchmean', 'mean'
        ], ('We only support reduction for \'batchmean\', \'sum\' '
            'and \'mean\'')
        assert base in ['e',
                        '2'], ('We only support log_base for \'e\' and \'2\'')
        self.reduction = reduction
        self.num_fake_feeded = self.num_images
        self.cal_kld = partial(
            gaussian_kld, weight=1, reduction='none', base=base)

    def prepare(self):
        """Prepare for evaluating models with this metric."""
        self.kld = []
        self.num_real_feeded = 0

    @torch.no_grad()
    def feed_op(self, batch, mode):
        """Feed data to the metric.

        Args:
            batch (Tensor): Input tensor.
            mode (str): The mode of current data batch. 'reals' or 'fakes'.
        """
        if mode == 'fakes':
            return
        assert isinstance(batch, dict), ('To calculate GaussianKLD loss, a '
                                         'dict contains probabilistic '
                                         'parameters is required.')
        # check required keys
        require_keys = [
            'mean_pred', 'mean_target', 'logvar_pred', 'logvar_target'
        ]
        if any([k not in batch for k in require_keys]):
            raise KeyError(f'The input dict must require {require_keys} at '
                           'the same time. But keys in the given dict are '
                           f'{batch.keys()}. Some of the requirements are '
                           'missing.')
        kld = self.cal_kld(batch['mean_target'], batch['mean_pred'],
                           batch['logvar_target'], batch['logvar_pred'])
        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(kld) for _ in range(ws)]
            dist.all_gather(placeholder, kld)
            kld = torch.cat(placeholder, dim=0)

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            self.kld.append(kld.cpu())

    @torch.no_grad()
    def summary(self):
        """Summarize the results.

        Returns:
            dict | list: Summarized results.
        """
        kld = torch.cat(self.kld, dim=0)
        assert kld.shape[0] >= self.num_images
        kld_np = kld.numpy()
        if self.reduction == 'sum':
            kld_result = np.sum(kld_np)
        elif self.reduction == 'mean':
            kld_result = np.mean(kld_np)
        else:
            kld_result = np.sum(kld_np) / kld_np.shape[0]
        self._result_str = (f'{kld_result:.4f}')
        return kld_result
