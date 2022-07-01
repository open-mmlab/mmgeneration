# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine import Visualizer
from torch import Tensor
from torchvision.utils import make_grid

from mmgen.registry import VISUALIZERS
from mmgen.typing import DataSetOutputs


@VISUALIZERS.register_module()
class GenVisualizer(Visualizer):
    """MMGeneration Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.

    Examples:
        >>> # Draw image
        >>> vis = GenVisualizer()
        >>> vis.add_datasample(
        >>>     'random_noise',
        >>>     gen_samples=torch.rand(2, 3, 10, 10),
        >>>     gt_samples=dict(imgs=torch.randn(2, 3, 10, 10)),
        >>>     gt_keys='imgs',
        >>>     vis_mode='image',
        >>>     n_rows=2,
        >>>     step=10)
    """

    def __init__(self,
                 name='visualizer',
                 vis_backends: Optional[List[Dict]] = None,
                 save_dir: Optional[str] = None) -> None:
        super().__init__(name, vis_backends=vis_backends, save_dir=save_dir)

    @staticmethod
    def _post_process_image(
            image: Tensor,
            color_order: str,
            mean: Optional[Sequence[Union[float, int]]] = None,
            std: Optional[Sequence[Union[float, int]]] = None) -> Tensor:
        """Post process images. First convert image to `rgb` order. And then
        de-norm image to fid `mean` and `std` if `mean` and `std` is passed.

        Args:
            image (Tensor): Image to pose process.
            color_order (str): The color order of the passed image.
            mean (Optional[Sequence[Union[float, int]]], optional): Target
                mean of the passed image. Defaults to None.
            std (Optional[Sequence[Union[float, int]]], optional): Target
                std of the passed image. Defaults to None.

        Returns:
            Tensor: Image in original value range and RGB color order.
        """
        if image.shape[1] == 1:
            image = torch.cat([image, image, image], dim=1)
        if color_order == 'bgr':
            image = image[:, [2, 1, 0], ...]
        if mean is not None and std is not None:
            image = image * std + mean
        return image

    @staticmethod
    def _get_padding_tensor(samples: Tuple[dict, Tensor],
                            n_rows: int) -> Optional[Tensor]:
        """Get tensor for padding the empty position."""

        if isinstance(samples, dict):
            sample_shape = next(iter(samples.values())).shape
        else:
            sample_shape = samples.shape

        n_samples = sample_shape[0]
        n_padding = n_samples % n_rows
        if n_padding:
            return -1.0 * torch.ones(n_padding, *sample_shape[1:])
        return None

    def _vis_image_sample(self, gen_samples: Tuple[dict,
                                                   Tensor], gt_samples: dict,
                          draw_gt: bool, gt_keys: Optional[Tuple[str,
                                                                 List[str]]],
                          color_order: str, target_mean: Sequence[Union[float,
                                                                        int]],
                          target_std: Sequence[Union[float, int]],
                          n_row: int) -> np.ndarray:
        """Visualize image sample.

        Args:
            gen_samples (Tuple[dict, Tensor]): Generated samples to visualize.
            gt_samples (dict): Real images to visualize.
            n_row (int): The number of samples in a row.

        Returns:
            np.ndarray: The visualization result.
        """

        # handle `gt_samples`
        gt_samples_ = dict()
        if draw_gt:
            assert gt_samples is not None, (
                '\'gt_sample\' must not be passed to visualize real images.')
            gt_keys = ['imgs'] if gt_keys is None else gt_keys
            for k in gt_keys:
                assert k in gt_samples['inputs'], (
                    f'Cannot find \'{k}\' not in \'gt_samples\'.')
                gt_samples_[k] = self._post_process_image(
                    gt_samples['inputs'][k].cpu(), color_order, target_mean,
                    target_std)

        # handle `gen_samples`
        if isinstance(gen_samples, dict):
            gen_samples_ = dict()
            for name, sample in gen_samples.items():
                gen_samples_[name] = self._post_process_image(
                    sample.cpu(), color_order, target_mean, target_std)
        else:
            gen_samples_ = self._post_process_image(gen_samples.cpu(),
                                                    color_order, target_mean,
                                                    target_std)

        padding_tensor = self._get_padding_tensor(gen_samples, n_row)

        vis_results = []

        for target_samples in [gt_samples_, gen_samples_]:
            if target_samples == {}:
                continue

            if isinstance(target_samples, dict):
                for sample in target_samples.values():
                    if padding_tensor is not None:
                        vis_results.append(
                            torch.cat([sample, padding_tensor], dim=0))
                    else:
                        vis_results.append(sample)
            else:
                vis_results.append(target_samples)

        # concatnate along batch size
        vis_results = torch.cat(vis_results, dim=0)
        vis_results = make_grid(vis_results, nrow=n_row).cpu().permute(1, 2, 0)
        vis_results = vis_results.numpy().astype(np.uint8)
        return vis_results

    def add_datasample(self,
                       name: str,
                       *,
                       gen_samples: Tuple[dict, Tensor],
                       gt_samples: Optional[DataSetOutputs] = None,
                       draw_gt: bool = False,
                       gt_keys: Optional[Tuple[str, List[str]]] = None,
                       vis_mode: Optional[str] = None,
                       n_rows: Optional[int] = None,
                       color_order: str = 'bgr',
                       target_mean: Sequence[Union[float, int]] = 127.5,
                       target_std: Sequence[Union[float, int]] = 127.5,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.

        Args:
            name (str): The image identifier.
            gen_samples ()
            gt_samples
            draw_gt
            gt_keys
            vis_mode
            n_rows
            input_color_order
            output_color_order
            target_mean
            target_std
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            step (int): Global step value to record. Defaults to 0.
        """

        # get visualize function
        if vis_mode is None:
            vis_func = self._vis_image_sample
        else:
            vis_func = getattr(self, f'_vis_{vis_mode}_sample')

        vis_sample = vis_func(gen_samples, gt_samples, draw_gt, gt_keys,
                              color_order, target_mean, target_std, n_rows)

        if show:
            self.show(vis_sample, win_name=name, wait_time=wait_time)

        self.add_image(name, vis_sample, step)
