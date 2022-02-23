# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from .builder import DATASETS


def create_real_pyramid(real, min_size, max_size, scale_factor_init):
    """Create image pyramid.

    This function is modified from the official implementation:
    https://github.com/tamarott/SinGAN/blob/master/SinGAN/functions.py#L221

    In this implementation, we adopt the rescaling function from MMCV.
    Args:
        real (np.array): The real image array.
        min_size (int): The minimum size for the image pyramid.
        max_size (int): The maximum size for the image pyramid.
        scale_factor_init (float): The initial scale factor.
    """

    num_scales = int(
        np.ceil(
            np.log(np.power(min_size / min(real.shape[0], real.shape[1]), 1)) /
            np.log(scale_factor_init))) + 1

    scale2stop = int(
        np.ceil(
            np.log(
                min([max_size, max([real.shape[0], real.shape[1]])]) /
                max([real.shape[0], real.shape[1]])) /
            np.log(scale_factor_init)))

    stop_scale = num_scales - scale2stop

    scale1 = min(max_size / max([real.shape[0], real.shape[1]]), 1)
    real_max = mmcv.imrescale(real, scale1)
    scale_factor = np.power(
        min_size / (min(real_max.shape[0], real_max.shape[1])),
        1 / (stop_scale))

    scale2stop = int(
        np.ceil(
            np.log(
                min([max_size, max([real.shape[0], real.shape[1]])]) /
                max([real.shape[0], real.shape[1]])) /
            np.log(scale_factor_init)))
    stop_scale = num_scales - scale2stop

    reals = []
    for i in range(stop_scale + 1):
        scale = np.power(scale_factor, stop_scale - i)
        curr_real = mmcv.imrescale(real, scale)
        reals.append(curr_real)

    return reals, scale_factor, stop_scale


@DATASETS.register_module()
class SinGANDataset(Dataset):
    """SinGAN Dataset.

    In this dataset, we create an image pyramid and save it in the cache.

    Args:
        img_path (str): Path to the single image file.
        min_size (int): Min size of the image pyramid. Here, the number will be
            set to the ``min(H, W)``.
        max_size (int): Max size of the image pyramid. Here, the number will be
            set to the ``max(H, W)``.
        scale_factor_init (float): Rescale factor. Note that the actual factor
            we use may be a little bit different from this value.
        num_samples (int, optional): The number of samples (length) in this
            dataset. Defaults to -1.
    """

    def __init__(self,
                 img_path,
                 min_size,
                 max_size,
                 scale_factor_init,
                 num_samples=-1):
        self.img_path = img_path
        assert mmcv.is_filepath(self.img_path)
        self.load_annotations(min_size, max_size, scale_factor_init)
        self.num_samples = num_samples

    def load_annotations(self, min_size, max_size, scale_factor_init):
        """Load annatations for SinGAN Dataset.

        Args:
            min_size (int): The minimum size for the image pyramid.
            max_size (int): The maximum size for the image pyramid.
            scale_factor_init (float): The initial scale factor.
        """
        real = mmcv.imread(self.img_path)
        self.reals, self.scale_factor, self.stop_scale = create_real_pyramid(
            real, min_size, max_size, scale_factor_init)

        self.data_dict = {}

        for i, real in enumerate(self.reals):
            self.data_dict[f'real_scale{i}'] = self._img2tensor(real)

        self.data_dict['input_sample'] = torch.zeros_like(
            self.data_dict['real_scale0'])

    def _img2tensor(self, img):
        img = torch.from_numpy(img).to(torch.float32).permute(2, 0,
                                                              1).contiguous()
        img = (img / 255 - 0.5) * 2

        return img

    def __getitem__(self, index):
        return self.data_dict

    def __len__(self):
        return int(1e6) if self.num_samples < 0 else self.num_samples
