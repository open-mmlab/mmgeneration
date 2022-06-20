# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
from mmengine.dataset import BaseDataset

from mmgen.registry import DATASETS


@DATASETS.register_module()
class UnconditionalImageDataset(BaseDataset):
    """Unconditional Image Dataset.

    This dataset contains raw images for training unconditional GANs. Given
    a root dir, we will recursively find all images in this root. The
    transformation on data is defined by the pipeline.

    Args:
        data_root (str): Root path for unconditional images.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool, optional): If True, the dataset will work in test
            mode. Otherwise, in train mode. Default to False.
    """

    _VALID_IMG_SUFFIX = ('.jpg', '.png', '.jpeg', '.JPEG')

    def __init__(self, data_root, pipeline, test_mode=False):
        super().__init__(
            data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    def load_data_list(self):
        """Load annotations."""
        # recursively find all of the valid images from data_root
        data_list = []
        imgs_list = mmcv.scandir(
            self.data_root, self._VALID_IMG_SUFFIX, recursive=True)
        data_list = [
            dict(img_path=osp.join(self.data_root, x)) for x in imgs_list
        ]
        return data_list

    def __repr__(self):
        dataset_name = self.__class__
        data_root = self.data_root
        num_imgs = len(self)
        return (f'dataset_name: {dataset_name}, total {num_imgs} images in '
                f'data_root: {data_root}')
