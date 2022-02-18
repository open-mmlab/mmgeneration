# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class UnconditionalImageDataset(Dataset):
    """Unconditional Image Dataset.

    This dataset contains raw images for training unconditional GANs. Given
    a root dir, we will recursively find all images in this root. The
    transformation on data is defined by the pipeline.

    Args:
        imgs_root (str): Root path for unconditional images.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool, optional): If True, the dataset will work in test
            mode. Otherwise, in train mode. Default to False.
    """

    _VALID_IMG_SUFFIX = ('.jpg', '.png', '.jpeg', '.JPEG')

    def __init__(self, imgs_root, pipeline, test_mode=False):
        super().__init__()
        self.imgs_root = imgs_root
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.load_annotations()

        # print basic dataset information to check the validity
        mmcv.print_log(repr(self), 'mmgen')

    def load_annotations(self):
        """Load annotations."""
        # recursively find all of the valid images from imgs_root
        imgs_list = mmcv.scandir(
            self.imgs_root, self._VALID_IMG_SUFFIX, recursive=True)
        self.imgs_list = [osp.join(self.imgs_root, x) for x in imgs_list]

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        results = dict(real_img_path=self.imgs_list[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        results = dict(real_img_path=self.imgs_list[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, idx):
        if not self.test_mode:
            return self.prepare_train_data(idx)

        return self.prepare_test_data(idx)

    def __repr__(self):
        dataset_name = self.__class__
        imgs_root = self.imgs_root
        num_imgs = len(self)
        return (f'dataset_name: {dataset_name}, total {num_imgs} images in '
                f'imgs_root: {imgs_root}')
