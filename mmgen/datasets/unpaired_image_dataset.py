# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import numpy as np
from mmcv import scandir
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class UnpairedImageDataset(Dataset):
    """General unpaired image folder dataset for image generation.

    It assumes that the training directory of images from domain A is
    '/path/to/data/trainA', and that from domain B is '/path/to/data/trainB',
    respectively. '/path/to/data' can be initialized by args 'dataroot'.
    During test time, the directory is '/path/to/data/testA' and
    '/path/to/data/testB', respectively.

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of unpaired
            images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        domain_a (str, optional): Domain of images in trainA / testA.
            Defaults to None.
        domain_b (str, optional): Domain of images in trainB / testB.
            Defaults to None.
    """

    def __init__(self,
                 dataroot,
                 pipeline,
                 test_mode=False,
                 domain_a=None,
                 domain_b=None):
        super().__init__()
        phase = 'test' if test_mode else 'train'
        self.dataroot_a = osp.join(str(dataroot), phase + 'A')
        self.dataroot_b = osp.join(str(dataroot), phase + 'B')
        self.data_infos_a = self.load_annotations(self.dataroot_a)
        self.data_infos_b = self.load_annotations(self.dataroot_b)
        self.len_a = len(self.data_infos_a)
        self.len_b = len(self.data_infos_b)
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        assert isinstance(domain_a, str)
        assert isinstance(domain_b, str)
        self.domain_a = domain_a
        self.domain_b = domain_b

    def load_annotations(self, dataroot):
        """Load unpaired image paths of one domain.

        Args:
            dataroot (str): Path to the folder root for unpaired images of
                one domain.

        Returns:
            list[dict]: List that contains unpaired image paths of one domain.
        """
        data_infos = []
        paths = sorted(self.scan_folder(dataroot))
        for path in paths:
            data_infos.append(dict(path=path))
        return data_infos

    def prepare_train_data(self, idx):
        """Prepare unpaired training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        idx_b = np.random.randint(0, self.len_b)
        img_b_path = self.data_infos_b[idx_b]['path']
        results = dict()
        results[f'img_{self.domain_a}_path'] = img_a_path
        results[f'img_{self.domain_b}_path'] = img_b_path
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare unpaired test data.

        Args:
            idx (int): Index of current batch.

        Returns:
            list[dict]: Prepared test data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        img_b_path = self.data_infos_b[idx % self.len_b]['path']
        results = dict()
        results[f'img_{self.domain_a}_path'] = img_a_path
        results[f'img_{self.domain_b}_path'] = img_b_path
        return self.pipeline(results)

    def __len__(self):
        return max(self.len_a, self.len_b)

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: Image list obtained from the given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = scandir(path, suffix=IMG_EXTENSIONS, recursive=True)
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        if not self.test_mode:
            return self.prepare_train_data(idx)

        return self.prepare_test_data(idx)
