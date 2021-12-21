# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from pathlib import Path

from mmcv import scandir
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class PairedImageDataset(Dataset):
    """General paired image folder dataset for image generation.

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'. '/path/to/data'
    can be initialized by args 'dataroot'. Each sample contains a pair of
    images concatenated in the w dimension (A|B).

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of paired images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        testdir (str): Subfolder of dataroot which contain test images.
            Default: 'test'.
    """

    def __init__(self, dataroot, pipeline, test_mode=False, testdir='test'):
        super().__init__()
        phase = testdir if test_mode else 'train'
        self.dataroot = osp.join(str(dataroot), phase)
        self.data_infos = self.load_annotations()
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        pair_paths = sorted(self.scan_folder(self.dataroot))
        for pair_path in pair_paths:
            data_infos.append(dict(pair_path=pair_path))

        return data_infos

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

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        if not self.test_mode:
            return self.prepare_train_data(idx)

        return self.prepare_test_data(idx)
