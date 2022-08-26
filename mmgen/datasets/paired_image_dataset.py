# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

from mmengine import scandir
from mmengine.dataset import BaseDataset

from mmgen.registry import DATASETS

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class PairedImageDataset(BaseDataset):
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

    def __init__(self, data_root, pipeline, test_mode=False, testdir='test'):
        phase = testdir if test_mode else 'train'
        self.data_root = osp.join(str(data_root), phase)
        super().__init__(
            data_root=self.data_root, pipeline=pipeline, test_mode=test_mode)
        # self.data_infos = self.load_annotations()

    def load_data_list(self):
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        pair_paths = sorted(self.scan_folder(self.data_root))
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
