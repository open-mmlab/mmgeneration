# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmcv import FileClient
from mmengine.dataset import BaseDataset

from mmgen.registry import DATASETS
from .utils import infer_io_backend


@DATASETS.register_module()
class UnconditionalImageDataset(BaseDataset):
    """Unconditional Image Dataset.

    This dataset contains raw images for training unconditional GANs. Given
    a root dir, we will recursively find all images in this root. The
    transformation on data is defined by the pipeline.

    Args:
        data_root (str): Root path for unconditional images.
        pipeline (list[dict | callable]): A sequence of data transforms.
        io_backend (str, optional): The storage backend type. Options are
            "disk", "ceph", "memcached", "lmdb", "http" and "petrel".
            Default: None.
        test_mode (bool, optional): If True, the dataset will work in test
            mode. Otherwise, in train mode. Default to False.
    """

    _VALID_IMG_SUFFIX = ('.jpg', '.png', '.jpeg', '.JPEG')

    def __init__(self,
                 data_root,
                 pipeline,
                 io_backend: Optional[str] = None,
                 test_mode=False):
        if io_backend is None:
            io_backend = infer_io_backend(data_root)
        self.file_client = FileClient(backend=io_backend)
        super().__init__(
            data_root=data_root, pipeline=pipeline, test_mode=test_mode)

    def load_data_list(self):
        """Load annotations."""
        # recursively find all of the valid images from data_root
        data_list = []
        imgs_list = self.file_client.list_dir_or_file(
            self.data_root,
            list_dir=False,
            suffix=self._VALID_IMG_SUFFIX,
            recursive=True)
        data_list = [
            dict(img_path=self.file_client.join_path(self.data_root, x))
            for x in imgs_list
        ]
        return data_list

    def __repr__(self):
        dataset_name = self.__class__
        data_root = self.data_root
        num_imgs = len(self)
        return (f'dataset_name: {dataset_name}, total {num_imgs} images in '
                f'data_root: {data_root}')
