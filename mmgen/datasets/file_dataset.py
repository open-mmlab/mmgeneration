# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class FileDataset(Dataset):
    """Uncoditional file Dataset.

    This dataset contains raw images for training unconditional GANs. Given
    the path of a file, we will load all image in this file. The
    transformation on data is defined by the pipeline. Please ensure that
    ``LoadImageFromFile`` is not in your pipeline configs because we directly
    get images in ``np.ndarray`` from the given file.

    Args:
        file_path (str): Path of the file.
        img_keys (str): Key of the images in npz file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool, optional): If True, the dataset will work in test
            mode. Otherwise, in train mode. Default to False.
        npz_keys (str | list[str], optional): Key of the images to load in the
            npz file. Must with the input file is as npz file.
    """

    _VALID_FILE_SUFFIX = ('.npz')

    def __init__(self, file_path, pipeline, test_mode=False):
        super().__init__()
        assert any([
            file_path.endswith(suffix) for suffix in self._VALID_FILE_SUFFIX
        ]), (f'We only support \'{self._VALID_FILE_SUFFIX}\' in this dataset, '
             f'but receive {file_path}.')

        self.file_path = file_path
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.load_annotations()

        # print basic dataset information to check the validity
        mmcv.print_log(repr(self), 'mmgen')

    def load_annotations(self):
        """Load annotations."""
        if self.file_path.endswith('.npz'):
            data_info, data_length = self._load_annotations_from_npz()
            data_fetch_fn = self._npz_data_fetch_fn

        self.data_infos = data_info
        self.data_fetch_fn = data_fetch_fn
        self.data_length = data_length

    def _load_annotations_from_npz(self):
        """Load annotations from npz file and check number of samples are
        consistent  among all items.

        Returns:
            tuple: dict and int
        """
        npz_file = np.load(self.file_path, mmap_mode='r')
        data_info_dict = dict()
        npz_keys = list(npz_file.keys())

        # checnk num samples
        num_samples = None
        for k in npz_keys:
            data_info_dict[k] = npz_file[k]
            # check number of samples
            if num_samples is None:
                num_samples = npz_file[k].shape[0]
            else:
                assert num_samples == npz_file[k].shape[0]
        return data_info_dict, num_samples

    @staticmethod
    def _npz_data_fetch_fn(data_infos, idx):
        """Fetch data from npz file by idx and package them to a dict.

        Args:
            data_infos (array, tuple, dict): Data infos in the npz file.
            idx (int): Index of current batch.

        Returns:
            dict: Data infos of the given idx.
        """
        data_dict = dict()
        for k in data_infos.keys():
            data_dict[k] = data_infos[k][idx]
        return data_dict

    def prepare_data(self, idx, data_fetch_fn=None):
        """Prepare data.

        Args:
            idx (int): Index of current batch.
            data_fetch_fn (callable): Function to fetch data.

        Returns:
            dict: Prepared training data batch.
        """
        if data_fetch_fn is None:
            data = self.data_infos[idx]
        else:
            data = data_fetch_fn(self.data_infos, idx)
        return self.pipeline(data)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return self.prepare_data(idx, self.data_fetch_fn)

    def __repr__(self):
        dataset_name = self.__class__
        file_path = self.file_path
        num_imgs = len(self)
        return (f'dataset_name: {dataset_name}, total {num_imgs} images in '
                f'file_path: {file_path}')
