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
            data_info_list = self._load_annotations_from_npz()
        self.data_infos = data_info_list

    def _load_annotations_from_npz(self):
        npz_file = np.load(self.file_path)
        data_info_dict = dict()
        data_info_list = []
        npz_keys = list(npz_file.keys())

        num_samples = None
        for k in npz_keys:
            data_info_dict[k] = npz_file[k]
            # check number of samples
            if num_samples is None:
                num_samples = npz_file[k].shape[0]
            else:
                assert num_samples == npz_file[k].shape[0]

        # save to list
        for idx in range(num_samples):
            data_info = dict()
            for k in npz_keys:
                var = data_info_dict[k][idx]
                if var.shape == ():
                    var = np.array([var])
                data_info[k] = var
            data_info_list.append(data_info)

        return data_info_list

    def prepare_data(self, idx):
        """Prepare data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        return self.pipeline(self.data_infos[idx])

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __repr__(self):
        dataset_name = self.__class__
        file_path = self.file_path
        num_imgs = len(self)
        return (f'dataset_name: {dataset_name}, total {num_imgs} images in '
                f'file_path: {file_path}')
