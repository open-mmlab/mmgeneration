import torch
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class QuickTestImageDataset(Dataset):
    """Uncoditional Image Dataset.

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

    def __init__(self, *args, size=None, **kwargs):
        super().__init__()
        self.size = size

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return dict(real_img=torch.randn(3, self.size[0], self.size[1]))
