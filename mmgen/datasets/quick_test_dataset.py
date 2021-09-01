# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.utils.data import Dataset

from .builder import DATASETS


@DATASETS.register_module()
class QuickTestImageDataset(Dataset):
    """Dataset for quickly testing the correctness.

    Args:
        size (tuple[int]): The size of the images. Defaults to `None`.
    """

    def __init__(self, *args, size=None, **kwargs):
        super().__init__()
        self.size = size
        self.img_tensor = torch.randn(3, self.size[0], self.size[1])

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return dict(real_img=self.img_tensor)
