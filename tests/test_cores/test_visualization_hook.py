import os.path as osp
import tempfile
from unittest.mock import MagicMock

import mmcv
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mmgen.core import VisualizationHook
from mmgen.utils import get_root_logger


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        img = torch.zeros((3, 10, 10))
        img[:, 2:9, :] = 1.
        results = dict(imgs=img)
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None

    def train_step(self, data_batch, optimizer):
        output = dict(results=dict(img=data_batch['imgs']))
        return output


def test_visual_hook():
    with pytest.raises(AssertionError):
        VisualizationHook('temp', [1, 2, 3])
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))

    img = torch.zeros((1, 3, 10, 10))
    img[:, :, 2:9, :] = 1.
    model = ExampleModel()
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        visual_hook = VisualizationHook('visual', ['img'], interval=8)
        runner = mmcv.runner.IterBasedRunner(
            model=model, work_dir=tmpdir, logger=get_root_logger())
        runner.register_hook(visual_hook)
        runner.run([data_loader], [('train', 10)], 10)
        img_saved = mmcv.imread(
            osp.join(tmpdir, 'visual', 'iter_8.png'), flag='unchanged')

        np.testing.assert_almost_equal(img_saved,
                                       img[0].permute(1, 2, 0) * 127 + 128)
