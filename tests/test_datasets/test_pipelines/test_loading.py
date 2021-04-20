from pathlib import Path

import mmcv
import numpy as np

from mmgen.datasets import LoadImageFromFile


def test_load_image_from_file():
    path_baboon = Path(
        __file__).parent / '..' / '..' / 'data' / 'image' / 'baboon.png'
    img_baboon = mmcv.imread(str(path_baboon), flag='color')

    # read gt image
    # input path is Path object
    results = dict(gt_path=path_baboon)
    config = dict(io_backend='disk', key='gt')
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results['gt'].shape == (480, 500, 3)
    np.testing.assert_almost_equal(results['gt'], img_baboon)
    assert results['gt_path'] == str(path_baboon)
    # input path is str
    results = dict(gt_path=str(path_baboon))
    results = image_loader(results)
    assert results['gt'].shape == (480, 500, 3)
    np.testing.assert_almost_equal(results['gt'], img_baboon)
    assert results['gt_path'] == str(path_baboon)

    assert repr(image_loader) == (
        image_loader.__class__.__name__ +
        ('(io_backend=disk, key=gt, '
         'flag=color, save_original_img=False)'))
