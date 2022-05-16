# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmgen.datasets.pipelines import Compose
from mmgen.datasets.pipelines import Resize
from mmgen.datasets.pipelines import PackGenInputs

def test_compose():
    with pytest.raises(TypeError):
        Compose('LoadAlpha')

    img = np.random.randn(256, 256, 3)
    results = dict(img=img, img_name='test_image.png')
    test_pipeline = [
        dict(type='Resize', scale=(64,64)),
        dict(type='PackGenInputs', meta_keys=['img_name'])
    ]
    compose = Compose(test_pipeline)
    compose_results = compose(results)
    assert compose_results['inputs'].shape == (3, 64, 64)
    assert compose_results['data_sample'].img_name == 'test_image.png'

    resize = Resize(scale=(64, 64))
    pack = PackGenInputs(meta_keys=['img_name'])
    compose = Compose([resize, pack])
    compose_results = compose(results)
    assert compose_results['inputs'].shape == (3, 64, 64)
    assert compose_results['data_sample'].img_name == 'test_image.png'
