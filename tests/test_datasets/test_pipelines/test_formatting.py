import numpy as np
import pytest
import torch

from mmgen.datasets.pipelines import Collect, ImageToTensor, ToTensor


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def test_to_tensor():
    to_tensor = ToTensor(['str'])
    with pytest.raises(TypeError):
        results = dict(str='0')
        to_tensor(results)

    target_keys = ['tensor', 'numpy', 'sequence', 'int', 'float']
    to_tensor = ToTensor(target_keys)
    ori_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1)

    results = to_tensor(ori_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, ori_results[key])

    # Add an additional key which is not in keys.
    ori_results = dict(
        tensor=torch.randn(2, 3),
        numpy=np.random.randn(2, 3),
        sequence=list(range(10)),
        int=1,
        float=0.1,
        str='test')
    results = to_tensor(ori_results)
    assert check_keys_contain(results.keys(), target_keys)
    for key in target_keys:
        assert isinstance(results[key], torch.Tensor)
        assert torch.equal(results[key].data, ori_results[key])

    assert repr(
        to_tensor) == to_tensor.__class__.__name__ + f'(keys={target_keys})'


def test_image_to_tensor():
    ori_results = dict(img=np.random.randn(256, 256, 3))
    keys = ['img']
    to_float32 = False
    image_to_tensor = ImageToTensor(keys)
    results = image_to_tensor(ori_results)
    assert results['img'].shape == torch.Size([3, 256, 256])
    assert isinstance(results['img'], torch.Tensor)
    assert torch.equal(results['img'].data, ori_results['img'])
    assert results['img'].dtype == torch.float32

    ori_results = dict(img=np.random.randint(256, size=(256, 256)))
    keys = ['img']
    to_float32 = True
    image_to_tensor = ImageToTensor(keys)
    results = image_to_tensor(ori_results)
    assert results['img'].shape == torch.Size([1, 256, 256])
    assert isinstance(results['img'], torch.Tensor)
    assert torch.equal(results['img'].data, ori_results['img'])
    assert results['img'].dtype == torch.float32

    assert repr(image_to_tensor) == (
        image_to_tensor.__class__.__name__ +
        f'(keys={keys}, to_float32={to_float32})')


def test_collect():
    inputs = dict(
        img=np.random.randn(256, 256, 3),
        label=[1],
        img_name='test_image.png',
        ori_shape=(256, 256, 3),
        img_shape=(256, 256, 3),
        pad_shape=(256, 256, 3),
        flip_direction='vertical',
        img_norm_cfg=dict(to_bgr=False))
    keys = ['img', 'label']
    meta_keys = ['img_shape', 'img_name', 'ori_shape']
    collect = Collect(keys, meta_keys=meta_keys)
    results = collect(inputs)
    assert set(list(results.keys())) == set(['img', 'label', 'meta'])
    inputs.pop('img')
    assert set(results['meta'].data.keys()) == set(meta_keys)
    for key in results['meta'].data:
        assert results['meta'].data[key] == inputs[key]

    assert repr(collect) == (
        collect.__class__.__name__ +
        f'(keys={keys}, meta_keys={collect.meta_keys})')
