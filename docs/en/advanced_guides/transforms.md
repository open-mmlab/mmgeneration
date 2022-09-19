# Data Transforms

In this tutorial, we introduce the design of transforms pipeline in MMGeneration.

The structure of this guide is as follows:

- [Data Transforms](#data-transforms)
  - [Design of Data pipelines](#design-of-data-pipelines)
  - [Customization data transformation](#customization-data-transformation)

## Design of Data pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading with multiple workers.
`Dataset` returns a dict of data items corresponding the arguments of models' forward method.

In 1.x version of MMGeneration, all data transformations are inherited from `BaseTransform`.
The input and output types of transformations are both dict. A simple example is as follow:

```python
>>> from mmgen.datasets.transforms import LoadPairedImageFromFile
>>> transforms = LoadPairedImageFromFile(
>>>     key='pair',
>>>     domain_a='horse',
>>>     domain_b='zebra',
>>>     flag='color'),
>>> data_dict = {'pair_path': './data/pix2pix/facades/train/1.png'}
>>> data_dict = transforms(data_dict)
>>> print(data_dict.keys())
dict_keys(['pair_path', 'pair', 'pair_ori_shape', 'img_mask', 'img_photo', 'img_mask_path', 'img_photo_path', 'img_mask_ori_shape', 'img_photo_ori_shape'])
```

Generally, the last step of the transforms pipeline must be `PackGenInputs`.
`PackGenInputs` will pack the processed data into a dict containing two fields: `inputs` and `data_samples`.
`inputs` is the variable you want to use as the model's input, which can be the type of `torch.Tensor`, dict of `torch.Tensor`, or any type you want.
`data_samples` is a list of `GenDataSample`. Each `GenDataSample` contains groundtruth and necessary information for corresponding input.

Here is a pipeline example for Pix2Pix training on aerial2maps dataset.

```python
source_domain = 'aerial'
target_domain = 'map'

pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': [f'img_{domain_a}', f'img_{domain_b}']},
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(
                type='mmgen.Resize', scale=(286, 286),
                interpolation='bicubic'),
            dict(type='mmgen.FixedCrop', crop_size=(256, 256))
        ]),
    dict(
        type='Flip',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        direction='horizontal'),
    dict(
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}', 'pair'],
        meta_keys=[
            'pair_path', 'sample_idx', 'pair_ori_shape',
            f'img_{domain_a}_path', f'img_{domain_b}_path',
            f'img_{domain_a}_ori_shape', f'img_{domain_b}_ori_shape', 'flip',
            'flip_direction'
        ])
]
```

## Customization data transformation

The customized data transformation must inherinted from `BaseTransform` and implement `transform` function.
Here we use a simple flipping transformation as example:

```python
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

Thus, we can instantiate a `MyFlip` object and use it to process the data dict.

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

Or, we can use `MyFlip` transformation in data pipeline in our config file.

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

Note that if you want to use `MyFlip` in config, you must ensure the file containing `MyFlip` is imported during the program run.
