# Tutorial 2: Customize Datasets

In this section, we will detail how to prepare data and adopt proper dataset in our repo for different methods.
## Datasets for unconditional models


**Data preparation for unconditional model** is simple. What you need to do is downloading the images and put them into a directory. Next, you should set a symlink in the `data` directory. For standard unconditional gans with static architectures, like DCGAN and StyleGAN2, `UnconditionalImageDataset` is designed to train such unconditional models. Here is an example config for FFHQ dataset:

```python
dataset_type = 'UnconditionalImageDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(type='Flip', keys=['real_img'], direction='horizontal'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

# `samples_per_gpu` and `imgs_root` need to be set.
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=dataset_type,
            imgs_root='data/ffhq/images',
            pipeline=train_pipeline)))

```
Here, we adopt `RepeatDataset` to avoid frequent dataloader reloading, which will accelerate the training procedure. As shown in the example, `pipeline` provides important data pipeline to process images, including loading from file system, resizing, cropping and transferring to `torch.Tensor`. All of supported data pipelines can be found in `mmgen/datasets/pipelines`.

For unconditional GANs with dynamic architectures like PGGAN and StyleGANv1, `GrowScaleImgDataset` is recommended to use for training. Since such dynamic architectures need real images in different scales, directly adopting `UnconditionalImageDataset` will bring heavy I/O cost for loading multiple high-resolution images. Here is an example we use for training PGGAN in CelebA-HQ dataset:

```python
dataset_type = 'GrowScaleImgDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(type='Flip', keys=['real_img'], direction='horizontal'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

# `samples_per_gpu` and `imgs_root` need to be set.
data = dict(
    # samples per gpu should be the same as the first scale, e.g. '4': 64
    # in this case
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # just an example
        imgs_roots={
            '64': './data/celebahq/imgs_64',
            '256': './data/celebahq/imgs_256',
            '512': './data/celebahq/imgs_512',
            '1024': './data/celebahq/imgs_1024'
        },
        pipeline=train_pipeline,
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=300000))
```
In this dataset, you should provide a dictionary of image paths to the `imgs_roots`. Thus, you should resize the images in the dataset in advance. For the resizing methods in the data pre-processing, we adopt bilinear interpolation methods in all of the experiments studied in MMGeneration.

Note that this dataset should be used with `PGGANFetchDataHook`. In this config file, this hook should be added in the customized hooks, as shown below.

```python
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(type='PGGANFetchDataHook', interval=1),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        priority='VERY_HIGH')
]
```
This fetching data hook helps the dataloader update the status of dataset to change the data source and batch size during training.

## Datasets for image translation models
**Data preparation for translation model** needs a little attention. You should organize the files in the way we told you in `quick_run.md`. Fortunately, for most official datasets like facades and summer2winter_yosemite, they already have the right format. Also, you should set a symlink in the `data` directory. For paired-data trained translation model like Pix2Pix , `PairedImageDataset` is designed to train such translation models. Here is an example config for facades dataset:

```python
train_dataset_type = 'PairedImageDataset'
val_dataset_type = 'PairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(type='FixedCrop', keys=['img_a', 'img_b'], crop_size=(256, 256)),
    dict(type='Flip', keys=['img_a', 'img_b'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize',
        keys=['img_a', 'img_b'],
        to_rgb=False,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize',
        keys=['img_a', 'img_b'],
        to_rgb=False,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
dataroot = 'data/paired/facades'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=train_dataset_type,
        dataroot=dataroot,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=val_dataset_type,
        dataroot=dataroot,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        dataroot=dataroot,
        pipeline=test_pipeline,
        test_mode=True))
```

Here, we adopt `LoadPairedImageFromFile` to load a paired image as the common loader does and crops
it into two images with the same shape in different domains. As shown in the example, `pipeline` provides important data pipeline to process images, including loading from file system, resizing, cropping, flipping and transferring to `torch.Tensor`. All of supported data pipelines can be found in `mmgen/datasets/pipelines`.

For unpaired-data trained translation model like CycleGAN , `UnpairedImageDataset` is designed to train such translation models. Here is an example config for horse2zebra dataset:

```python
train_dataset_type = 'UnpairedImageDataset'
val_dataset_type = 'UnpairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_a',
        flag='color'),
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_b',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=['img_a', 'img_b'],
        crop_size=(256, 256),
        random_crop=True),
    dict(type='Flip', keys=['img_a'], direction='horizontal'),
    dict(type='Flip', keys=['img_b'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize',
        keys=['img_a', 'img_b'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_a',
        flag='color'),
    dict(
        type='LoadImageFromFile', io_backend='disk', key='img_b',
        flag='color'),
    dict(
        type='Resize',
        keys=['img_a', 'img_b'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize',
        keys=['img_a', 'img_b'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['img_a', 'img_b']),
    dict(
        type='Collect',
        keys=['img_a', 'img_b'],
        meta_keys=['img_a_path', 'img_b_path'])
]
data_root = './data/horse2zebra/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
    val_samples_per_gpu=1,
    val_workers_per_gpu=0,
    train=dict(
        type=train_dataset_type,
        dataroot=data_root,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True))

```
Here, `UnpairedImageDataset` will load both images (domain A and B) from different paths and transform them at the same time.
