# Tutorial 5: How to Extract Inception State for FID Evaluation

In MMGeneration, we provide a [script](https://github.com/open-mmlab/mmgeneration/blob/master/tools/utils/inception_stat.py) to extract the inception state of the dataset. In this doc, we provide a brief introduction on how to use this script.

<!-- TOC -->

- [Load images](#load-images)
  - [Load from directory](#load-from-directory)
  - [Load with dataset config](#load-with-dataset-config)
- [Define the version of Inception Net](#define-the-version-of-inception-net)
- [Control number of images to calculate inception state](#control-number-of-images-to-calculate-inception-state)
- [Control the shuffle operation in data loading](#control-the-shuffle-operation-in-data-loading)
- [Note on inception state extraction between various code bases](#note-on-inception-state-extraction-between-various-code-bases)

<!-- TOC -->

## Load Images

We provide two ways to load real data, namely, pass the path of directory that contains real images and pass the dataset config file you want to use.

### Load from Directory

If you want to pass the path of real images, you can use `--imgsdir` arguments as the follow command.

```shell
python tools/utils/inception_stat.py --imgsdir ${IMGS_PATH} --pklname ${PKLNAME} --size ${SIZE} --flip ${FLIP}
```

Then a pre-defined pipeline will be used to load images in `${IMGS_PATH}`.

```python
pipeline = [
    dict(type='LoadImageFromFile', key='real_img'),
    dict(
        type='Resize', keys=['real_img'], scale=SIZE,
        keep_ratio=False),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=True),  # default to RGB images
    dict(type='Collect', keys=['real_img'], meta_keys=[]),
    dict(type='ImageToTensor', keys=['real_img'])
]
```

If `${FLIP}` is set as `True`, the following config of horizontal flip operation would be added to the end of the pipeline.

```python
dict(type='Flip', keys=['real_img'], direction='horizontal')
```

If you want to use a specific pipeline otherwise the pre-defined ones, you can use `--pipeline-cfg` to pass a config file contains the data pipeline you want to use.

```shell
python tools/utils/inception_stat.py --imgsdir ${IMGS_PATH} --pklname ${PKLNAME} --pipeline-cfg ${PIPELINE}
```

To be noted that, the name of the pipeline dict in `${PIPELINE}` should be fixed as `inception_pipeline`. For example,

```python
# an example of ${PIPELINE}
inception_pipeline = [
    dict(type='LoadImageFromFile', key='real_img'),
    ...
]
```

### Load with Dataset Config

If you want to use a dataset config, you can use `--data-config` arguments as the following command.

```shell
python tools/utils/inception_stat.py --data-config ${CONFIG} --pklname ${PKLNAME} --subset ${SUBSET}
```

Then a dataset will be instantiated following the `${SUBSET}` in the configs, and defaults to `test`. Take the following dataset config as example,

```python
# from `imagenet_128x128_inception_stat.py`
data = dict(
    samples_per_gpu=None,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
```

If not defined, the config in `data['test']` would be used in data loading process. If you want to extract the inception state of the training set, you can set `--subset train` in the command. Then the dataset would be built under the guidance of config in `data['train']` and images under `data/imagenet/train` and process pipeline of `train_pipeline` would be used.

## Define the Version of Inception Net

In the aforementioned command, the script will take the [PyTorch InceptionV3](https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py) by default. If you want the [Tero's InceptionV3](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt), you will need to switch to the script module:

```shell
python tools/utils/inception_stat.py --imgsdir ${IMGS_PATH} --pklname ${PKLNAME} --size ${SIZE} \
    --inception-style stylegan --inception-pth ${PATH_SCRIPT_MODULE}
```

## Control Number of Images to Calculate Inception State

In `inception_stat.py`, we provide `--num-samples` argument to control the number of images used to calculate inception state.

```shell
python tools/utils/inception_stat.py --data-config ${CONFIG} --pklname ${PKLNAME} --num-samples ${NUMS}
```

If `${NUMS}` is set as `-1`, all images in the defined dataset would be used.

## Control the Shuffle Operation in Data Loading

In `inception_stat.py`, we provide `--no-shuffle` argument to avoid the shuffle operation in images loading process. For example, you can use the following command:

```shell
python tools/utils/inception_stat.py --data-config ${CONFIG} --pklname ${PKLNAME} --no-shuffle
```

## Note on Inception State Extraction between Various Code Bases

For FID evaluation, differences between [PyTorch Studio GAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) and ours are mainly on the selection of real samples. In MMGen, we follow the pipeline of [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch), where the whole training set is adopted to extract inception statistics. Besides, we also use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.

You can download the preprocessed inception state by the following url:

- [CIFAR10](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/cifar10.pkl)
- [ImageNet1k](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/imagenet.pkl)
- [ImageNet1k-64x64](https://download.openmmlab.com/mmgen/evaluation/fid_inception_pkl/imagenet_64x64.pkl)

You can use following commands to extract those inception states by yourself as well.

```shell
# For CIFAR10
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/cifar10_inception_stat.py --pklname cifar10.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train

# For ImageNet1k
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/imagenet_128x128_inception_stat.py --pklname imagenet.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train

# For ImageNet1k-64x64
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/imagenet_64x64_inception_stat.py --pklname imagenet_64x64.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train
```
