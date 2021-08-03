# dataset settings
dataset_type = 'mmcls.ImageNet'

# This config is set for extract inception state of ImageNet dataset.
# Following the pipeline of BigGAN, we center crop and resize images to 128x128
# before feeding them to the Inception Net. Please refer to
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/utils/prepare_data.sh
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/make_hdf5.py
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/calculate_inception_moments.py  # noqa

# Importantly, the `to_rgb` is set to `True` to convert image orders to RGB.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(128, 128), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCropLongEdge', keys=['img']),
    dict(type='Resize', size=(128, 128), backend='pillow'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

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
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=test_pipeline))
