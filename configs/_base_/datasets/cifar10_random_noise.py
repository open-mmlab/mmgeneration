dataset_type = 'mmcls.CIFAR10'

# cifar dataset without augmentation
# different from mmcls, we adopt the setting used in BigGAN
# Note that the pipelines below are from MMClassification. Importantly, the
# `to_rgb` is set to `True` to convert image to BGR orders. The default order
# in Cifar10 is RGB. Thus, we have to convert it to BGR.

# Follow the pipeline in
# https://github.com/pfnet-research/sngan_projection/blob/master/datasets/cifar10.py
# Only `RandomImageNoise` augmentation is adopted.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomImgNoise', keys=['img']),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# Different from the classification task, the val/test split also use the
# training part, which is the same to StyleGAN-ADA.
data = dict(
    samples_per_gpu=None,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type, data_prefix='data/cifar10',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline),
    test=dict(
        type=dataset_type, data_prefix='data/cifar10', pipeline=test_pipeline))
