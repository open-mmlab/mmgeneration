dataset_type = 'mmcls.CIFAR10'

# different from mmcls, we adopt the setting used in BigGAN
# Note that the pipelines below are from MMClassification. Importantly, the
# `to_rgb` is set to `True` to convert image to BGR orders. The default order
# in Cifar10 is RGB. Thus, we have to convert it to BGR.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
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
