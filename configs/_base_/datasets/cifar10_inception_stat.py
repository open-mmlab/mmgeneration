dataset_type = 'mmcls.CIFAR10'

# This config is set for extract inception state of CIFAR dataset.

# Different from mmcls, we adopt the setting used in BigGAN.
# Note that the pipelines below are from MMClassification.
# The default order in Cifar10 is RGB. Thus, we set `to_rgb` as `False`.
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
train_pipeline = [
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
