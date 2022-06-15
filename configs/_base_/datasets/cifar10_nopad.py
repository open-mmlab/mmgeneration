custom_imports = dict(
    imports=['mmcls.datasets.pipelines'], allow_failed_imports=False)
dataset_type = 'mmcls.CIFAR10'
cifar_pipeline = [
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.PackClsInputs')
]
cifar_dataset = dict(
    type=dataset_type,
    data_prefix='data/cifar10',
    test_mode=False,
    pipeline=cifar_pipeline)

train_dataloader = dict(
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    persistent_workers=True)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=cifar_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
