# dataset settings
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
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['img_a', 'img_b']),
    dict(
        type='Normalize', keys=['img_a', 'img_b'], to_rgb=True,
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=train_dataset_type,
        dataroot=None,
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(
        type=val_dataset_type,
        dataroot=None,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=val_dataset_type,
        dataroot=None,
        pipeline=test_pipeline,
        test_mode=True))
