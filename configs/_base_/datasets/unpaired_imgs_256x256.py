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
data_root = None
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
