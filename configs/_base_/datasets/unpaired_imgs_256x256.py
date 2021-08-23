train_dataset_type = 'UnpairedImageDataset'
val_dataset_type = 'UnpairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
style_a = 'photo'
style_b = 'mask'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{style_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{style_b}',
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='Crop',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        crop_size=(256, 256),
        random_crop=True),
    dict(type='Flip', keys=[f'img_{style_a}'], direction='horizontal'),
    dict(type='Flip', keys=[f'img_{style_b}'], direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=[f'img_{style_a}', f'img_{style_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=[f'img_{style_a}', f'img_{style_b}']),
    dict(
        type='Collect',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        meta_keys=[f'img_{style_a}_path', f'img_{style_b}_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile', io_backend='disk', key='image',
        flag='color'),
    dict(
        type='Resize',
        keys=['image'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=['image']),
    dict(
        type='Normalize',
        keys=['image'],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['image']),
    dict(type='Collect', keys=['image'], meta_keys=['image_path'])
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
        test_mode=False,
        style_a=style_a,
        style_b=style_b),
    val=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        style_a=style_a,
        style_b=style_b),
    test=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        style_a=style_a,
        style_b=style_b))
