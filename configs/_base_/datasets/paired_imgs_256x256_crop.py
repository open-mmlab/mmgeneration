# dataset settings
train_dataset_type = 'PairedImageDataset'
val_dataset_type = 'PairedImageDataset'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
style_a = 'photo'
style_b = 'mask'
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        style_a=style_a,
        style_b=style_b,
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='FixedCrop',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        crop_size=(256, 256)),
    dict(
        type='Flip',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=[f'img_{style_a}', f'img_{style_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        to_rgb=False,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=[f'img_{style_a}', f'img_{style_b}']),
    dict(
        type='Collect',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        meta_keys=[f'img_{style_a}_path', f'img_{style_b}_path'])
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='image',
        style_a=style_a,
        style_b=style_b,
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        scale=(256, 256),
        interpolation='bicubic'),
    dict(type='RescaleToZeroOne', keys=[f'img_{style_a}', f'img_{style_b}']),
    dict(
        type='Normalize',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        to_rgb=False,
        **img_norm_cfg),
    dict(type='ImageToTensor', keys=[f'img_{style_a}', f'img_{style_b}']),
    dict(
        type='Collect',
        keys=[f'img_{style_a}', f'img_{style_b}'],
        meta_keys=[f'img_{style_a}_path', f'img_{style_b}_path'])
]

data = dict(
    samples_per_gpu=1,
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
