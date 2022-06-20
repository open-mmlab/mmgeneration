# dataset settings
dataset_type = 'PairedImageDataset'
domain_a = 'photo'
domain_b = 'mask'
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(286, 286),
        interpolation='bicubic'),
    dict(
        type='FixedCrop',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        crop_size=(256, 256)),
    dict(
        type='Flip',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        direction='horizontal')
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='image',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='Resize',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        scale=(256, 256),
        interpolation='bicubic')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
