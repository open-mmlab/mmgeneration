dataset_type = 'GrowScaleImgDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Resize', scale=(128, 128)),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackGenInputs')
]

train_dataloader = dict(
    num_workers=4,
    batch_size=None,  # initialize batch size
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        # data_roots=None,
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=-1),
    sampler=dict(type='InfiniteSampler', shuffle=True))

val_dataloader = test_dataloader = train_dataloader
