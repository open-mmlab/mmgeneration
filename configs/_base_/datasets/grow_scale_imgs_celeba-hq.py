dataset_type = 'GrowScaleImgDataset'

# TODO: do we use flip in test/val config
pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackGenInputs')
]

train_dataloader = dict(
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_roots={
            '64': './data/celebahq/imgs_64',
            '256': './data/celebahq/imgs_256',
            '512': './data/celebahq/imgs_512',
            '1024': './data/celebahq/imgs_1024'
        },
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=300000,
        pipeline=pipeline),
    sampler=dict(type='InfiniteSampler', shuffle=True))

val_dataloader = test_dataloader = train_dataloader
