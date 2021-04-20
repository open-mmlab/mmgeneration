dataset_type = 'GrowScaleImgDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(type='Flip', keys=['real_img'], direction='horizontal'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

# `samples_per_gpu` and `imgs_root` need to be set.
data = dict(
    # samples per gpu should be the same as the first scale, e.g. '4': 64
    # in this case
    samples_per_gpu=None,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # just an example
        imgs_roots={
            '64': './data/celebahq/imgs_64',
            '256': './data/celebahq/imgs_256',
            '512': './data/celebahq/imgs_512',
            '1024': './data/celebahq/imgs_1024'
        },
        pipeline=train_pipeline,
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
        len_per_stage=300000))
