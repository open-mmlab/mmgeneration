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
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='GrowScaleImgDataset',
        imgs_roots=dict({
            '1024': './data/ffhq/images',
            '256': './data/ffhq/ffhq_imgs/ffhq_256',
            '64': './data/ffhq/ffhq_imgs/ffhq_64'
        }),
        pipeline=train_pipeline,
        gpu_samples_base=4,
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4,
            '128': 4,
            '256': 4,
            '512': 4,
            '1024': 4
        },
        len_per_stage=300000))
