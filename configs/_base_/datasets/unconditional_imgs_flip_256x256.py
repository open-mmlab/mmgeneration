dataset_type = 'UnconditionalImageDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='real_img', io_backend='disk'),
    dict(type='Resize', keys=['real_img'], scale=(256, 256)),
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
    samples_per_gpu=None,
    workers_per_gpu=4,
    train=dict(type=dataset_type, imgs_root=None, pipeline=train_pipeline))
