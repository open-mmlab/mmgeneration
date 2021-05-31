dataset_type = 'UnconditionalImageDataset'

# To be noted that, `Resize` operation with `pillow` backend and
# `bicubic` interpolation is the must for correct IS evaluation
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(
        type='Resize',
        keys=['real_img'],
        scale=(299, 299),
        backend='pillow',
        interpolation='bicubic'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=True),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

data = dict(
    samples_per_gpu=None,
    workers_per_gpu=4,
    val=dict(type=dataset_type, imgs_root=None, pipeline=val_pipeline))
