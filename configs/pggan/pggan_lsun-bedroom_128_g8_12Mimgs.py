_base_ = [
    '../_base_/models/pggan/pggan_128x128.py',
    '../_base_/datasets/grow_scale_imgs_128x128.py',
    '../_base_/default_runtime.py'
]

optimizer = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)

data = dict(
    samples_per_gpu=64,
    train=dict(
        imgs_roots={'128': './data/lsun/bedroom_train'},
        gpu_samples_base=4,
        # note that this should be changed with total gpu number
        gpu_samples_per_scale={
            '4': 64,
            '8': 32,
            '16': 16,
            '32': 8,
            '64': 4
        },
    ))

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(type='PGGANFetchDataHook', interval=1),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        priority='VERY_HIGH')
]

lr_config = None

total_iters = 280000

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 128, 128)))
