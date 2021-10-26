_base_ = [
    '../_base_/models/dcgan/dcgan_64x64.py',
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/default_runtime.py'
]

# output single channel
model = dict(generator=dict(out_channels=1), discriminator=dict(in_channels=1))

# define dataset
# modify train_pipeline to load gray scale images
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        flag='grayscale',
        io_backend='disk'),
    dict(type='Resize', keys=['real_img'], scale=(64, 64)),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5],
        std=[127.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=128,
    train=dict(imgs_root='data/mnist_64/train', pipeline=train_pipeline),
    val=None)

# adjust running config
lr_config = None
checkpoint_config = dict(interval=500, by_epoch=False)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=100)
]

log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])

total_iters = 5000

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 64, 64)))

optimizer = dict(
    generator=dict(type='Adam', lr=0.0004, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999)))
