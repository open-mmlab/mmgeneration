_base_ = [
    '../_base_/models/dcgan/dcgan_64x64.py',
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/default_runtime.py'
]

# define dataset
# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=128, train=dict(imgs_root='data/lsun/bedroom_train'))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=100000, by_epoch=False)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000)
]

total_iters = 1500002

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 64, 64)))
