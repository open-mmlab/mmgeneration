_base_ = [
    '../_base_/datasets/unconditional_imgs_128x128.py',
    '../_base_/models/wgangp/wgangp_base.py'
]

data = dict(
    samples_per_gpu=64,
    train=dict(imgs_root='./data/celeba-cropped/cropped_images_aligned_png/'))

checkpoint_config = dict(interval=10000, by_epoch=False)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)
]

lr_config = None
total_iters = 160000

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 128, 128)))
