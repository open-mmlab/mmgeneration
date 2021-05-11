_base_ = [
    '../_base_/datasets/unconditional_imgs_64x64.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='StaticUnconditionalGAN',
    generator=dict(type='LSGANGenerator', output_scale=64),
    discriminator=dict(type='LSGANDiscriminator', input_scale=64),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))
train_cfg = dict(disc_steps=1)
test_cfg = None
# define dataset
# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=128,
    train=dict(imgs_root='data/lsun/bedroom_train'),
    val=dict(imgs_root='data/lsun/bedroom_train'))

optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99)),
    discriminator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.99)))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000)
]

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=dict(
        type='FID', num_images=50000, inception_pkl=None, bgr2rgb=True),
    sample_kwargs=dict(sample_model='orig'))

total_iters = 160000
# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

metrics = dict(
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 64, 64)),
    fid50k=dict(type='FID', num_images=50000, inception_pkl=None))
