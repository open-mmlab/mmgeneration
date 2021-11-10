# In this config, we follow the setting `launch_SAGAN_bz128x2_ema.sh` from
# BigGAN's repo. Please refer to https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/launch_SAGAN_bs128x2_ema.sh  # noqa
# In summary, in this config:
# 1. use eps=1e-8 for Spectral Norm
# 2. not use syncBN
# 3. not use  Spectral Norm for embedding layers in cBN
# 4. start EMA at iterations
# 5. use xavier_uniform for weight initialization
# 6. no data augmentation

_base_ = [
    '../_base_/models/sagan/sagan_128x128.py',
    '../_base_/datasets/imagenet_noaug_128.py', '../_base_/default_runtime.py'
]

init_cfg = dict(type='BigGAN')
model = dict(
    num_classes=1000,
    generator=dict(
        num_classes=1000,
        init_cfg=init_cfg,
        norm_eps=1e-5,
        sn_eps=1e-8,
        auto_sync_bn=False,
        with_embedding_spectral_norm=False),
    discriminator=dict(num_classes=1000, init_cfg=init_cfg, sn_eps=1e-8),
)

n_disc = 1
train_cfg = dict(disc_step=n_disc, use_ema=True)

lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema'),
        interval=n_disc,
        start_iter=2000 * n_disc,
        interp_cfg=dict(momentum=0.999),
        priority='VERY_HIGH')
]

inception_pkl = './work_dirs/inception_pkl/imagenet.pkl'

evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(
            type='FID',
            num_images=50000,
            inception_pkl=inception_pkl,
            bgr2rgb=True,
            inception_args=dict(type='StyleGAN')),
        dict(type='IS', num_images=50000)
    ],
    best_metric=['fid', 'is'],
    sample_kwargs=dict(sample_model='ema'))

total_iters = 1000000 * n_disc
# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        inception_args=dict(type='StyleGAN')),
    IS50k=dict(type='IS', num_images=50000))

optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999)),
    discriminator=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999)))

# train on 8 gpus
data = dict(samples_per_gpu=32)
