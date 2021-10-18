_base_ = [
    '../_base_/models/sagan/sagan_128x128.py',
    '../_base_/datasets/imagenet_128.py', '../_base_/default_runtime.py'
]

init_cfg = dict(type='studio')
model = dict(
    generator=dict(num_classes=1000, init_cfg=init_cfg),
    discriminator=dict(num_classes=1000, init_cfg=init_cfg),
)

lr_config = None
checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)
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

n_disc = 1
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

# train on 4 gpus
data = dict(samples_per_gpu=64)
