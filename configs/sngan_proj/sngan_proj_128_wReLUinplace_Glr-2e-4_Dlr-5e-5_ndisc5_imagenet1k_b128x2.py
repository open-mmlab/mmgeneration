_base_ = [
    '../_base_/models/sngan_proj/sngan_proj_128x128.py',
    '../_base_/datasets/imagenet_128.py', '../_base_/default_runtime.py'
]

num_classes = 1000
init_cfg = dict(type='studio')
model = dict(
    num_classes=num_classes,
    generator=dict(
        num_classes=num_classes,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=init_cfg),
    discriminator=dict(
        num_classes=num_classes,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=init_cfg))

n_disc = 5
train_cfg = dict(disc_steps=n_disc)
lr_config = None

checkpoint_config = dict(interval=50000, by_epoch=False, max_keep_ckpts=20)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000)
]

log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

inception_pkl = './work_dirs/inception_pkl/imagenet.pkl'

evaluation = dict(
    type='GenerativeEvalHook',
    interval=dict(milestones=[800000], interval=[10000, 4000]),
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
    sample_kwargs=dict(sample_model='orig'))

total_iters = 500000 * n_disc
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
    generator=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999)),
    discriminator=dict(type='Adam', lr=0.00005, betas=(0.0, 0.999)))

# train on 2 gpus
data = dict(samples_per_gpu=128)
