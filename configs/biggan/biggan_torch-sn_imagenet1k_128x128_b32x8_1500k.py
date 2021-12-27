_base_ = [
    '../_base_/models/biggan/biggan_128x128.py',
    '../_base_/datasets/imagenet_noaug_128.py', '../_base_/default_runtime.py'
]

# define dataset
# you must set `samples_per_gpu`
data = dict(samples_per_gpu=32, workers_per_gpu=8)

model = dict(
    generator=dict(sn_style='torch'), discriminator=dict(sn_style='torch'))

# adjust running config
lr_config = None
checkpoint_config = dict(interval=5000, by_epoch=False, max_keep_ckpts=10)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=10000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=8,
        start_iter=160000,
        interp_cfg=dict(momentum=0.9999, momentum_nontrainable=0.9999),
        priority='VERY_HIGH')
]

# Traning sets' datasize 1,281,167
total_iters = 1500000

# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)

# Note set your inception_pkl's path
inception_pkl = 'work_dirs/inception_pkl/imagenet.pkl'
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(
            type='FID',
            num_images=50000,
            inception_pkl=inception_pkl,
            bgr2rgb=True),
        dict(type='IS', num_images=50000)
    ],
    sample_kwargs=dict(sample_model='ema'),
    best_metric=['fid', 'is'])

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=50000,
        inception_pkl=inception_pkl,
        bgr2rgb=True,
        inception_args=dict(type='StyleGAN')),
    is50k=dict(type='IS', num_images=50000))
