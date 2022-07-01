_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/cifar10_noaug.py',
    '../_base_/models/sagan/sagan_32x32.py'
]

# MODEL
disc_step = 5
init_cfg = dict(type='studio')
model = dict(
    # CIFAR images are RGB, convert to BGR
    data_preprocessor=dict(rgb_to_bgr=True),
    generator=dict(init_cfg=init_cfg),
    discriminator=dict(init_cfg=init_cfg))

# TRAIN
train_cfg = dict(max_iters=100000 * disc_step)
train_dataloader = dict(batch_size=64)

optim_wrapper = dict(
    generator=dict(optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))))

# VIS_HOOK
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        sample_kwargs_list=dict(type='GAN', name='fake_img'))
]

# METRICS
metrics = [
    dict(
        type='InceptionScore',
        prefix='IS-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig'),
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
