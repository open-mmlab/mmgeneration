_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/pggan/pggan_1024.py',
    '../_base_/datasets/grow_scale_imgs_celeba-hq.py',
]

# MODEL
model_wrapper_cfg = dict(find_unused_parameters=True)

# TRAIN
train_cfg = dict(max_iters=280000)

optim_wrapper = dict(
    constructor='PGGANOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    lr_schedule=dict(
        generator={
            '128': 0.0015,
            '256': 0.002,
            '512': 0.003,
            '1024': 0.003
        },
        discriminator={
            '128': 0.0015,
            '256': 0.002,
            '512': 0.003,
            '1024': 0.003
        }))

# VIS_HOOK + DATAFETCH
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=1,
        fixed_input=True,
        sample_kwargs_list=dict(type='GAN', name='fake_img')),
    dict(type='PGGANFetchDataHook')
]

# METRICS
metrics = [
    dict(
        type='SWD', fake_nums=16384, image_shape=(3, 1024, 1024),
        prefix='SWD'),
    dict(type='MS_SSIM', fake_nums=10000, prefix='MS-SSIM')
]

# do not evaluate in training
val_cfg = val_evaluator = val_dataloader = None
test_evaluator = dict(metrics=metrics)
