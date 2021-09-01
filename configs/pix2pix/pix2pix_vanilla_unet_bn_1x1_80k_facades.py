_base_ = [
    '../_base_/models/pix2pix_vanilla_unet_bn.py',
    '../_base_/datasets/paired_imgs_256x256_crop.py',
    '../_base_/default_runtime.py'
]
train_cfg = dict(direction='b2a')
test_cfg = dict(direction='b2a')
dataroot = 'data/paired/facades'
data = dict(
    train=dict(dataroot=dataroot),
    val=dict(dataroot=dataroot),
    test=dict(dataroot=dataroot))
# optimizer
optimizer = dict(
    generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))

# learning policy
lr_config = None

# checkpoint saving
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['fake_b'],
        interval=5000)
]
runner = None
use_ddp_wrapper = True

# runtime settings
total_iters = 80000
workflow = [('train', 1)]
exp_name = 'pix2pix_facades'
work_dir = f'./work_dirs/experiments/{exp_name}'
metrics = dict(
    FID=dict(type='FID', num_images=106, image_shape=(3, 256, 256)),
    IS=dict(type='IS', num_images=106, image_shape=(3, 256, 256)))
