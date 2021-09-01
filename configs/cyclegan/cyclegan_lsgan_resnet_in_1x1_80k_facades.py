_base_ = [
    '../_base_/models/cyclegan_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_runtime.py'
]
train_cfg = dict(direction='b2a', buffer_size=50)
test_cfg = dict(test_direction='b2a', show_input=False)
dataroot = './data/unpaired_facades'
data = dict(
    train=dict(dataroot=dataroot),
    val=dict(dataroot=dataroot),
    test=dict(dataroot=dataroot))

optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
lr_config = None
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
total_iters = 80000
workflow = [('train', 1)]
exp_name = 'cyclegan_facades'
work_dir = f'./work_dirs/experiments/{exp_name}'
metrics = dict(
    FID=dict(type='FID', num_images=106, image_shape=(3, 256, 256)),
    IS=dict(type='IS', num_images=106, image_shape=(3, 256, 256)))
