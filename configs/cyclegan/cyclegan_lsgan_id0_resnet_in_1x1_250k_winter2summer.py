_base_ = [
    '../_base_/models/cyclegan_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_runtime.py'
]
test_cfg = dict(test_direction='b2a', show_input=False)
model = dict(id_loss=dict(type='L1Loss', loss_weight=0, reduction='mean'))
dataroot = './data/summer2winter_yosemite'
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
        res_name_list=['fake_a'],
        interval=5000)
]

runner = None
use_ddp_wrapper = True
total_iters = 250000
workflow = [('train', 1)]
exp_name = 'cyclegan_summer2winter_id0'
work_dir = f'./work_dirs/experiments/{exp_name}'
# testA: 309, testB:238
metrics = dict(
    FID=dict(type='FID', num_images=309, image_shape=(3, 256, 256)),
    IS=dict(type='IS', num_images=309, image_shape=(3, 256, 256)))
