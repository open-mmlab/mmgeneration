_base_ = [
    '../_base_/models/cyclegan_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_runtime.py'
]
train_cfg = dict(direction='b2a', buffer_size=50)
test_cfg = dict(test_direction='b2a', show_input=False)
model = dict(id_loss=dict(type='L1Loss', loss_weight=0, reduction='mean'))
dataroot = './data/unpaired_facades'
data = dict(
    train=dict(dataroot=dataroot),
    val=dict(dataroot=dataroot),
    test=dict(dataroot=dataroot))

optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))
lr_config = None
checkpoint_config = dict(interval=100, save_optimizer=True, by_epoch=False)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])

total_iters = 80000
cudnn_benchmark = True
workflow = [('train', 1)]
exp_name = 'cyclegan_facades_id0'
work_dir = f'./work_dirs/{exp_name}'
metrics = dict(
    FID=dict(type='FID', num_images=106, image_shape=(3, 256, 256)),
    IS=dict(type='IS', num_images=106, image_shape=(3, 256, 256)))
