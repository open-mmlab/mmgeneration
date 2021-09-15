_base_ = [
    '../_base_/models/cyclegan/cyclegan_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_runtime.py'
]
train_cfg = dict(buffer_size=50)
test_cfg = None

domain_a = 'photo'
domain_b = 'mask'
model = dict(
    default_domain=domain_a,
    reachable_domains=[domain_a, domain_b],
    related_domains=[domain_a, domain_b],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_b}',
                target=f'real_{domain_b}',
            ),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_b}', target=f'real_{domain_b}'),
            reduction='mean')
    ])
dataroot = './data/unpaired_facades'
data = dict(
    train=dict(dataroot=dataroot),
    val=dict(dataroot=dataroot),
    test=dict(dataroot=dataroot))

optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=40000, interval=400)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='training_samples',
        res_name_list=['fake_photo'],
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
