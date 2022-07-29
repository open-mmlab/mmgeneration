_base_ = [
    '../_base_/models/cyclegan/cyclegan_lsgan_resnet.py',
    '../_base_/datasets/unpaired_imgs_256x256.py',
    '../_base_/default_runtime.py'
]
train_cfg = dict(max_iters=80000)

domain_a = 'photo'
domain_b = 'mask'
model = dict(
    loss_config=dict(cycle_loss_weight=10., id_loss_weight=0.),
    default_domain=domain_a,
    reachable_domains=[domain_a, domain_b],
    related_domains=[domain_a, domain_b])

param_scheduler = dict(
    type='LinearLrInterval',
    interval=400,
    by_epoch=False,
    start_factor=0.0002,
    end_factor=0,
    begin=40000,
    end=80000)

dataroot = './data/unpaired_facades'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': [f'img_{domain_a}', f'img_{domain_b}']},
        auto_remap=True,
        share_random_params=True,
        transforms=dict(
            type='Resize', scale=(286, 286), interpolation='bicubic'),
    ),
    dict(
        type='Crop',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        crop_size=(256, 256),
        random_crop=True),
    dict(type='Flip', keys=[f'img_{domain_a}'], direction='horizontal'),
    dict(type='Flip', keys=[f'img_{domain_b}'], direction='horizontal'),
    dict(
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_a}',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key=f'img_{domain_b}',
        flag='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': [f'img_{domain_a}', f'img_{domain_b}']},
        auto_remap=True,
        share_random_params=True,
        transforms=dict(
            type='Resize', scale=(286, 286), interpolation='bicubic'),
    ),
    dict(
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=dataroot,
        pipeline=train_pipeline,
        domain_a=domain_a,
        domain_b=domain_b))

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=dataroot,
        pipeline=test_pipeline,
        test_mode=True,
        domain_a=domain_a,
        domain_b=domain_b))

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=dataroot,
        pipeline=test_pipeline,
        test_mode=True,
        domain_a=domain_a,
        domain_b=domain_b))

optim_wrapper = dict(
    generators=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))),
    discriminators=dict(
        optimizer=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999))))

custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=[
            dict(type='Translation', name='trans'),
            dict(type='TranslationVal', name='trans_val')
        ])
]

# learning policy
num_images = 106
metrics = [
    dict(
        type='TransIS',
        prefix='IS-Full',
        fake_nums=num_images,
        fake_key=f'fake_{domain_a}',
        inception_style='PyTorch'),
    dict(
        type='TransFID',
        prefix='FID-Full',
        fake_nums=num_images,
        inception_style='PyTorch',
        real_key=f'img_{domain_a}',
        fake_key=f'fake_{domain_a}')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
