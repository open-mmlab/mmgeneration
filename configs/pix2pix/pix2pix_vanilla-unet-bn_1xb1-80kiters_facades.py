_base_ = [
    '../_base_/models/pix2pix/pix2pix_vanilla_unet_bn.py',
    '../_base_/datasets/paired_imgs_256x256_crop.py',
    '../_base_/default_runtime.py'
]
source_domain = 'mask'
target_domain = 'photo'
# model settings
model = dict(
    default_domain=target_domain,
    reachable_domains=[target_domain],
    related_domains=[target_domain, source_domain])

train_cfg = dict(max_iters=80000)

# dataset settings
domain_a = target_domain
domain_b = source_domain
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': [f'img_{domain_a}', f'img_{domain_b}']},
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(
                type='mmgen.Resize', scale=(286, 286),
                interpolation='bicubic'),
            dict(type='mmgen.FixedCrop', crop_size=(256, 256))
        ]),
    dict(
        type='Flip',
        keys=[f'img_{domain_a}', f'img_{domain_b}'],
        direction='horizontal'),
    dict(
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}', 'pair'],
        meta_keys=[
            'pair_path', 'sample_idx', 'pair_ori_shape',
            f'img_{domain_a}_path', f'img_{domain_b}_path',
            'img_photo_ori_shape', 'img_mask_ori_shape', 'flip',
            'flip_direction'
        ])
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        key='pair',
        domain_a=domain_a,
        domain_b=domain_b,
        flag='color'),
    dict(
        type='TransformBroadcaster',
        mapping={'img': [f'img_{domain_a}', f'img_{domain_b}']},
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(
                type='mmgen.Resize', scale=(256, 256), interpolation='bicubic')
        ]),
    dict(
        type='PackGenInputs',
        keys=[f'img_{domain_a}', f'img_{domain_b}', 'pair'],
        meta_keys=[f'img_{domain_a}_path', f'img_{domain_b}_path'])
]

dataroot = 'data/pix2pix/facades'
train_dataloader = dict(
    dataset=dict(data_root=dataroot, pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(test_mode=True, data_root=dataroot, pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(test_mode=True, data_root=dataroot, pipeline=test_pipeline))

# optimizer
optim_wrapper = dict(
    generators=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))),
    discriminators=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))))

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

fake_nums = 106
metrics = [
    dict(
        type='TransIS',
        prefix='IS-Full',
        fake_nums=fake_nums,
        inception_style='PyTorch',
        fake_key=f'fake_{target_domain}',
        sample_model='orig'),
    dict(
        type='TransFID',
        prefix='FID-Full',
        fake_nums=fake_nums,
        inception_style='PyTorch',
        real_key=f'img_{target_domain}',
        fake_key=f'fake_{target_domain}',
        sample_model='orig')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
