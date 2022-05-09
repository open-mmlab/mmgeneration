# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcv.runner import obj_from_dict

from mmgen.models import (GANLoss, L1Loss, PatchDiscriminator, ResnetGenerator,
                          build_model)


def test_cyclegan():

    model_cfg = dict(
        type='CycleGAN',
        default_domain='photo',
        reachable_domains=['photo', 'mask'],
        related_domains=['photo', 'mask'],
        generator=dict(
            type='ResnetGenerator',
            in_channels=3,
            out_channels=3,
            base_channels=64,
            norm_cfg=dict(type='IN'),
            use_dropout=False,
            num_blocks=9,
            padding_mode='reflect',
            init_cfg=dict(type='normal', gain=0.02)),
        discriminator=dict(
            type='PatchDiscriminator',
            in_channels=3,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='IN'),
            init_cfg=dict(type='normal', gain=0.02)),
        gan_loss=dict(
            type='GANLoss',
            gan_type='lsgan',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1.0),
        gen_auxiliary_loss=[
            dict(
                type='L1Loss',
                loss_weight=10.0,
                data_info=dict(pred='cycle_photo', target='real_photo'),
                reduction='mean'),
            dict(
                type='L1Loss',
                loss_weight=10.0,
                data_info=dict(
                    pred='cycle_mask',
                    target='real_mask',
                ),
                reduction='mean'),
            dict(
                type='L1Loss',
                loss_weight=0.5,
                data_info=dict(pred='identity_photo', target='real_photo'),
                reduction='mean'),
            dict(
                type='L1Loss',
                loss_weight=0.5,
                data_info=dict(pred='identity_mask', target='real_mask'),
                reduction='mean')
        ])

    train_cfg = None
    test_cfg = None

    # build synthesizer
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert synthesizer.__class__.__name__ == 'CycleGAN'
    assert isinstance(synthesizer.generators['photo'], ResnetGenerator)
    assert isinstance(synthesizer.generators['mask'], ResnetGenerator)
    assert isinstance(synthesizer.discriminators['photo'], PatchDiscriminator)
    assert isinstance(synthesizer.discriminators['mask'], PatchDiscriminator)
    assert isinstance(synthesizer.gan_loss, GANLoss)
    for loss_module in synthesizer.gen_auxiliary_losses:
        assert isinstance(loss_module, L1Loss)

    # prepare data
    inputs = torch.rand(1, 3, 64, 64)
    targets = torch.rand(1, 3, 64, 64)
    data_batch = {'img_mask': inputs, 'img_photo': targets}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }

    # test forward_test
    with torch.no_grad():
        outputs = synthesizer(inputs, target_domain='photo', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_mask'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    with torch.no_grad():
        outputs = synthesizer(targets, target_domain='mask', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_photo'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    # test forward_train
    with torch.no_grad():
        outputs = synthesizer(inputs, target_domain='photo', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_mask'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    with torch.no_grad():
        outputs = synthesizer(targets, target_domain='mask', test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_photo'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 64, 64)

    # test train_step
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    for v in [
            'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
            'loss_gan_g_photo', 'loss_l1'
    ]:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['real_photo'],
                       data_batch['img_photo'])
    assert torch.equal(outputs['results']['real_mask'], data_batch['img_mask'])
    assert torch.is_tensor(outputs['results']['fake_mask'])
    assert torch.is_tensor(outputs['results']['fake_photo'])
    assert outputs['results']['fake_mask'].size() == (1, 3, 64, 64)
    assert outputs['results']['fake_photo'].size() == (1, 3, 64, 64)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        synthesizer = synthesizer.cuda()
        optimizer = {
            'generators':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(synthesizer, 'generators').parameters())),
            'discriminators':
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(
                    params=getattr(synthesizer,
                                   'discriminators').parameters()))
        }
        data_batch_cuda = copy.deepcopy(data_batch)
        data_batch_cuda['img_mask'] = inputs.cuda()
        data_batch_cuda['img_photo'] = targets.cuda()

        # forward_test
        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_mask'],
                target_domain='photo',
                test_mode=True)
        assert torch.equal(outputs['source'],
                           data_batch_cuda['img_mask'].cpu())
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_photo'],
                target_domain='mask',
                test_mode=True)
        assert torch.equal(outputs['source'],
                           data_batch_cuda['img_photo'].cpu())
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        # test forward_train
        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_mask'],
                target_domain='photo',
                test_mode=False)
        assert torch.equal(outputs['source'], data_batch_cuda['img_mask'])
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_photo'],
                target_domain='mask',
                test_mode=False)
        assert torch.equal(outputs['source'], data_batch_cuda['img_photo'])
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 64, 64)

        # train_step
        outputs = synthesizer.train_step(data_batch_cuda, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        print(outputs['log_vars'].keys())
        assert isinstance(outputs['results'], dict)
        for v in [
                'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
                'loss_gan_g_photo', 'loss_l1'
        ]:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_photo'],
                           data_batch_cuda['img_photo'].cpu())
        assert torch.equal(outputs['results']['real_mask'],
                           data_batch_cuda['img_mask'].cpu())
        assert torch.is_tensor(outputs['results']['fake_mask'])
        assert torch.is_tensor(outputs['results']['fake_photo'])
        assert outputs['results']['fake_mask'].size() == (1, 3, 64, 64)
        assert outputs['results']['fake_photo'].size() == (1, 3, 64, 64)

    # test disc_steps and disc_init_steps
    data_batch['img_mask'] = inputs.cpu()
    data_batch['img_photo'] = targets.cpu()
    train_cfg = dict(disc_steps=2, disc_init_steps=2)
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }

    # iter 0, 1
    for i in range(2):
        outputs = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['results'], dict)
        for v in ['loss_gan_g_mask', 'loss_gan_g_photo', 'loss_l1']:
            assert outputs['log_vars'].get(v) is None
        assert isinstance(outputs['log_vars']['loss_gan_d_mask'], float)
        assert isinstance(outputs['log_vars']['loss_gan_d_photo'], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_photo'],
                           data_batch['img_photo'])
        assert torch.equal(outputs['results']['real_mask'],
                           data_batch['img_mask'])
        assert torch.is_tensor(outputs['results']['fake_mask'])
        assert torch.is_tensor(outputs['results']['fake_photo'])
        assert outputs['results']['fake_mask'].size() == (1, 3, 64, 64)
        assert outputs['results']['fake_photo'].size() == (1, 3, 64, 64)
        assert synthesizer.iteration == i + 1

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        assert synthesizer.iteration == i
        outputs = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['results'], dict)
        log_check_list = [
            'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
            'loss_gan_g_photo', 'loss_l1'
        ]
        if i % 2 == 1:
            log_None_list = ['loss_gan_g_mask', 'loss_gan_g_photo', 'loss_l1']
            for v in log_None_list:
                assert outputs['log_vars'].get(v) is None
                log_check_list.remove(v)
        for v in log_check_list:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_mask'],
                           data_batch['img_mask'])
        assert torch.equal(outputs['results']['real_photo'],
                           data_batch['img_photo'])
        assert torch.is_tensor(outputs['results']['fake_mask'])
        assert torch.is_tensor(outputs['results']['fake_photo'])
        assert outputs['results']['fake_mask'].size() == (1, 3, 64, 64)
        assert outputs['results']['fake_photo'].size() == (1, 3, 64, 64)
        assert synthesizer.iteration == i + 1

    # test GAN image buffer size = 0
    data_batch['img_mask'] = inputs.cpu()
    data_batch['img_photo'] = targets.cpu()
    train_cfg = dict(buffer_size=0)
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    optimizer = {
        'generators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'generators').parameters())),
        'discriminators':
        obj_from_dict(
            optim_cfg, torch.optim,
            dict(params=getattr(synthesizer, 'discriminators').parameters()))
    }
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    for v in [
            'loss_gan_d_mask', 'loss_gan_d_photo', 'loss_gan_g_mask',
            'loss_gan_g_photo', 'loss_l1'
    ]:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['real_mask'], data_batch['img_mask'])
    assert torch.equal(outputs['results']['real_photo'],
                       data_batch['img_photo'])
    assert torch.is_tensor(outputs['results']['fake_mask'])
    assert torch.is_tensor(outputs['results']['fake_photo'])
    assert outputs['results']['fake_mask'].size() == (1, 3, 64, 64)
    assert outputs['results']['fake_photo'].size() == (1, 3, 64, 64)
    assert synthesizer.iteration == 1
