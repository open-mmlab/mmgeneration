# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcv.runner import obj_from_dict

from mmgen.models import GANLoss, L1Loss, build_model
from mmgen.models.architectures.pix2pix import (PatchDiscriminator,
                                                UnetGenerator)


def test_pix2pix():
    # model settings
    model_cfg = dict(
        type='Pix2Pix',
        generator=dict(
            type='UnetGenerator',
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type='BN'),
            use_dropout=True,
            init_cfg=dict(type='normal', gain=0.02)),
        discriminator=dict(
            type='PatchDiscriminator',
            in_channels=6,
            base_channels=64,
            num_conv=3,
            norm_cfg=dict(type='BN'),
            init_cfg=dict(type='normal', gain=0.02)),
        gan_loss=dict(
            type='GANLoss',
            gan_type='vanilla',
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1.0),
        default_domain='photo',
        reachable_domains=['photo'],
        related_domains=['photo', 'mask'],
        gen_auxiliary_loss=dict(
            type='L1Loss',
            loss_weight=100.0,
            data_info=dict(pred='fake_photo', target='real_photo'),
            reduction='mean'))

    train_cfg = None
    test_cfg = None

    # build synthesizer
    synthesizer = build_model(
        model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    # test attributes
    assert synthesizer.__class__.__name__ == 'Pix2Pix'
    assert isinstance(synthesizer.generators['photo'], UnetGenerator)
    assert isinstance(synthesizer.discriminators['photo'], PatchDiscriminator)
    assert isinstance(synthesizer.gan_loss, GANLoss)
    assert isinstance(synthesizer.gen_auxiliary_losses[0], L1Loss)
    assert synthesizer.test_cfg is None

    # prepare data
    img_mask = torch.rand(1, 3, 256, 256)
    img_photo = torch.rand(1, 3, 256, 256)
    data_batch = {'img_mask': img_mask, 'img_photo': img_photo}

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
    domain = 'photo'
    with torch.no_grad():
        outputs = synthesizer(img_mask, target_domain=domain, test_mode=True)
    assert torch.equal(outputs['source'], data_batch['img_mask'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 256, 256)

    # test forward_train
    outputs = synthesizer(img_mask, target_domain=domain, test_mode=False)
    assert torch.equal(outputs['source'], data_batch['img_mask'])
    assert torch.is_tensor(outputs['target'])
    assert outputs['target'].size() == (1, 3, 256, 256)

    # test train_step
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    for v in ['loss_gan_d_fake', 'loss_gan_d_real', 'loss_gan_g', 'loss_l1']:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1

    assert torch.equal(outputs['results']['real_mask'], data_batch['img_mask'])
    assert torch.equal(outputs['results']['real_photo'],
                       data_batch['img_photo'])
    assert torch.is_tensor(outputs['results']['fake_photo'])
    assert outputs['results']['fake_photo'].size() == (1, 3, 256, 256)

    # test cuda
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
        data_batch_cuda['img_mask'] = img_mask.cuda()
        data_batch_cuda['img_photo'] = img_photo.cuda()

        # forward_test
        with torch.no_grad():
            outputs = synthesizer(
                data_batch_cuda['img_mask'],
                target_domain=domain,
                test_mode=True)
        assert torch.equal(outputs['source'],
                           data_batch_cuda['img_mask'].cpu())
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 256, 256)

        # test forward_train
        outputs = synthesizer(
            data_batch_cuda['img_mask'], target_domain=domain, test_mode=False)
        assert torch.equal(outputs['source'], data_batch_cuda['img_mask'])
        assert torch.is_tensor(outputs['target'])
        assert outputs['target'].size() == (1, 3, 256, 256)

        # train_step
        outputs = synthesizer.train_step(data_batch_cuda, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['results'], dict)
        for v in [
                'loss_gan_d_fake', 'loss_gan_d_real', 'loss_gan_g', 'loss_l1'
        ]:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_mask'],
                           data_batch_cuda['img_mask'].cpu())
        assert torch.equal(outputs['results']['real_photo'],
                           data_batch_cuda['img_photo'].cpu())
        assert torch.is_tensor(outputs['results']['fake_photo'])
        assert outputs['results']['fake_photo'].size() == (1, 3, 256, 256)

    # test disc_steps and disc_init_steps
    data_batch['img_mask'] = img_mask.cpu()
    data_batch['img_photo'] = img_photo.cpu()
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
        assert outputs['log_vars'].get('loss_gan_g') is None
        assert outputs['log_vars'].get('loss_l1') is None
        for v in ['loss_gan_d_fake', 'loss_gan_d_real']:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_mask'],
                           data_batch['img_mask'])
        assert torch.equal(outputs['results']['real_photo'],
                           data_batch['img_photo'])
        assert torch.is_tensor(outputs['results']['fake_photo'])
        assert outputs['results']['fake_photo'].size() == (1, 3, 256, 256)
        assert synthesizer.iteration == i + 1

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        assert synthesizer.iteration == i
        outputs = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs['log_vars'], dict)
        assert isinstance(outputs['results'], dict)
        log_check_list = [
            'loss_gan_d_fake', 'loss_gan_d_real', 'loss_gan_g', 'loss_l1'
        ]
        if i % 2 == 1:
            assert outputs['log_vars'].get('loss_gan_g') is None
            assert outputs['log_vars'].get('loss_pixel') is None
            log_check_list.remove('loss_gan_g')
            log_check_list.remove('loss_l1')
        for v in log_check_list:
            assert isinstance(outputs['log_vars'][v], float)
        assert outputs['num_samples'] == 1
        assert torch.equal(outputs['results']['real_mask'],
                           data_batch['img_mask'])
        assert torch.equal(outputs['results']['real_photo'],
                           data_batch['img_photo'])
        assert torch.is_tensor(outputs['results']['fake_photo'])
        assert outputs['results']['fake_photo'].size() == (1, 3, 256, 256)
        assert synthesizer.iteration == i + 1

    # test without pixel loss
    model_cfg_ = copy.deepcopy(model_cfg)
    model_cfg_.pop('gen_auxiliary_loss')
    synthesizer = build_model(model_cfg_, train_cfg=None, test_cfg=None)
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
    data_batch['img_mask'] = img_mask.cpu()
    data_batch['img_photo'] = img_photo.cpu()
    outputs = synthesizer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs['log_vars'], dict)
    assert isinstance(outputs['results'], dict)
    assert outputs['log_vars'].get('loss_pixel') is None
    for v in ['loss_gan_d_fake', 'loss_gan_d_real', 'loss_gan_g']:
        assert isinstance(outputs['log_vars'][v], float)
    assert outputs['num_samples'] == 1
    assert torch.equal(outputs['results']['real_mask'], data_batch['img_mask'])
    assert torch.equal(outputs['results']['real_photo'],
                       data_batch['img_photo'])
    assert torch.is_tensor(outputs['results']['fake_photo'])
    assert outputs['results']['fake_photo'].size() == (1, 3, 256, 256)
