# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcv.runner import obj_from_dict
from mmengine.logging import MessageHub
from mmengine.optim import OptimWrapper, OptimWrapperDict

from mmgen.models import GANDataPreprocessor, Pix2Pix
from mmgen.models.architectures.pix2pix import (PatchDiscriminator,
                                                UnetGenerator)


def test_pix2pix():
    # model settings
    model_cfg = dict(
        data_preprocessor=GANDataPreprocessor(),
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
        default_domain='photo',
        reachable_domains=['photo'],
        related_domains=['photo', 'mask'])

    # build synthesizer
    synthesizer = Pix2Pix(**model_cfg)
    # test attributes
    assert synthesizer.__class__.__name__ == 'Pix2Pix'
    assert isinstance(synthesizer.generators['photo'], UnetGenerator)
    assert isinstance(synthesizer.discriminators['photo'], PatchDiscriminator)

    # prepare data
    img_mask = torch.rand(1, 3, 256, 256)
    img_photo = torch.rand(1, 3, 256, 256)
    data_batch = {'img_mask': img_mask, 'img_photo': img_photo}

    # prepare optimizer
    optim_cfg = dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))
    optimizer = OptimWrapperDict(
        generators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(synthesizer, 'generators').parameters()))),
        discriminators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(
                    params=getattr(synthesizer,
                                   'discriminators').parameters()))))

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
    message_hub = MessageHub.get_instance('mmgen')
    message_hub.update_info('iter', 0)
    log_vars = synthesizer.train_step(data_batch, optimizer)
    print(log_vars.keys())
    assert isinstance(log_vars, dict)
    for v in ['loss_gan_d_fake', 'loss_gan_d_real', 'loss_gan_g']:
        assert isinstance(log_vars[v].item(), float)

    # test cuda
    if torch.cuda.is_available():
        synthesizer = synthesizer.cuda()
        optimizer = OptimWrapperDict(
            generators=OptimWrapper(
                obj_from_dict(
                    optim_cfg, torch.optim,
                    dict(
                        params=getattr(synthesizer,
                                       'generators').parameters()))),
            discriminators=OptimWrapper(
                obj_from_dict(
                    optim_cfg, torch.optim,
                    dict(
                        params=getattr(synthesizer,
                                       'discriminators').parameters()))))
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
        message_hub.update_info('iter', 0)
        log_vars = synthesizer.train_step(data_batch_cuda, optimizer)
        assert isinstance(log_vars, dict)
        for v in ['loss_gan_d_fake', 'loss_gan_d_real', 'loss_gan_g']:
            assert isinstance(log_vars[v].item(), float)

    # test disc_steps and disc_init_steps
    data_batch['img_mask'] = img_mask.cpu()
    data_batch['img_photo'] = img_photo.cpu()
    synthesizer = Pix2Pix(
        **model_cfg, discriminator_steps=2, disc_init_steps=2)
    optimizer = OptimWrapperDict(
        generators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(params=getattr(synthesizer, 'generators').parameters()))),
        discriminators=OptimWrapper(
            obj_from_dict(
                optim_cfg, torch.optim,
                dict(
                    params=getattr(synthesizer,
                                   'discriminators').parameters()))))

    # iter 0, 1
    for i in range(2):
        message_hub.update_info('iter', i)
        log_vars = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(log_vars, dict)
        assert log_vars.get('loss_gan_g') is None
        for v in ['loss_gan_d_fake', 'loss_gan_d_real']:
            assert isinstance(log_vars[v].item(), float)

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        message_hub.update_info('iter', i)
        log_vars = synthesizer.train_step(data_batch, optimizer)
        assert isinstance(log_vars, dict)
        log_check_list = ['loss_gan_d_fake', 'loss_gan_d_real', 'loss_gan_g']
        if (i + 1) % 2 == 1:
            assert log_vars.get('loss_gan_g') is None
            log_check_list.remove('loss_gan_g')

        for v in log_check_list:
            assert isinstance(log_vars[v].item(), float)
