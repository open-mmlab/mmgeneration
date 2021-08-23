# model settings
model = dict(
    type='Pix2Pix',
    default_style='photo',
    reachable_styles=['photo'],
    related_styles=['photo', 'mask'],
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
    gen_auxiliary_loss=dict(
        type='L1Loss',
        loss_weight=100.0,
        data_info=dict(pred='style_photo', target='src_photo'),
        reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = None
