# model settings
model = dict(
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
    pixel_loss=dict(type='L1Loss', loss_weight=100.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(direction='a2b')  # model default: a2b
test_cfg = dict(direction='a2b', show_input=True)
