model = dict(
    type='BasicGaussianDiffusion',
    num_timesteps=1000,
    data_preprocessor=dict(type='GANDataPreprocessor'),
    betas_cfg=dict(type='cosine'),
    denoising=dict(
        type='DenoisingUnet',
        image_size=64,
        in_channels=3,
        base_channels=192,
        resblocks_per_downsample=3,
        attention_res=(32, 16, 8),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.1,
        num_classes=1000,
        use_fp16=False,
        resblock_updown=True,
        attention_cfg=dict(
            type='MultiHeadAttentionBlock',
            num_heads=4,
            num_head_channels=64,
            use_new_attention_order=True),
        use_scale_shift_norm=True),
    timestep_sampler=dict(type='UniformTimeStepSampler'),
    ddpm_loss=[
        dict(
            type='DDPMVLBLoss',
            rescale_mode='constant',
            rescale_cfg=dict(scale=4000 / 1000),
            data_info=dict(
                mean_pred='mean_pred',
                mean_target='mean_posterior',
                logvar_pred='logvar_pred',
                logvar_target='logvar_posterior'),
            log_cfgs=[
                dict(
                    type='quartile',
                    prefix_name='loss_vlb',
                    total_timesteps=4000),
                dict(type='name')
            ]),
        dict(
            type='DDPMMSELoss',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=4000),
        )
    ],
)
