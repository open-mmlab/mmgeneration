model = dict(
    type='BasicGaussianDiffusion',
    num_timesteps=4000,
    data_preprocessor=dict(type='GANDataPreprocessor'),
    betas_cfg=dict(type='cosine'),
    denoising=dict(
        type='DenoisingUnet',
        image_size=32,
        in_channels=3,
        base_channels=128,
        resblocks_per_downsample=3,
        attention_res=[16, 8],
        use_scale_shift_norm=True,
        dropout=0.3,
        num_heads=4,
        use_rescale_timesteps=True,
        output_cfg=dict(mean='eps', var='learned_range')),
    ema_config=dict(momentum=0.9999),
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
