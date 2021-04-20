metrics = dict(
    fid50k=dict(type='FID', num_images=50000),
    pr50k3=dict(type='PR', num_images=50000, k=3),
    is50k=dict(type='IS', num_images=50000),
    ppl_zfull=dict(type='PPL', space='Z', sampling='full', num_images=50000),
    ppl_wfull=dict(type='PPL', space='W', sampling='full', num_images=50000),
    ppl_zend=dict(type='PPL', space='Z', sampling='end', num_images=50000),
    ppl_wend=dict(type='PPL', space='W', sampling='end', num_images=50000),
    ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    swd16k=dict(type='SWD', num_images=16384))
