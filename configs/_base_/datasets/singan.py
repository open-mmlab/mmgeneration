dataset_type = 'SinGANDataset'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=False,
    train=dict(
        type=dataset_type,
        img_path=None,  # need to set
        min_size=25,
        max_size=250,
        scale_factor_init=0.75))
