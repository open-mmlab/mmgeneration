_base_ = ['./pggan_celeba-cropped_128_g8_12Mimgs.py']

# Overwrite data configs
data_roots = {'128': './data/lsun/images/bedroom_train'}
train_dataloader = dict(batch_size=64, dataset=dict(data_roots=data_roots))
test_dataloader = dict(dataset=dict(data_root=data_roots['128']))
