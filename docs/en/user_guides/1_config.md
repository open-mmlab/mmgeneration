# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

## Modify config through script arguments

When submitting jobs using "tools/train.py" or "tools/test.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options test_cfg.use_ema=False` changes the default sampling model to the original generator.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromWebcam'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromWebcam`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark " is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config File Structure

There are 3 basic component types under `config/_base_`, dataset, model, default_runtime.
Many methods could be easily constructed with one of each like StyleGAN2, CycleGAN, SinGAN.
Configs consisting of components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made base on StyleGAN2, user may first inherit the basic StyleGAN2 structure by specifying `_base_ = ../styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `xxxgan` under `configs`,

Please refer to [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{model}_[module setting]_{training schedule}_{dataset}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `stylegan`, `dcgan`, etc. Settings referred in the original paper are included in this field as well (e.g., Stylegan2-config-f)
- `[module setting]`: specific setting for some modules, including learning rate (e.g., `Glr4e-4_Dlr1e-4` for dcgan), loss terms (`gamma32.8` for stylegan3) and network structures (e.g., `woReLUInplace` in sagan). In this section, information from different submodules (e.g., generator and discriminator) are connected with `_`.
- `[batch_per_gpu x gpu]`: GPUs and samples per GPU, `b4x8` is used by default in stylegan2.
- `{schedule}`: training schedule. Including learning rate (e.g., `lr1e-3`), number of gpu and batch size is used (e.g., `8xb32`), and total iterations (e.g., `160kiter`) or number of images shown in the discriminator (e.g., `12Mimgs`).
- `{dataset}`: dataset like `ffhq`, `lsun-car`, `celeba-hq`.

## An Example of StyleGAN2

To help the users have a basic idea of a complete config and the modules in a modern GAN model,
we make brief comments on the config of Stylegan2 at 256x256 scale.
For more detailed usage and the corresponding alternative for each module, please refer to the API documentation and the [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md).

```python

_base_ = [
    '../_base_/datasets/ffhq_flip.py',
    '../_base_/models/stylegan/stylegan2_base.py',
    '../_base_/default_runtime.py'
]  # base config file which we build new config file on.

# reg params
d_reg_interval = 16
g_reg_interval = 4

g_reg_ratio = g_reg_interval / (g_reg_interval + 1)
d_reg_ratio = d_reg_interval / (d_reg_interval + 1)

ema_half_life = 10.  # G_smoothing_kimg

# Define model
model = dict(
    generator=dict(out_size=1024),
    discriminator=dict(in_size=1024),
    ema_config=dict(
        type='ExponentialMovingAverage',
        interval=1,
        momentum=0.5**(32. / (ema_half_life * 1000.))),
    loss_config=dict(
        r1_loss_weight=10. / 2. * d_reg_interval,
        r1_interval=d_reg_interval,
        norm_mode='HWC',
        g_reg_interval=g_reg_interval,
        g_reg_weight=2. * g_reg_interval,
        pl_batch_shrink=2))

# define training config required by runner
train_cfg = dict(max_iters=800002)  # total iterations to train the model

# define optimizator for generator and discriminator
optim_wrapper = dict(
    generator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * g_reg_ratio, betas=(0,
                                                        0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * d_reg_ratio, betas=(0,
                                                        0.99**d_reg_ratio))))

batch_size = 4
data_root = './data/ffhq/images'

train_dataloader = dict(
    batch_size=batch_size,  # specify the number of samples on each GPU
    dataset=dict(data_root=data_root))

val_dataloader = dict(
    batch_size=batch_size,  # specify the number of samples on each GPU
    dataset=dict(data_root=data_root))

test_dataloader = dict(
    batch_size=batch_size,  # specify the number of samples on each GPU
    dataset=dict(data_root=data_root))

# add customized hooks for training
custom_hooks = [
    dict(
        type='GenVisualizationHook',  # visualize training samples for GANs
        interval=5000,  # the interval of calling this hook
        fixed_input=True,  # whether fix the input when generate images
        vis_kwargs_list=dict(type='GAN', name='fake_img')  # pre-defined visualize config for GAN models
    )
]

# metrics we used to test this model
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        inception_pkl='work_dirs/inception_pkl/ffhq256-full.pkl',  # provide the inception pkl for FID
        sample_model='ema'  # use ema model to evaluate the metric
    ),
    dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K'),
    dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
]
default_hooks = dict(checkpoint=dict(save_best='FID-Full-50k/fid'))  # save checkpoint has the best FID metric

val_evaluator = dict(metrics=metrics)  # define metric in evaluation process
test_evaluator = dict(metrics=metrics)  # define metric in test process
```
