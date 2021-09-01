# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

## Modify config through script arguments

When submitting jobs using "tools/train.py" or "tools/evaluation.py", you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options test_cfg.use_ema=False` changes the default sampling model to the original generator.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromWebcam'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=LoadImageFromWebcam`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark \" is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Config File Structure

There are 4 basic component types under `config/_base_`, dataset, model, default_metrics, default_runtime.
Many methods could be easily constructed with one of each like StyleGAN2, CycleGAN, SinGAN.
Configs consisting of components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from existing methods.
For example, if some modification is made base on StyleGAN2, user may first inherit the basic StyleGAN2 structure by specifying `_base_ = ../styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder `xxxgan` under `configs`,

Please refer to [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#config) for detailed documentation.

## Config Name Style

We follow the below style to name config files. Contributors are advised to follow the same style.

```
{model}_[model setting]_{dataset}_[batch_per_gpu x gpu]_{schedule}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type like `stylegan`, `dcgan`, etc.
- `[model setting]`: specific setting for some model, like `c2` for `stylegan2`, etc.
- `{dataset}`: dataset like `ffhq`, `lsun-car`, `celeba-hq`.
- `[batch_per_gpu x gpu]`: GPUs and samples per GPU, `b4x8` is used by default in stylegan2.
- `{schedule}`: training schedule. Following Tero's convention, we recommend to use the number of images shown to the discriminator, like 5M, 800k. Of course, you can use 5e indicating 5 epochs or 80k-iters for 80k iterations.


## An Example of StyleGAN2

To help the users have a basic idea of a complete config and the modules in a modern detection system,
we make brief comments on the config of Stylegan2 at 256x256 scale.
For more detailed usage and the corresponding alternative for each module, please refer to the API documentation and the [tutorial in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md).

```python
_base_ = [
    '../_base_/datasets/ffhq_flip.py', '../_base_/models/stylegan2_base.py',
    '../_base_/default_runtime.py', '../_base_/default_metrics.py'
]  # base config file which we build new config file on.

model = dict(generator=dict(out_size=256), discriminator=dict(in_size=256))  # update the `out_size` and `in_size` arguments.

data = dict(
    samples_per_gpu=4,  # specify the number of samples on each GPU
    train=dict(dataset=dict(imgs_root='./data/ffhq/ffhq_imgs/ffhq_256')))  # provide root path for dataset

ema_half_life = 10.  # G_smoothing_kimg

custom_hooks = [  # add customized hooks for training
    dict(
        type='VisualizeUnconditionalSamples',  # visualize training samples for GANs
        output_dir='training_samples',  # define output path
        interval=5000),  # the interval of calling this hook
    dict(
        type='ExponentialMovingAverageHook',  # EMA hook for better generator
        module_keys=('generator_ema', ),  # get the ema model and the original model should be named as `generator`
        interval=1,  # the interval of calling this hook
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),  # args for updating params for ema model
        priority='VERY_HIGH')  # define the priority of this hook
]

metrics = dict(  # metrics we used to test this model
    fid50k=dict(
        inception_pkl='work_dirs/inception_pkl/ffhq-256-50k-rgb.pkl',  # provdie the inception pkl for FID
        bgr2rgb=True))  # change the order of the image channel when extracting inception features

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)  # define checkpoint hook
lr_config = None  # remove lr scheduler

log_config = dict(  # define log hook
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

total_iters = 800002  # define the total number of iterations

```

## FAQ

### Ignore some fields in the base configs

Sometimes, you may set `_delete_=True` to ignore some of fields in base configs.
You may refer to [mmcv](https://mmcv.readthedocs.io/en/latest/utils.html#inherit-from-base-config-with-ignored-fields) for simple illustration.

You may have a careful look at [this tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md) for better understanding of this feature.

### Use intermediate variables in configs

Some intermediate variables are used in the config files, like `train_pipeline`/`test_pipeline` in datasets.
It's worth noting that when modifying intermediate variables in the children configs, users need to pass the intermediate variables into corresponding fields again. An intuitive example can be found in [this tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/config.md).
