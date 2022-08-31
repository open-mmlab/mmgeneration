# Tutorial 1: Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
If you wish to inspect the config file, you may run `python tools/misc/print_config.py /PATH/TO/CONFIG` to see the complete config.

The structure of this guide are as follows:

- [Tutorial 1: Learn about Configs](#tutorial-1-learn-about-configs)
  - [Modify config through script arguments](#modify-config-through-script-arguments)
  - [Config File Structure](#config-file-structure)
  - [Config Name Style](#config-name-style)
  - [An Example of StyleGAN2](#an-example-of-stylegan2)
    - [Model config](#model-config)
    - [Dataset and evaluator config](#dataset-and-evaluator-config)
    - [Training and testing config](#training-and-testing-config)
    - [Optimization config](#optimization-config)
    - [Hook config](#hook-config)
    - [Runtime config](#runtime-config)

## Modify config through script arguments

When submitting jobs using `tools/train.py` or `tools/test.py`, you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict chains.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options test_cfg.use_ema=False` changes the default sampling model to the original generator.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `train_dataloader.dataset.pipeline` is normally a list
  e.g. `[dict(type='LoadImageFromFile'), ...]`. If you want to change `'LoadImageFromFile'` to `'LoadImageFromWebcam'` in the pipeline,
  you may specify `--cfg-options train_dataloader.dataset.pipeline.0.type=LoadImageFromWebcam`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. You can set `--cfg-options key="[a,b]"` or `--cfg-options key=a,b`. It also allows nested list/tuple values, e.g., `--cfg-options key="[(a,b),(c,d)]"`. Note that the quotation mark " is necessary to support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

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

To help the users have a basic idea of a complete config and the modules in a modern GAN model.
Taking [Stylegan2 at 1024x1024 scale](../../../configs/styleganv2/stylegan2_c2_8xb4-fp16-global-800kiters_quicktest-ffhq-256x256.py) as an example, we will introduce each field in the config according to different function modules.
For more detailed usage and the corresponding alternative for each module, please refer to the API documentation and the [tutorial in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/config.md).

### Model config

In MMGeneration's config, we use model to setup generation algorithm components.
In addition to neural network components such as generator, discriminator etc, it also requires `data_preprocessor`, `loss_config`, and some of them contains `ema_config`.
`data_preprocessor` is responsible for processing a batch of data output by dataloader.
`loss_config` is responsible for weight of loss terms.
`ema_config` is responsible for exponential moving average (EMA) operation for generator.

```python
model = dict(
    type='StyleGAN2',  # The name of the model
    data_preprocessor=dict(type='GANDataPreprocessor'),  # The config of data preprocessor, usually includs image normalization and padding
    generator=dict(  # The config for generator
        type='StyleGANv2Generator',  # The name of the generator
        out_size=1024,  # The output resolution of the generator
        style_channels=512),  # The number of style channels of the generator
    discriminator=dict(  # The config for discriminator
        type='StyleGAN2Discriminator',  # The name of the discriminator
        in_size=1024),  # The input resolution of the discriminator
    ema_config=dict(  # The config for EMA
        type='ExponentialMovingAverage',  # Specific the type of Average model
        interval=1,  # The interval of EMA operation
        momentum=0.9977843871238888),  # The momentum of EMA operation
    loss_config=dict(  # The config for loss terms
        r1_loss_weight=80.0,  # The weight for r1 gradient penalty
        r1_interval=16,  # The interval of r1 gradient penalty
        norm_mode='HWC',  # The normalization mode for r1 gradient penalty
        g_reg_interval=4,  # The interval for generator's regularization
        g_reg_weight=8.0,  # The weight for generator's regularization
        pl_batch_shrink=2))  # The factor of shrinking the batch size in path length regularization
```

### Dataset and evaluator config

[Dataloaders](https://pytorch.org/docs/stable/data.html?highlight=data%20loader#torch.utils.data.DataLoader) are required for the training, validation, and testing of the [runner](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).
Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs.

```python
dataset_type = 'UnconditionalImageDataset'  # Dataset type, this will be used to define the dataset
data_root = './data/ffhq/'  # Root path of data

train_pipeline = [  # Training data process pipeline
    dict(type='LoadImageFromFile', key='img'),  # First pipeline to load images from file path
    dict(type='Flip', keys=['img'], direction='horizontal'),  # Argumentation pipeline that flip the images
    dict(type='PackGenInputs', keys=['img'], meta_keys=['img_path'])  # The last pipeline that formats the annotation data (if have) and decides which keys in the data should be packed into data_samples
]
val_pipeline = [
    dict(type='LoadImageFromFile', key='img'),  # First pipeline to load images from file path
    dict(type='PackGenInputs', keys=['img'], meta_keys=['img_path'])  # The last pipeline that formats the annotation data (if have) and decides which keys in the data should be packed into data_samples
]
train_dataloader = dict(  # The config of train dataloader
    batch_size=4,  # Batch size of a single GPU
    num_workers=8,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # If ``True``, the dataloader will not shutdown the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(  # The config of training data sampler
        type='InfiniteSampler',  # InfiniteSampler for iteratiion-based training. Refers to https://github.com/open-mmlab/mmengine/blob/fe0eb0a5bbc8bf816d5649bfdd34908c258eb245/mmengine/dataset/sampler.py#L107
        shuffle=True),  # Whether randomly shuffle the training data
    dataset=dict(  # The config of the training dataset
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline))
val_dataloader = dict(  # The config of validation dataloader
    batch_size=4,  # Batch size of a single GPU
    num_workers=8,  # Worker to pre-fetch data for each single GPU
    dataset=dict(  # The config of the validation dataset
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline),
    sampler=dict(  # The config of validatioin data sampler
        type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/fe0eb0a5bbc8bf816d5649bfdd34908c258eb245/mmengine/dataset/sampler.py#L14
        shuffle=False),  # Whether randomly shuffle the validation data
    persistent_workers=True)
test_dataloader = val_dataloader  # The config of the testing dataloader
```

[Evaluators](https://mmengine.readthedocs.io/en/latest/tutorials/metric_and_evaluator.html) are used to compute the metrics of the trained model on the validation and testing datasets.
The config of evaluators consists of one or a list of metric configs:

```python
val_evaluator = dict(  # The config for validation evaluator
    type='GenEvaluator',  # The type of evaluation
    metrics=[  # The config for metrics
        dict(
            type='FrechetInceptionDistance',
            prefix='FID-Full-50k',
            fake_nums=50000,
            inception_style='StyleGAN',
            sample_model='ema'),
        dict(type='PrecisionAndRecall', fake_nums=50000, prefix='PR-50K'),
        dict(type='PerceptualPathLength', fake_nums=50000, prefix='ppl-w')
    ])
test_evaluator = val_evaluator  # The config for testing evaluator
```

### Training and testing config

MMEngine's runner uses Loop to control the training, validation, and testing processes.
Users can set the maximum training iteration and validation intervals with these fields.

```python
train_cfg = dict(  # The config for training
    by_epoch=False,  # Set `by_epoch` as False to use iteration-based training
    val_begin=1,  # Which iteration to start the validation
    val_interval=10000,  # Validation intervals
    max_iters=800002)  # Maximum training iterations
val_cfg = dict(type='GenValLoop')  # The validation loop type
test_cfg = dict(type='GenTestLoop')  # The testing loop type
```

### Optimization config

`optim_wrapper` is the field to configure optimization related settings.
The optimizer wrapper not only provides the functions of the optimizer, but also supports functions such as gradient clipping, mixed precision training, etc. Find more in [optimizer wrapper tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html).

```python
optim_wrapper = dict(
    constructor='GenOptimWrapperConstructor',
    generator=dict(
        optimizer=dict(type='Adam', lr=0.0016, betas=(0, 0.9919919678228657))),
    discriminator=dict(
        optimizer=dict(
            type='Adam',
            lr=0.0018823529411764706,
            betas=(0, 0.9905854573074332))))
```

`param_scheduler` is a field that configures methods of adjusting optimization hyperparameters such as learning rate and momentum.
Users can combine multiple schedulers to create a desired parameter adjustment strategy.
Find more in [parameter scheduler tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/param_scheduler.html).
Since StyleGAN2 do not use parameter scheduler, we use config in [CycleGAN](../../../configs/cyclegan/cyclegan_lsgan-id0-resnet-in_1xb1-250kiters_summer2winter.py) as an example:

```python
# parameter scheduler in CycleGAN config
param_scheduler = dict(
    type='LinearLrInterval',  # The type of scheduler
    interval=400,  # The interval to update the learning rate
    by_epoch=False,  # The scheduler is called by iteration
    start_factor=0.0002,  # The number we multiply parameter value in the first iteration
    end_factor=0,  # The number we multiply parameter value at the end of linear changing process.
    begin=40000,  # The start iteration of the scheduler
    end=80000)  # The end iteration of the scheduler
```

### Hook config

Users can attach hooks to training, validation, and testing loops to insert some operations during running. There are two different hook fields, one is `default_hooks` and the other is `custom_hooks`.

`default_hooks` is a dict of hook configs. `default_hooks` are the hooks must required at runtime. They have default priority which should not be modified. If not set, runner will use the default values. To disable a default hook, users can set its config to `None`.

```python
default_hooks = dict(
    timer=dict(type='GenIterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        less_keys=['FID-Full-50k/fid'],
        greater_keys=['IS-50k/is'],
        save_optimizer=True,
        save_best='FID-Full-50k/fid'))
```

`custom_hooks` is a list of hook configs. Users can develop there own hooks and insert them in this field.

```python
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
```

### Runtime config

```python
default_scope = 'mmgen'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html

# config for environment
env_cfg = dict(
    cudnn_benchmark=True,  # whether to enable cudnn benchmark.
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # set multi process parameters.
    dist_cfg=dict(backend='nccl'),  # set distributed parameters.
)

log_level = 'INFO'  # The level of logging
log_processor = dict(
    type='GenLogProcessor',  # log processor to process runtime logs
    by_epoch=False)  # print log by iteration
load_from = None  # load model checkpoint as a pre-trained model for a given path
resume = False  # Whether to resume from the checkpoint define in `load_from`. If `load_from` is `None`, it will resume the latest checkpoint in `work_dir`
```
