# Migration from MMGeneration 0.x

In 1.x version of MMGeneration, we redesign distributed training and mixed precision training, the usage of optimizer and data flow based on MMEngine. This document will help users of the 0.x version to quickly migrate to the newest version.

- [Migration from MMGeneration 0.x](#migration-from-mmgeneration-0x)
  - [New dependencies](#new-dependencies)
  - [1. Runner and schedule](#1-runner-and-schedule)
  - [2. Evaluation and testing setting](#2-evaluation-and-testing-setting)
  - [3. Distributed Training setting](#3-distributed-training-setting)
  - [4. Optimizer](#4-optimizer)
  - [5. Learining rate schedule](#5-learining-rate-schedule)
  - [6. Visualization setting](#6-visualization-setting)
  - [7. AMP (auto mixed precision) training](#7-amp-auto-mixed-precision-training)
  - [8. Data settings](#8-data-settings)
  - [9. Runtime setting](#9-runtime-setting)

## New dependencies

MMGeneration 1.x depends on some new packages, you can prepare a new clean environment and install again according to the [install tutorial](./get_started.md). Or install the below packages manually.

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is the core the OpenMMLab 2.0 architecture, and we splited many compentents unrelated to computer vision from MMCV to MMEngine.
2. [MMCV](https://github.com/open-mmlab/mmcv/tree/dev-2.x): The computer vision package of OpenMMLab. This is not a new dependency, but you need to upgrade it to above 2.0.0rc0 version.
3. [rich](https://github.com/Textualize/rich): A terminal formatting package, and we use it to beautify some outputs in the terminal.

## 1. Runner and schedule

In 0.x version, MMGeneration use `total_iters` fields to control the total training iteration and use `DynamicIterBasedRunner` to handle the training process.
In 1.x version, we use [`Runner`](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/runner.md) and [`Loops`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py) provided by `MMEngine` and use `train_cfg.max_iters` field to control to define the total training iteration and use `train_cfg.val_interval` to control the evaluation interval.

And to evaluate and test the model correctly, we need to set specific loop in `val_cfg` and `test_cfg`.

<table class="docutils">
<thead>
  <tr>
    <th> Static Model in 0.x Version </th>
    <th> Static Model in 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
total_iters = 1000000

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
```

</td>

<td valign="top">

```python
train_cfg = dict(
    by_epoch=False,  # use iteration based training
    max_iters=1000000,  # max training iteration
    val_begin=1,
    val_interval=10000)  # evaluation interval
val_cfg = dict(type='GenValLoop')  # specific loop in validation
test_cfg = dict(type='GenTestLoop')  # specific loop in testing
```

</td>

</tr>
</thead>
</table>

## 2. Evaluation and testing setting

The evaluation field is splited to `val_evaluator` and `test_evaluator`. And it won't supports `interval` and `save_best` arguments. The `interval` is moved to `train_cfg.val_interval`, see [the schedule settings](#1-runner-and-schedule) and the `save_best` is moved to `default_hooks.checkpoint.save_best`.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
evaluation = dict(
    type='GenerativeEvalHook',
    interval=10000,
    metrics=[
        dict(
            type='FID',
            num_images=50000,
            bgr2rgb=True,
            inception_args=dict(type='StyleGAN')),
        dict(type='IS', num_images=50000)
    ],
    best_metric=['fid', 'is'],
    sample_kwargs=dict(sample_model='ema'))
```

</td>

<td valign="top">

```python
val_evaluator = dict(
    type='GenEvaluator',
    metrics=[
        dict(
            type='FrechetInceptionDistance',
            prefix='FID-Full-50k',
            fake_nums=50000,
            inception_style='StyleGAN',
            sample_model='orig')
        dict(
            type='InceptionScore',
            prefix='IS-50k',
            fake_nums=50000)])
# set best config
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        less_keys=['FID-Full-50k/fid'],
        greater_keys=['IS-50k/is'],
        save_optimizer=True,
        save_best=['FID-Full-50k/fid', 'IS-50k/is'],
        rule=['less', 'greater']))
test_evaluator = val_evaluator
```

</td>

</tr>
</thead>
</table>

## 3. Distributed Training setting

In 0.x version, MMGeneration uses `DDPWrapper` and `DynamicRunner` to train of static and dynamic model (e.g., PGGAN and StyleGANv2) respectively. In 1.x version, we use `MMSeparateDistributedDataParallel` provided by MMEngine to implement distributed training.

The configuration differences are shown below:

<table class="docutils">
<thead>
  <tr>
    <th> Static Model in 0.x Version </th>
    <th> Static Model in 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
# Use DDPWrapper
use_ddp_wrapper = True
find_unused_parameters = False

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False)
```

</td>

<td valign="top">

```python
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)
```

</td>

</tr>
</thead>
</table>

<table class="docutils">
<thead>
  <tr>
    <th> Dynamic Model in 0.x Version </th>
    <th> Dynamic Model in 1.x Version </th>
<tbody>
<tr>

<td valign="top">

```python
use_ddp_wrapper = False
find_unused_parameters = False

# Use DynamicRunner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True)
```

</td>

<td valign="top">

```python
model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=True) # set `find_unused_parameters` for dynamic models
```

</td>

</tr>
</thead>
</table>

## 4. Optimizer

In version 0.x, MMGeneration uses PyTorch's native Optimizer, which only provides general parameter optimization.
In version 1.x, we use `OptimizerWrapper` provided by MMEngine.

Compared to PyTorch's `Optimizer`, `OptimizerWrapper` supports the following features:

- `OptimizerWrapper.update_params` implement `zero_grad`, `backward` and `step` in a single function.
- Support gradient accumulation automatically.
- Provide a context manager named `OptimizerWrapper.optim_context` to warp the forward process. `optim_context` can automatically call `torch.no_sync` according to current number of updating iteration. In AMP (auto mixed precision) training, `autocast` is called in `optim_context` as well.

For GAN models, generator and discriminator use different optimizer and training schedule.
To ensure that the GAN model's function signature of `train_step` is consistent with other models, we use `OptimWrapperDict`, inherited from `OptimizerWrapper`, to wrap the optimizer of the generator and discriminator.
To automate this process MMGeneration implement `GenOptimWrapperContructor`.
And you should specify this constructor in your config is you want to train GAN model.

The config for the 0.x and 1.x versions are shown below:

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
optimizer = dict(
    generator=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6),
    discriminator=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6))
```

</td>

<td valign="top">

```python
optim_wrapper = dict(
    # Use constructor implemented by MMGeneration
    constructor='GenOptimWrapperConstructor',
    generator=dict(optimizer=dict(type='Adam', lr=0.0002, betas=(0.0, 0.999), eps=1e-6)),
    discriminator=dict(
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6)))
```

</td>

</tr>
</thead>
</table>

> Note that, in the 1.x, MMGeneration uses `OptimWrapper` to realize gradient accumulation. This make the config of `discriminator_steps` (training trick for updating the generator once after multiple updates of the discriminator) and gradient accumulation different between 0.x and 1.x version.

- In 0.x version,  we use `disc_steps`, `gen_steps` and `batch_accumulation_steps` in configs. `disc_steps` and `batch_accumulation_steps` are counted by the number of calls of `train_step` (is also the number of data reads from the dataloader). Therefore the number of consecutive updates of the discriminator is `disc_steps // batch_accumulation_steps`. And for generators, `gen_steps` is the number of times the generator actually updates continuously.
- In 1.x version, we use `discriminator_steps`, `generator_steps` and `accumulative_counts` in configs. `discriminator_steps` and `generator_steps` are the number of consecutive updates to itself before updating other modules.

Take config of BigGAN-128 as example.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
model = dict(
    type='BasiccGAN',
    generator=dict(
        type='BigGANGenerator',
        output_scale=128,
        noise_size=120,
        num_classes=1000,
        base_channels=96,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        split_noise=True,
        auto_sync_bn=False),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=128,
        num_classes=1000,
        base_channels=96,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True),
    gan_loss=dict(type='GANLoss', gan_type='hinge'))

# continuous update discriminator for `disc_steps // batch_accumulation_steps = 8 // 8 = 1` times
# continuous update generator for `gen_steps = 1` times
# generators and discriminators perform `batch_accumulation_steps = 8` times gradient accumulations before each update
train_cfg = dict(
    disc_steps=8, gen_steps=1, batch_accumulation_steps=8, use_ema=True)
```

</td>

<td valign="top">

```python
model = dict(
    type='BigGAN',
    num_classes=1000,
    data_preprocessor=dict(type='GANDataPreprocessor'),
    generator=dict(
        type='BigGANGenerator',
        output_scale=128,
        noise_size=120,
        num_classes=1000,
        base_channels=96,
        shared_dim=128,
        with_shared_embedding=True,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        split_noise=True,
        auto_sync_bn=False),
    discriminator=dict(
        type='BigGANDiscriminator',
        input_scale=128,
        num_classes=1000,
        base_channels=96,
        sn_eps=1e-6,
        init_type='ortho',
        act_cfg=dict(type='ReLU', inplace=True),
        with_spectral_norm=True),
    # continuous update discriminator for `discriminator_steps = 1` times
    # continuous update generator for `generator_steps = 1` times
    generator_steps=1,
    discriminator_steps=1)

optim_wrapper = dict(
    constructor='GenOptimWrapperConstructor',
    generator=dict(
        # generator perform `accumulative_counts = 8` times gradient accumulations before each update
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6)),
    discriminator=dict(
        # discriminator perform `accumulative_counts = 8` times gradient accumulations before each update
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6)))
```

</td>

</tr>
</thead>
</table>

## 5. Learining rate schedule

In 0.x version, MMGeneration use `lr_config` field to define the learning reate scheduler. In 1.x version, we use `param_scheduler` to replace it.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
lr_config = dict(
    policy='Linear',
    by_epoch=False,
    target_lr=0,
    start=135000,
    interval=1350)
```

</td>
<td valign="top">

```python
param_scheduler = dict(
    type='LinearLrInterval',
    interval=1350,
    by_epoch=False,
    start_factor=0.0002,
    end_factor=0,
    begin=135000,
    end=270000)
```

</td>
</tr>
</thead>
</table>

## 6. Visualization setting

In 0.x, MMGeneration use `MMGenVisualizationHook` and `VisualizeUnconditionalSamples` to visualization generating results in training process. In 1.x version, we unify the function of those hooks into `GenVisualizationHook`. Additionally, follow the design of MMEngine, we implement `GenVisualizer` and a group of `VisBackend` to draw and save the visualization results.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=1000)
]
```

</td>

<td valign="top">

```python
custom_hooks = [
    dict(
        type='GenVisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=dict(type='GAN', name='fake_img'))
]
vis_backends = [dict(type='GenVisBackend')]
visualizer = dict(type='GenVisualizer', vis_backends=vis_backends)
```

</td>

</tr>
</thead>
</table>

To learn more about the visualization function, please refers to [this tutorial](./user_guides/5_visualization.md).

## 7. AMP (auto mixed precision) training

In 0.x, MMGeneration do not support AMP training for the entire forward process.
Instead, users must use `auto_fp16` decorator to warp the specific submodule and convert the parameter of submodule to fp16.
This allows for fine-grained control of the model parameters, but is more cumbersome to use.
In addition, users need to handle operations such as scaling of the loss function during the training process by themselves.

In 1.x version, MMGeneration use `AmpOptimWrapper` provided by MMEngine.
In `AmpOptimWrapper.update_params`, gradient scaling and `GradScaler` updating is automatically performed.
And in `optim_context` context manager, `auto_cast` is applied to the entire forward process.

Specifically, the difference between the 0.x and 1.x is as follows:

<table class="docutils">
<thead>
  <tr>
    <th> 0.x version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
# config
runner = dict(fp16_loss_scaler=dict(init_scale=512))
```

```python
# code
import torch.nn as nn
from mmgen.models.builder import build_model
from mmgen.core.runners.fp16_utils import auto_fp16


class DemoModule(nn.Module):
    def __init__(self, cfg):
        self.net = build_model(cfg)

    @auto_fp16
    def forward(self, x):
        return self.net(x)

class DemoModel(nn.Module):

    def __init__(self, cfg):
        super().__init__(self)
        self.demo_network = DemoModule(cfg)

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        # get data from data_batch
        inputs = data_batch['img']
        output = self.demo_network(inputs)

        optimizer.zero_grad()
        loss, log_vars = self.get_loss(data_dict_)

        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(loss_disc, optimizer,
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer)
            loss_scaler.step(optimizer)
        else:
            optimizer.step()
```

</td>

<td valign="top">

```python
# config
optim_wrapper = dict(
    constructor='GenOptimWrapperConstructor',
    generator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-06),
        type='AmpOptimWrapper',  # use amp wrapper
        loss_scale='dynamic'),
    discriminator=dict(
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-06),
        type='AmpOptimWrapper',  # use amp wrapper
        loss_scale='dynamic'))
```

```python
# code
import torch.nn as nn
from mmgen.registry import MODULES
from mmengine.model import BaseModel


class DemoModule(nn.Module):
    def __init__(self, cfg):
        self.net = MODULES.build(cfg)

    def forward(self, x):
        return self.net(x)

class DemoModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(self)
        self.demo_network = DemoModule(cfg)

    def train_step(self, data, optim_wrapper):
        # get data from data_batch
        data = self.data_preprocessor(data, True)
        inputs = data['inputs']

        with optim_wrapper.optim_context(self.discriminator):
            output = self.demo_network(inputs)
        loss_dict = self.get_loss(output)
        # use parse_loss provide by `BaseModel`
        loss, log_vars = self.parse_loss(loss_dict)
        optimizer_wrapper.update_params(loss)

        return log_vars
```

</td>

</tr>
</thead>
</table>

To avoid user modifications to the configuration file, MMGeneration provides the `--amp` option in `train.py`, which allows the user to start AMP training without modifying the configuration file.
Users can start AMP training by following command:

```bash
bash tools/dist_train.sh CONFIG GPUS --amp

# for slurm users
bash tools/slurm_train.sh PARTITION JOB_NAME CONFIG WORK_DIR --amp
```

## 8. Data settings

MMGeneration redesign data flow and data transforms pipelien based on MMCV 2.x and MMEngine.

Changes in `data`:

1. The original `data` field is splited to `train_dataloader`, `val_dataloader` and `test_dataloader`. This allows us to configure them in fine-grained. For example, you can specify different sampler and batch size during training and test.
2. The `samples_per_gpu` is renamed to `batch_size`.
3. The `workers_per_gpu` is renamed to `num_workers`.

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
data = dict(
    samples_per_gpu=None,
    workers_per_gpu=4,
    train=dict(...),
    val=dict(...),
    test=dict(...))
```

</td>

<td valign="top">

```python
# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(...))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)
```

</td>

</tr>
</thead>
</table>

Changes in `pipeline`

1. Normalization, color space transforms are no longer performed in transforms pipelines, but converted to `data_preprocessor`.
2. Data is packed to `GenDataSample` by `PackGenInputs` in the last step of transforms pipeline. To know more about datasample please refers to [this tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/data_element.md).

Take config for FFHQ-Flip dataset as example:

<table class="docutils">
<thead>
  <tr>
    <th> 0.x Version </th>
    <th> 1.x Version </th>
<tbody>
<tr>
<td valign="top">

```python
dataset_type = 'UnconditionalImageDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(type='Flip', keys=['real_img'], direction='horizontal'),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=False),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='real_img',
        io_backend='disk',
    ),
    dict(
        type='Normalize',
        keys=['real_img'],
        mean=[127.5] * 3,
        std=[127.5] * 3,
        to_rgb=True),
    dict(type='ImageToTensor', keys=['real_img']),
    dict(type='Collect', keys=['real_img'], meta_keys=['real_img_path'])
]
```

</td>

<td valign="top">

```python
dataset_type = 'UnconditionalImageDataset'

train_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='Flip', keys=['img'], direction='horizontal'),
    dict(type='PackGenInputs', keys=['img'], meta_keys=['img_path'])
]

val_pipeline = [
    dict(type='LoadImageFromFile', key='img'),
    dict(type='PackGenInputs', keys=['img'], meta_keys=['img_path'])
]
data_preprocessor = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], bgr_to_rgb=False)
```

</td>

</tr>
</thead>
</table>

## 9. Runtime setting

Changes in **`checkpoint_config`** and **`log_config`**:

The `checkpoint_config` are moved to `default_hooks.checkpoint` and the `log_config` are moved to `default_hooks.logger`.
And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.

```python
default_hooks = dict(
    # record time of every iteration.
    timer=dict(type='GenIterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    # save checkpoint per 10000 iterations
    checkpoint=dict(
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        less_keys=['FID-Full-50k/fid'],
        greater_keys=['IS-50k/is'],
        save_optimizer=True))
```

In addition, we splited the original logger to logger and visualizer. The logger is used to record
information and the visualizer is used to show the logger in different backends, like terminal, TensorBoard
and Wandb.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
```

</td>
<tr>
<td>New</td>
<td>

```python
default_hooks = dict(
    ...
    logger=dict(type='LoggerHook', interval=100),
)

```

</td>
</tr>
</table>

Changes in **`load_from`** and **`resume_from`**:

- The `resume_from` is removed. And we use `resume` and `load_from` to replace it.
  - If `resume=True` and `load_from` is not None, resume training from the checkpoint in `load_from`.
  - If `resume=True` and `load_from` is None, try to resume from the latest checkpoint in the work directory.
  - If `resume=False` and `load_from` is not None, only load the checkpoint, not resume training.
  - If `resume=False` and `load_from` is None, do not load nor resume.

Changes in **`dist_params`**: The `dist_params` field is a sub field of `env_cfg` now. And there are some new
configurations in the `env_cfg`.

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'))
```

Changes in **`workflow`**: `workflow` related functionalities are removed.

New field **`default_scope`**: The start point to search module for all registries. The `default_scope` in MMGeneration is `mmgen`. See [the registry tutorial](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md) for more details.
