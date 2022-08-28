# Migration from MMGen 0.x

In 1.x version of MMGeneration, we redesign distributed training and mixed precision training, the usage of optimizer and data flow based on MMEngine. This document will help users of the 0.x version to quickly migrate to the newest version.

## 1. Distributed Training

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
    <th> Dynamic Model in 0,x Version </th>
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

## 2. Optimizer

In version 0.x, MMGeneration uses PyTorch's native Optimizer, which only provides general parameter optimization.
In version 1.x, we use `OptimizerWrapper` provided by MMEngine.

Compared to PyTorch's `Optimizer`, `OptimizerWrapper` supports the following features:

- `OptimizerWrapper.update_params` implement `zero_grad`, `backward` and `step` in a single function.
- Support gradient accumulation automatically.
- Provide a context manager named `OptimizerWrapper.optim_context` to warp the forward process. `optim_context` can automatically call `torch.no_sync` according to current number of updating iteration. In AMP (auto mixed precision) training, `autocast` is called in `optim_context` as well.

For GAN models, generator and discriminator use different optimizer and training schedule.
To ensure that the GAN model's function signature of `train_step` is consistent with other models, we use `OptimWrapperDict`, which is inherited from from `OptimizerWrapper`, to wrap the optimizer of the generator and discriminator.
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

> Note that, in the 1.x, MMGeneration uses `OptimWrapper` to realize gradient accumulation.  This make the configure of `discriminator_steps` (training trick for updating the generator once after multiple updates of the discriminator) and gradient accumulation different between 0.x and 1.x version.

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
        # generators  perform `accumulative_counts = 8` times gradient accumulations before each update
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0001, betas=(0.0, 0.999), eps=1e-6)),
    discriminator=dict(
        # generators  perform `accumulative_counts = 8` times gradient accumulations before each update
        accumulative_counts=8,
        optimizer=dict(type='Adam', lr=0.0004, betas=(0.0, 0.999), eps=1e-6)))
```

</td>

</tr>
</thead>
</table>

## 3. AMP (auto mixed precision) training

In 0.x, MMGeneration do not support AMP training for the entire forward process.
Instead, Users must use `auto_fp16` decorator to warp the specific submodule and convert the parameter of submodule to fp16.
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

## 4. 数据集

MMGeneration redesign data flow and data transforms pipelien based on MMCV 2.x and MMEngine.

The main differences between 0.x and 1.x are as follow:

1. Normalization, color space transforms are no longer performed in transforms pipelines, but converted to `data_preprocessor`.
2. Data is packed to `GenDataSample` by `PackGenInputs` in the last step of transforms pipeline. (More about datasample please refers to this tutorial. TODO:)

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

# `samples_per_gpu` and `imgs_root` need to be set.
data = dict(
    samples_per_gpu=None,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=100,
        dataset=dict(
            type=dataset_type, imgs_root=None, pipeline=train_pipeline)),
    val=dict(type=dataset_type, imgs_root=None, pipeline=val_pipeline))
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

# `batch_size` and `data_root` need to be set.
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=None,  # set by user
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True)

# define data_preprocessor in model's config
model = dict(data_preprocessor=dict(type='GANDataPreprocessor'))
```
