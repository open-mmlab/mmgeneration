# Design Your Own Models

MMGeneration is built upon MMEngine and MMCV, which enables users to design new models quickly, train and evaluate them easily. In this section, you will learn how to design your own models.

## Overview of models in MMGeneration

In MMGeneration, one algorithm can be splited two compents: **Model** and **Architecture**(s).

- **Model** are topmost wrappers and always inherint from `BaseModel` provided in MMEngine. **Model** is responsible to network forward, loss calculation and backward, parameters updating, etc. In MMGeneration, **Model** should be registered as `MODELS`.
- **Architecture**(s) is the neural network to train or inference, and always as element(s) of **Model**. In MMGeneration, **Architecture** should be registered as **MODULES**.

Take BigGAN model as an example, [generator and discriminator](https://github.com/open-mmlab/mmgeneration/blob/test-1.x/mmgen/models/architectures/biggan/generator_discriminator.py), [`BigGAN`](https://github.com/open-mmlab/mmgeneration/blob/test-1.x/mmgen/models/gans/biggan.py) is the **Model**, which take data from dataloader and train generator and discriminator alternatively.

You can find implementation **Model** for [GAN](https://github.com/open-mmlab/mmgeneration/tree/test-1.x/mmgen/models/gans) and [Diffusion](https://github.com/open-mmlab/mmgeneration/tree/test-1.x/mmgen/models/architectures) models here. And implementation of **Architecture**(s) can be find [here](https://github.com/open-mmlab/mmgeneration/tree/test-1.x/mmgen/models/architectures).

Here, we take the implementation of the classical gan model, DCGAN \[1\], as an example.

To implement DCGAN, you need to follow these steps:

- [Step 1: Define your own network **Architecture**(s)](#step-1-define-your-own-network-architectures)
- [Step 2: Define your own **Model**](#step-2-define-the-forward-loop-of-your-model)
- [Step 3: Start training](#step-3-start-training)

## Step 1: Define your own network **Architecture**(s)

DCGAN is a classical image generative adversarial network \[1\]. To implement the network architecture of DCGAN, we need to create a new file `mmgen/models/architectures/dcgan/generator_discriminator.py` and implement generator (`class DCGANGenerator`) and discriminator (`class DCGANDiscriminator`).

In this step, we implement `class DCGANGenerator`, `class DCGANDiscriminator` and define the network architecture in `__init__` function.
In particular, we need to use `@MODULES.register_module()` to add the generator and discriminator into the registration of MMGeneration.

Take the following code as example:

```python
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmengine.logging import MMLogger
from mmengine.model.utils import normal_init

from mmgen.models.builder import MODULES
from ..common import get_module_device


@MODULES.register_module()
class DCGANGenerator(nn.Module):
    def __init__(self,
                 output_scale,
                 out_channels=3,
                 base_channels=1024,
                 input_scale=4,
                 noise_size=100,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='Tanh'),
                 pretrained=None):
        super().__init__()
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size

        # the number of times for upsampling
        self.num_upsamples = int(np.log2(output_scale // input_scale))

        # output 4x4 feature map
        self.noise2feat = ConvModule(
            noise_size,
            base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg)

        # build up upsampling backbone (excluding the output layer)
        upsampling = []
        curr_channel = base_channels
        for _ in range(self.num_upsamples - 1):
            upsampling.append(
                ConvModule(
                    curr_channel,
                    curr_channel // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))

            curr_channel //= 2

        self.upsampling = nn.Sequential(*upsampling)

        # output layer
        self.output_layer = ConvModule(
            curr_channel,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=None,
            act_cfg=out_act_cfg)
```

Then, we implement the `forward` function of  `DCGANGenerator`, which takes as `noise` tensor or `num_batches` and then returns the results from `DCGANGenerator`.

```python
    def forward(self, noise, num_batches=0, return_noise=False):
        noise_batch = noise_batch.to(get_module_device(self))
        x = self.noise2feat(noise_batch)
        x = self.upsampling(x)
        x = self.output_layer(x)
        return x
```

If you want to implement specific weights initialization method for you network, you need add `init_weights` function by yourself.

```python
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, _BatchNorm):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
```

After the implementation of class `DCGANGenerator`, we need to update the model list in `mmgen/models/architectures/__init__.py`, so that we can import and use class `DCGANGenerator` by `mmgen.models.architectures`.

Implementation of Class `DCGANDiscriminator` follows the similar logic, and you can find the implementation [here](https://github.com/open-mmlab/mmgeneration/blob/test-1.x/mmgen/models/architectures/dcgan/generator_discriminator.py#L195).

## Step 2: Define the your **Model**

After the implementation of the network **Architecture**, we need to define our **Model** class `DCGAN`.

In **Model**, you should implement three function `train_step`, `val_step` and `test_step`.

- `train_step`: This function is responsible to update the parameters of the network and called by `IterBasedTrainLoop` or `EpochBasedTrainLoop`. `train_step` take data batch as input and return a dict of log.
- `val_step`: This function is responsible for getting output for validation during the training process. and is called by `GenValLoop`.
- `test_step`: This function is responsible for getting output in test process and is called by `GenTestLoop`.

For simplify using, we provide `BaseGAN` class im MMGeneration, which implement generic `train_step`, `val_step` and `test_step` function for GAN models. With `BaseGAN`, each specific GAN algorithm only need to implement `train_generator` and `train_discriminator`.

In `train_step`, we support data preprocessing, gradient accumulation (realized by [`OptimWrapper`](https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/optim_wrapper.md)) and expontial moving averate (EMA) realized by [(`ExponentialMovingAverage`)](https://github.com/open-mmlab/mmgeneration/blob/test-1.x/mmgen/models/averaged_model.py#L19). With `BaseGAN.train_step`, each specific GAN algorithm only need to implement `train_generator` and `train_discriminator`.

```python
    def train_step(self, data: dict,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        data = self.data_preprocessor(data, True)
        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        # train discriminator, use context manager provided by MMEngine
        with disc_optimizer_wrapper.optim_context(self.discriminator):
            # train_discriminator should be implemented!
            log_vars = self.train_discriminator(
                **data, optimizer_wrapper=disc_optimizer_wrapper)

        # add 1 to `curr_iter` because iter is updated in train loop.
        # Whether to update the generator. We update generator with
        # discriminator is fully updated for `self.n_discriminator_steps`
        # iterations. And one full updating for discriminator contains
        # `disc_accu_counts` times of grad accumulations.
        if (curr_iter + 1) % (self.discriminator_steps * disc_accu_iters) == 0:
            set_requires_grad(self.discriminator, False)
            gen_optimizer_wrapper = optim_wrapper['generator']
            gen_accu_iters = gen_optimizer_wrapper._accumulative_counts

            log_vars_gen_list = []
            # init optimizer wrapper status for generator manually
            gen_optimizer_wrapper.initialize_count_status(
                self.generator, 0, self.generator_steps * gen_accu_iters)
            # update generator, use context manager provided by MMEngine
            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    # train_generator should be implemented!
                    log_vars_gen = self.train_generator(
                        **data, optimizer_wrapper=gen_optimizer_wrapper)

                log_vars_gen_list.append(log_vars_gen)
            log_vars_gen = gather_log_vars(log_vars_gen_list)
            log_vars_gen.pop('loss', None)  # remove 'loss' from gen logs

            set_requires_grad(self.discriminator, True)

            # only do ema after generator update
            if self.with_ema_gen and (curr_iter + 1) >= (
                    self.ema_start * self.discriminator_steps *
                    disc_accu_iters):
                self.generator_ema.update_parameters(
                    self.generator.module
                    if is_model_wrapper(self.generator) else self.generator)

            log_vars.update(log_vars_gen)

        # return the log dict
        return log_vars
```

In `val_step` and `test_step`, we call data preprocessing and `BaseGAN.forward` progressively.

```python
    def val_step(self, data: dict) -> SampleList:
        data = self.data_preprocessor(data)
        # call `forward`
        outputs = self(**data)
        return outputs

    def test_step(self, data: dict) -> SampleList:
        data = self.data_preprocessor(data)
        # call `orward`
        outputs = self(**data)
        return outputs
```

Then, we implement `train_generator` and `train_discriminator` in `DCGAN` class.

```python
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmgen.registry import MODELS
from .base_gan import BaseGAN


@MODELS.register_module()
class DCGAN(BaseGAN):
    def disc_loss(self, disc_pred_fake: Tensor,
                  disc_pred_real: Tensor) -> Tuple:
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.binary_cross_entropy_with_logits(
            disc_pred_real, 1. * torch.ones_like(disc_pred_real))

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake: Tensor) -> Tuple:
        losses_dict = dict()
        losses_dict['loss_gen'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 1. * torch.ones_like(disc_pred_fake))
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def train_discriminator(
            self, inputs, data_sample,
            optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        real_imgs = inputs['img']

        num_batches = real_imgs.shape[0]

        noise_batch = self.noise_fn(num_batches=num_batches)
        with torch.no_grad():
            fake_imgs = self.generator(noise=noise_batch, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        disc_pred_real = self.discriminator(real_imgs)

        parsed_losses, log_vars = self.disc_loss(disc_pred_fake,
                                                 disc_pred_real)
        optimizer_wrapper.update_params(parsed_losses)
        return log_vars

    def train_generator(self, inputs, data_sample,
                        optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        # num_batches = inputs['real_imgs'].shape[0]
        num_batches = inputs['img'].shape[0]

        # >>> new setting
        noise = self.noise_fn(num_batches=num_batches)
        fake_imgs = self.generator(noise=noise, return_noise=False)

        disc_pred_fake = self.discriminator(fake_imgs)
        parsed_loss, log_vars = self.gen_loss(disc_pred_fake)

        optimizer_wrapper.update_params(parsed_loss)
        return log_vars
```

After the implementation of `class DCGAN`, we need to update the model list in `mmgen/models/__init__.py`, so that we can import and use `class DCGAN` by `mmgen.models`.

## Step 3: Start training

After implementing the network architecture and the forward loop of DCGAN,
now we can create a new file `configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py`
to set the configurations needed by training DCGAN.

In the configuration file, we need to specify the parameters of our model, `class DCGAN`, including the generator network architecture, loss function and data preprocessor of input tensors.

```python
# model settings
model = dict(
    type='DCGAN',
    noise_size=100,
    data_preprocessor=dict(type='GANDataPreprocessor'),
    generator=dict(type='DCGANGenerator', output_scale=64, base_channels=1024),
    discriminator=dict(
        type='DCGANDiscriminator',
        input_scale=64,
        output_scale=4,
        out_channels=1))
```

We also need to specify the training dataloader and testing dataloader according to [create your own dataloader](advanced_tutorial/dataset.md).
Finally we can start training our own model by：

```python
python train.py configs/dcgan/dcgan_1xb128-5epoches_lsun-bedroom-64x64.py
```

## References

1. Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
