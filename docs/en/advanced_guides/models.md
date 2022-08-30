# Design Your Own Models

MMGeneration is built upon MMEngine and MMCV, which enables users to design new models quickly, train and evaluate them easily.
In this section, you will learn how to design your own models.
Here, we take the implementation of the classical gan model, DCGAN \[1\], as an example.

To implement DCGAN, you need to follow these steps:

- [Step 1: Define your own network architectures](#step-1-define-your-own-network-architectures)
- [Step 2: Define the train step](#step-2-define-the-forward-loop-of-your-model)
- [Step 3: Start training](#step-3-start-training)

## Step 1: Define your own network architectures

DCGAN is a classical image generative adversarial network \[1\].
To implement the network architecture of DCGAN,
we need to create a new file `mmgen/models/architectures/dcgan/generator_discriminator.py` and implement `class DCGANGenerator`, `class DCGANDiscriminator`.

In this step, we implement `class DCGANGenerator`, `class DCGANDiscriminator` and define the network architecture in `__init__` function.
In particular, we need to use `@MODULES.register_module()` to add the modules into the registration of MMGeneration.

**note**
MMGeneration has two basic categories, `MODULES` and `MODELS`, exist in our repo. In other words, each module will be registered as `MODULES` or `MODELS`.

`MODELS` only contains all of the topmost wrappers for generative models. It supports the commonly used `train_step` and other sampling interface, which can be directly called during training. For static architectures in unconditional GANs, `StaticUnconditionalGAN` is the model that you can use for training your generator and discriminator.

All of the other modules in `MMGeneration` will be registered as `MODULES`, including components of loss functions, generators and discriminators.

```python
import numpy as np
import torch
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
    """Generator for DCGAN.

    Implementation Details for DCGAN architecture:

    #. Adopt transposed convolution in the generator;
    #. Use batchnorm in the generator except for the final output layer;
    #. Use ReLU in the generator in addition to the final output layer.

    More details can be found in the original paper:
    Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks
    http://arxiv.org/abs/1511.06434

    Args:
        output_scale (int | tuple[int]): Output scale for the generated
            image. If only a integer is provided, the output image will
            be a square shape. The tuple of two integers will set the
            height and width for the output image, respectively.
        out_channels (int, optional): The channel number of the output feature.
            Default to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Default to 1024.
        input_scale (int | tuple[int], optional): Output scale for the
            generated image. If only a integer is provided, the input feature
            ahead of the convolutional generator will be a square shape. The
            tuple of two integers will set the height and width for the input
            convolutional feature, respectively. Defaults to 4.
        noise_size (int, optional): Size of the input noise
            vector. Defaults to 100.
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='ReLU')``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
    """

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

Then, we implement the `forward` function of  `class DCGANGenerator`, which takes as `noise` tensor or `num_batches` and then returns the results from `DCGANGenerator`.

```python
    def forward(self, noise, num_batches=0, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img`` and
                ``noise_batch`` will be returned.
        """
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            if noise.ndim == 2:
                noise_batch = noise[:, :, None, None]
            elif noise.ndim == 4:
                noise_batch = noise
            else:
                raise ValueError('The noise should be in shape of (n, c) or '
                                 f'(n, c, 1, 1), but got {noise.shape}')
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size, 1, 1))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.noise_size, 1, 1))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))

        x = self.noise2feat(noise_batch)
        x = self.upsampling(x)
        x = self.output_layer(x)

        if return_noise:
            return dict(fake_img=x, noise_batch=noise_batch)

        return x
```

After the implementation of `class DCGANGenerator`, we need to update the model list in `mmgen/models/architectures/__init__.py`, so that we can import and use `class DCGANGenerator` by `mmgen.models.architectures`.

## Step 2: Define the forward loop of your model

After the implementation of the network architecture,
we need to define our model `class DCGAN` and implement its forward loop.

To implement `class DCGAN`,
we create a new file in `mmgen/models/gans/dcgan.py`.
Specifically, `class DCGAN` inherits from `mmgen.models.BaseGAN`.
In the `__init__` function of `BaseGAN`, we define the loss functions, training and testing configurations, networks of `class BaseEditModel`.

```python
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from copy import deepcopy
from typing import Dict, Optional, Union

import torch.nn as nn
from mmengine import Config
from mmengine.model import BaseModel
from torch import Tensor

from mmgen.registry import MODULES
from ..common import get_valid_noise_size

ModelType = Union[Dict, nn.Module]
TrainInput = Union[dict, Tensor]


class BaseGAN(BaseModel, metaclass=ABCMeta):
    """Base class for GAN models.

    Args:
        generator (ModelType): The config or model of the generator.
        discriminator (Optional[ModelType]): The config or model of the
            discriminator. Defaults to None.
        data_preprocessor (Optional[Union[dict, Config]]): The pre-process
            config or :class:`~mmgen.models.GANDataPreprocessor`.
        generator_steps (int): The number of times the generator is completely
            updated before the discriminator is updated. Defaults to 1.
        discriminator_steps (int): The number of times the discriminator is
            completely updated before the generator is updated. Defaults to 1.
        ema_config (Optional[Dict]): The config for generator's exponential
            moving average setting. Defaults to None.
    """

    def __init__(self,
                 generator: ModelType,
                 discriminator: Optional[ModelType] = None,
                 data_preprocessor: Optional[Union[dict, Config]] = None,
                 generator_steps: int = 1,
                 discriminator_steps: int = 1,
                 noise_size: Optional[int] = None,
                 ema_config: Optional[Dict] = None):
        super().__init__(data_preprocessor=data_preprocessor)

        # get valid noise_size
        self.noise_size = get_valid_noise_size(noise_size, generator)

        # build generator
        if isinstance(generator, dict):
            self._gen_cfg = deepcopy(generator)
            # build generator with default `noise_size` and `num_classes`
            gen_args = dict()
            if self.noise_size:
                gen_args['noise_size'] = self.noise_size
            if hasattr(self, 'num_classes'):
                gen_args['num_classes'] = self.num_classes
            generator = MODULES.build(generator, default_args=gen_args)
        self.generator = generator

        # build discriminator
        if discriminator:
            if isinstance(discriminator, dict):
                self._disc_cfg = deepcopy(discriminator)
                # build discriminator with default `num_classes`
                disc_args = dict()
                if hasattr(self, 'num_classes'):
                    disc_args['num_classes'] = self.num_classes
                discriminator = MODULES.build(
                    discriminator, default_args=disc_args)
        self.discriminator = discriminator

        self._gen_steps = generator_steps
        self._disc_steps = discriminator_steps

        if ema_config is None:
            self._ema_config = None
            self._with_ema_gen = False
        else:
            self._ema_config = deepcopy(ema_config)
            self._init_ema_model(self._ema_config)
            self._with_ema_gen = True

```

Since `mmengine.model.BaseModel` provides the basic functions of the algorithmic model,
such as weights initialize, batch inputs preprocess, parse losses, and update model parameters.
Therefore, the subclasses inherit from BaseModel, i.e., `class DCGAN` in this example,
only need to implement the `train_step`, which implements the logic of one train step.

```python
    def train_step(self, data: TrainStepInputs,
                   optim_wrapper: OptimWrapperDict) -> Dict[str, Tensor]:
        """Train GAN model. In the training of GAN models, generator and
        discriminator are updated alternatively. In MMGeneration's design,
        `self.train_step` is called with data input. Therefore we always update
        discriminator, whose updating is relay on real data, and then determine
        if the generator needs to be updated based on the current number of
        iterations. More details about whether to update generator can be found
        in :meth:`should_gen_update`.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optim_wrapper (OptimWrapperDict): OptimWrapperDict instance
                contains OptimWrapper of generator and discriminator.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        inputs_dict, data_sample = self.data_preprocessor(data, True)

        disc_optimizer_wrapper: OptimWrapper = optim_wrapper['discriminator']
        disc_accu_iters = disc_optimizer_wrapper._accumulative_counts

        with disc_optimizer_wrapper.optim_context(self.discriminator):
            log_vars = self.train_discriminator(inputs_dict, data_sample,
                                                disc_optimizer_wrapper)

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
            for _ in range(self.generator_steps * gen_accu_iters):
                with gen_optimizer_wrapper.optim_context(self.generator):
                    log_vars_gen = self.train_generator(
                        inputs_dict, data_sample, gen_optimizer_wrapper)

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

        return log_vars
```

TODO: the main problem is that this train_step contains too much behaviours.

```python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmgen.registry import MODELS
from .base_gan import BaseGAN


@MODELS.register_module()
class DCGAN(BaseGAN):
    """Impelmentation of ``.

    Paper link:

    Detailed architecture can be found in
    :class:~`mmgen.models.architectures.dcgan.generator_discriminator.DCGANGenerator`  # noqa
    and
    :class:~`mmgen.models.architectures.dcgan.generator_discriminator.DCGANDiscriminator`  # noqa
    """

    def disc_loss(self, disc_pred_fake: Tensor,
                  disc_pred_real: Tensor) -> Tuple:
        r"""Get disc loss. DCGAN use the vanilla gan loss to train
        the discriminator.

        .. math::


        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.
            disc_pred_real (Tensor): Discriminator's prediction of the real
                images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_disc_fake'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 0. * torch.ones_like(disc_pred_fake))
        losses_dict['loss_disc_real'] = F.binary_cross_entropy_with_logits(
            disc_pred_real, 1. * torch.ones_like(disc_pred_real))

        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def gen_loss(self, disc_pred_fake: Tensor) -> Tuple:
        """Get gen loss. DCGAN use the vanilla gan loss to train the generator.

        .. math::

        Args:
            disc_pred_fake (Tensor): Discriminator's prediction of the fake
                images.

        Returns:
            tuple[Tensor, dict]: Loss value and a dict of log variables.
        """
        losses_dict = dict()
        losses_dict['loss_gen'] = F.binary_cross_entropy_with_logits(
            disc_pred_fake, 1. * torch.ones_like(disc_pred_fake))
        loss, log_var = self.parse_losses(losses_dict)
        return loss, log_var

    def train_discriminator(
            self, inputs, data_sample,
            optimizer_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Train discriminator.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (List[GenDataSample]): Data samples from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.
        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
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
        """Train generator.

        Args:
            inputs (TrainInput): Inputs from dataloader.
            data_samples (List[GenDataSample]): Data samples from dataloader.
                Do not used in generator's training.
            optim_wrapper (OptimWrapper): OptimWrapper instance used to update
                model parameters.

        Returns:
            Dict[str, Tensor]: A ``dict`` of tensor for logging.
        """
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

After the implementation of `class DCGAN`,
we need to update the model list in `mmgen/models/__init__.py`,
so that we can import and use `class DCGAN` by `mmgen.models`.

```python
from mmgen.models import DCGAN
```

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
