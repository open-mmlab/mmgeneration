# Tutorial 3: Customize Models

We basically categorize our supported models into 3 main streams according to tasks:

- Unconditional GANs:
    - Static architectures: DCGAN, StyleGANv2
    - Dynamic architectures: PGGAN, StyleGANv1
- Image Translation Models: Pix2Pix, CycleGAN
- Internal Learning (Single Image Model): SinGAN

Of course, some methods, like WGAN-GP, focus on the design of loss functions or learning schedule may be adopted into multiple generative models. Different from `MMDetection`, only two basic categories, `MODULES` and `MODELS`, exist in our repo. In other words, each module will be registered as `MODULES` or `MODELS`.

`MODELS` only contains all of the topmost wrappers for generative models. It supports the commonly used `train_step` and other sampling interface, which can be directly called during training. For static architectures in unconditional GANs, `StaticUnconditionalGAN` is the model that you can use for training your generator and discriminator.

All of the other modules in `MMGeneration` will be registered as `MODULES`, including components of loss functions, generators and discriminators.

## Develop new components

In all of the related repos in OpenMMLab, users may follow the similar steps to build up a new components:


- Implement a class
- Decorate the class with one of the register (`MODELS` or `MODULES` in our repo)
- Import this component in related `__init__.py` files
- Modify the configs and train your models

In the following part, we will show how to add a new generator in `MMGeneration`.

### Implement a class

Here is an standard template for define a new component with PyTorch. Users may insert their codes to define their generator or other components.

```python
import torch.nn as nn

class NewGenerator(nn.Module):

    def __init__(self, *args, **kwargs):
        super(NewGenerator, self).__init__()
        # insert your codes

    def forward(self, x):
        pass
        # insert your codes
```

### Decoate new class with register

In this step, users should import the proper register from `MMGeneration` and decorate their new modules with the `register_module` function.

```python
import torch.nn as nn

from mmgen.models import MODULES


@MODULES.register_module()
class NewGenerator(nn.Module):

    def __init__(self, *args, **kwargs):
        super(NewGenerator, self).__init__()
        # insert your codes

    def forward(self, x):
        pass
        # insert your codes
```

### Import new component in `__init__.py`

Only decorating the new class will **NOT** register the new class into our register. The most important thing you should do is to explicitly import this class in `__init__.py` files.

```python

from .new_generators import NewGenerator

__all__ = ['NewGenerator]
```

If you have already import some modules in a `__init__.py` file, the code still meets `cannot import error`, though. You may try to import this module from the parent package's `__init__.py` file.

### Modify config file to use new model

As discussed in the [tutorial for our config system](https://github.com/open-mmlab/mmgeneration/blob/master/docs/tutorials/config.md), users are recommended to create a new config file based on existing standard configs. Here, we show how to modify the [StyleGAN2 model](https://github.com/open-mmlab/mmgeneration/blob/master/configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py) with our new generator:

```python
_base_ = ['configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py']

model = dict(generator=dict(type='NewGenerator'))
```

Defining the new config file in this way will help us to modify the generator to our new architecture while keeping other configuration unchanged. However, if you do not want to inherit other arguments defined in the `_base_` config file, you can apply the `_delete_` keyword in this way:

```python
_base_ = ['configs/styleganv2/stylegan2_c2_ffhq_256_b4x8_800k.py']

model = dict(generator=dict(_delete_=True, type='NewGenerator'))
```

In `MMCV`, we will automatically discard all of the data for `generator` coming from the `_base_` config file. That is, your generator is just built by `dict(generator=dict(type='NewGenerator))`.
