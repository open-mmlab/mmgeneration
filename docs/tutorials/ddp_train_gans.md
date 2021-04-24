# Tutorial 5: DDP Training in MMGeneration

In this section, we will discuss the `DDP` (Distributed Data-Parallel) training for generative models, especially for GANs.

## Summary of ways for DDP Training

|             DDP Model              | find_unused_parameters | Static GANs | Dynamic GANs |
| :--------------------------------: | :--------------------: | :---------: | :----------: |
|         MMDDP/PyTorch DDP          |         False          |    Error    |    Error     |
|         MMDDP/PyTorch DDP          |          True          |    Error    |    Error     |
|            DDP Wrapper             |         False          | **No Bugs** |    Error     |
|            DDP Wrapper             |          True          | **No Bugs** | **No Bugs**  |
| MMDDP/PyTorch DDP + Dynamic Runner |          True          | **No Bugs** | **No Bugs**  |

In this table, we summarize the ways of DDP training for GANs. [`MMDDP/PyTorch DDP`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/distributed.py) denotes directly wrapping the GAN model (containing the generator, discriminator, and loss modules) with `MMDistributedDataPrarallel`. However, in such a way, we cannot train the GAN models with the adversarial training schedule. The main reason is that we always need to backward the losses for partial models (only for discriminator or generator) in `train_step` function.

Another way to use DDP is adopting the [DDP Wrapper](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/core/ddp_wrapper.py) to wrap each component in the GAN model with `MMDDP`, which is widely used in current literature, e.g., `MMEditing` and [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch). In this way, there is an important argument, `find_unused_parameters`. As shown in the table, users must set `True` in this argument for training dynamic architectures, like PGGAN and StyleGANv1. However, once set `True` in `find_unused_parameters`, the model will rebuild the bucket for synchronizing gradients and information after each forward process. This step will help the backward procedure to track which tensors are needed in the current computation graph.

In `MMGeneration`, we design another way for users to adopt `DDP` training, i.e., `MMDDP/PyTorch DDP + Dynamic Runner`. Before specifying the details of this new design, we first clarify why users should switch to it. In spite of achieving training in dynamic GANs with `DDP Wrapper`, we still spot some inconvenience and disadvantages:

- `DDP Wrapper` prevents users from calling the function or obtaining the attribute of the component in GANs, e.g., generator and discriminator. After adopting `DDP Wrapper`, if we want to call the function in `generator`, we have to use `generator.module.xxx()`.
- `DDP Wrapper` will cause redundant bucket rebuilding. The true reason for avoiding ddp error by adopting `DDP Wrapper` is that each component in the GAN model will rebuild the bucket for backward right after calling their `forward` function. However, as known in GAN literature, there are many cases that we need not build a bucket for backward, e.g., building the bucket for the generator when updating discriminators.

To solve these points, we try to find a way to directly adopt `MMDDP` and support dynamic GAN training. In `MMGeneration`, `DynamicIterBasedRunner` helps us to achieve this. Importantly, only `<10` line modification will solve the problem.

## MMDDP/PyTorch DDP + Dynamic Runner

The key point of adopting DDP in static/dynamic GAN training is to construct (or check) the bucket used for backward before backward (discriminator backward and generator backward). Since the parameters that need gradients in these two backward are from different parts of the GAN model. Thus, our solution is just explicitly rebuilding the bucket right before each backward procedure.

In [mmgen/core/runners/dynamic_iterbased_runner.py](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/core/runners/dynamic_iterbased_runner.py), we obtain the `reducer` by using **PyTorch private API**:

```python
if self.is_dynamic_ddp:
    kwargs.update(dict(ddp_reducer=self.model.reducer))
outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
```
The reducer can help us to rebuild the bucket for current backward path by just adding this line in the `train_step` function:

```python
if ddp_reducer is not None:
    ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))
```

A complete using case is:

```python
loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

# prepare for backward in ddp. If you do not call this function before
# back propagation, the ddp will not dynamically find the used params
# in current computation.
if ddp_reducer is not None:
    ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

loss_disc.backward()
```
That is, users should add reducer preparation in between the loss calculation and loss backward.

In our `MMGeneration`, this feature is adoptted as the default way to train DDP model. In configs, users should only add the following configuration to use dynamic ddp runner:

```python
# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True,
    pass_training_status=True)
```

*We have to admit that this implementation will use the private interface in PyTorch and will keep maintaining this feature.*



## DDP Wrapper

Of course, we still support using the `DDP Wrapper` to train your GANs. If you want to switch to use DDP Wrapper, you should modify the config file like this:

```python
# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = True  # True for dynamic model, False for static model

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)
```

In [`dcgan config file`](https://github.com/open-mmlab/mmgeneration/tree/master/configs/dcgan/dcgan_celeba-cropped_64_b128x1_300k.py), we have already provided an example for using `DDPWrapper` in MMGeneration.
