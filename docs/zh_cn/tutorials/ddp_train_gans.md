# Tutorial 5: MMGeneration 中的分布式训练

在本节中，我们将讨论生成模型的 `DDP`（分布式数据并行）训练，特别是 GANs 的训练。

## 分布式数据并行的训练方式总结

|             DDP Model              | find_unused_parameters | Static GANs | Dynamic GANs |
| :--------------------------------: | :--------------------: | :---------: | :----------: |
|         MMDDP/PyTorch DDP          |         False          |    Error    |    Error     |
|         MMDDP/PyTorch DDP          |          True          |    Error    |    Error     |
|            DDP Wrapper             |         False          | **No Bugs** |    Error     |
|            DDP Wrapper             |          True          | **No Bugs** | **No Bugs**  |
| MMDDP/PyTorch DDP + Dynamic Runner |          True          | **No Bugs** | **No Bugs**  |

在这个表格中，我们总结了生成对抗网络(GANs)的 DDP 训练方式。[`MMDDP/PyTorch DDP`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/distributed.py)表示用 `MMDistributedDataPrarallel` 直接封装 GAN 模型（包含生成器、判别器和损失模块）。然而，在这种方式下，我们无法对 GAN 模型应用对抗训练。主要原因是我们总是需要在 `train_step` 函数中对部分模型（只对判别器或生成器）的损失进行反向传播。

另一种使用 DDP 的方式是采用 [DDP Wrapper](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/core/ddp_wrapper.py)，用 `MMDDP` 封装 GAN 模型中的每个模块，这在目前的实践中被广泛使用，例如，`MMEditing` 和 [StyleGAN2-ADA-PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch)。这样一来，就有了一个重要的参数，`find_unused_parameters`。如表所示，对于训练动态架构的模型，如 PGGAN 和 StyleGANv1，用户必须设置这个参数为 `True`。 然而，一旦 `find_unused_parameters` 设置为 `True`，模型将在每个前向传播后重建 `bucket` 以同步梯度和信息，从而在反向传播过程中追踪计算图所需的张量。

在 `MMGeneration` 中，我们为用户设计了另一种采用 `DDP` 训练的方式，即 `MMDDP/PyTorch DDP + Dynamic Runner`。在具体说明这个新设计的细节之前，我们首先解释为什么用户应该使用它。尽管通过 `DDP Wrapper` 实现了动态 GAN 的训练，我们仍然发现了一些不便和缺点。

- `DDP Wrapper` 使用户无法调用或获得 GANs 中模块的函数或属性，例如，生成器和判别器。采用 `DDP Wrapper` 后，如果我们想调用 `generator` 中的函数，我们必须使用 `generator.module.xxx()`。
- `DDP Wrapper` 将导致多余的桶重建。通过采用 `DDP Wrapper` 来避免 ddp 错误的真正原因是，GAN 模型中的每个模块在调用它们的 `forward` 函数后，会立即为反向传播重建桶。然而，正如 GAN 实践中所知道的，有很多情况下我们不需要为反向传播建立一个桶，例如，在更新判别器时为生成器建桶。

为了解决这些问题，我们试图找到一种方法来直接采用 `MMDDP` 并支持动态的 GAN 训练。在 `MMGeneration` 中，`DynamicIterBasedRunner` 帮助我们实现这一目标。重要的是，只需要少于十行的修改就能解决这个问题。

## MMDDP/PyTorch DDP + Dynamic Runner

在静态/动态GAN训练中采用 DDP 的关键点是在反向传播（判别器和生成器）之前构建（或检查）桶。因为这两个反向中需要梯度的参数来自 GAN 模型的不同部分。因此，我们的解决方案只是在每个反向传播之前显示地重建桶。

在[mmgen/core/runners/dynamic_iterbased_runner.py](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/core/runners/dynamic_iterbased_runner.py)中，我们通过使用 **PyTorch private API** 获得 `reducer`。

```python
if self.is_dynamic_ddp:
    kwargs.update(dict(ddp_reducer=self.model.reducer))
outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
```

通过如下对 train_step 的修改，reducer 可以帮助我们在当前反传中重建桶：

```python
if ddp_reducer is not None:
    ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))
```

一个完整用例如下:

```python
loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

# prepare for backward in ddp. If you do not call this function before
# back propagation, the ddp will not dynamically find the used params
# in current computation.
if ddp_reducer is not None:
    ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

loss_disc.backward()
```

也就是说，用户应该在损失计算和损失反传之间准备 reducer。

在我们的 `MMGeneration` 中，这个功能被作为训练 DDP 模型的默认方式。在配置文件中，用户只需要添加以下配置来使用动态 ddp runner。

```python
# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=True,
    pass_training_status=True)
```

*这个实现将使用 PyTorch 中的私有接口，我们将继续维护这一实现。*

## DDP Wrapper

当然，我们仍然支持使用 `DDP Wrapper` 来训练你的 GANs。如果你想切换到使用 DDP Wrapper，你应该这样修改配置文件。

```python
# use ddp wrapper for faster training
use_ddp_wrapper = True
find_unused_parameters = True  # True for dynamic model, False for static model

runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,  # Note that this flag should be False.
    pass_training_status=True)
```

在[`dcgan config file`](https://github.com/open-mmlab/mmgeneration/tree/master/configs/dcgan/dcgan_celeba-cropped_64_b128x1_300k.py)中，我们已经提供了一个在 MMGeneration 中使用 `DDP Wrapper` 的例子。
