# Tutorial 4: Design of Our Loss Modules

As shown in the last tutorial for customizing models, `losses` are regarded/registered as `MODULES` in `MMGeneration`. Customizing losses is similar to customizing any other models. This section is mainly for clarifying the design of loss modules in our repo. Importantly, when writing your own loss modules, you should follow the same design, so that the new loss module can be adopted in our framework without extra efforts.

## Design of loss modules

In general, to implement a loss module, we will write a function implementation and then wrap it with a class implementation. However, in `MMGeneration`, we provide another unified interface `data_info` for users to define the mapping between the input argument and data items.

```python
@weighted_loss
def disc_shift_loss(pred):
    return pred**2

@MODULES.register_module()
class DiscShiftLoss(nn.Module):

    def __init__(self, loss_weight=1.0, data_info=None):
        super(DiscShiftLoss, self).__init__()
        # codes can be found in ``mmgen/models/losses/disc_auxiliary_loss.py``

    def forward(self, *args, **kwargs):
        # codes can be found in ``mmgen/models/losses/disc_auxiliary_loss.py``
```


The goal of this design for loss modules is to allow for using it automatically in the generative models (`MODELS`), without other complex codes to define the mapping between data and keyword arguments. Thus, different from other frameworks in `OpenMMLab`, our loss modules contain a special keyword, `data_info`, which is a dictionary defining the mapping between the input arguments and data from the generative models. Taking the `DiscShiftLoss` as an example, when writing the config file, users may use this loss as follows:

```python
dict(type='DiscShiftLoss',
    loss_weight=0.001 * 0.5,
    data_info=dict(pred='disc_pred_real')
```
The information in `data_info` tells the module to use the `disc_pred_real` data as the input tensor for `pred` arguments. Once the `data_info` is not `None`, our loss module will automatically build up the computational graph.

```python
@MODULES.register_module()
class DiscShiftLoss(nn.Module):

    def __init__(self, loss_weight=1.0, data_info=None):
        super(DiscShiftLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info

    def forward(self, *args, **kwargs):
        # use data_info to build computational path
        if self.data_info is not None:
            # parse the args and kwargs
            if len(args) == 1:
                assert isinstance(args[0], dict), (
                    'You should offer a dictionary containing network outputs '
                    'for building up computational graph of this loss module.')
                outputs_dict = args[0]
            elif 'outputs_dict' in kwargs:
                assert len(args) == 0, (
                    'If the outputs dict is given in keyworded arguments, no'
                    ' further non-keyworded arguments should be offered.')
                outputs_dict = kwargs.pop('outputs_dict')
            else:
                raise NotImplementedError(
                    'Cannot parsing your arguments passed to this loss module.'
                    ' Please check the usage of this module')
            # link the outputs with loss input args according to self.data_info
            loss_input_dict = {
                k: outputs_dict[v]
                for k, v in self.data_info.items()
            }
            kwargs.update(loss_input_dict)
            kwargs.update(dict(weight=self.loss_weight))
            return disc_shift_loss(**kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return disc_shift_loss(*args, weight=self.loss_weight, **kwargs)

    @staticmethod
    def loss_name():
        return 'loss_disc_shift'

```

As shown in this part of codes, once users set the `data_info`, the loss module will receive a dictionary containing all of the necessary data and modules, which is provided by the `MODELS` in the training procedure. If this dictionary is given as a non-keyword argument, it should be offered as the first argument. If you are using a keyword argument, please name it as `outputs_dict`.

To build the computational graph, the generative models have to provide a dictionary containing all kinds of data. Having a close look at any generative model, you will find that we collect all kinds of features and modules into a dictionary. The following codes are from our `ProgressiveGrowingGAN`:

```python
    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None)
        # ...

        # get data dict to compute losses for disc
        data_dict_ = dict(
            iteration=curr_iter,
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            curr_scale=self.curr_scale[0],
            transition_weight=transition_weight,
            gen_partial=partial(
                self.generator,
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight),
            disc_partial=partial(
                self.discriminator,
                curr_scale=self.curr_scale[0],
                transition_weight=transition_weight))

        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # ...
```

Here, the `_get_disc_loss` defined in [BaseGAN](https://github.com/open-mmlab/mmgeneration/tree/master/mmgen/models/gans/base_gan.py) will help to combine all kinds of losses automatically.

```python
    def _get_disc_loss(self, outputs_dict):
        # Construct losses dict. If you hope some items to be included in the
        # computational graph, you have to add 'loss' in its name. Otherwise,
        # items without 'loss' in their name will just be used to print
        # information.
        losses_dict = {}
        # gan loss
        losses_dict['loss_disc_fake'] = self.gan_loss(
            outputs_dict['disc_pred_fake'], target_is_real=False, is_disc=True)
        losses_dict['loss_disc_real'] = self.gan_loss(
            outputs_dict['disc_pred_real'], target_is_real=True, is_disc=True)

        # disc auxiliary loss
        if self.with_disc_auxiliary_loss:
            for loss_module in self.disc_auxiliary_losses:
                loss_ = loss_module(outputs_dict)
                if loss_ is None:
                    continue

                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses_dict:
                    losses_dict[loss_module.loss_name(
                    )] = losses_dict[loss_module.loss_name()] + loss_
                else:
                    losses_dict[loss_module.loss_name()] = loss_
        loss, log_var = self._parse_losses(losses_dict)

        return loss, log_var

```

Therefore, as long as users design the loss module with the same rules, any kind of loss can be inserted in the training of generative models, without other modifications in the code of models. What you only need to do is just defining the `data_info` in the config files.
