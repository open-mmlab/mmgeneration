# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmgen.models.builder import MODULES
from .utils import weighted_loss

_reduction_modes = ['none', 'mean', 'sum', 'batchmean', 'flatmean']


@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target (Tensor): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target (Tensor): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def gaussian_kld(mean_target, mean_pred, logvar_target, logvar_pred, base='e'):
    r"""Calculate KLD (Kullback-Leibler divergence) of two gaussian
    distribution.
    To be noted that in this function, KLD is calcuated in base `e`.

    .. math::
        :nowrap:

        \begin{align}
        KLD(p||q) &= -\int{p(x)\log{q(x)} dx} + \int{p(x)\log{p(x)} dx} \\
            &= \frac{1}{2}\log{(2\pi \sigma_2^2)} +
            \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} -
            \frac{1}{2}(1 + \log{2\pi \sigma_1^2}) \\
            &= \log{\frac{\sigma_2}{\sigma_1}} +
            \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
        \end{align}

    Args:
        mean_target (torch.Tensor): Mean of the target (or the first)
            distribution.
        mean_pred (torch.Tensor): Mean of the predicted (or the second)
            distribution.
        logvar_target (torch.Tensor): Log variance of the target (or the first)
            distribution
        logvar_pred (torch.Tensor): Log variance of the predicted (or the
            second) distribution.
        base (str, optional): The log base of calculated KLD. We support
            ``'e'`` (for ln) and ``'2'`` (for log_2). Defaults to ``'e'``.

    Returns:
        torch.Tensor: KLD between two given distribution.
    """
    if base not in ['e', '2']:
        raise ValueError('Only support 2 and e for log base, but receive '
                         f'{base}')
    kld = 0.5 * (-1.0 + logvar_pred - logvar_target +
                 torch.exp(logvar_target - logvar_pred) +
                 ((mean_target - mean_pred)**2) * torch.exp(-logvar_pred))
    if base == '2':
        return kld / np.log(2.0)
    return kld


def approx_gaussian_cdf(x):
    r"""Approximate the cumulative distribution function of the gaussian distribution.

    Refers to:
    Approximations to the Cumulative Normal Function and its Inverse for Use on a Pocket Calculator  # noqa

    https://www.jstor.org/stable/2346872?origin=crossref

    .. math::
        :nowrap:
        \begin{eqnarray}
            \Phi(x) &\approx \frac{1}{2} \left ( 1 + \tanh(y) \right ) \\
            y &= \sqrt{\frac{2}{\pi}}(x+0.044715 x^3)
        \end{eqnarray}

    Args:
        x (torch.Tensor): Input data.

    Returns:
        torch.Tensor: Calculated cumulative distribution.

    """
    factor = np.sqrt(2.0 / np.pi)
    y = factor * (x + 0.044715 * torch.pow(x, 3))
    phi = 0.5 * (1 + torch.tanh(y))
    return phi


@weighted_loss
def discretized_gaussian_log_likelihood(x, mean, logvar, base='e'):
    r"""Calculate gaussian log-likelihood for a discretized input. We assume
    that the input `x` are ranged in [-1, 1], the likelihood term can be
    calculated as the following equation:

    .. math::
     :nowrap:
        \begin{equarray}
            p_{\theta}(\mathbf{x}_0 | \mathbf{x}_1) =
                \prod_{i=1}^{D} \int_{\delta_{-}(x_0^i)}^{\delta_{+}(x_0^i)}
                {\mathcal{N}(x; \mu_{\theta}^i(\mathbf{x}_1, 1),
                \sigma_{1}^2)}dx\\
            \delta_{+}(x)= \begin{cases}
                \infty & \text{if } x = 1 \\
                x + \frac{1}{255} & \text{if } x < 1
            \end{cases}
            \quad
            \delta_{-}(x)= \begin{cases}
                -\infty & \text{if } x = -1 \\
                x - \frac{1}{255} & \text{if } x > -1
            \end{cases}
        \end{equarray}

    When calculating this loss term, we first normalize `x` to normal
    distribution and calculate the above integral by the cumulative
    distribution function of normal distribution. Then rescale results to the
    target ones.

    Args:
        x (torch.Tensor): Target `x_0` to be modeled. Range in [-1, 1].
        mean (torch.Tensor): Predicted mean of `x_0`.
        logvar (torch.Tensor): Predicted log variance of `x_0`.
        base (str, optional): The log base of calculated KLD. Support ``'e'``
            and ``'2'``. Defaults to ``'e'``.

    Returns:
        torch.Tensor: Calculated log likelihood.
    """
    if base not in ['e', '2']:
        raise ValueError('Only support 2 and e for log base, but receive '
                         f'{base}')

    inv_std = torch.exp(-logvar * 0.5)
    x_centered = x - mean

    lower_bound = (x_centered - 1.0 / 255.0) * inv_std
    upper_bound = (x_centered + 1.0 / 255.0) * inv_std
    cdf_to_lower = approx_gaussian_cdf(lower_bound)
    cdf_to_upper = approx_gaussian_cdf(upper_bound)

    log_cdf_upper = torch.log(cdf_to_upper.clamp(min=1e-12))
    log_one_minus_cdf_lower = torch.log((1.0 - cdf_to_lower).clamp(min=1e-12))
    log_cdf_delta = torch.log((cdf_to_upper - cdf_to_lower).clamp(min=1e-12))

    log_probs = torch.where(
        x < -0.999, log_cdf_upper,
        torch.where(x > 0.999, log_one_minus_cdf_lower, log_cdf_delta))

    if base == '2':
        return log_probs / np.log(2.0)
    return log_probs


@MODULES.register_module()
class MSELoss(nn.Module):
    """MSE loss.

    **Note for the design of ``data_info``:**
    In ``MMGeneration``, almost all of loss modules contain the argument
    ``data_info``, which can be used for constructing the link between the
    input items (needed in loss calculation) and the data from the generative
    model. For example, in the training of GAN model, we will collect all of
    important data/modules into a dictionary:

    .. code-block:: python
        :caption: Code from StaticUnconditionalGAN, train_step
        :linenos:

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we may need to provide ``pred`` and ``target`` as input.
    Thus, an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            pred='fake_imgs',
            target='real_imgs')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_mse'.
    """

    def __init__(self, loss_weight=1.0, data_info=None, loss_name='loss_mse'):
        super().__init__()
        self.loss_weight = loss_weight
        self.data_info = data_info
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function, ``mse_loss``.
        """
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
            return mse_loss(**kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return mse_loss(*args, weight=self.loss_weight, **kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODULES.register_module()
class L1Loss(nn.Module):
    """L1 loss.

    **Note for the design of ``data_info``:**
    In ``MMGeneration``, almost all of loss modules contain the argument
    ``data_info``, which can be used for constructing the link between the
    input items (needed in loss calculation) and the data from the generative
    model. For example, in the training of GAN model, we will collect all of
    important data/modules into a dictionary:

    .. code-block:: python
        :caption: Code from StaticUnconditionalGAN, train_step
        :linenos:

        data_dict_ = dict(
            gen=self.generator,
            disc=self.discriminator,
            disc_pred_fake=disc_pred_fake,
            disc_pred_real=disc_pred_real,
            fake_imgs=fake_imgs,
            real_imgs=real_imgs,
            iteration=curr_iter,
            batch_size=batch_size)

    But in this loss, we may need to provide ``pred`` and ``target`` as input.
    Thus, an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            pred='fake_imgs',
            target='real_imgs')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (float | None, optional): Average factor when computing the
            mean of losses. Defaults to ``None``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If ``None``, this module will
            directly pass the input data to the loss function.
            Defaults to None.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_l1'.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 avg_factor=None,
                 data_info=None,
                 loss_name='loss_l1'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.data_info = data_info
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function, ``l1_loss``.
        """
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
            kwargs.update(
                dict(weight=self.loss_weight, reduction=self.reduction))
            return l1_loss(**kwargs)
        else:
            # if you have not define how to build computational graph, this
            # module will just directly return the loss as usual.
            return l1_loss(
                *args,
                weight=self.loss_weight,
                reduction=self.reduction,
                avg_factor=self.avg_factor,
                **kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODULES.register_module()
class GaussianKLDLoss(nn.Module):
    """GaussianKLD loss.

    **Note for the design of ``data_info``:**
    In ``MMGeneration``, almost all of loss modules contain the argument
    ``data_info``, which can be used for constructing the link between the
    input items (needed in loss calculation) and the data from the generative
    model. For example, in the training of GAN model, we will collect all of
    important data/modules into a dictionary:

    .. code-block:: python
        :caption: Code from BaseDiffusion, train_step
        :linenos:

        data_dict_ = dict(
            denoising=denoising,
            real_imgs=torch.Tensor([N, C, H, W]),
            mean_pred=torch.Tensor([N, C, H, W]),
            mean_target=torch.Tensor([N, C, H, W]),
            logvar_pred=torch.Tensor([N, C, H, W]),
            logvar_target=torch.Tensor([N, C, H, W]),
            timesteps=torch.Tensor([N,]),
            iteration=curr_iter,
            batch_size=batch_size)

    In this loss, we may need to provide ``mean_pred``, ``mean_target``,
    ``logvar_pred`` and ``logvar_target`` as input. Thus, an example of the
    ``data_info`` is:

    .. code-block:: python
        :linenos:

        data_info = dict(
            mean_pred='mean_pred',
            mean_target='mean_target',
            logvar_pred='logvar_pred',
            logvar_target='logvar_target')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        reduction (str, optional): Same as built-in losses of PyTorch. Noted
            that 'batchmean' mode given the correct KL divergence where losses
            are averaged over batch dimension only. Defaults to 'mean'.
        avg_factor (float | None, optional): Average factor when computing the
            mean of losses. Defaults to ``None``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If not passed,
            ``_default_data_info`` would be used. Defaults to None.
        base (str, optional): The log base of calculated KLD. Support
            ``'e'`` and ``'2'``. Defaults to ``'e'``.
        only_update_var (bool, optional): If true, only `logvar_pred` will be
            updated and variable in output_dict corresponding to `mean_pred`
            will be detached. Defaults to False.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_l1'.
    """

    _default_data_info = dict(
        mean_pred='mean_pred',
        mean_target='mean_target',
        logvar_pred='logvar_pred',
        logvar_target='logvar_target')

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 avg_factor=None,
                 data_info=None,
                 base='e',
                 only_update_var=False,
                 loss_name='loss_GaussianKLD'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.data_info = self._default_data_info if data_info is None \
            else data_info
        self.base = base
        self.only_update_var = only_update_var
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``gaussian_kld_loss``.
        """

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
        loss_input_dict = dict()
        for k, v in self.data_info.items():
            if 'mean_pred' == k and self.only_update_var:
                loss_input_dict[k] = outputs_dict[v].detach()
            else:
                loss_input_dict[k] = outputs_dict[v]

        kwargs.update(loss_input_dict)
        kwargs.update(
            dict(
                weight=self.loss_weight,
                reduction=self.reduction,
                base=self.base))
        return gaussian_kld(**kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


# TODO: this name is toooooo long.
@MODULES.register_module()
class DiscretizedGaussianLogLikelihoodLoss(nn.Module):
    r"""Discretized-Gaussian-Log-Likelihood Loss.

    **Note for the design of ``data_info``:**
    In ``MMGeneration``, almost all of loss modules contain the argument
    ``data_info``, which can be used for constructing the link between the
    input items (needed in loss calculation) and the data from the generative
    model. For example, in the training of GAN model, we will collect all of
    important data/modules into a dictionary:

    .. code-block:: python
        :caption: Code from BaseDiffusion, train_step
        :linenos:

        data_dict_ = dict(
            denoising=denoising,
            real_imgs=torch.Tensor([N, C, H, W]),
            mean_pred=torch.Tensor([N, C, H, W]),
            mean_target=torch.Tensor([N, C, H, W]),
            logvar_pred=torch.Tensor([N, C, H, W]),
            logvar_target=torch.Tensor([N, C, H, W]),
            timesteps=torch.Tensor([N,]),
            iteration=curr_iter,
            batch_size=batch_size)

    In this loss, we may need to provide ``mean``, ``logvar`` and ``x``. Thus,
    an example of the ``data_info`` is:

    .. code-block:: python
        :linenos:
        data_info = dict(
            x='real_imgs',
            mean='mean_pred',
            logvar='logvar_pred')

    Then, the module will automatically construct this mapping from the input
    data dictionary.

    Args:
        loss_weight (float, optional): Weight of this loss item.
            Defaults to ``1.``.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (float | None, optional): Average factor when computing the
            mean of losses. Defaults to ``None``.
        data_info (dict, optional): Dictionary contains the mapping between
            loss input args and data dictionary. If not passed,
            ``_default_data_info`` would be used. Defaults to None.
        base (str, optional): The log base of calculated KLD. Support
            ``'e'`` and ``'2'``. Defaults to ``'e'``.
        only_update_var (bool, optional): If true, only `logvar_pred` will be
            updated and variable in output_dict corresponding to `mean_pred`
            will be detached. Defaults to False.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_l1'.
    """

    _default_data_info = dict(
        x='real_imgs', mean='mean_pred', logvar='logvar_pred')

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 avg_factor=None,
                 data_info=None,
                 base='e',
                 only_update_var=False,
                 loss_name='loss_DiscGaussianLogLikelihood'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.data_info = self._default_data_info if data_info is None \
            else data_info
        self.base = base
        self.only_update_var = only_update_var
        self._loss_name = loss_name

    def forward(self, *args, **kwargs):
        """Forward function.

        If ``self.data_info`` is not ``None``, a dictionary containing all of
        the data and necessary modules should be passed into this function.
        If this dictionary is given as a non-keyword argument, it should be
        offered as the first argument. If you are using keyword argument,
        please name it as `outputs_dict`.

        If ``self.data_info`` is ``None``, the input argument or key-word
        argument will be directly passed to loss function,
        ``gaussian_kld_loss``.
        """

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
        loss_input_dict = dict()
        for k, v in self.data_info.items():
            if k == 'mean' and self.only_update_var:
                loss_input_dict[k] = outputs_dict[v].detach()
            else:
                loss_input_dict[k] = outputs_dict[v]

        kwargs.update(loss_input_dict)
        kwargs.update(
            dict(
                weight=self.loss_weight,
                reduction=self.reduction,
                base=self.base))

        return discretized_gaussian_log_likelihood(**kwargs)

    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
