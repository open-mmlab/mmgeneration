# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy

import mmcv
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ExponentialMovingAverageHook(Hook):
    """Exponential Moving Average Hook.

    Exponential moving average is a trick that widely used in current GAN
    literature, e.g., PGGAN, StyleGAN, and BigGAN. This general idea of it is
    maintaining a model with the same architecture, but its parameters are
    updated as a moving average of the trained weights in the original model.
    In general, the model with moving averaged weights achieves better
    performance.

    Args:
        module_keys (str | tuple[str]): The name of the ema model. Note that we
            require these keys are followed by '_ema' so that we can easily
            find the original model by discarding the last four characters.
        interp_mode (str, optional): Mode of the interpolation method.
            Defaults to 'lerp'.
        interp_cfg (dict | None, optional): Set arguments of the interpolation
            function. Defaults to None.
        interval (int, optional): Evaluation interval (by iterations).
            Default: -1.
        start_iter (int, optional): Start iteration for ema. If the start
            iteration is not reached, the weights of ema model will maintain
            the same as the original one. Otherwise, its parameters are updated
            as a moving average of the trained weights in the original model.
            Default: 0.
        momentum_policy (str, optional): Policy of the momentum updating
            method. Defaults to 'fixed'.
        momentum_cfg (dict | None, optional): Set arguments of the momentum
            updater function. Defaults to None.
    """

    _registered_interp_funcs = ['lerp']
    _registered_momentum_updaters = ['rampup', 'fixed']

    def __init__(self,
                 module_keys,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 start_iter=0,
                 momentum_policy='fixed',
                 momentum_cfg=None):
        super().__init__()
        # check args
        assert interp_mode in self._registered_interp_funcs, (
            'Supported '
            f'interpolation functions are {self._registered_interp_funcs}, '
            f'but got {interp_mode}')

        assert momentum_policy in self._registered_momentum_updaters, (
            'Supported momentum policy are'
            f'{self._registered_momentum_updaters},'
            f' but got {momentum_policy}')

        assert isinstance(module_keys, str) or mmcv.is_tuple_of(
            module_keys, str)
        self.module_keys = (module_keys, ) if isinstance(module_keys,
                                                         str) else module_keys
        # sanity check for the format of module keys
        for k in self.module_keys:
            assert k.endswith(
                '_ema'), 'You should give keys that end with "_ema".'
        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(
            interp_cfg)
        self.interval = interval
        self.start_iter = start_iter

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = getattr(self, interp_mode)

        self.momentum_cfg = dict() if momentum_cfg is None else deepcopy(
            momentum_cfg)
        self.momentum_policy = momentum_policy
        if momentum_policy != 'fixed':
            assert hasattr(
                self, momentum_policy
            ), f'Currently, we do not support {self.momentum_policy} for EMA.'
            self.momentum_updater = getattr(self, momentum_policy)

    @staticmethod
    def lerp(a, b, momentum=0.999, momentum_nontrainable=0., trainable=True):
        """Does a linear interpolation of two parameters/ buffers.

        Args:
            a (torch.Tensor): Interpolation start point, refer to orig state.
            b (torch.Tensor): Interpolation end point, refer to ema state.
            momentum (float, optional): The weight for the interpolation
                formula. Defaults to 0.999.
            momentum_nontrainable (float, optional): The weight for the
                interpolation formula used for nontrainable parameters.
                Defaults to 0..
            trainable (bool, optional): Whether input parameters is trainable.
                If set to False, momentum_nontrainable will be used.
                Defaults to True.

        Returns:
            torch.Tensor: Interpolation result.
        """
        m = momentum if trainable else momentum_nontrainable
        return a + (b - a) * m

    @staticmethod
    def rampup(runner, ema_kimg=10, ema_rampup=0.05, batch_size=4, eps=1e-8):
        """Ramp up ema momentum.

        Ref: https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/training/training_loop.py#L300-L308 # noqa

        Args:
            runner (_type_): _description_
            ema_kimg (int, optional): Half-life of the exponential moving
                average of generator weights. Defaults to 10.
            ema_rampup (float, optional): EMA ramp-up coefficient.If set to
                None, then rampup will be disabled. Defaults to 0.05.
            batch_size (int, optional): Total batch size for one training
                iteration. Defaults to 4.
            eps (float, optional): Epsiolon to avoid ``batch_size`` divided by
                zero. Defaults to 1e-8.

        Returns:
            dict: Updated momentum.
        """
        cur_nimg = (runner.iter + 1) * batch_size
        ema_nimg = ema_kimg * 1000
        if ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
        ema_beta = 0.5**(batch_size / max(ema_nimg, eps))
        return dict(momentum=ema_beta)

    def every_n_iters(self, runner, n):
        if runner.iter < self.start_iter:
            return True
        return (runner.iter + 1 - self.start_iter) % n == 0 if n > 0 else False

    @torch.no_grad()
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model

        # update momentum
        _interp_cfg = deepcopy(self.interp_cfg)
        if self.momentum_policy != 'fixed':
            _updated_args = self.momentum_updater(runner, **self.momentum_cfg)
            _interp_cfg.update(_updated_args)

        for key in self.module_keys:
            # get current ema states
            ema_net = getattr(model, key)
            states_ema = ema_net.state_dict(keep_vars=False)
            # get currently original states
            net = getattr(model, key[:-4])
            states_orig = net.state_dict(keep_vars=True)

            for k, v in states_orig.items():
                if runner.iter < self.start_iter:
                    states_ema[k].data.copy_(v.data)
                else:
                    states_ema[k] = self.interp_func(
                        v,
                        states_ema[k],
                        trainable=v.requires_grad,
                        **_interp_cfg).detach()
            ema_net.load_state_dict(states_ema, strict=True)

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(
            runner.model) else runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not hasattr(model, k) and not hasattr(model, k[:-4]):
                raise RuntimeError(
                    f'Cannot find both {k[:-4]} and {k} network for EMA hook.')
            if not hasattr(model, k) and hasattr(model, k[:-4]):
                setattr(model, k, deepcopy(getattr(model, k[:-4])))
                warnings.warn(
                    f'We do not suggest construct and initialize EMA model {k}'
                    ' in hook. You may explicitly define it by yourself.')
