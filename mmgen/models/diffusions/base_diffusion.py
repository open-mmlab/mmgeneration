# Copyright (c) OpenMMLab. All rights reserved.
import sys
from abc import ABCMeta
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from ..architectures.common import get_module_device
from ..builder import MODELS, build_module
from .utils import _get_label_batch, _get_noise_batch, var_to_tensor


@MODELS.register_module()
class BasicGaussianDiffusion(nn.Module, metaclass=ABCMeta):
    """Basic module for gaussian Diffusion Denoising Probabilistic Models. A
    diffusion probabilistic model (which we will call a 'diffusion model' for
    brevity) is a parameterized Markov chain trained using variational
    inference to produce samples matching the data after finite time.

    The design of this module implements DDPM and improve-DDPM according to
    "Denoising Diffusion Probabilistic Models" (2020) and "Improved Denoising
    Diffusion Probabilistic Models" (2021).

    Args:
        denoising (dict): Config for denoising model.
        ddpm_loss (dict): Config for losses of DDPM.
        betas_cfg (dict): Config for betas in diffusion process.
        num_timesteps (int, optional): The number of timesteps of the diffusion
            process. Defaults to 1000.
        num_classes (int | None, optional): The number of conditional classes.
            Defaults to None.
        sample_method (string, optional): Sample method for the denoising
            process. Support 'DDPM' and 'DDIM'. Defaults to 'DDPM'.
        timesteps_sampler (string, optional): How to sample timesteps in
            training process. Defaults to `UniformTimeStepSampler`.
        train_cfg (dict | None, optional): Config for training schedule.
            Defaults to None.
        test_cfg (dict | None, optional): Config for testing schedule. Defaults
            to None.
    """

    def __init__(self,
                 denoising,
                 ddpm_loss,
                 betas_cfg,
                 num_timesteps=1000,
                 num_classes=0,
                 sample_method='DDPM',
                 timestep_sampler='UniformTimeStepSampler',
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.fp16_enable = False
        # build denoising module in this function
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.sample_method = sample_method
        self._denoising_cfg = deepcopy(denoising)
        self.denoising = build_module(
            denoising,
            default_args=dict(
                num_classes=num_classes, num_timesteps=num_timesteps))

        # get output-related configs from denoising
        self.denoising_var_mode = self.denoising.var_mode
        self.denoising_mean_mode = self.denoising.mean_mode
        # output_channels in denoising may be double, therefore we
        # get number of channels from config
        image_channels = self._denoising_cfg['in_channels']
        # image_size should be the attribute of denoising network
        image_size = self.denoising.image_size

        image_shape = torch.Size([image_channels, image_size, image_size])
        self.image_shape = image_shape
        self.get_noise = partial(
            _get_noise_batch,
            image_shape=image_shape,
            num_timesteps=self.num_timesteps)
        self.get_label = partial(
            _get_label_batch, num_timesteps=self.num_timesteps)

        # build sampler
        if timestep_sampler is not None:
            self.sampler = build_module(
                timestep_sampler,
                default_args=dict(num_timesteps=num_timesteps))
        else:
            self.sampler = None

        # build losses
        if ddpm_loss is not None:
            self.ddpm_loss = build_module(
                ddpm_loss, default_args=dict(sampler=self.sampler))
            if not isinstance(self.ddpm_loss, nn.ModuleList):
                self.ddpm_loss = nn.ModuleList([self.ddpm_loss])
        else:
            self.ddpm_loss = None

        self.betas_cfg = deepcopy(betas_cfg)

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parse_test_cfg()

        self.prepare_diffusion_vars()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        if self.train_cfg is None:
            self.train_cfg = dict()
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            self.denoising_ema = deepcopy(self.denoising)

        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        if self.use_ema:
            self.denoising_ema = deepcopy(self.denoising)

    def _get_loss(self, outputs_dict):
        losses_dict = {}

        # forward losses
        for loss_fn in self.ddpm_loss:
            losses_dict[loss_fn.loss_name()] = loss_fn(outputs_dict)

        loss, log_vars = self._parse_losses(losses_dict)

        # update collected log_var from loss_fn
        for loss_fn in self.ddpm_loss:
            if hasattr(loss_fn, 'log_vars'):
                log_vars.update(loss_fn.log_vars)
        return loss, log_vars

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensor')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self,
                   data,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        """The iteration step during training.

        This method defines an iteration step during training. Different from
        other repo in **MM** series, we allow the back propagation and
        optimizer updating to directly follow the iterative training schedule
        of DDPMs.
        Of course, we will show that you can also move the back
        propagation outside of this method, and then optimize the parameters
        in the optimizer hook. But this will cause extra GPU memory cost as a
        result of retaining computational graph. Otherwise, the training
        schedule should be modified in the detailed implementation.


        Args:
            optimizer (dict): Dict contains optimizer for denoising network.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.
        """

        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        real_imgs = data[self.real_img_key]
        # denoising training
        optimizer['denoising'].zero_grad()
        denoising_dict_ = self.reconstruction_step(
            data,
            timesteps=self.sampler,
            sample_model='orig',
            return_noise=True)
        denoising_dict_['iteration'] = curr_iter
        denoising_dict_['real_imgs'] = real_imgs
        denoising_dict_['loss_scaler'] = loss_scaler

        loss, log_vars = self._get_loss(denoising_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss, optimizer['denoising'],
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            loss.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['denoising'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['denoising'])
            # loss_scaler.update will be called in runner.train()
        else:
            optimizer['denoising'].step()

        # image used for vislization
        results = dict(
            real_imgs=real_imgs,
            x_0_pred=denoising_dict_['x_0_pred'],
            x_t=denoising_dict_['diffusion_batches'],
            x_t_1=denoising_dict_['fake_img'])
        outputs = dict(
            log_vars=log_vars, num_samples=real_imgs.shape[0], results=results)

        if hasattr(self, 'iteration'):
            self.iteration += 1

        return outputs

    def reconstruction_step(self,
                            data_batch,
                            noise=None,
                            label=None,
                            timesteps=None,
                            sample_model='orig',
                            return_noise=False,
                            **kwargs):
        """Reconstruction step at corresponding `timestep`. To be noted that,
        denoisint target ``x_t`` for each timestep are all generated from real
        images, but not the denoising result from denoising network.

        ``sample_from_noise`` focus on generate samples start from **random
        (or given) noise**. Therefore, we design this function to realize a
        reconstruction process for the given images.

        If `timestep` is None, automatically perform reconstruction at all
        timesteps.

        Args:
            data_batch (dict): Input data from dataloader.
            noise (torch.Tensor | callable | None): Noise used in diffusion
                process. You can directly give a batch of noise through a
                ``torch.Tensor`` or offer a callable function to sample a
                batch of noise data. Otherwise, the ``None`` indicates to use
                the default noise sampler. Defaults to None.
            label (torch.Tensor | None , optional): The conditional label of
                the input image. Defaults to None.
            timestep (int | list | torch.Tensor | callable | None): Target
                timestep to perform reconstruction.
            sampel_model (str, optional): Use which model to sample fake
                images. Defaults to `'orig'`.
            return_noise (bool, optional): If True,``noise_batch``, ``label``
                and all other intermedia variables will be returned together
                with ``fake_img`` in a dict. Defaults to False.

        Returns:
            torch.Tensor | dict: The output may be the direct synthesized
                images in ``torch.Tensor``. Otherwise, a dict with required
                data , including generated images, will be returned.
        """
        assert sample_model in [
            'orig', 'ema'
        ], ('We only support \'orig\' and \'ema\' for '
            f'\'reconstruction_step\', but receive \'{sample_model}\'.')

        denoising_model = self.denoising if sample_model == 'orig' \
            else self.denoising_ema
        # 0. prepare for timestep, noise and label
        device = get_module_device(self)
        real_imgs = data_batch[self.real_img_key]
        num_batches = real_imgs.shape[0]

        if timesteps is None:
            # default to performing the whole reconstruction process
            timesteps = torch.LongTensor([
                t for t in range(self.num_timesteps)
            ]).view(self.num_timesteps, 1)
            timesteps = timesteps.repeat([1, num_batches])
        if isinstance(timesteps, (int, list)):
            timesteps = torch.LongTensor(timesteps)
        elif callable(timesteps):
            timestep_generator = timesteps
            timesteps = timestep_generator(num_batches)
        else:
            assert isinstance(timesteps, torch.Tensor), (
                'we only support int list tensor or a callable function')
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.to(get_module_device(self))

        if noise is not None:
            assert 'noise' not in data_batch, (
                'Receive \'noise\' in both data_batch and passed arguments.')
        if noise is None:
            noise = data_batch['noise'] if 'noise' in data_batch else None

        if self.num_classes > 0:
            if label is not None:
                assert 'label' not in data_batch, (
                    'Receive \'label\' in both data_batch '
                    'and passed arguments.')
            if label is None:
                label = data_batch['label'] if 'label' in data_batch else None
            label_batches = self.get_label(
                label, num_batches=num_batches).to(device)
        else:
            label_batches = None

        output_dict = defaultdict(list)
        # loop all timesteps
        for timestep in timesteps:
            # 1. get diffusion results and parameters
            noise_batches = self.get_noise(
                noise, num_batches=num_batches).to(device)

            diffusion_batches = self.q_sample(real_imgs, timestep,
                                              noise_batches)
            # 2. get denoising results.
            denoising_batches = self.denoising_step(
                denoising_model,
                diffusion_batches,
                timestep,
                label=label_batches,
                return_noise=return_noise,
                clip_denoised=not self.training)
            # 3. get ground truth by q_posterior
            target_batches = self.q_posterior_mean_variance(
                real_imgs, diffusion_batches, timestep, logvar=True)
            if return_noise:
                output_dict_ = dict(
                    timesteps=timestep,
                    noise=noise_batches,
                    diffusion_batches=diffusion_batches)
                if self.num_classes > 0:
                    output_dict_['label'] = label_batches
                output_dict_.update(denoising_batches)
                output_dict_.update(target_batches)
            else:
                output_dict_ = dict(fake_img=denoising_batches)
            # update output of `timestep` to output_dict
            for k, v in output_dict_.items():
                if k in output_dict:
                    output_dict[k].append(v)
                else:
                    output_dict[k] = [v]

        # 4. concentrate list to tensor
        for k, v in output_dict.items():
            output_dict[k] = torch.cat(v, dim=0)

        # 5. return results
        if return_noise:
            return output_dict
        return output_dict['fake_img']

    def sample_from_noise(self,
                          noise,
                          num_batches=0,
                          sample_model='ema/orig',
                          label=None,
                          **kwargs):
        """Sample images from noises by using Denoising model.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional):  The number of batch size.
                Defaults to 0.
            sample_model (str, optional): The model to sample. If ``ema/orig``
                is passed, this method will try to sample from ema (if
                ``self.use_ema == True``) and orig model. Defaults to
                'ema/orig'.
            label (torch.Tensor | None , optional): The conditional label.
                Defaults to None.

        Returns:
            torch.Tensor | dict: The output may be the direct synthesized
                images in ``torch.Tensor``. Otherwise, a dict with queried
                data, including generated images, will be returned.
        """
        # get sample function by name
        sample_fn_name = f'{self.sample_method.upper()}_sample'
        if not hasattr(self, sample_fn_name):
            raise AttributeError(
                f'Cannot find sample method [{sample_fn_name}] correspond '
                f'to [{self.sample_method}].')
        sample_fn = getattr(self, sample_fn_name)

        if sample_model == 'ema':
            assert self.use_ema
            _model = self.denoising_ema
        elif sample_model == 'ema/orig' and self.use_ema:
            _model = self.denoising_ema
        else:
            _model = self.denoising

        outputs = sample_fn(
            _model,
            noise=noise,
            num_batches=num_batches,
            label=label,
            **kwargs)

        if isinstance(outputs, dict) and 'noise_batch' in outputs:
            # return_noise is True
            noise = outputs['x_t']
            label = outputs['label']
            kwargs['timesteps_noise'] = outputs['noise_batch']
            fake_img = outputs['fake_img']
        else:
            fake_img = outputs

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.denoising
            outputs_ = sample_fn(
                _model, noise=noise, num_batches=num_batches, **kwargs)
            if isinstance(outputs_, dict) and 'noise_batch' in outputs_:
                # return_noise is True
                fake_img_ = outputs_['fake_img']
            else:
                fake_img_ = outputs_
            if isinstance(fake_img, dict):
                # save_intermedia is True
                fake_img = {
                    k: torch.cat([fake_img[k], fake_img_[k]], dim=0)
                    for k in fake_img.keys()
                }
            else:
                fake_img = torch.cat([fake_img, fake_img_], dim=0)

        return fake_img

    @torch.no_grad()
    def DDPM_sample(self,
                    model,
                    noise=None,
                    num_batches=0,
                    label=None,
                    save_intermedia=False,
                    timesteps_noise=None,
                    return_noise=False,
                    show_pbar=False,
                    **kwargs):
        """DDPM sample from random noise.
        Args:
            model (torch.nn.Module): Denoising model used to sample images.
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            label (torch.Tensor | None , optional): The conditional label.
                Defaults to None.
            save_intermedia (bool, optional): Whether to save denoising result
                of intermedia timesteps. If set as True, will return a dict
                which key and value are denoising timestep and denoising
                result. Otherwise, only the final denoising result will be
                returned. Defaults to False.
            timesteps_noise (torch.Tensor, optional): Noise term used in each
                denoising timestep. If given, the input noise will be shaped to
                [num_timesteps, b, c, h, w]. If set as None, noise of each
                denoising timestep will be randomly sampled. Default as None.
            return_noise (bool, optional): If True, a dict contains
                ``noise_batch``, ``x_t`` and ``label`` will be returned
                together with the denoising results, and the key of denoising
                results is ``fake_img``. To be noted that ``noise_batches``
                will shape as [num_timesteps, b, c, h, w]. Defaults to False.
            show_pbar (bool, optional): If True, a progress bar will be
                displayed. Defaults to False.
        Returns:
            torch.Tensor | dict: If ``save_intermedia``, a dict contains
                denoising results of each timestep will be returned.
                Otherwise, only the final denoising result will be returned.
        """
        device = get_module_device(self)
        noise = self.get_noise(noise, num_batches=num_batches).to(device)
        x_t = noise.clone()
        if save_intermedia:
            # save input
            intermedia = {self.num_timesteps: x_t.clone()}

        # use timesteps noise if defined
        if timesteps_noise is not None:
            timesteps_noise = self.get_noise(
                timesteps_noise, num_batches=num_batches,
                timesteps_noise=True).to(device)

        batched_timesteps = torch.arange(self.num_timesteps - 1, -1,
                                         -1).long().to(device)
        if show_pbar:
            pbar = mmcv.ProgressBar(self.num_timesteps)
        for t in batched_timesteps:
            batched_t = t.expand(x_t.shape[0])
            step_noise = timesteps_noise[t, ...] \
                if timesteps_noise is not None else None

            x_t = self.denoising_step(
                model, x_t, batched_t, noise=step_noise, label=label, **kwargs)
            if save_intermedia:
                intermedia[int(t)] = x_t.cpu().clone()
            if show_pbar:
                pbar.update()
        denoising_results = intermedia if save_intermedia else x_t

        if show_pbar:
            sys.stdout.write('\n')

        if return_noise:
            return dict(
                noise_batch=timesteps_noise,
                x_t=noise,
                label=label,
                fake_img=denoising_results)

        return denoising_results

    def prepare_diffusion_vars(self):
        """Prepare for variables used in the diffusion process."""
        self.betas = self.get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_bar = np.cumproduct(self.alphas, axis=0)
        self.alphas_bar_prev = np.append(1.0, self.alphas_bar[:-1])
        self.alphas_bar_next = np.append(self.alphas_bar[1:], 0.0)

        # calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alphas_bar = np.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1.0 - self.alphas_bar)
        self.log_one_minus_alphas_bar = np.log(1.0 - self.alphas_bar)
        self.sqrt_recip_alplas_bar = np.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = np.sqrt(1.0 / self.alphas_bar - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.tilde_betas_t = self.betas * (1 - self.alphas_bar_prev) / (
            1 - self.alphas_bar)
        # clip log var for tilde_betas_0 = 0
        self.log_tilde_betas_t_clipped = np.log(
            np.append(self.tilde_betas_t[1], self.tilde_betas_t[1:]))
        self.tilde_mu_t_coef1 = np.sqrt(
            self.alphas_bar_prev) / (1 - self.alphas_bar) * self.betas
        self.tilde_mu_t_coef2 = np.sqrt(
            self.alphas) * (1 - self.alphas_bar_prev) / (1 - self.alphas_bar)

    def get_betas(self):
        """Get betas by defined schedule method in diffusion process."""
        self.betas_schedule = self.betas_cfg.pop('type')
        if self.betas_schedule == 'linear':
            return self.linear_beta_schedule(self.num_timesteps,
                                             **self.betas_cfg)
        elif self.betas_schedule == 'cosine':
            return self.cosine_beta_schedule(self.num_timesteps,
                                             **self.betas_cfg)
        else:
            raise AttributeError(f'Unknown method name {self.beta_schedule}'
                                 'for beta schedule.')

    @staticmethod
    def linear_beta_schedule(diffusion_timesteps, beta_0=1e-4, beta_T=2e-2):
        r"""Linear schedule from Ho et al, extended to work for any number of
        diffusion steps.

        Args:
            diffusion_timesteps (int): The number of betas to produce.
            beta_0 (float, optional): `\beta` at timestep 0. Defaults to 1e-4.
            beta_T (float, optional): `\beta` at timestep `T` (the final
                diffusion timestep). Defaults to 2e-2.

        Returns:
            np.ndarray: Betas used in diffusion process.
        """
        scale = 1000 / diffusion_timesteps
        beta_0 = scale * beta_0
        beta_T = scale * beta_T
        return np.linspace(
            beta_0, beta_T, diffusion_timesteps, dtype=np.float64)

    @staticmethod
    def cosine_beta_schedule(diffusion_timesteps, max_beta=0.999, s=0.008):
        r"""Create a beta schedule that discretizes the given alpha_t_bar
        function, which defines the cumulative product of `(1-\beta)` over time
        from `t = [0, 1]`.

        Args:
            diffusion_timesteps (int): The number of betas to produce.
            max_beta (float, optional): The maximum beta to use; use values
                lower than 1 to prevent singularities. Defaults to 0.999.
            s (float, optional): Small offset to prevent `\beta` from being too
                small near `t = 0` Defaults to 0.008.

        Returns:
            np.ndarray: Betas used in diffusion process.
        """

        def f(t, T, s):
            return np.cos((t / T + s) / (1 + s) * np.pi / 2)**2

        betas = []
        for t in range(diffusion_timesteps):
            alpha_bar_t = f(t + 1, diffusion_timesteps, s)
            alpha_bar_t_1 = f(t, diffusion_timesteps, s)
            betas_t = 1 - alpha_bar_t / alpha_bar_t_1
            betas.append(min(betas_t, max_beta))
        return np.array(betas)

    def q_sample(self, x_0, t, noise=None):
        r"""Get diffusion result at timestep `t` by `q(x_t | x_0)`.

        Args:
            x_0 (torch.Tensor): Original image without diffusion.
            t (torch.Tensor): Target diffusion timestep.
            noise (torch.Tensor, optional): Noise used in reparameteration
                trick. Default to None.

        Returns:
            torch.tensor: Diffused image `x_t`.
        """
        device = get_module_device(self)
        num_batches = x_0.shape[0]
        tar_shape = x_0.shape
        noise = self.get_noise(noise, num_batches=num_batches)
        mean = var_to_tensor(self.sqrt_alphas_bar, t, tar_shape, device)
        std = var_to_tensor(self.sqrt_one_minus_alphas_bar, t, tar_shape,
                            device)

        return x_0 * mean + noise * std

    def q_mean_log_variance(self, x_0, t):
        r"""Get mean and log_variance of diffusion process `q(x_t | x_0)`.

        Args:
            x_0 (torch.tensor): The original image before diffusion, shape as
                [bz, ch, H, W].
            t (torch.tensor): Target timestep, shape as [bz, ].

        Returns:
            Tuple(torch.tensor): Tuple contains mean and log variance.
        """
        device = get_module_device(self)
        tar_shape = x_0.shape
        mean = var_to_tensor(self.sqrt_alphas_bar, t, tar_shape, device) * x_0
        logvar = var_to_tensor(self.log_one_minus_alphas_bar, t, tar_shape,
                               device)
        return mean, logvar

    def q_posterior_mean_variance(self,
                                  x_0,
                                  x_t,
                                  t,
                                  need_var=True,
                                  logvar=False):
        r"""Get mean and variance of diffusion posterior
            `q(x_{t-1} | x_t, x_0)`.

        Args:
            x_0 (torch.tensor): The original image before diffusion, shape as
                [bz, ch, H, W].
            t (torch.tensor): Target timestep, shape as [bz, ].
            need_var (bool, optional): If set as ``True``, this function will
                return a dict contains ``var``. Otherwise, only mean will be
                returned, ``logvar`` will be ignored. Defaults to True.
            logvar (bool, optional): If set as ``True``, the returned dict
                will additionally contain ``logvar``. This argument will be
                considered only if ``var == True``. Defaults to False.

        Returns:
            torch.Tensor | dict: If ``var``, will return a dict contains
                ``mean`` and ``var``. Otherwise, only mean will be returned.
                If ``var`` and ``logvar`` set at as True simultaneously, the
                returned dict will additional contain ``logvar``.
        """
        device = get_module_device(self)
        tar_shape = x_0.shape
        tilde_mu_t_coef1 = var_to_tensor(self.tilde_mu_t_coef1, t, tar_shape,
                                         device)
        tilde_mu_t_coef2 = var_to_tensor(self.tilde_mu_t_coef2, t, tar_shape,
                                         device)
        posterior_mean = tilde_mu_t_coef1 * x_0 + tilde_mu_t_coef2 * x_t
        # do not need variance, just return mean
        if not need_var:
            return posterior_mean
        posterior_var = var_to_tensor(self.tilde_betas_t, t, tar_shape, device)
        out_dict = dict(
            mean_posterior=posterior_mean, var_posterior=posterior_var)
        if logvar:
            posterior_logvar = var_to_tensor(self.log_tilde_betas_t_clipped, t,
                                             tar_shape, device)
            out_dict['logvar_posterior'] = posterior_logvar
        return out_dict

    def p_mean_variance(self,
                        denoising_output,
                        x_t,
                        t,
                        clip_denoised=True,
                        denoised_fn=None):
        r"""Get mean, variance, log variance of denoising process
        `p(x_{t-1} | x_{t})` and predicted `x_0`.

        Args:
            denoising_output (dict[torch.Tensor]): The output from denoising
                model.
            x_t (torch.Tensor): Diffused image at timestep `t` to denoising.
            t (torch.Tensor): Current timestep.
            clip_denoised (bool, optional): Whether cliped sample results into
                [-1, 1]. Defaults to True.
            denoised_fn (callable, optional): If not None, a function which
                applies to the predicted ``x_0`` before it is passed to the
                following sampling procedure. Noted that this function will be
                applies before ``clip_denoised``. Defaults to None.

        Returns:
            dict: A dict contains ``var_pred``, ``logvar_pred``, ``mean_pred``
                and ``x_0_pred``.
        """
        target_shape = x_t.shape
        device = get_module_device(self)
        # prepare for var and logvar
        if self.denoising_var_mode.upper() == 'LEARNED':
            # NOTE: the output actually LEARNED_LOG_VAR
            logvar_pred = denoising_output['logvar']
            varpred = torch.exp(logvar_pred)

        elif self.denoising_var_mode.upper() == 'LEARNED_RANGE':
            # NOTE: the output actually LEARNED_FACTOR
            var_factor = denoising_output['factor']
            lower_bound_logvar = var_to_tensor(self.log_tilde_betas_t_clipped,
                                               t, target_shape, device)
            upper_bound_logvar = var_to_tensor(
                np.log(self.betas), t, target_shape, device)
            logvar_pred = var_factor * upper_bound_logvar + (
                1 - var_factor) * lower_bound_logvar
            varpred = torch.exp(logvar_pred)

        elif self.denoising_var_mode.upper() == 'FIXED_LARGE':
            # use betas as var
            varpred = var_to_tensor(
                np.append(self.tilde_betas_t[1], self.betas), t, target_shape,
                device)
            logvar_pred = torch.log(varpred)

        elif self.denoising_var_mode.upper() == 'FIXED_SMALL':
            # use posterior (tilde_betas)  as var
            varpred = var_to_tensor(self.tilde_betas_t, t, target_shape,
                                    device)
            logvar_pred = var_to_tensor(self.log_tilde_betas_t_clipped, t,
                                        target_shape, device)
        else:
            raise AttributeError('Unknown denoising var output type '
                                 f'[{self.denoising_var_mode}].')

        def process_x_0(x):
            if denoised_fn is not None and callable(denoised_fn):
                x = denoised_fn(x)
            return x.clamp(-1, 1) if clip_denoised else x

        # prepare for mean and x_0
        if self.denoising_mean_mode.upper() == 'EPS':
            eps_pred = denoising_output['eps_t_pred']
            # We can get x_{t-1} with eps in two following approaches:
            # 1. eps --(Eq 15)--> \hat{x_0} --(Eq 7)--> \tilde_mu --> x_{t-1}
            # 2. eps --(Eq 11)--> \mu_{\theta} --(Eq 7)--> x_{t-1}
            # We can verify \tilde_mu in method 1 and \mu_{\theta} in method 2
            # are almost same (error of 1e-4) with the same eps input.
            # In our implementation, we use method (1) to consistent with
            # the official ones.
            # If you want to calculate \mu_{\theta} with method 2, you can
            # use the following code:
            # coef1 = var_to_tensor(
            #     np.sqrt(1.0 / self.alphas), t, tar_shape)
            # coef2 = var_to_tensor(
            #     self.betas / self.sqrt_one_minus_alphas_bar, t, tar_shape)
            # mu_theta = coef1 * (x_t - coef2 * eps)
            x_0_pred = process_x_0(self.pred_x_0_from_eps(eps_pred, x_t, t))
            mean_pred = self.q_posterior_mean_variance(
                x_0_pred, x_t, t, need_var=False)
        elif self.denoising_mean_mode.upper() == 'START_X':
            x_0_pred = process_x_0(denoising_output['x_0_pred'])
            mean_pred = self.q_posterior_mean_variance(
                x_0_pred, x_t, t, need_var=False)
        elif self.denoising_mean_mode.upper() == 'PREVIOUS_X':
            # NOTE: the output actually PREVIOUS_X_MEAN (MU_THETA)
            # because this actually predict \mu_{\theta}
            mean_pred = denoising_output['x_tm1_pred']
            x_0_pred = process_x_0(self.pred_x_0_from_x_tm1(mean_pred, x_t, t))
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean_mode}].')

        output_dict = dict(
            var_pred=varpred,
            logvar_pred=logvar_pred,
            mean_pred=mean_pred,
            x_0_pred=x_0_pred)
        # avoid return duplicate variables
        return {
            k: output_dict[k]
            for k in output_dict.keys() if k not in denoising_output
        }

    def denoising_step(self,
                       model,
                       x_t,
                       t,
                       noise=None,
                       label=None,
                       clip_denoised=True,
                       denoised_fn=None,
                       model_kwargs=None,
                       return_noise=False):
        """Single denoising step. Get `x_{t-1}` from ``x_t`` and ``t``.

        Args:
            model (torch.nn.Module): Denoising model used to sample images.
            x_t (torch.Tensor): Input diffused image.
            t (torch.Tensor): Current timestep.
            noise (torch.Tensor | callable | None): Noise for
                reparameterization trick. You can directly give a batch of
                noise through a ``torch.Tensor`` or offer a callable function
                to sample a batch of noise data. Otherwise, the ``None``
                indicates to use the default noise sampler.
            label (torch.Tensor | callable | None): You can directly give a
                batch of label through a ``torch.Tensor`` or offer a callable
                function to sample a batch of label data. Otherwise, the
                ``None`` indicates to use the default label sampler.
            clip_denoised (bool, optional): Whether to clip sample results into
                [-1, 1]. Defaults to False.
            denoised_fn (callable, optional): If not None, a function which
                applies to the predicted ``x_0`` prediction before it is used
                to sample. Applies before ``clip_denoised``. Defaults to None.
            model_kwargs (dict, optional): Arguments passed to denoising model.
                Defaults to None.
            return_noise (bool, optional): If True, ``noise_batch``, outputs
                from denoising model and ``p_mean_variance`` will be returned
                in a dict with ``fake_img``. Defaults to False.

        Return:
            torch.Tensor | dict: If not ``return_noise``, only the denoising
                image will be returned. Otherwise, the dict contains
                ``fake_image``, ``noise_batch`` and outputs from denoising
                model and ``p_mean_variance`` will be returned.
        """
        # init model_kwargs as dict if not passed
        if model_kwargs is None:
            model_kwargs = dict()
        model_kwargs.update(dict(return_noise=return_noise))

        denoising_output = model(x_t, t, label=label, **model_kwargs)
        p_output = self.p_mean_variance(denoising_output, x_t, t,
                                        clip_denoised, denoised_fn)
        mean_pred = p_output['mean_pred']
        var_pred = p_output['var_pred']

        num_batches = x_t.shape[0]
        device = get_module_device(self)
        # get noise for reparameterization
        noise = self.get_noise(noise, num_batches=num_batches).to(device)
        nonzero_mask = ((t != 0).float().view(-1,
                                              *([1] * (len(x_t.shape) - 1))))

        # Here we directly use var_pred instead logvar_pred,
        # only error of 1e-12.
        # logvar_pred = p_output['logvar_pred']
        # sample = mean_pred + \
        #     nonzero_mask * torch.exp(0.5 * logvar_pred) * noise
        sample = mean_pred + nonzero_mask * torch.sqrt(var_pred) * noise
        if return_noise:
            return dict(
                fake_img=sample,
                noise_repar=noise,
                **denoising_output,
                **p_output)
        return sample

    def pred_x_0_from_eps(self, eps, x_t, t):
        r"""Predict x_0 from eps by Equ 15 in DDPM paper:

        .. math::
            x_0 = \frac{(x_t - \sqrt{(1-\bar{\alpha}_t)} * eps)}
            {\sqrt{\bar{\alpha}_t}}

        Args:
            eps (torch.Tensor)
            x_t (torch.Tensor)
            t (torch.Tensor)

        Returns:
            torch.tensor: Predicted ``x_0``.
        """
        device = get_module_device(self)
        tar_shape = x_t.shape
        coef1 = var_to_tensor(self.sqrt_recip_alplas_bar, t, tar_shape, device)
        coef2 = var_to_tensor(self.sqrt_recipm1_alphas_bar, t, tar_shape,
                              device)
        return x_t * coef1 - eps * coef2

    def pred_x_0_from_x_tm1(self, x_tm1, x_t, t):
        r"""
        Predict `x_0` from `x_{t-1}`. (actually from `\mu_{\theta}`).
        `(\mu_{\theta} - coef2 * x_t) / coef1`, where `coef1` and `coef2`
        are from Eq 6 of the DDPM paper.

        NOTE: This function actually predict ``x_0`` from ``mu_theta`` (mean
        of ``x_{t-1}``).

        Args:
            x_tm1 (torch.Tensor): `x_{t-1}` used to predict `x_0`.
            x_t (torch.Tensor): `x_{t}` used to predict `x_0`.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Predicted `x_0`.

        """
        device = get_module_device(self)
        tar_shape = x_t.shape
        coef1 = var_to_tensor(self.tilde_mu_t_coef1, t, tar_shape, device)
        coef2 = var_to_tensor(self.tilde_mu_t_coef2, t, tar_shape, device)
        x_0 = (x_tm1 - coef2 * x_t) / coef1
        return x_0

    def forward_train(self, data, **kwargs):
        """Deprecated forward function in training."""
        raise NotImplementedError(
            'In MMGeneration, we do NOT recommend users to call'
            'this function, because the train_step function is designed for '
            'the training process.')

    def forward_test(self, data, **kwargs):
        """Testing function for Diffusion Denosing Probability Models.

        Args:
            data (torch.Tensor | dict | None): Input data. This data will be
                passed to different methods.
        """
        mode = kwargs.pop('mode', 'sampling')
        if mode == 'sampling':
            return self.sample_from_noise(data, **kwargs)
        elif mode == 'reconstruction':
            # this mode is design for evaluation likelood metrics
            return self.reconstruction_step(data, **kwargs)

        raise NotImplementedError('Other specific testing functions should'
                                  ' be implemented by the sub-classes.')

    def forward(self, data, return_loss=False, **kwargs):
        """Forward function.

        Args:
            data (dict | torch.Tensor): Input data dictionary.
            return_loss (bool, optional): Whether in training or testing.
                Defaults to False.

        Returns:
            dict: Output dictionary.
        """
        if return_loss:
            return self.forward_train(data, **kwargs)

        return self.forward_test(data, **kwargs)
