# Copyright (c) OpenMMLab. All rights reserved.
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Union

import mmengine
# import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmengine import MessageHub
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.optim import OptimWrapperDict
from torch import Tensor

from mmgen.registry import MODELS, MODULES
from mmgen.structures import GenDataSample, PixelData
from mmgen.utils.typing import ForwardInputs, NoiseVar, SampleList
from ..architectures.common import get_module_device
from ..common import get_valid_num_batches, noise_sample_fn
from .utils import cosine_beta_schedule, linear_beta_schedule, var_to_tensor

TrainInput = Union[dict, Tensor]
ModelType = Union[Dict, nn.Module]


@MODELS.register_module()
class BasicGaussianDiffusion(BaseModel):

    def __init__(self,
                 denoising: ModelType,
                 betas_cfg,
                 num_timesteps=1000,
                 timestep_sampler: str = 'UniformTimeStepSampler',
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 ddpm_loss: Optional[List[dict]] = None,
                 ema_config: Optional[dict] = None):
        super().__init__(data_preprocessor)

        # build generator
        self.num_timesteps = num_timesteps
        if isinstance(denoising, dict):
            self._denoising_cfg = deepcopy(denoising)
            denoising_args = dict(num_timesteps=num_timesteps)
            denoising = MODULES.build(denoising, default_args=denoising_args)
        self.denoising = denoising

        # get output-related configs from denoising
        self.denoising_var_mode = self.denoising.var_mode
        self.denoising_mean_mode = self.denoising.mean_mode

        # output_channels in denoising may be double, therefore we
        # get number of channels from config
        image_channels = self._denoising_cfg['in_channels']
        # image_size should be the attribute of denoising network
        image_size = self.denoising.image_size
        self.image_shape = [image_channels, image_size, image_size]

        # build time sampler
        if timestep_sampler is not None:
            self.sampler = MODULES.build(
                timestep_sampler,
                default_args=dict(num_timesteps=num_timesteps))
        else:
            self.sampler = None

        self.betas_cfg = deepcopy(betas_cfg)

        # ema configs
        if ema_config is None:
            self._ema_config = None
            self._with_ema_denoising = False
        else:
            self._ema_config = deepcopy(ema_config)
            self._init_ema_model(self._ema_config)
            self._with_ema_denoising = True

        # build losses
        if ddpm_loss is not None:
            self.ddpm_loss = [
                MODULES.build(cfg_, default_args=dict(sampler=self.sampler))
                for cfg_ in ddpm_loss
            ]
            if not isinstance(self.ddpm_loss, nn.ModuleList):
                self.ddpm_loss = nn.ModuleList(self.ddpm_loss)
        else:
            self.ddpm_loss = None

        self.prepare_diffusion_vars()

    @property
    def device(self) -> torch.device:
        """Get current device of the model.

        Returns:
            torch.device: The current device of the model.
        """
        return next(self.parameters()).device

    @property
    def with_ema_denoising(self) -> bool:
        """Whether the denoising adopts exponential moving average.

        Returns:
            bool: If `True`, means this denoising model is adopted to
                exponential moving average and vice versa.
        """
        return self._with_ema_denoising

    def _init_ema_model(self, ema_config: dict):
        """Initialize a EMA model corresponding to the given `ema_config`. If
        `ema_config` is an empty dict or `None`, EMA model will not be
        initialized.

        Args:
            ema_config (dict): Config to initialize the EMA model.
        """
        ema_config.setdefault('type', 'ExponentialMovingAverage')
        self.ema_start = ema_config.pop('start_iter', 0)
        src_model = self.denoising.module if is_model_wrapper(
            self.denoising) else self.denoising
        self.denoising_ema = MODELS.build(
            ema_config, default_args=dict(model=src_model))

    def noise_fn(self,
                 noise: NoiseVar,
                 num_batches: int = 1,
                 timesteps_noise: bool = False) -> Tensor:
        if timesteps_noise:
            if isinstance(noise, Tensor):
                if noise.ndim == 4:
                    # ignore num_batches
                    # duplicate [bz, ch, h, w] to [n, bz, ch, h, w]
                    noise_batch = noise.repeat(
                        [self.num_timesteps, 1, 1, 1, 1])
                elif noise.ndim == 3:
                    # unsqueeze along batch_size
                    num_batches = max(1, num_batches)
                    noise_batch = noise.repeat([num_batches, 1, 1, 1])
                    # duplicate [bz, ch, h, w] to [n, bz, ch, h, w]
                    noise_batch = noise_batch.repeat(
                        [self.num_timesteps, 1, 1, 1, 1])
                else:
                    assert noise.ndim == 5, (
                        'When \'timesteps_noise\' is True, the dimension of '
                        '\'noise\' tensor must be 3, 4 or 5. But receive '
                        f'noise whose shape is {noise.shape}')
                    noise_batch = noise

                # security check for the noise
                assert ((noise_batch.shape[0] == self.num_timesteps)
                        and (noise_batch.shape[-3:] == (self.image_shape))), (
                            'Cannot convert input noise tensor with shape '
                            f'\'{noise.shape}\' to a timesteps noise.')
            else:
                # generate random timestep noise with noise_sample_fn
                noise_batch = torch.stack([
                    noise_sample_fn(
                        noise,
                        num_batches=num_batches,
                        noise_size=self.image_shape,
                    ) for _ in range(self.num_timesteps)
                ],
                                          dim=0)
            # move to target device manually
            return noise_batch.to(self.device)
        else:
            return noise_sample_fn(
                noise=noise,
                num_batches=num_batches,
                noise_size=self.image_shape,
                device=self.device)

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
            return linear_beta_schedule(self.num_timesteps, **self.betas_cfg)
        elif self.betas_schedule == 'cosine':
            return cosine_beta_schedule(self.num_timesteps, **self.betas_cfg)
        else:
            raise AttributeError(f'Unknown method name {self.beta_schedule}'
                                 'for beta schedule.')

    def _get_valid_model(self, batch_inputs: ForwardInputs) -> str:
        """Try get the valid forward mode from inputs.

        - If forward model is defined by one of `batch_inputs`, it will be
          used as forward model.
        - If forward model is not defined by any input, 'ema' will returned if
          :property:`with_ema_denoising` is true. Otherwise, 'orig' will be
          returned.

        Args:
            batch_inputs (ForwardInputs): Inputs passed to :meth:`forward`.

        Returns:
            str: Forward model to generate image. ('orig', 'ema' or
                'ema/orig').
        """
        if isinstance(batch_inputs, dict):
            sample_model = batch_inputs.get('sample_model', None)
        else:  # batch_inputs is a Tensor
            sample_model = None

        # set default value
        if sample_model is None:
            sample_model = 'ema' if self.with_ema_denoising else 'orig'

        # security checking
        assert sample_model in [
            'ema', 'ema/orig', 'orig'
        ], ('Only support \'ema\', \'ema/orig\', \'orig\' '
            f'in {self.__class__.__name__}\'s image sampling.')
        if sample_model in ['ema', 'ema/orig']:
            assert self.with_ema_denoising, (
                f'\'{self.__class__.__name__}\' do not have EMA model.')
        return sample_model

    def forward(self,
                inputs: ForwardInputs,
                data_samples: Optional[list] = None,
                mode: Optional[str] = None):

        # parse batch inputs
        if isinstance(inputs, Tensor):
            noise = inputs
            forward_mode = 'sample'
            sample_kwargs = {}
        else:
            noise = inputs.get('noise', None)
            num_batches = get_valid_num_batches(inputs)
            noise = self.noise_fn(noise=noise, num_batches=num_batches)
            forward_mode = inputs.get('forward_mode', 'sampling')
            sample_kwargs = inputs.get('sample_kwargs', dict())
        assert forward_mode in [
            'sampling', 'recon'
        ], ('Only support \'sampling\' and \'recon\' for \'forward_mode\'. '
            f'But receive \'{forward_mode}\'.')

        # get sample model
        sample_model = self._get_valid_model(inputs)

        # forward_method = self.ddpm_sampling if forward_mode == 'sampling' \
        #     else self.reconstruction_step
        assert forward_mode == 'sampling', (
            'Only support DDPM sampling currently.')
        forward_method = self.ddpm_sampling

        if sample_model in ['ema', 'ema/orig']:
            denoising = self.denoising_ema
        else:  # mode is 'orig'
            denoising = self.denoising

        outputs = forward_method(
            denoising, noise=noise, return_as_datasample=True, **sample_kwargs)

        if sample_model == 'ema/orig':
            denoising = self.denoising
            outputs_orig = forward_method(
                denoising,
                noise=noise,
                num_batches=num_batches,
                **sample_kwargs)
            outputs = dict(ema=outputs, orig=outputs_orig)

        if isinstance(outputs, dict):
            batch_sample_list = []
            for idx in range(num_batches):
                ema_sample = outputs['ema'][idx]
                ema_sample.sample_model = 'ema'
                orig_sample = outputs['orig'][idx]
                orig_sample.sample_model = 'orig'
                gen_sample = GenDataSample(
                    ema=ema_sample,
                    orig=orig_sample,
                    sample_model='ema/orig',
                    forward_mode=forward_mode,
                    sample_kwargs=sample_kwargs)
                batch_sample_list.append(gen_sample)
        else:
            batch_sample_list = []
            for idx in range(num_batches):
                out = outputs[idx]
                out.sample_model = sample_model
                batch_sample_list.append(out)
        return batch_sample_list

    @torch.no_grad()
    def ddpm_sampling(
            self,
            model,
            noise: Tensor,
            # num_batches=0,
            label=None,
            save_intermedia=False,
            timesteps_noise=None,
            show_pbar=False,
            return_as_datasample=False,
            # return_noise=False,
            **kwargs):

        device = get_module_device(self)
        num_batches = noise.shape[0]
        x_t = noise.clone()
        if save_intermedia:
            # save input
            # intermedia = {self.num_timesteps: x_t.clone()}
            intermedia = [x_t.clone()]

        # use timesteps noise if defined
        if timesteps_noise is not None:
            timesteps_noise = self.noise_fn(
                timesteps_noise, num_batches=num_batches, timesteps_noise=True)

        batched_timesteps = torch.arange(self.num_timesteps - 1, -1,
                                         -1).long().to(device)
        if show_pbar:
            pbar = mmengine.ProgressBar(self.num_timesteps)
        for t in batched_timesteps:
            batched_t = t.expand(x_t.shape[0])
            step_noise = timesteps_noise[t, ...] \
                if timesteps_noise is not None else None

            x_t = self.denoising_step(
                model, x_t, batched_t, noise=step_noise, label=label, **kwargs)
            if save_intermedia:
                intermedia.append(x_t.clone())
                # intermedia[int(t)] = x_t.cpu().clone()
            if show_pbar:
                pbar.update()
        denoising_results = torch.stack(intermedia, dim=1) \
            if save_intermedia else x_t

        if show_pbar:
            sys.stdout.write('\n')

        # convert to data sample if need
        if return_as_datasample:
            batch_sample_list = []
            for res in denoising_results:
                gen_sample = GenDataSample(fake_img=PixelData(data=res))
                batch_sample_list.append(gen_sample)
            return batch_sample_list

        # return tensor
        return denoising_results

    def val_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data.

        Calls ``self.data_preprocessor(data)`` and
        ``self(inputs, data_sample, mode=None)`` in order. Return the
        generated results which will be passed to evaluator.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            SampleList: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        # outputs = self(inputs_dict, data_sample)
        outputs = self(**data)
        return outputs

    def test_step(self, data: dict) -> SampleList:
        """Gets the generated image of given data. Same as :meth:`val_step`.

        Args:
            data (dict): Data sampled from metric specific
                sampler. More detials in `Metrics` and `Evaluator`.

        Returns:
            SampleList: Generated image or image dict.
        """
        data = self.data_preprocessor(data)
        # outputs = self(inputs_dict, data_sample)
        outputs = self(**data)
        return outputs

    def denoising_loss(self, outputs_dict):
        losses_dict = {}

        # forward losses
        for loss_fn in self.ddpm_loss:
            losses_dict[loss_fn.loss_name()] = loss_fn(outputs_dict)

        loss, log_vars = self.parse_losses(losses_dict)

        # update collected log_var from loss_fn
        for loss_fn in self.ddpm_loss:
            if hasattr(loss_fn, 'log_vars'):
                log_vars.update(loss_fn.log_vars)
        return loss, log_vars

    def train_step(self, data: dict, optim_wrapper: OptimWrapperDict):

        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')

        # real_imgs, data_samples = self.data_preprocessor(data)
        data = self.data_preprocessor(data)
        real_imgs = data['inputs']
        denoising_dict_ = self.reconstruction_step(
            self.denoising,
            real_imgs,
            timesteps=self.sampler,
            return_noise=True)
        denoising_dict_['real_imgs'] = real_imgs
        loss, log_vars = self.denoising_loss(denoising_dict_)
        optim_wrapper['denoising'].update_params(loss)

        # update EMA
        if self.with_ema_denoising and (curr_iter + 1) >= self.ema_start:
            self.denoising_ema.update_parameters(
                self.denoising_ema.
                module if is_model_wrapper(self.denoising) else self.denoising)
            # if not update buffer, copy buffer from orig model
            if not self.denoising_ema.update_buffers:
                self.denoising_ema.sync_buffers(
                    self.denoising.module
                    if is_model_wrapper(self.denoising) else self.denoising)
        elif self.with_ema_denoising:
            # before ema, copy weights from orig
            self.denoising_ema.sync_parameters(
                self.denoising.
                module if is_model_wrapper(self.denoising) else self.denoising)

        return log_vars

    def reconstruction_step(self,
                            model,
                            real_imgs,
                            noise=None,
                            label=None,
                            timesteps=None,
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
            real_imgs (Tensor): Real images from dataloader.
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
        # 0. prepare for timestep, noise and label
        num_batches = real_imgs.shape[0]
        device = self.data_preprocessor.device

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

        output_dict = defaultdict(list)
        # loop all timesteps
        for timestep in timesteps:
            # 1. get diffusion results and parameters
            noise_batches = self.noise_fn(
                noise, num_batches=num_batches).to(device)

            diffusion_batches = self.q_sample(real_imgs, timestep,
                                              noise_batches)
            # 2. get denoising results.
            denoising_batches = self.denoising_step(
                model,
                diffusion_batches,
                timestep,
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
        # get noise for reparameterization
        noise = self.noise_fn(noise, num_batches=num_batches)
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
        noise = self.noise_fn(noise, num_batches=num_batches)
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
