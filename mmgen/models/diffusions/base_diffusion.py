from abc import ABCMeta
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from ..architectures.common import get_module_device
from ..builder import MODELS, build_module
from .utils import _get_label_batch, _get_noise_batch, var_to_tensor


@MODELS.register_module()
class BasicGaussianDiffusion(nn.Module, metaclass=ABCMeta):
    """BasicGaussianDiffusion Module.

    Args:
        denoising (dict): Config for denoising model.
        ddpm_loss (dict): Config for DDPM.
        betas_cfg (dict): Config for betas in diffusion process.
        num_timesteps (int, optional): Number of timesteps of the diffusion
            process. Defaults to 1000.
        num_classes (int | None, optional): The number of conditional classes.
            Defaults to None.
        sample_method (string, optional): Sample method for the denoising
            process.  Defaults to 'DDPM'.
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
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.fp16_enable = False
        # build denoising in this function
        self.num_classes = num_classes
        self.sample_method = sample_method
        self._denoising_cfg = deepcopy(denoising)
        self.denoising = build_module(
            denoising,
            default_args=dict(
                num_classes=num_classes, num_timesteps=num_timesteps))

        # get output-related configs from denoising
        self.denoising_var = self.denoising.var_cfg
        self.denoising_mean = self.denoising.mean_cfg
        # output_channels in denoising may be double, therefore we
        # get number of channels from config
        image_channels = self._denoising_cfg.get('in_channels')
        # image_size should be the attribute of denoising network
        image_size = self.denoising.image_size

        image_shape = torch.Size([image_channels, image_size, image_size])
        self.get_noise = partial(
            _get_noise_batch,
            image_shape=image_shape,
            num_timesteps=self.num_timesteps)
        self.get_label = partial(
            _get_label_batch, num_timesteps=self.num_timesteps)
        # self.var_to_tensor = partial(var_to_tensor, tar_shape=image_size)

        # build diffusion
        if ddpm_loss is not None:
            self.ddpm_loss = build_module(ddpm_loss)
            if not isinstance(self.ddpm_loss, nn.ModuleList):
                self.ddpm_loss = nn.ModuleList([self.ddpm_loss])
        else:
            self.ddpm_loss = None

        self.betas_cfg = deepcopy(betas_cfg)
        self.num_timesteps = num_timesteps

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self._parse_train_cfg()
        if test_cfg is not None:
            self._parser_test_cfg()

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

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part --> what should we do here?

    def _get_loss(self, outputs_dict):
        pass

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
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
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
        denoising_dict_ = self.reconstruction_step(data, return_noise=True)
        denoising_dict_['iteration'] = curr_iter
        denoising_dict_['real_imgs'] = data[self.real_img_key]
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

        # image used for vislization
        results = dict(
            real_imgs=real_imgs,
            pred_x_0=denoising_dict_[''],
            x_t=denoising_dict_[''],
            x_t_1=denoising_dict_[''])
        outputs = dict(
            log_vars=log_vars, num_samples=real_imgs.shape[0], results=results)
        if hasattr(self, 'iteration'):
            self.iteration += 1
        return outputs

    def reconstruction_step(self,
                            data_batch,
                            noise=None,
                            label=None,
                            timestep=None,
                            return_noise=False):
        """Reconstruction step.

        ``sample_from_noise`` focus on generate samples start from **random
        (or given) noise**. Therefore, we design this function to release a
        reconstruction with respect to the given images.

        Args:
            data_batch (dict): Input data from dataloader.
            noise (torch.Tensor | callable)
            timestep (int | list | torch.Tensor | callable): Target
                timestep to perform reconstruction.
        """
        # 0. prepare for timestep, noise and label
        real_imgs = data_batch[self.real_img_key]
        num_batch, noise_size = real_imgs.shape[0], real_imgs.shape[2]
        if timestep is None:
            timestep = torch.Tensor([t for t in range(self.num_timesteps)
                                     ]).expand([num_batch, 1])
        if isinstance(timestep, (int, list)):
            timestep = torch.Tensor(timestep)
        elif callable(timestep):
            timestep_generator = timestep
            timestep = timestep_generator((num_batch, ))
        else:
            assert isinstance(timestep, torch.Tensor), (
                'we only support int list tensor or a callable function')
        timestep = timestep.to(get_module_device(self))

        if noise is not None:
            assert 'noise' not in data_batch, (
                'Receive \'noise\' in both data_batch and passed arguments.')
        if noise is None:
            noise = data_batch['noise'] if 'noise' in data_batch else None
        if label is not None:
            assert 'label' not in data_batch, (
                'Receive \'label\' in both data_batch and passed arguments.')
        if label is None:
            label = data_batch['label'] if 'label' in data_batch else None
        noise_batch = self._get_noise_batch(noise, num_batch, noise_size)

        # 1. get diffusion results and parameters
        diffusion_batches = self.q_sample(real_imgs, noise_batch)
        # 2. get denoising results. Set return_noise=False here for ``label``
        # and ``t`` can be directly accessed
        denoising_batches = self.denoising(
            diffusion_batches, timestep, label, return_noise=False)

        if return_noise:
            # return t and target_noise for calculate loss terms or metrics
            output_dict = dict(
                t=timestep,
                noise=noise_batch,
                label=label,
                diffusion_batches=diffusion_batches)
            output_dict.update(noise_batch)
            return output_dict

        return denoising_batches

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
                is passed, this method would try to sample from ema (if
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
            noise = outputs['noise_batch']

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.denoising
            outputs_ = sample_fn(
                _model, noise=noise, num_batches=num_batches, **kwargs)

            if isinstance(outputs_, dict):
                outputs['fake_img'] = torch.cat(
                    [outputs['fake_img'], outputs_['fake_img']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)

        return outputs

    @torch.no_grad()
    def DDPM_sample(self,
                    model,
                    noise=None,
                    num_batches=0,
                    label=None,
                    save_intermedia=False,
                    timesteps_noise=None,
                    **kwargs):
        """DDPM sample from random noise.
        Args:
            model (torch.nn.Module): Denoising model used to sample images.
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional):
            label (torch.Tensor, optional):
            save_intermedia (bool, optional): Whether to save denoising result
                of intermedia timesteps. If set as True, would return a list
                contains denoising results of each timestep. Otherwise, only
                the final denoising result would be returned. Defaults to
                False.
            timesteps_noise (torch.Tensor, optional): Noise term used in each
                denoising timesteps. If passed, the input noise would shape as
                [num_timesteps, b, c, h, w]. If set as None, noise of each
                denoising timestep would be randomly sampled. Default as None.
        Returns:
            torch.Tensor | list[torch.Tensor]: If ``save_intermedia``, a list
                contains denoising results of each timestep would be returned.
                Otherwise, only the final denoising result would be returned.
        """
        x_t = self._get_noise_batch(noise, num_batches)
        if save_intermedia:
            # save input
            intermedia = [x_t.clone()]

        # use timesteps noise if defined
        if timesteps_noise is not None:
            timesteps_noise = self._get_noise_batch(
                timesteps_noise, num_batches, timesteps_noise=True)

        # TODO: remove self for batched_timesteps
        if not hasattr(self, 'batched_timesteps'):
            self.batched_timesteps = torch.arange(self.num_timesteps - 1, -1,
                                                  -1).long()
            if torch.cuda.is_available():
                self.batched_timesteps = self.batched_timesteps.cuda()

        for t in self.batched_timesteps:
            batched_t = t.expand(x_t.shape[0])
            step_noise = timesteps_noise[t, ...] \
                if timesteps_noise is not None else None

            x_t = self.denoising_step(
                model, x_t, batched_t, noise=step_noise, label=label, **kwargs)
            if save_intermedia:
                intermedia.append(x_t.clone())
        if save_intermedia:
            return intermedia
        return x_t

    @torch.no_grad()
    def DDIM_sample(self,
                    model,
                    noise=None,
                    num_batches=0,
                    save_intermedia=False,
                    **kwargs):
        """"""
        # TODO: finish later
        pass

    def prepare_diffusion_vars(self):
        """Prepare for variables used in the diffusion process.

        TODO: we should use cupy or torch for speeding up?
        """
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
        num_batches = x_0.shape[0]
        tar_shape = x_0.shape
        # noise = self._get_noise_batch(noise, num_batches, noise_size)
        noise = self.get_noise(noise, num_batches=num_batches)
        mean = var_to_tensor(self.sqrt_alphas_bar, t, tar_shape)
        std = var_to_tensor(self.sqrt_one_minus_alphas_bar, t, tar_shape)

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
        tar_shape = x_0.shape
        mean = var_to_tensor(self.sqrt_alphas_bar, t, tar_shape) * x_0
        log_var = var_to_tensor(self.log_one_minus_alphas_bar, t, tar_shape)
        return mean, log_var

    def q_posterior_mean_variance(self, x_0, x_t, t, var=True, log_var=False):
        r"""Get mean and variance of diffusion posterior
            `q(x_{t-1} | x_t, x_0)`.

        Args:
            x_0 (torch.tensor): The original image before diffusion, shape as
                [bz, ch, H, W].
            t (torch.tensor): Target timestep, shape as [bz, ].
            var (bool, optional): If set as ``True``, would return a dict
                contains ``var``. Otherwise, only mean would be returned,
                ``log_var`` would be ignored. Defaults to True.
            log_var (bool, optional): If set as ``True``, the returned dict
                would additional contain ``log_var``. This argument would be
                considered only if ``var == True``. Defaults to False.

        Returns:
            torch.Tensor | dict: If ``var``, would return a dict contains
                ``mean`` and ``var``. Otherwise, only mean would be returned.
                If ``var`` and ``log_var`` set at as True simultaneously, the
                returned dict would additional contain ``log_var``.
        """

        tar_shape = x_0.shape
        tilde_mu_t_coef1 = var_to_tensor(self.tilde_mu_t_coef1, t, tar_shape)
        tilde_mu_t_coef2 = var_to_tensor(self.tilde_mu_t_coef2, t, tar_shape)
        posterior_mean = tilde_mu_t_coef1 * x_0 + tilde_mu_t_coef2 * x_t
        # do not need variance, just return mean
        if not var:
            return posterior_mean
        posterior_var = var_to_tensor(self.tilde_betas_t, t, tar_shape)
        out_dict = dict(mean=posterior_mean, var=posterior_var)
        if log_var:
            posterior_log_var = var_to_tensor(self.log_tilde_betas_t_clipped,
                                              t, tar_shape)
            out_dict['log_var'] = posterior_log_var
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
                applies to the predicted ``x_0`` prediction before it is used
                to sample. Applies before ``clip_denoised``. Defaults to None.

        Returns:
            dict: A dict contains ``var_pred``, ``log_var_pred``, ``mean_pred``
                and ``x_0_pred``.
        """
        tar_shape = x_t.shape
        # prepare for var and log_var
        if self.denoising_var.upper() == 'LEARNED':
            # TODO: maybe change to LEARNED_LOG_VAR
            log_var_pred = denoising_output['log_var']
            var_pred = torch.exp(log_var_pred)

        elif self.denoising_var.upper() == 'LEARNED_RANGE':
            # TODO: maybe change to LEARNED_FACTOR ?
            var_factor = denoising_output['factor']
            lower_bound_log_var = var_to_tensor(self.log_tilde_betas_t_clipped,
                                                t, tar_shape)
            upper_bound_log_var = var_to_tensor(
                np.log(self.betas), t, tar_shape)
            log_var_pred = var_factor * upper_bound_log_var + (
                1 - var_factor) * lower_bound_log_var
            var_pred = torch.exp(log_var_pred)

        elif self.denoising_var.upper() == 'FIXED_LARGE':
            # use betas as var
            var_pred = var_to_tensor(
                np.append(self.tilde_betas_t[1], self.betas), t, tar_shape)
            log_var_pred = torch.log(var_pred)

        elif self.denoising_var.upper() == 'FIXED_SMALL':
            # use posterior (tilde_betas)  as var
            var_pred = var_to_tensor(self.tilde_betas_t, t, tar_shape)
            log_var_pred = var_to_tensor(self.log_tilde_betas_t_clipped, t,
                                         tar_shape)
        else:
            raise AttributeError('Unknown denoising var output type '
                                 f'[{self.denoising_var}].')

        def process_x_0(x):
            if denoised_fn is not None and callable(denoised_fn):
                x = denoised_fn(x)
            return x.clamp(-1, 1) if clip_denoised else x

        # prepare for mean and x_0
        if self.denoising_mean.upper() == 'EPS':
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
                x_0_pred, x_t, t, var=False)
        elif self.denoising_mean.upper() == 'START_X':
            x_0_pred = process_x_0(denoising_output['x_0_pred'])
            mean_pred = self.q_posterior_mean_variance(
                x_0_pred, x_t, t, var=False)
        elif self.denoising_mean.upper() == 'PREVIOUS_X':
            # TODO: maybe we should call this PREVIOUS_X_MEAN or MU_THETA
            # because this actually predict \mu_{\theta}
            mean_pred = denoising_output['x_tm1_pred']
            x_0_pred = process_x_0(self.pred_x_0_from_x_tm1(mean_pred, x_t, t))
        else:
            raise AttributeError('Unknown denoising mean output type '
                                 f'[{self.denoising_mean}].')

        return dict(
            var_pred=var_pred,
            log_var_pred=log_var_pred,
            mean_pred=mean_pred,
            x_0_pred=x_0_pred)

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
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            label (torch.Tensor | callable | None): You can directly give a
                batch of label through a ``torch.Tensor`` or offer a callable
                function to sample a batch of label data. Otherwise, the
                ``None`` indicates to use the default label sampler.
            clip_denoised (bool, optional): Whether cliped sample results into
                [-1, 1]. Defaults to True.
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
                image would be returned. Otherwise, the dict contains
                ``fake_image``, ``noise_batch`` and outputs from denoising
                model and ``p_mean_variance`` would be returned.
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
        noise = self._get_noise_batch(noise, num_batches)
        nonzero_mask = ((t != 0).float().view(-1,
                                              *([1] * (len(x_t.shape) - 1))))

        # TODO: directly use var_pred instead log_var_pred,
        # only error of 1e-12.
        # log_var_pred = p_output['log_var_pred']
        # sample = mean_pred + \
        #     nonzero_mask * torch.exp(0.5 * log_var_pred) * noise
        sample = mean_pred + nonzero_mask * torch.sqrt(var_pred) * noise
        if return_noise:
            return dict(
                fake_image=sample,
                noise_batch=noise,
                label=label,
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
        tar_shape = x_t.shape
        coef1 = var_to_tensor(self.sqrt_recip_alplas_bar, t, tar_shape)
        coef2 = var_to_tensor(self.sqrt_recipm1_alphas_bar, t, tar_shape)
        return x_t * coef1 - eps * coef2

    def pred_x_0_from_x_tm1(self, x_tm1, x_t, t):
        r"""
        TODO: may be we should change the signature of this function to
        ``pred_x_0_from_mu_theta(self, mu_theta, x_t, t)``

        Predict `x_0` from `x_{t-1}`. (actually from `\mu_{\theta}`).
        `(\mu_{\theta} - coef2 * x_t) / coef1`, where `coef1` and `coef2`
        are from Eq 6 of the DDPM paper.

        Args:
            x_tm1 (torch.Tensor): `x_{t-1}` used to predict `x_0`.
            x_t (torch.Tensor): `x_{t}` used to predict `x_0`.
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Predicted `x_0`.

        """
        tar_shape = x_t.shape
        coef1 = var_to_tensor(self.tilde_mu_t_coef1, t, tar_shape)
        coef2 = var_to_tensor(self.tilde_mu_t_coef2, t, tar_shape)
        x_0 = (x_tm1 - coef2 * x_t) / coef1
        return x_0

    def forward_train(self):
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
            kwargs['return_noise'] = True
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
