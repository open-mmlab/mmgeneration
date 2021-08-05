from abc import ABCMeta
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from mmgen.models import build_model


class BaseGaussianDiffusion(nn.Module, metaclass=ABCMeta):
    """BaseGaussianDiffusion Module."""

    def __init__(self, denoising, diffusion, train_cfg=None, test_cfg=None):
        """"""

        super().__init__()
        self.fp16_enable = False
        # build denoising in this function
        self._denoising_cfg = deepcopy(denoising)
        self.denoising = build_model(denoising)

        # build diffusion
        self._diffusion_cfg = deepcopy(diffusion)
        self.betas_cfg = getattr(self._diffusion_cfg, 'betas_cfg')

        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        self.prepare_diffusion_vars()

    def _parse_train_cfg(self):
        """ TODO: training part would be finished later~~~
        """
        pass

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def train_step(self, data, optimizer, ddp_reducer=None):
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

        TODO: training would be supported later ~~~
        """
        pass

    def sample_from_noise(self,
                          noise,
                          num_batches=0,
                          sample_model='ema/orig',
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

        Returns:
            torch.Tensor | dict: The output may be the direct synthesized
                images in ``torch.Tensor``. Otherwise, a dict with queried
                data, including generated images, will be returned.
        """
        # get sample function by name
        sample_fn_name = f'{self.sample_method.upper()}_sample'
        if not hasattr(sample_fn_name):
            raise AttributeError(
                f'Cannot find sample method [{sample_fn_name}] correspond '
                f'to [{self.sample_method}].')
        sample_fn = partial(
            getattr(self, sample_fn_name), noise, num_batches, **kwargs)

        if sample_model == 'ema':
            assert self.use_ema
            _model = self.denoising_ema
        elif sample_model == 'ema/orig' and self.use_ema:
            _model = self.denoising_ema
        else:
            _model = self.denoising

        outputs = sample_fn(
            noise, num_batches=num_batches, model=_model, **kwargs)

        if isinstance(outputs, dict) and 'noise_batch' in outputs:
            noise = outputs['noise_batch']

        if sample_model == 'ema/orig' and self.use_ema:
            _model = self.denoising
            outputs_ = sample_fn(noise, num_batches, model=_model, **kwargs)

            if isinstance(outputs_, dict):
                outputs['fake_img'] = torch.cat(
                    [outputs['fake_img'], outputs_['fake_img']], dim=0)
            else:
                outputs = torch.cat([outputs, outputs_], dim=0)

        return outputs

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

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.tilde_betas_t = self.betas * (1 - self.alphas_bar_prev) / (
            1 - self.alphas_bar)
        self.tilde_mu_t_coef1 = np.sqrt(
            self.alphas_bar_prev) / (1 - self.alphas_bar) * self.betas
        self.tilde_mu_t_coef2 = self.sqrt_alphas_bar * (
            1 - self.alphas_bar_prev) / (1 - self.alphas_bar)

    def get_betas(self):
        """Get betas by defined schedule method in diffusion process."""
        self.betas_schedule = self.betas_cfgs.pop('type')
        if self.beta_schedule == 'linear':
            return self.linear_beta_schedule(self.num_diffusion_timesteps,
                                             **self.betas_cfgs)
        elif self.beta_schedule == 'cosine':
            return self.cosine_beta_schedule(self.num_diffusion_timesteps,
                                             **self.betas_cfgs)
        else:
            raise AttributeError(f'Unknown method name {self.beta_schedule}'
                                 'for beta schedule.')

    @staticmethod
    def linear_beta_schedule(diffusion_timesteps, beta_0=1e-4, beta_T=2e-2):
        """Linear schedule from Ho et al, extended to work for any number of
        diffusion steps."""
        scale = 1000 / diffusion_timesteps
        beta_0 = scale * beta_0
        beta_T = scale * beta_T
        return np.linspace(
            beta_0, beta_T, diffusion_timesteps, dtype=np.float64)

    @staticmethod
    def cosine_beta_schedule(diffusion_timesteps, max_beta=0.999, s=0.02):
        """Create a beta schedule that discretizes the given alpha_t_bar
        function, which defines the cumulative product of (1-beta) over time
        from t = [0, 1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """

        def f(t, T, s):
            return np.cos((t / T + s) / (1 + s) * np.pi / 2)**2

        betas = []
        for t in range(diffusion_timesteps):
            alpha_bar_t = f(t, diffusion_timesteps, s)
            alpha_bar_t_1 = f(t - 1, diffusion_timesteps, s)
            betas_t = 1 - alpha_bar_t / alpha_bar_t_1
            betas.append(betas_t)
        return np.array(betas)

    @staticmethod
    def _get_noise_batch(noise, num_batches, noise_size):
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            # assert noise.shape[1] == self.noise_size
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
            noise_batch = noise_generator((num_batches, noise_size, 1, 1))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, noise_size, 1, 1))
        return noise_batch

    @staticmethod
    def var_to_tensor(var, index, tar_shape):
        """Function used to extract variables by given index, and convert into
        tensor as given shape."""
        # we must move var to cuda for it's ndarray in current design
        var_indexed = torch.from_numpy(var)[index]
        if torch.cuda.is_available():
            var_indexed = var_indexed.cuda()

        while len(var_indexed) < len(tar_shape):
            var_indexed = var_indexed[..., None]
        return var_indexed.expand(tar_shape)

    def q_sample(self, x_0, t, noise=None):
        """Get diffusion result at timestep t q(x_t | x_0)."""
        num_batches, noise_size = x_0.shape[0], x_0.shape[2]
        tar_shape = x_0.shape
        noise = self._get_noise_batch(noise, num_batches, noise_size)
        mean = self.var_to_tensor(self.sqrt_alphas_bar, t, tar_shape)
        std = self.var_to_tensor(self.sqrt_one_minus_alphas_bar, t, tar_shape)

        return x_0 * mean + noise * std

    def q_mean_log_variance(self, x_0, t):
        """Get mean and variance of diffusion process q(x_t | x_0)
        Args:
            x_0 (torch.tensor): shape as [bz, ch, H, W]
            t (torch.tensor): shape as [bz, ]

        Returns:
            Tuple(torch.tensor, torch.tensor)
        """
        tar_shape = x_0.shape
        mean = self.var_to_tensor(self.sqrt_alphas_bar, t, tar_shape)
        log_var = self.var_to_tensor(self.log_one_minus_alphas_bar, t,
                                     tar_shape)
        return mean, log_var

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """Get mean and variance of diffusion posterior q(x_{t-1} | x_t,
        x_0)."""

        pass

    def p_mean_variance(self):
        pass

    def get_diffusion(self):
        pass

    def get_denoising(self):
        pass

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
        if kwargs.pop('mode', 'sampling') == 'sampling':
            return self.sample_from_noise(data, **kwargs)

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
