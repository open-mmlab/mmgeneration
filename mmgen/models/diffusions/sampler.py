import numpy as np
import torch

from ..builder import MODULES


@MODULES.register_module()
class UniformTimeStepSampler:
    """Timestep sampler for DDPM-based models. This sampler sample all
    timesteps with the same probabilistic.

    Args:
        num_timesteps (int): Total timesteps of the diffusion process.
    """

    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def sample(self, batch_size):
        """Sample timesteps.
        Args:
            batch_size (int): The desired batch size of the sampled timesteps.

        Returns:
            torch.Tensor: Sampled timesteps.
        """
        p = [1 / self.num_timesteps for _ in range(self.num_timesteps)]
        return torch.from_numpy(
            np.random.choice(self.num_timesteps, size=(batch_size, ), p=p))

    def __call__(self, batch_size):
        return self.sample(batch_size)
