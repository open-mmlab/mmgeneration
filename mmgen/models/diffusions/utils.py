# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def linear_beta_schedule(diffusion_timesteps: int,
                         beta_0: float = 1e-4,
                         beta_T: float = 2e-2) -> np.ndarray:
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
    return np.linspace(beta_0, beta_T, diffusion_timesteps, dtype=np.float64)


def cosine_beta_schedule(diffusion_timesteps: int,
                         max_beta: float = 0.999,
                         s: float = 0.008) -> np.ndarray:
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


def var_to_tensor(var, index, target_shape=None, device=None):
    """Function used to extract variables by given index, and convert into
    tensor as given shape.
    Args:
        var (np.array): Variables to be extracted.
        index (torch.Tensor): Target index to extract.
        target_shape (torch.Size, optional): If given, the indexed variable
            will expand to the given shape. Defaults to None.
        device (str): If given, the indexed variable will move to the target
            device. Otherwise, indexed variable will on cpu. Defaults to None.

    Returns:
        torch.Tensor: Converted variable.
    """
    # we must move var to cuda for it's ndarray in current design
    var_indexed = torch.from_numpy(var)[index].float()

    if device is not None:
        var_indexed = var_indexed.to(device)

    while len(var_indexed.shape) < len(target_shape):
        var_indexed = var_indexed[..., None]
    return var_indexed
