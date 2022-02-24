# Copyright (c) OpenMMLab. All rights reserved.
import torch


def _get_noise_batch(noise,
                     image_shape,
                     num_timesteps=0,
                     num_batches=0,
                     timesteps_noise=False):
    """Get noise batch. Support get sequeue of noise along timesteps.

    We support the following use cases ('bz' denotes ```num_batches`` and 'n'
    denotes ``num_timesteps``):

    If timesteps_noise is True, we output noise which dimension is 5.
    - Input is [bz, c, h, w]: Expand to [n, bz, c, h, w]
    - Input is [n, c, h, w]: Expand to [n, bz, c, h, w]
    - Input is [n*bz, c, h, w]: View to [n, bz, c, h, w]
    - Dim of the input is 5: Return the input, ignore ``num_batches`` and
      ``num_timesteps``
    - Callable or None: Generate noise shape as [n, bz, c, h, w]
    - Otherwise: Raise error

    If timestep_noise is False, we output noise which dimension is 4 and
    ignore ``num_timesteps``.
    - Dim of the input is 3: Unsqueeze to [1, c, h, w], ignore ``num_batches``
    - Dim of the input is 4: Return input, ignore ``num_batches``
    - Callable or None: Generate noise shape as [bz, c, h, w]
    - Otherwise: Raise error

    It's to be noted that, we do not move the generated label to target device
    in this function because we can not get which device the noise should move
    to.

    Args:
        noise (torch.Tensor | callable | None): You can directly give a
            batch of noise through a ``torch.Tensor`` or offer a callable
            function to sample a batch of noise data. Otherwise, the
            ``None`` indicates to use the default noise sampler.
        image_shape (torch.Size): Size of images in the diffusion process.
        num_timesteps (int, optional): Total timestpes of the diffusion and
            denoising process. Defaults to 0.
        num_batches (int, optional): The number of batch size. To be noted that
            this argument only work when the input ``noise`` is callable or
            ``None``. Defaults to 0.
        timesteps_noise (bool, optional): If True, returned noise will shape
            as [n, bz, c, h, w], otherwise shape as [bz, c, h, w].
            Defaults to False.
        device (str, optional): If not ``None``, move the generated noise to
            corresponding device.
    Returns:
        torch.Tensor: Generated noise with desired shape.
    """
    if isinstance(noise, torch.Tensor):
        # conduct sanity check for the last three dimension
        assert noise.shape[-3:] == image_shape
        if timesteps_noise:
            if noise.ndim == 4:
                assert num_batches > 0 and num_timesteps > 0
                # noise shape as [n, c, h, w], expand to [n, bz, c, h, w]
                if noise.shape[0] == num_timesteps:
                    noise_batch = noise.view(num_timesteps, 1, *image_shape)
                    noise_batch = noise_batch.expand(-1, num_batches, -1, -1,
                                                     -1)
                # noise shape as [bz, c, h, w], expand to [n, bz, c, h, w]
                elif noise.shape[0] == num_batches:
                    noise_batch = noise.view(1, num_batches, *image_shape)
                    noise_batch = noise_batch.expand(num_timesteps, -1, -1, -1,
                                                     -1)
                # noise shape as [n*bz, c, h, w], reshape to [b, bz, c, h, w]
                elif noise.shape[0] == num_timesteps * num_batches:
                    noise_batch = noise.view(num_timesteps, -1, *image_shape)
                else:
                    raise ValueError(
                        'The timesteps noise should be in shape of '
                        '(n, c, h, w), (bz, c, h, w), (n*bz, c, h, w) or '
                        f'(n, bz, c, h, w). But receive {noise.shape}.')

            elif noise.ndim == 5:
                # direct return noise
                noise_batch = noise
            else:
                raise ValueError(
                    'The timesteps noise should be in shape of '
                    '(n, c, h, w), (bz, c, h, w), (n*bz, c, h, w) or '
                    f'(n, bz, c, h, w). But receive {noise.shape}.')
        else:
            if noise.ndim == 3:
                # reshape noise to [1, c, h, w]
                noise_batch = noise[None, ...]
            elif noise.ndim == 4:
                # do nothing
                noise_batch = noise
            else:
                raise ValueError(
                    'The noise should be in shape of (n, c, h, w) or'
                    f'(c, h, w), but got {noise.shape}')
    # receive a noise generator and sample noise.
    elif callable(noise):
        assert num_batches > 0
        noise_generator = noise
        if timesteps_noise:
            assert num_timesteps > 0
            # generate noise shape as [n, bz, c, h, w]
            noise_batch = noise_generator(
                (num_timesteps, num_batches, *image_shape))
        else:
            # generate noise shape as [bz, c, h, w]
            noise_batch = noise_generator((num_batches, *image_shape))
    # otherwise, we will adopt default noise sampler.
    else:
        assert num_batches > 0
        if timesteps_noise:
            assert num_timesteps > 0
            # generate noise shape as [n, bz, c, h, w]
            noise_batch = torch.randn(
                (num_timesteps, num_batches, *image_shape))
        else:
            # generate noise shape as [bz, c, h, w]
            noise_batch = torch.randn((num_batches, *image_shape))

    return noise_batch


def _get_label_batch(label,
                     num_timesteps=0,
                     num_classes=0,
                     num_batches=0,
                     timesteps_noise=False):
    """Get label batch. Support get sequeue of label along timesteps.

    We support the following use cases ('bz' denotes ```num_batches`` and 'n'
    denotes ``num_timesteps``):

    If num_classes <= 0, return None.

    If timesteps_noise is True, we output label which dimension is 2.
    - Input is [bz, ]: Expand to [n, bz]
    - Input is [n, ]: Expand to [n, bz]
    - Input is [n*bz, ]: View to [n, bz]
    - Dim of the input is 2: Return the input, ignore ``num_batches`` and
      ``num_timesteps``
    - Callable or None: Generate label shape as [n, bz]
    - Otherwise: Raise error

    If timesteps_noise is False, we output label which dimension is 1 and
    ignore ``num_timesteps``.
    - Dim of the input is 1: Unsqueeze to [1, ], ignore ``num_batches``
    - Dim of the input is 2: Return the input. ignore ``num_batches``
    - Callable or None: Generate label shape as [bz, ]
    - Otherwise: Raise error

    It's to be noted that, we do not move the generated label to target device
    in this function because we can not get which device the noise should move
    to.

    Args:
        label (torch.Tensor | callable | None): You can directly give a
            batch of noise through a ``torch.Tensor`` or offer a callable
            function to sample a batch of noise data. Otherwise, the
            ``None`` indicates to use the default noise sampler.
        num_timesteps (int, optional): Total timestpes of the diffusion and
            denoising process. Defaults to 0.
        num_batches (int, optional): The number of batch size. To be noted that
            this argument only work when the input ``noise`` is callable or
            ``None``. Defaults to 0.
        timesteps_noise (bool, optional): If True, returned noise will shape
            as [n, bz, c, h, w], otherwise shape as [bz, c, h, w].
            Defaults to False.
    Returns:
        torch.Tensor: Generated label with desired shape.
    """
    # no labels output if num_classes is 0
    if num_classes == 0:
        assert label is None, ('\'label\' should be None '
                               'if \'num_classes == 0\'.')
        return None

    # receive label and conduct sanity check.
    if isinstance(label, torch.Tensor):
        if timesteps_noise:
            if label.ndim == 1:
                assert num_batches > 0 and num_timesteps > 0
                # [n, ] to [n, bz]
                if label.shape[0] == num_timesteps:
                    label_batch = label.view(num_timesteps, 1)
                    label_batch = label_batch.expand(-1, num_batches)
                # [bz, ] to [n, bz]
                elif label.shape[0] == num_batches:
                    label_batch = label.view(1, num_batches)
                    label_batch = label_batch.expand(num_timesteps, -1)
                # [n*bz, ] to [n, bz]
                elif label.shape[0] == num_timesteps * num_batches:
                    label_batch = label.view(num_timesteps, -1)
                else:
                    raise ValueError(
                        'The timesteps label should be in shape of '
                        '(n, ), (bz,), (n*bz, ) or (n, bz, ). But receive '
                        f'{label.shape}.')

            elif label.ndim == 2:
                # dimension is 2, direct return
                label_batch = label
            else:
                raise ValueError(
                    'The timesteps label should be in shape of '
                    '(n, ), (bz,), (n*bz, ) or (n, bz, ). But receive '
                    f'{label.shape}.')
        else:
            # dimension is 0, expand to [1, ]
            if label.ndim == 0:
                label_batch = label[None, ...]
            # dimension is 1, do nothing
            elif label.ndim == 1:
                label_batch = label
            else:
                raise ValueError(
                    'The label should be in shape of (bz, ) or'
                    f'zero-dimension tensor, but got {label.shape}')
    # receive a noise generator and sample noise.
    elif callable(label):
        assert num_batches > 0
        label_generator = label
        if timesteps_noise:
            assert num_timesteps > 0
            # generate label shape as [n, bz]
            label_batch = label_generator((num_timesteps, num_batches))
        else:
            # generate label shape as [bz, ]
            label_batch = label_generator((num_batches, ))
    # otherwise, we will adopt default label sampler.
    else:
        assert num_batches > 0
        if timesteps_noise:
            assert num_timesteps > 0
            # generate label shape as [n, bz]
            label_batch = torch.randint(0, num_classes,
                                        (num_timesteps, num_batches))
        else:
            # generate label shape as [bz, ]
            label_batch = torch.randint(0, num_classes, (num_batches, ))

    return label_batch


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
