import torch


def _get_noise_batch(noise,
                     image_shape,
                     num_timesteps=0,
                     num_batches=0,
                     timesteps_noise=False):
    """Get noise batch. Support get sequeue of noise along timesteps.
    If passed noise is a torch.Tensor,
        1. timesteps_noise == True
            1.1. dimension of noise is 4
                1.1.1. noise shape as [bz, c, h, w]
                    --> expand noise to [n, bz, c, h, w]
                1.1.2. noise shape as [n, c, h, w]
                    --> expand noise to [n, bz, c, h, w]
                1.1.3. noise shape as [n*bz, c, h, w]
                    --> view noise to [n, bz, c, h, w]
            1.2 dimension of noise is 5
                --> return noise
        2. timesteps_noise == False
            2.1 dimension of noise is 3 --> view to [1, c, h, w]
            2.2 dimension of noise is 4 --> do nothing
    If passed noise is a callable function or None
        1. timesteps_noise == True
            --> generate noise shape as [n, bz, c, h, w]
        2. timesteps_noise == False
            --> generate noise shape as [bz, c, h, w]
    To be noted that, we do not move the generated noise to target device in
    this function because we can not get which device the noise should move to.

    Args:
        noise (torch.Tensor | callable | None): You can directly give a
            batch of noise through a ``torch.Tensor`` or offer a callable
            function to sample a batch of noise data. Otherwise, the
            ``None`` indicates to use the default noise sampler.
        image_shape (torch.Size): Size of images in the diffusion process.
        num_timesteps (int, optional): Total timestpes of the diffusion and
            denoising process. Defaults to 0.
        num_batches (int, optional): The number of batch size. Defaults to 0.
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
                if noise.shape[0] == num_timesteps:
                    noise_batch = noise.view(num_timesteps, 1, *image_shape)
                    noise_batch = noise_batch.expand(-1, num_batches, -1, -1)
                elif noise.shape[0] == num_batches:
                    noise_batch = noise.view(1, num_batches, *image_shape)
                    noise_batch = noise_batch.expand(num_timesteps, -1, -1, -1)
                elif noise.shape[0] == num_timesteps * num_batches:
                    noise_batch = noise.view(num_timesteps, -1, *image_shape)
                else:
                    raise ValueError(
                        'The timesteps noise should be in shape of '
                        '(n, c, h, w), (bz, c, h, w), (n*bz, c, h, w) or '
                        f'(n, bz, c, h, w). But receive {noise.shape}.')

            elif noise.ndim == 5:
                noise_batch = noise
            else:
                raise ValueError(
                    'The timesteps noise should be in shape of '
                    '(n, c, h, w), (bz, c, h, w), (n*bz, c, h, w) or '
                    f'(n, bz, c, h, w). But receive {noise.shape}.')
        else:
            if noise.ndim == 3:
                noise_batch = noise[None, ...]
            elif noise.ndim == 4:
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
            noise_batch = noise_generator(
                (num_timesteps, num_batches, *image_shape))
        else:
            noise_batch = noise_generator((num_batches, *image_shape))
    # otherwise, we will adopt default noise sampler.
    else:
        assert num_batches > 0
        if timesteps_noise:
            assert num_timesteps > 0
            noise_batch = torch.randn(
                (num_timesteps, num_batches, *image_shape))
        else:
            noise_batch = torch.randn((num_batches, *image_shape))

    return noise_batch


def _get_label_batch(label,
                     num_timesteps=0,
                     num_classes=0,
                     num_batches=0,
                     timesteps_noise=False):
    """Get label batch. Support get sequeue of label along timesteps.
    If passed label is a torch.Tensor,
        0. num_classes <= 0
            --> return None
        1. timesteps_noise == True
            1.1. dimension of label is 1
                1.1.1. label shape as [bz,]
                    --> expand label to [n, bz]
                1.1.2. label shape as [n, ]
                    --> expand label to [n, bz,]
                1.1.3. label shape as [n*bz, ]
                    --> view label to [n, bz,]
            1.2 dimension of label is 2
                --> return label
        2. timesteps_noise == False
            2.1 dimension of label is 0 --> view to [1, ]
            2.2 dimension of noise is 2 --> do nothing
    If passed label is a callable function or None
        1. timesteps_noise == True
            --> generate label shape as [n, bz, ]
        2. timesteps_noise == False
            --> generate label shape as [bz, ]
    To be noted that, we do not move the generated label to target device in
    this function because we can not get which device the noise should move to.

    Args:
        label (torch.Tensor | callable | None): You can directly give a
            batch of noise through a ``torch.Tensor`` or offer a callable
            function to sample a batch of noise data. Otherwise, the
            ``None`` indicates to use the default noise sampler.
        num_timesteps (int, optional): Total timestpes of the diffusion and
            denoising process. Defaults to 0.
        num_batches (int, optional): The number of batch size. Defaults to 0.
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
                if label.shape[0] == num_timesteps:
                    label_batch = label.view(num_timesteps, 1)
                    label_batch = label_batch.expand(-1, num_batches)
                elif label.shape[0] == num_batches:
                    label_batch = label.view(1, num_batches)
                    label_batch = label_batch.expand(num_timesteps, -1)
                elif label.shape[0] == num_timesteps * num_batches:
                    label_batch = label.view(num_timesteps, -1)
                else:
                    raise ValueError(
                        'The timesteps label should be in shape of '
                        '(n, ), (bz,), (n*bz, ) or (n, bz, ). But receive '
                        f'{label.shape}.')

            elif label.ndim == 2:
                label_batch = label
            else:
                raise ValueError(
                    'The timesteps label should be in shape of '
                    '(n, ), (bz,), (n*bz, ) or (n, bz, ). But receive '
                    f'{label.shape}.')
        else:
            if label.ndim == 0:
                label_batch = label[None, ...]
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
            label_batch = label_generator((num_batches))
        else:
            label_batch = label_generator((num_timesteps, num_batches))
    # otherwise, we will adopt default label sampler.
    else:
        assert num_batches > 0
        if timesteps_noise:
            assert num_timesteps > 0
            label_batch = torch.randint(0, num_classes,
                                        (num_timesteps, num_batches))
        else:
            label_batch = torch.randn(0, num_classes, (num_batches, ))

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
