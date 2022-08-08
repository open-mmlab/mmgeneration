# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmgen.utils.typing import ForwardInputs


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class GANImageBuffer:
    """This class implements an image buffer that stores previously generated
    images.

    This buffer allows us to update the discriminator using a history of
    generated images rather than the ones produced by the latest generator
    to reduce model oscillation.

    Args:
        buffer_size (int): The size of image buffer. If buffer_size = 0,
            no buffer will be created.
        buffer_ratio (float): The chance / possibility  to use the images
            previously stored in the buffer.
    """

    def __init__(self, buffer_size, buffer_ratio=0.5):
        self.buffer_size = buffer_size
        # create an empty buffer
        if self.buffer_size > 0:
            self.img_num = 0
            self.image_buffer = []
        self.buffer_ratio = buffer_ratio

    def query(self, images):
        """Query current image batch using a history of generated images.

        Args:
            images (Tensor): Current image batch without history information.
        """
        if self.buffer_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # if the buffer is not full, keep inserting current images
            if self.img_num < self.buffer_size:
                self.img_num = self.img_num + 1
                self.image_buffer.append(image)
                return_images.append(image)
            else:
                use_buffer = np.random.random() < self.buffer_ratio
                # by self.buffer_ratio, the buffer will return a previously
                # stored image, and insert the current image into the buffer
                if use_buffer:
                    random_id = np.random.randint(0, self.buffer_size)
                    image_tmp = self.image_buffer[random_id].clone()
                    self.image_buffer[random_id] = image
                    return_images.append(image_tmp)
                # by (1 - self.buffer_ratio), the buffer will return the
                # current image
                else:
                    return_images.append(image)
        # collect all the images and return
        return_images = torch.cat(return_images, 0)
        return return_images


def get_valid_noise_size(noise_size: Optional[int],
                         generator: Union[Dict, nn.Module]) -> Optional[int]:
    """Get the value of `noise_size` from input, `generator` and check the
    consistency of these values. If no conflict is found, return that value.

    Args:
        noise_size (Optional[int]): `noise_size` passed to
            `BaseGAN_refactor`'s initialize function.
        generator (ModelType): The config or the model of generator.

    Returns:
        int | None: The noise size feed to generator.
    """
    if isinstance(generator, dict):
        model_noise_size = generator.get('noise_size', None)
    else:
        model_noise_size = getattr(generator, 'noise_size', None)

    # get noise_size
    if noise_size is not None and model_noise_size is not None:
        assert noise_size == model_noise_size, (
            'Input \'noise_size\' is unconsistency with '
            f'\'generator.noise_size\'. Receive \'{noise_size}\' and '
            f'\'{model_noise_size}\'.')
    else:
        noise_size = noise_size or model_noise_size

    return noise_size


def get_valid_num_batches(batch_inputs: ForwardInputs) -> int:
    """Try get the valid batch size from inputs.

    - If some values in `batch_inputs` are `Tensor` and 'num_batches' is in
      `batch_inputs`, we check whether the value of 'num_batches' and the the
      length of first dimension of all tensors are same. If the values are not
      same, `AssertionError` will be raised. If all values are the same,
      return the value.
    - If no values in `batch_inputs` is `Tensor`, 'num_batches' must be
      contained in `batch_inputs`. And this value will be returned.
    - If some values are `Tensor` and 'num_batches' is not contained in
      `batch_inputs`, we check whether all tensor have the same length on the
      first dimension. If the length are not same, `AssertionError` will be
      raised. If all length are the same, return the length as batch size.
    - If batch_inputs is a `Tensor`, directly return the length of the first
      dimension as batch size.

    Args:
        batch_inputs (ForwardInputs): Inputs passed to :meth:`forward`.

    Returns:
        int: The batch size of samples to generate.
    """
    if isinstance(batch_inputs, Tensor):
        return batch_inputs.shape[0]

    # get num_batces from batch_inputs
    num_batches_dict = {
        k: v.shape[0]
        for k, v in batch_inputs.items() if isinstance(v, Tensor)
    }
    if 'num_batches' in batch_inputs:
        num_batches_dict['num_batches'] = batch_inputs['num_batches']

    # ensure num_batches is not None
    assert len(num_batches_dict.keys()) > 0, (
        'Cannot get \'num_batches\' form preprocessed input '
        f'(\'{batch_inputs}\').')

    # ensure all num_batches are same
    num_batches = list(num_batches_dict.values())[0]
    assert all([
        bz == num_batches for bz in num_batches_dict.values()
    ]), ('\'num_batches\' is inconsistency among the preprocessed input. '
         f'\'num_batches\' parsed resutls: {num_batches_dict}')

    return num_batches
