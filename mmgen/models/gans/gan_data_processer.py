# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine import BaseDataElement
from mmengine.model import ImgDataPreprocessor, stack_batch
from torch import Tensor

from mmgen.registry import MODELS
from mmgen.utils.typing import PreprocessInputs, PreprocessOutputs

CollectOutputTyping = Tuple[Union[Tensor, Dict[str, Union[Tensor, str, int]]],
                            list]


@MODELS.register_module()
class GANDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for GAN models. This class provide normalization and
    bgr to rgb conversion for image tensor inputs. Besides to process tensor
    input, this class support dict as input.

    - If the value is `Tensor` and the corresponding key is not contained in
    :attr:`_NON_IMAGE_KEYS`, it will be processed as image tensor.
    - If the value is `Tensor` and the corresponding key belongs to
    :attr:`_NON_IMAGE_KEYS`, it will not remains unchanged.
    - If value is string or integer, it will not remains unchanged.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of image
            channels. If ``bgr_to_rgb=True`` it means the mean value of R,
            G, B channels. If it is not specified, images will not be
            normalized. Defaults None.
        std (Sequence[float or int], optional): The pixel standard deviation of
            image channels. If ``bgr_to_rgb=True`` it means the standard
            deviation of R, G, B channels. If it is not specified, images will
            not be normalized. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
    """
    _NON_IMAGE_KEYS = ['noise']
    _NON_CONCENTATE_KEYS = ['num_batches', 'mode', 'sample_kwargs', 'eq_cfg']

    def __init__(self,
                 mean: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 std: Sequence[Union[float, int]] = (127.5, 127.5, 127.5),
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_image_keys: Optional[Tuple[str, List[str]]] = None,
                 non_concentate_keys: Optional[Tuple[str, List[str]]] = None):

        super().__init__(mean, std, pad_size_divisor, pad_value, bgr_to_rgb,
                         rgb_to_bgr)
        # get color order
        if bgr_to_rgb:
            input_color_order, output_color_order = 'bgr', 'rgb'
        elif rgb_to_bgr:
            input_color_order, output_color_order = 'rgb', 'bgr'
        else:
            # 'bgr' order as default
            input_color_order = output_color_order = 'bgr'
        self.input_color_order = input_color_order
        self.output_color_order = output_color_order

        # add user defined keys
        if non_image_keys is not None:
            if not isinstance(non_image_keys, list):
                non_image_keys = [non_image_keys]
            self._NON_IMAGE_KEYS += non_image_keys
        if non_concentate_keys is not None:
            if not isinstance(non_concentate_keys, list):
                non_concentate_keys = [non_concentate_keys]
            self._NON_CONCENTATE_KEYS += non_concentate_keys

    def _check_keys_consistency(self, data) -> None:
        """Ensure keys in all inputs are consistency."""
        first_data_keys = data[0].keys()
        assert all([data_.keys() == first_data_keys for data_ in data
                    ]), ('Keys must be consistency in \'data\'.')

    def collate_data(self, data: Sequence[dict]) -> CollectOutputTyping:
        """NOTE: support two special suitation.
        support input keys other than `inputs` and `data_sample`.
        - If value is `torch.Tensor` and key belongs to `_NON_IMAGE_KEYS`, this
            element will be concentrated.
        - If value is `torch.Tensor` and key not belongs to `_NON_IMAGE_KEYS`,
            this element will be preprocessed and concentrated.
        - If value is int, we will concentrate them as torch.Tensor.
        - If value is str, we assert all string are same and remain only one
        in the inputs dict.

        Example 1:
            >>> data = [dict(inputs=dict(noise=..., mode='ema', num_batches=1),
                            data_sample=xxx),
                        dict(inputs=dict(noise=..., mode='ema', num_batches=1),
                            data_sample=xxx)]
            >>> self.collect_data(data)
            >>> (dict(noise=..., mode='ema', num_batches=1), DATA_SAMPLE)

        Example 2:
            >>> data = [dict(inputs=Tensor, data_sample=xxx), ...]
            >>> self.collect_data(data)
            >>> (Tensor, [data_sample_1, data_sample_2, ...])

        Args:
            data (Sequence[dict]): The data need to collect.

        Returns:
            CollectOutputTyping: Collected results.
        """

        self._check_keys_consistency(data)
        batch_data_samples: List[BaseDataElement] = []
        # Allow no `data_samples` in data
        for data_ in data:
            if 'data_sample' in data_:
                batch_data_samples.append(data_['data_sample'])

        inputs_list = [data_['inputs'] for data_ in data]
        if isinstance(inputs_list[0], Tensor):
            batch_inputs = [input_.to(self.device) for input_ in inputs_list]
        else:
            # inputs is a dict
            self._check_keys_consistency(inputs_list)
            inputs_keys = inputs_list[0].keys()
            batch_inputs = dict()

            for k in inputs_keys:

                if k in self._NON_CONCENTATE_KEYS:
                    first_value = inputs_list[0][k]
                    assert all([
                        input_[k] == first_value for input_ in inputs_list
                    ]), (f'NON_CONCENTATE_KEY \'{k}\' should be consistency '
                         'among the data list.')
                    # Do not move to corresponding device.
                    batch_inputs[k] = first_value
                else:
                    # Move data from CPU to corresponding device.
                    batch_inputs[k] = [(inputs_[k]).to(self.device)
                                       for inputs_ in inputs_list]

        # Move data from CPU to corresponding device.
        batch_data_samples = [
            data_sample.to(self.device) for data_sample in batch_data_samples
        ]
        return batch_inputs, batch_data_samples

    def _preprocess_image_tensor(self, inputs: List[Tensor]) -> Tensor:
        """Process image tensor.

        Args:
            inputs (List[Tensor]): List of image tensor to process.

        Returns:
            Tensor: Processed and stacked image tensor.
        """
        # bgr to rgb if need
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        # Normalization.
        inputs = [(_input - self.mean) / self.std for _input in inputs]
        # Pad and stack Tensor.
        batch_inputs = stack_batch(inputs, self.pad_size_divisor,
                                   self.pad_value)
        return batch_inputs

    def forward(self,
                data: PreprocessInputs,
                training: bool = False) -> PreprocessOutputs:
        """Performs normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (PreprocessInputs): Input data to process.
            training (bool): Whether to enable training time augmentation.
                This is ignored for :class:`GANDataPreprocessor`. Defaults to
                False.
        Returns:
            PreprocessOutputs: Data in the same format as the model input.
        """

        if isinstance(data, dict):
            return data, []

        inputs, batch_data_samples = self.collate_data(data)

        if isinstance(inputs, list):
            batch_inputs = self._preprocess_image_tensor(inputs)
        else:  # inputs is `dict`
            batch_inputs = dict()
            for k, _input in inputs.items():
                if k in self._NON_IMAGE_KEYS:
                    batch_inputs[k] = torch.stack(_input, dim=0)
                elif k in self._NON_CONCENTATE_KEYS:
                    batch_inputs[k] = _input
                else:
                    batch_inputs[k] = self._preprocess_image_tensor(_input)

        return batch_inputs, batch_data_samples
