# Copyright (c) OpenMMLab. All rights reserved.

from numbers import Number
from typing import Sequence, Union

import mmengine
import numpy as np
import torch
from mmengine.structures import BaseDataElement, LabelData

from .pixel_data import PixelData


def format_label(value: Union[torch.Tensor, np.ndarray, Sequence, int],
                 num_classes: int = None) -> LabelData:
    """Convert label of various python types to :obj:`mmengine.LabelData`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.
        num_classes (int, optional): The number of classes. If not None, set
            it to the metainfo. Defaults to None.

    Returns:
        :obj:`mmengine.LabelData`: The foramtted label data.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')

    metainfo = {}
    if num_classes is not None:
        metainfo['num_classes'] = num_classes
        if value.max() >= num_classes:
            raise ValueError(f'The label data ({value}) should not '
                             f'exceed num_classes ({num_classes}).')
    label = LabelData(label=value, metainfo=metainfo)
    return label


class GenDataSample(BaseDataElement):
    """A data structure interface of Generation task.

    The attributes in ``GenDataSample`` are divided into several parts:
        - ``gt_img``: Ground truth image(s).
        - ``fake_img``: Generated fake image(s).
        - ``noise``: Input noise of GAN models to generate fake image(s).

    Meta field:
        img_shape (Tuple): The shape of the corresponding input image.
            Used for visualization.
        ori_shape (Tuple): The original shape of the corresponding image.
            Used for visualization.
        num_classes (int): The number of all categories.
            Used for label format conversion.

    Data field:
        gt_img (PixelData): Input real images
        fake_img (PixelData): Output fake images

        gt_label (LabelData): The ground truth label.
        pred_label (LabelData): The predicted label.
        scores (torch.Tensor): The outputs of model.
        logits (torch.Tensor): The outputs of model without softmax nor
            sigmoid.

    Examples:
        >>> import torch
        >>> from mmgen.structures import GenDataSample
        >>>
        >>> img_meta = dict(img_shape=(960, 720), num_classes=5)
        >>> data_sample = GenDataSample(metainfo=img_meta)
        >>> data_sample.set_gt_label(3)
        >>> print(data_sample)
        <GenDataSample(
           META INFORMATION
           num_classes = 5
           img_shape = (960, 720)
           DATA FIELDS
           gt_label: <LabelData(
                   META INFORMATION
                   num_classes: 5
                   DATA FIELDS
                   label: tensor([3])
               ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
        >>> # For multi-label data
        >>> data_sample.set_gt_label([0, 1, 4])
        >>> print(data_sample.gt_label)
        <LabelData(
            META INFORMATION
            num_classes: 5
            DATA FIELDS
            label: tensor([0, 1, 4])
        ) at 0x7fd7d1b41970>
        >>> # Set one-hot format score
        >>> score = torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1])
        >>> data_sample.set_pred_score(score)
        >>> print(data_sample.pred_label)
        <LabelData(
            META INFORMATION
            num_classes: 5
            DATA FIELDS
            score: tensor([0.1, 0.1, 0.6, 0.1, 0.1])
        ) at 0x7fd7d1b41970>
    """

    @property
    def gt_img(self) -> PixelData:
        return self._gt_img

    @gt_img.setter
    def gt_img(self, value: PixelData):
        self.set_field(value, '_gt_img', dtype=PixelData)

    @gt_img.deleter
    def gt_img(self):
        del self._gt_img

    @property
    def gt_samples(self) -> 'GenDataSample':
        return self._gt_samples

    @gt_samples.setter
    def gt_samples(self, value: 'GenDataSample'):
        self.set_field(value, '_gt_samples', dtype=GenDataSample)

    @gt_samples.deleter
    def gt_samples(self):
        del self._gt_samples

    @property
    def noise(self) -> torch.Tensor:
        return self._noise

    @noise.setter
    def noise(self, value: PixelData):
        self.set_field(value, '_noise', dtype=torch.Tensor)

    @noise.deleter
    def noise(self):
        del self._noise

    @property
    def fake_img(self) -> PixelData:
        return self._fake_img

    @fake_img.setter
    def fake_img(self, value: PixelData):
        self.set_field(value, '_fake_img', dtype=PixelData)

    @fake_img.deleter
    def fake_img(self):
        del self._fake_img

    @property
    def sample_model(self) -> str:
        return self._sample_model

    @sample_model.setter
    def sample_model(self, value: str):
        self.set_field(value, '_sample_model', dtype=str)

    @sample_model.deleter
    def sample_model(self):
        del self._sample_model

    @property
    def ema(self) -> 'GenDataSample':
        return self._ema

    @ema.setter
    def ema(self, value: 'GenDataSample'):
        self.set_field(value, '_ema', dtype=GenDataSample)

    @ema.deleter
    def ema(self):
        del self._ema

    @property
    def orig(self) -> 'GenDataSample':
        return self._orig

    @orig.setter
    def orig(self, value: 'GenDataSample'):
        self.set_field(value, '_orig', dtype=GenDataSample)

    @orig.deleter
    def orig(self):
        del self._orig

    def set_gt_label(
        self, value: Union[np.ndarray, torch.Tensor, Sequence[Number], Number]
    ) -> 'GenDataSample':
        """Set label of ``gt_label``."""
        label = format_label(value, self.get('num_classes'))
        if 'gt_label' in self:
            self.gt_label.label = label.label
        else:
            self.gt_label = label
        return self

    @property
    def gt_label(self):
        return self._gt_label

    @gt_label.setter
    def gt_label(self, value: LabelData):
        self.set_field(value, '_gt_label', dtype=LabelData)

    @gt_label.deleter
    def gt_label(self):
        del self._gt_label
