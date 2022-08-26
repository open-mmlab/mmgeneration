# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine import is_list_of

from mmgen.registry import TRANSFORMS
from mmgen.structures import GenDataSample


@TRANSFORMS.register_module()
class PackGenInputs(BaseTransform):
    """Pack the inputs data for the image generation.

    Args:
        keys (str): Target keys to pack. Defaults to 'img'.
        pack_all (bool): If true all keys will be packed. Defaults to False.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """

    def __init__(self,
                 keys: str = 'img',
                 pack_all: bool = False,
                 meta_keys: Optional[Sequence[str]] = None):
        self.pack_all = pack_all
        if not self.pack_all:
            if isinstance(keys, str):
                self.keys = [keys]
            elif is_list_of(keys, str):
                self.keys = keys
            else:
                raise TypeError(
                    'keys is supported to be a string or a list of string')
        self.meta_keys = [] if meta_keys is None else meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict(inputs=dict())
        pack_keys = results.keys() if self.pack_all else self.keys
        for key in pack_keys:
            if key in results:
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                packed_results['inputs'][key] = to_tensor(img)

        data_sample = GenDataSample()

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(key={self.keys}, meta_keys={self.meta_keys})')
