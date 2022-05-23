# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmcv.utils import is_list_of

from mmgen.core.data_structures import GenDataSample
from mmgen.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackGenInputs(BaseTransform):

    def __init__(self, keys='img', meta_keys=None):
        if isinstance(keys, str):
            self.keys = [keys]
        elif is_list_of(keys, str):
            self.keys = keys
        else:
            raise TypeError(
                'keys is supported to be a string or a list of string')
        self.meta_keys = meta_keys

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
        for key in self.keys:
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
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(key={self.key}, meta_keys={self.meta_keys})')
