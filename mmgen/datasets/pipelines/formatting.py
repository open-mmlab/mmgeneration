# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms import BaseTransform

from mmgen.core.data_structures import GenDataSample
from mmgen.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackGenInputs(BaseTransform):

    def __init__(self, key='img', meta_keys=None):
        self.key = key
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
        packed_results = dict()
        if self.key in results:
            img = results[self.key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        data_sample = GenDataSample()

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(keys={self.keys}, meta_keys={self.meta_keys})')
