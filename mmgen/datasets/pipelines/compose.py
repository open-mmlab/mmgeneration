# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence
from copy import deepcopy

from mmgen.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                # add support for using pipelines from `MMClassification`
                if transform['type'].startswith('mmcls.'):
                    try:
                        from mmcls.datasets import \
                            TRANSFORMS as MMCLSTRANSFORMS
                    except ImportError:
                        raise ImportError('Please install mmcls to use '
                                          f'{transform["type"]} dataset.')
                    transform_source = MMCLSTRANSFORMS
                    # remove prefix
                    transform_cfg = deepcopy(transform)
                    transform_cfg['type'] = transform_cfg['type'][6:]
                else:
                    transform_source = TRANSFORMS
                    transform_cfg = deepcopy(transform)
                transform = transform_source.build(transform_cfg)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function.

        Args:
            data (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
