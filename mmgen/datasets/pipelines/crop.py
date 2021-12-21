# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class Crop:
    """Crop data to specific size for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        random_crop (bool): If set to True, it will random crop
            image. Otherwise, it will work as center crop.
    """

    def __init__(self, keys, crop_size, random_crop=True):
        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')

        self.keys = keys
        self.crop_size = crop_size
        self.random_crop = random_crop

    def _crop(self, data):
        if not isinstance(data, list):
            data_list = [data]
        else:
            data_list = data

        crop_bbox_list = []
        data_list_ = []

        for item in data_list:
            data_h, data_w = item.shape[:2]
            crop_h, crop_w = self.crop_size
            crop_h = min(data_h, crop_h)
            crop_w = min(data_w, crop_w)

            if self.random_crop:
                x_offset = np.random.randint(0, data_w - crop_w + 1)
                y_offset = np.random.randint(0, data_h - crop_h + 1)
            else:
                x_offset = max(0, (data_w - crop_w)) // 2
                y_offset = max(0, (data_h - crop_h)) // 2

            crop_bbox = [x_offset, y_offset, crop_w, crop_h]
            item_ = item[y_offset:y_offset + crop_h,
                         x_offset:x_offset + crop_w, ...]
            crop_bbox_list.append(crop_bbox)
            data_list_.append(item_)

        if not isinstance(data, list):
            return data_list_[0], crop_bbox_list[0]
        return data_list_, crop_bbox_list

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for k in self.keys:
            data_, crop_bbox = self._crop(results[k])
            results[k] = data_
            results[k + '_crop_bbox'] = crop_bbox
        results['crop_size'] = self.crop_size
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, '
                     f'random_crop={self.random_crop}')

        return repr_str


@PIPELINES.register_module()
class FixedCrop:
    """Crop paired data (at a specific position) to specific size for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        crop_pos (Tuple[int]): Specific position (x, y). If set to None,
            random initialize the position to crop paired data batch.
    """

    def __init__(self, keys, crop_size, crop_pos=None):
        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')
        if not mmcv.is_tuple_of(crop_pos, int) and (crop_pos is not None):
            raise TypeError(
                'Elements of crop_pos must be int and crop_pos must be'
                f' tuple or None, but got {type(crop_pos[0])} in '
                f'{type(crop_pos)}')

        self.keys = keys
        self.crop_size = crop_size
        self.crop_pos = crop_pos

    def _crop(self, data, x_offset, y_offset, crop_w, crop_h):
        crop_bbox = [x_offset, y_offset, crop_w, crop_h]
        data_ = data[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w,
                     ...]
        return data_, crop_bbox

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        data_h, data_w = results[self.keys[0]].shape[:2]
        crop_h, crop_w = self.crop_size
        crop_h = min(data_h, crop_h)
        crop_w = min(data_w, crop_w)

        if self.crop_pos is None:
            x_offset = np.random.randint(0, data_w - crop_w + 1)
            y_offset = np.random.randint(0, data_h - crop_h + 1)
        else:
            x_offset, y_offset = self.crop_pos
            crop_w = min(data_w - x_offset, crop_w)
            crop_h = min(data_h - y_offset, crop_h)

        for k in self.keys:
            # In fixed crop for paired images, sizes should be the same
            if (results[k].shape[0] != data_h
                    or results[k].shape[1] != data_w):
                raise ValueError(
                    'The sizes of paired images should be the same. Expected '
                    f'({data_h}, {data_w}), but got ({results[k].shape[0]}, '
                    f'{results[k].shape[1]}).')
            data_, crop_bbox = self._crop(results[k], x_offset, y_offset,
                                          crop_w, crop_h)
            results[k] = data_
            results[k + '_crop_bbox'] = crop_bbox
        results['crop_size'] = self.crop_size
        results['crop_pos'] = self.crop_pos
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, '
                     f'crop_pos={self.crop_pos}')
        return repr_str
