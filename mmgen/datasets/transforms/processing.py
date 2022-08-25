# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
import numpy as np
from mmcls.registry import TRANSFORMS as CLS_TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms import Resize as MMCV_Resize

from mmgen.registry import TRANSFORMS


# TODO: remove the item since mmcv.transforms already contain
@TRANSFORMS.register_module()
class Flip(BaseTransform):
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are
    "flip", "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys (list[str]): The images to be flipped.
        flip_ratio (float): The propability to flip the images.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys, flip_ratio=0.5, direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                if isinstance(results[key], list):
                    for v in results[key]:
                        mmcv.imflip_(v, self.direction)
                else:
                    mmcv.imflip_(results[key], self.direction)

        results['flip'] = flip
        results['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, flip_ratio={self.flip_ratio}, '
                     f'direction={self.direction})')
        return repr_str


@TRANSFORMS.register_module()
class Resize(MMCV_Resize):

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if len(results['img'].shape) == 2:
            results['img'] = np.expand_dims(results['img'], axis=2)
        return results


@TRANSFORMS.register_module()
class NumpyPad(BaseTransform):
    """Numpy Padding.

    In this augmentation, numpy padding is adopted to customize padding
    augmentation. Please carefully read the numpy manual in:
    https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    If you just hope a single dimension to be padded, you must set ``padding``
    like this:

    ::

        padding = ((2, 2), (0, 0), (0, 0))

    In this case, if you adopt an input with three dimension, only the first
    dimension will be padded.

    Args:
        padding (int | tuple(int)): Please refer to the args ``pad_width`` in
            ``numpy.pad``.
    """

    def __init__(self, padding, **kwargs):
        self.padding = padding
        self.kwargs = kwargs

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        results['img'] = np.pad(results['img'], self.padding, **self.kwargs)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(padding={self.padding}, kwargs={self.kwargs})')
        return repr_str


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class RandomCropLongEdge(BaseTransform):
    """Random crop the given image by the long edge.

    Args:
        keys (list[str]): The images to be cropped.
    """

    def __init__(self, keys):
        assert keys, 'Keys should not be empty.'
        self.keys = keys

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            img = results[key]
            img_height, img_width = img.shape[:2]
            crop_size = min(img_height, img_width)
            y1 = 0 if img_height == crop_size else \
                np.random.randint(0, img_height - crop_size)
            x1 = 0 if img_width == crop_size else \
                np.random.randint(0, img_width - crop_size)
            y2, x2 = y1 + crop_size - 1, x1 + crop_size - 1

            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            results[key] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')
        return repr_str


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class CenterCropLongEdge(BaseTransform):
    """Center crop the given image by the long edge.

    Args:
        keys (list[str]): The images to be cropped.
    """

    def __init__(self, keys):
        assert keys, 'Keys should not be empty.'
        self.keys = keys

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            img = results[key]
            img_height, img_width = img.shape[:2]
            crop_size = min(img_height, img_width)
            y1 = 0 if img_height == crop_size else \
                int(round(img_height - crop_size) / 2)
            x1 = 0 if img_width == crop_size else \
                int(round(img_width - crop_size) / 2)
            y2 = y1 + crop_size - 1
            x2 = x1 + crop_size - 1

            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            results[key] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')
        return repr_str


@TRANSFORMS.register_module()
class Crop(BaseTransform):
    """Crop data to specific size for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        random_crop (bool): If set to True, it will random crop
            image. Otherwise, it will work as center crop.
    """

    def __init__(self, keys, crop_size, random_crop=True):
        # if not mmcv.is_tuple_of(crop_size, int):
        if not mmengine.is_tuple_of(crop_size, int):
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

    def transform(self, results):
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


@TRANSFORMS.register_module()
class FixedCrop(BaseTransform):
    """Crop paired data (at a specific position) to specific size for training.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        crop_pos (Tuple[int]): Specific position (x, y). If set to None,
            random initialize the position to crop paired data batch.
    """

    def __init__(self, crop_size, crop_pos=None):
        if not mmengine.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')
        if not mmengine.is_tuple_of(crop_pos, int) and (crop_pos is not None):
            raise TypeError(
                'Elements of crop_pos must be int and crop_pos must be'
                f' tuple or None, but got {type(crop_pos[0])} in '
                f'{type(crop_pos)}')

        self.crop_size = crop_size
        self.crop_pos = crop_pos

    def _crop(self, data, x_offset, y_offset, crop_w, crop_h):
        crop_bbox = [x_offset, y_offset, crop_w, crop_h]
        data_ = data[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w,
                     ...]
        return data_, crop_bbox

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        default_key = 'img'
        data_h, data_w = results[default_key].shape[:2]
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

        # In fixed crop for paired images, sizes should be the same
        # hardcode to use f-string
        if (results[default_key].shape[0] != data_h
                or results[default_key].shape[1] != data_w):
            raise ValueError(
                'The sizes of paired images should be the same. Expected '
                f'({data_h}, {data_w}), but got '
                f'({results[default_key].shape[0]}, '
                f'{results[default_key].shape[1]}).')
        data_, crop_bbox = self._crop(results[default_key], x_offset, y_offset,
                                      crop_w, crop_h)
        results[default_key] = data_
        results['crop_bbox'] = crop_bbox
        results['crop_size'] = self.crop_size
        results['crop_pos'] = self.crop_pos
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'crop_size={self.crop_size}, '
                     f'crop_pos={self.crop_pos}')
        return repr_str
