# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmcls.datasets import PIPELINES as CLS_PIPELINE
from mmcv.transforms import Resize as MMCV_Resize

from mmgen.registry import TRANSFORMS
from .base import BaseTransform

# TODO: remove the item since mmcv.transforms already contain
@TRANSFORMS.register_module()
class Flip:
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
    """Resize data to a specific size for training or resize the images to fit
    the network input regulation for testing.

    When used for resizing images to fit network input regulation, the case is
    that a network may have several downsample and then upsample operation,
    then the input height and width should be divisible by the downsample
    factor of the network.
    For example, the network would downsample the input for 5 times with
    stride 2, then the downsample factor is 2^5 = 32 and the height
    and width should be divisible by 32.

    Required keys are the keys in attribute "keys", added or modified keys are
    "keep_ratio", "scale_factor", "interpolation" and the
    keys in attribute "keys".

    All keys in "keys" should have the same shape. "test_trans" is used to
    record the test transformation to align the input's shape.

    Args:
        keys (list[str]): The images to be resized.
        scale (float | Tuple[int]): If scale is Tuple(int), target spatial
            size (h, w). Otherwise, target spatial size is scaled by input
            size. If any of scale is -1, we will rescale short edge.
            Note that when it is used, `scale_factor` and `max_size` are
            useless. Default: None
        keep_ratio (bool): If set to True, images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: False.
            Note that it is used togher with `scale`.
        scale_factor (int): Let the output shape be a multiple of scale_factor.
            Default:None.
            Note that when it is used, `scale` should be set to None and
            `keep_ratio` should be set to False.
        max_size (int): The maximum size of the longest side of the output.
            Default:None.
            Note that it is used togher with `scale_factor`.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear" | "bicubic" | "area" | "lanczos".
            Default: "bilinear".
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.
    """

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
    diemansion will be padded.

    Args:
        keys (list[str]): The images to be resized.
        padding (int | tuple(int)): Please refer to the args ``pad_width`` in
            ``numpy.pad``.
    """

    def __init__(self, keys, padding, **kwargs):
        self.keys = keys
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

        for k in self.keys:
            results[k] = np.pad(results[k], self.padding, **self.kwargs)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, padding={self.padding}, kwargs={self.kwargs})'
        )
        return repr_str


@CLS_PIPELINE.register_module()
@TRANSFORMS.register_module()
class RandomImgNoise(BaseTransform):
    """Add random noise with specific distribution and range to the input
    image.

    Args:
        keys (list[str]): The images to be added random noise.
        lower_bound (float, optional): The lower bound of the noise.
            Default to ``0.``.
        upper_bound (float, optional): The upper bound of the noise.
            Default to ``1 / 128.``.
        distribution (str, optional): The probability distribution of the
            noise. Default to 'uniform'.
    """

    def __init__(self,
                 keys,
                 lower_bound=0,
                 upper_bound=1 / 128.,
                 distribution='uniform'):
        assert keys, 'Keys should not be empty.'

        self.keys = keys
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if distribution not in ['uniform', 'normal']:
            raise KeyError('Only support \'uniform\' distribution and '
                           '\'normal\' distribution, receive '
                           f'{distribution}.')
        self.distribution = distribution

    def transform(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.distribution == 'uniform':
            dist_fn = np.random.rand
        else:  # self.distribution == 'normal
            dist_fn = np.random.randn

        for key in self.keys:
            img_size = results[key].shape
            noise = dist_fn(*img_size)
            scale = noise.max() - noise.min()
            noise = noise - noise.min()
            noise = noise / scale * (self.upper_bound - self.lower_bound)
            noise = noise + self.lower_bound
            results[key] += noise

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, lower_bound={self.lower_bound}, '
                     f'upper_bound={self.upper_bound})')
        return repr_str


@CLS_PIPELINE.register_module()
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


@CLS_PIPELINE.register_module()
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
