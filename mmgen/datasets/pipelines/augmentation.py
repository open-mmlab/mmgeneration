import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
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

    def __call__(self, results):
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


@PIPELINES.register_module()
class Resize:
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
            Note that when it is used, `size_factor` and `max_size` are
            useless. Default: None
        keep_ratio (bool): If set to True, images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: False.
            Note that it is used togher with `scale`.
        size_factor (int): Let the output shape be a multiple of size_factor.
            Default:None.
            Note that when it is used, `scale` should be set to None and
            `keep_ratio` should be set to False.
        max_size (int): The maximum size of the longest side of the output.
            Default:None.
            Note that it is used togher with `size_factor`.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear" | "bicubic" | "area" | "lanczos".
            Default: "bilinear".
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.
    """

    def __init__(self,
                 keys,
                 scale=None,
                 keep_ratio=False,
                 size_factor=None,
                 max_size=None,
                 interpolation='bilinear',
                 backend=None):
        assert keys, 'Keys should not be empty.'
        if size_factor:
            assert scale is None, ('When size_factor is used, scale should ',
                                   f'be None. But received {scale}.')
            assert keep_ratio is False, ('When size_factor is used, '
                                         'keep_ratio should be False.')
        if max_size:
            assert size_factor is not None, (
                'When max_size is used, '
                f'size_factor should also be set. But received {size_factor}.')
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif mmcv.is_tuple_of(scale, int):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        elif scale is not None:
            raise TypeError(
                f'Scale must be None, float or tuple of int, but got '
                f'{type(scale)}.')
        self.keys = keys
        self.scale = scale
        self.size_factor = size_factor
        self.max_size = max_size
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.backend = backend

    def _resize(self, img):
        if self.keep_ratio:
            img, self.scale_factor = mmcv.imrescale(
                img,
                self.scale,
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img,
                self.scale,
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend)
            self.scale_factor = np.array((w_scale, h_scale), dtype=np.float32)
        return img

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.size_factor:
            h, w = results[self.keys[0]].shape[:2]
            new_h = h - (h % self.size_factor)
            new_w = w - (w % self.size_factor)
            if self.max_size:
                new_h = min(self.max_size - (self.max_size % self.size_factor),
                            new_h)
                new_w = min(self.max_size - (self.max_size % self.size_factor),
                            new_w)
            self.scale = (new_w, new_h)
        for key in self.keys:
            results[key] = self._resize(results[key])
            if len(results[key].shape) == 2:
                results[key] = np.expand_dims(results[key], axis=2)

        results['scale_factor'] = self.scale_factor
        results['keep_ratio'] = self.keep_ratio
        results['interpolation'] = self.interpolation

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, scale={self.scale}, '
            f'keep_ratio={self.keep_ratio}, size_factor={self.size_factor}, '
            f'max_size={self.max_size},interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class NumpyPad:
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

    def __call__(self, results):
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
