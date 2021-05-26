import mmcv
import numpy as np
from mmcv.fileio import FileClient

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load image from file.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `None`. If backend is None, the
            global imread_backend specified by ``mmcv.use_backend()`` will be
            used. Default: None.
        save_original_img (bool): If True, maintain a copy of the image in
            ``results`` dict with name of ``f'ori_{key}'``. Default: False.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 backend=None,
                 save_original_img=False,
                 **kwargs):
        self.io_backend = io_backend
        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.backend = backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        filepath = str(results[f'{self.key}_path'])
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(
            img_bytes,
            flag=self.flag,
            channel_order=self.channel_order,
            backend=self.backend)  # HWC

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(io_backend={self.io_backend}, key={self.key}, '
            f'flag={self.flag}, save_original_img={self.save_original_img})')
        return repr_str


@PIPELINES.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load a pair of images from file.

    Each sample contains a pair of images, which are concatenated in the w
    dimension (a|b). This is a special loading class for generation paired
    dataset. It loads a pair of images as the common loader does and crops
    it into two images with the same shape in different domains.

    Required key is "pair_path". Added or modified keys are "pair",
    "pair_ori_shape", "ori_pair", "img_a", "img_b", "img_a_path",
    "img_b_path", "img_a_ori_shape", "img_b_ori_shape", "ori_img_a" and
    "ori_img_b".

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        filepath = str(results[f'{self.key}_path'])
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(img_bytes, flag=self.flag)  # HWC, BGR
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        # crop pair into a and b
        w = img.shape[1]
        if w % 2 != 0:
            raise ValueError(
                f'The width of image pair must be even number, but got {w}.')
        new_w = w // 2
        img_a = img[:, :new_w, :]
        img_b = img[:, new_w:, :]

        results['img_a'] = img_a
        results['img_b'] = img_b
        results['img_a_path'] = filepath
        results['img_b_path'] = filepath
        results['img_a_ori_shape'] = img_a.shape
        results['img_b_ori_shape'] = img_b.shape
        if self.save_original_img:
            results['ori_img_a'] = img_a.copy()
            results['ori_img_b'] = img_b.copy()

        return results
