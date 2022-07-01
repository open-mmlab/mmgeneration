# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Optional, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.config import Config
from mmengine.fileio import dump
from mmengine.visualization import BaseVisBackend
from mmengine.visualization.vis_backend import force_init_env

from mmgen.registry import VISBACKENDS


@VISBACKENDS.register_module()
class GenVisBackend(BaseVisBackend):
    """Generation visualization backend class. It can write image, config,
    scalars, etc. to the local hard disk and ceph path. You can get the drawing
    backend through the experiment property for custom drawing.

    Examples:
        >>> from mmgen.visualization import GenVisBackend
        >>> import numpy as np
        >>> vis_backend = GenVisBackend(save_dir='temp_dir',
        >>>                             ceph_path='s3://temp-bucket')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> vis_backend.add_image('img', img)
        >>> vis_backend.add_scalar('mAP', 0.6)
        >>> vis_backend.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis_backend.add_config(cfg)
    Args:
        save_dir (str): The root directory to save the files produced by the
            visualizer.
        img_save_dir (str): The directory to save images.
            Default to 'vis_image'.
        config_save_file (str): The file name to save config.
            Default to 'config.py'.
        scalar_save_file (str):  The file name to save scalar values.
            Default to 'scalars.json'.
        ceph_path (Optional[str]): The remote path of Ceph cloud storage.
            Defaults to None.
    """

    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json',
                 ceph_path: Optional[str] = None):
        assert config_save_file.split('.')[-1] == 'py'
        assert scalar_save_file.split('.')[-1] == 'json'
        super().__init__(save_dir)
        self._img_save_dir = img_save_dir
        self._config_save_file = config_save_file
        self._scalar_save_file = scalar_save_file

        self._ceph_path = ceph_path
        self._file_client = None

    def _init_env(self):
        """Init save dir."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)
        self._img_save_dir = osp.join(
            self._save_dir,  # type: ignore
            self._img_save_dir)
        self._config_save_file = osp.join(
            self._save_dir,  # type: ignore
            self._config_save_file)
        self._scalar_save_file = osp.join(
            self._save_dir,  # type: ignore
            self._scalar_save_file)

        if self._ceph_path is not None:
            file_client_args = dict(
                path_mapping={self._save_dir: self._ceph_path})
            self._file_client = mmcv.FileClient(
                backend='petrel', **file_client_args)

    @property  # type: ignore
    @force_init_env
    def experiment(self) -> 'GenVisBackend':
        """Return the experiment object associated with this visualization
        backend."""
        return self

    @force_init_env
    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to disk.

        Args:
            config (Config): The Config object
        """
        assert isinstance(config, Config)
        config.dump(self._config_save_file)
        self._upload(self._config_save_file)

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.array,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        assert image.dtype == np.uint8
        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(self._img_save_dir, exist_ok=True)
        save_file_name = f'{name}_{step}.png'
        cv2.imwrite(osp.join(self._img_save_dir, save_file_name), drawn_image)
        self._upload(osp.join(self._img_save_dir, save_file_name))

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to disk.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._dump({name: value, 'step': step}, self._scalar_save_file, 'json')
        self._upload(f'{self._scalar_save_file}.json')

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars to disk.

        The scalar dict will be written to the default and
        specified files if ``file_path`` is specified.
        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values. The value must be dumped
                into json format.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the ``file_path`` file at the same time
                if the ``file_path`` parameter is specified.
                Default to None.
        """
        assert isinstance(scalar_dict, dict)
        scalar_dict.setdefault('step', step)

        if file_path is not None:
            assert file_path.split('.')[-1] == 'json'
            new_save_file_path = osp.join(
                self._save_dir,  # type: ignore
                file_path)
            assert new_save_file_path != self._scalar_save_file, \
                '``file_path`` and ``scalar_save_file`` have the ' \
                'same name, please set ``file_path`` to another value'
            self._dump(scalar_dict, new_save_file_path, 'json')
        self._dump(scalar_dict, self._scalar_save_file, 'json')
        self._upload(f'{self._scalar_save_file}.json')

    def _dump(self, value_dict: dict, file_path: str,
              file_format: str) -> None:
        """dump dict to file.

        Args:
           value_dict (dict) : The dict data to saved.
           file_path (str): The file path to save data.
           file_format (str): The file format to save data.
        """
        with open(file_path, 'a+') as f:
            dump(value_dict, f, file_format=file_format)
            f.write('\n')

    def _upload(self, path: str) -> None:
        """Upload file at path to remote.

        Args:
            path (str): Path of file to upload.
        """
        if self._file_client is None:
            return
        with open(path, 'rb') as file:
            self._file_client.put(file, path)
        if self._delete_local:
            os.remove(path)
