# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
from mmcv.runner import HOOKS, Hook, master_only


@HOOKS.register_module()
class PetrelUploadHook(Hook):
    """Upload Data with Petrel.

    With this hook, users can easily upload data to the cloud server for
    saving local spaces. Please read the notes below for using this hook,
    especially for the declaration of ``petrel``.

    One of the major functions is to transfer the checkpoint files from the
    local directory to the cloud server.

    .. note::

        ``petrel`` is a private package containing several commonly used
        ``AWS`` python API. Currently, this package is only for internal usage
        and will not be released to the public. We will support ``boto3`` in
        the future. We think this hook is an easy template for you to transfer
        to ``boto3``.

    Args:
        data_path (str, optional): Relative path of the data according to
            current working directory. Defaults to 'ckpt'.
        suffix (str, optional): Suffix for the data files. Defaults to '.pth'.
        ceph_path (str | None, optional): Path in the cloud server.
            Defaults to None.
        interval (int, optional): Uploading interval (by iterations).
            Default: -1.
        upload_after_run (bool, optional): Whether to upload after running.
            Defaults to True.
        rm_orig (bool, optional): Whether to removing the local files after
            uploading. Defaults to True.
    """

    cfg_path = '~/petreloss.conf'

    def __init__(self,
                 data_path='ckpt',
                 suffix='.pth',
                 ceph_path=None,
                 interval=-1,
                 upload_after_run=True,
                 rm_orig=True):
        super().__init__()
        self.interval = interval
        self.upload_after_run = upload_after_run
        self.data_path = data_path
        self.suffix = suffix
        self.ceph_path = ceph_path
        self.rm_orig = rm_orig

        # setup petrel client
        try:
            from petrel_client.client import Client
        except ImportError:
            raise ImportError('Please install petrel in advance.')
        self.client = Client(self.cfg_path)

    @staticmethod
    def upload_dir(client,
                   local_dir,
                   remote_dir,
                   exp_name=None,
                   suffix=None,
                   remove_local_file=True):
        """Upload a directory to the cloud server.

        Args:
            client (obj): AWS client.
            local_dir (str): Path for the local data.
            remote_dir (str): Path for the remote server.
            exp_name (str, optional): The experiment name. Defaults to None.
            suffix (str, optional): Suffix for the data files.
                Defaults to None.
            remove_local_file (bool, optional): Whether to removing the local
                files after uploading. Defaults to True.
        """
        files = mmcv.scandir(local_dir, suffix=suffix, recursive=False)
        files = [os.path.join(local_dir, x) for x in files]
        # remove the rebundant symlinks in the data directory
        files = [x for x in files if not os.path.islink(x)]

        # get the actual exp_name in work_dir
        if exp_name is None:
            exp_name = local_dir.split('/')[-1]

        mmcv.print_log(f'Uploading {len(files)} files to ceph.', 'mmgen')

        for file in files:
            with open(file, 'rb') as f:
                data = f.read()
                _path_splits = file.split('/')
                idx = _path_splits.index(exp_name)
                _rel_path = '/'.join(_path_splits[idx:])
                _ceph_path = os.path.join(remote_dir, _rel_path)
                client.put(_ceph_path, data)

            # remove the local file to save space
            if remove_local_file:
                os.remove(file)

    @master_only
    def after_run(self, runner):
        """The behavior after the whole running.

        Args:
            runner (object): The runner.
        """
        if not self.upload_after_run:
            return

        _data_path = os.path.join(runner.work_dir, self.data_path)
        # get the actual exp_name in work_dir
        exp_name = runner.work_dir.split('/')[-1]

        self.upload_dir(
            self.client,
            _data_path,
            self.ceph_path,
            exp_name=exp_name,
            suffix=self.suffix,
            remove_local_file=self.rm_orig)
