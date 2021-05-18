import os

import mmcv
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class PetrelUploadHook(Hook):

    cfg_path = '~/petreloss.conf'

    def __init__(self,
                 data_path='ckpt',
                 suffix='.pth',
                 ceph_path=None,
                 num_retry=5,
                 interval=-1,
                 upload_after_run=True,
                 rm_orig=True):
        super().__init__()
        self.num_retry = num_retry
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

    def after_run(self, runner):
        if not self.upload_after_run:
            return

        _data_path = os.path.join(runner.work_dir, self.data_path)

        files = mmcv.scandir(_data_path, suffix=self.suffix, recursive=False)
        files = [os.path.join(_data_path, x) for x in files]
        # remove the rebundant symlinks in the data directory
        files = [x for x in files if not os.path.islink(x)]

        # get the actual exp_name in work_dir
        exp_name = runner.work_dir.split('/')[-1]

        mmcv.print_log(f'Uploading {len(files)} files to ceph.', 'mmgen')

        for file in files:
            with open(file, 'rb') as f:
                data = f.read()
                _path_splits = file.split('/')
                idx = _path_splits.index(exp_name)
                _rel_path = '/'.join(_path_splits[idx:])
                _ceph_path = os.path.join(self.ceph_path, _rel_path)
                self.client.put(_ceph_path, data)

            if self.rm_orig:
                os.remove(file)
