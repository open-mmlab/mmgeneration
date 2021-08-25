# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import pickle

import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only


@HOOKS.register_module()
class PickleDataHook(Hook):
    """Pickle Useful Data Hook.

    This hook will be used in SinGAN training for saving some important data
    that will be used in testing or inference.

    Args:
        output_dir (str): The output path for saving pickled data.
        data_name_list (list[str]): The list contains the name of results in
            outputs dict.
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        before_run (bool, optional): Whether to save before running.
            Defaults to False.
        after_run (bool, optional): Whether to save after running.
            Defaults to False.
        filename_tmpl (str, optional): Format string used to save images. The
            output file name will be formatted as this args.
            Defaults to 'iter_{}.pkl'.
    """

    def __init__(self,
                 output_dir,
                 data_name_list,
                 interval=-1,
                 before_run=False,
                 after_run=False,
                 filename_tmpl='iter_{}.pkl'):
        assert mmcv.is_list_of(data_name_list, str)
        self.output_dir = output_dir
        self.data_name_list = data_name_list
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self._before_run = before_run
        self._after_run = after_run

    @master_only
    def after_run(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if self._after_run:
            self._pickle_data(runner)

    @master_only
    def before_run(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if self._before_run:
            self._pickle_data(runner)

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        self._pickle_data(runner)

    def _pickle_data(self, runner):
        filename = self.filename_tmpl.format(runner.iter + 1)
        if not hasattr(self, '_out_dir'):
            self._out_dir = os.path.join(runner.work_dir, self.output_dir)
        mmcv.mkdir_or_exist(self._out_dir)
        file_path = os.path.join(self._out_dir, filename)
        with open(file_path, 'wb') as f:
            data = runner.outputs['results']
            not_find_keys = []
            data_dict = {}
            for k in self.data_name_list:
                if k in data.keys():
                    data_dict[k] = self._get_numpy_data(data[k])
                else:
                    not_find_keys.append(k)
            pickle.dump(data_dict, f)
            mmcv.print_log(f'Pickle data in {filename}', 'mmgen')

            if len(not_find_keys) > 0:
                mmcv.print_log(
                    f'Cannot find keys for pickling: {not_find_keys}',
                    'mmgen',
                    level=logging.WARN)
            f.flush()

    def _get_numpy_data(self, data):
        if isinstance(data, list):
            return [self._get_numpy_data(x) for x in data]

        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()

        return data
