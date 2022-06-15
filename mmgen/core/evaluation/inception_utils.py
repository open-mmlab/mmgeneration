# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os.path as osp
import pickle
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmengine import is_filepath, print_log
from mmengine.dataset import BaseDataset, Compose
from mmengine.dist import all_gather, get_world_size, is_main_process
from mmengine.evaluator import BaseMetric
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.models.inception import inception_v3

from mmgen.models.architectures import InceptionV3
from mmgen.utils import MMGEN_CACHE_DIR
from mmgen.utils.io_utils import download_from_url

ALLOWED_INCEPTION = ['StyleGAN', 'PyTorch']
TERO_INCEPTION_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'  # noqa


@contextmanager
def disable_gpu_fuser_on_pt19():
    """On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run.

    Refers to:
      https://github.com/GaParmar/clean-fid/blob/5e1e84cdea9654b9ac7189306dfa4057ea2213d8/cleanfid/inception_torchscript.py#L9  # noqa
      https://github.com/GaParmar/clean-fid/issues/5
      https://github.com/pytorch/pytorch/issues/64062
    """
    if torch.__version__.startswith('1.9.'):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith('1.9.'):
        torch._C._jit_override_can_fuse_on_gpu(old_val)


def load_inception(inception_args, metric):
    """Load Inception Model from given ``inception_args`` and ``metric``. This
    function would try to load Inception under the guidance of 'type' given in
    `inception_args`, if not given, we would try best to load Tero's ones. In
    detailly, we would first try to load the model from disk with the given
    'inception_path', and then try to download the checkpoint from
    'inception_url'. If both method are failed, pytorch version of Inception
    would be loaded.

    Args:
        inception_args (dict): Keyword args for inception net.
        metric (string): Metric to use the Inception. This argument would
            influence the pytorch's Inception loading.
    Returns:
        model (torch.nn.Module): Loaded Inception model.
        style (string): The version of the loaded Inception.
    """

    if not isinstance(inception_args, dict):
        raise TypeError('Receive invalid \'inception_args\': '
                        f'\'{inception_args}\'')

    _inception_args = deepcopy(inception_args)
    inceptoin_type = _inception_args.pop('type', None)

    if torch.__version__ < '1.6.0':
        print_log(
            'Current Pytorch Version not support script module, load '
            'Inception Model from torch model zoo. If you want to use '
            'Tero\' script model, please update your Pytorch higher '
            f'than \'1.6\' (now is {torch.__version__})', 'current')
        return _load_inception_torch(_inception_args, metric), 'pytorch'

    # load pytorch version is specific
    if inceptoin_type != 'StyleGAN':
        return _load_inception_torch(_inception_args, metric), 'pytorch'

    # try to load Tero's version
    path = _inception_args.get('inception_path', TERO_INCEPTION_URL)

    # try to parse `path` as web url and download
    if 'http' not in path:
        model = _load_inception_from_path(path)
        if isinstance(model, torch.nn.Module):
            return model, 'StyleGAN'

    # try to parse `path` as path on disk
    model = _load_inception_from_url(path)
    if isinstance(model, torch.nn.Module):
        return model, 'StyleGAN'

    raise RuntimeError('Cannot Load Inception Model, please check the input '
                       f'`inception_args`: {inception_args}')


def _load_inception_from_path(inception_path):
    print_log(
        'Try to load Tero\'s Inception Model from '
        f'\'{inception_path}\'.', 'current')
    try:
        model = torch.jit.load(inception_path)
        print_log('Load Tero\'s Inception Model successfully.', 'current')
    except Exception as e:
        model = None
        print_log('Load Tero\'s Inception Model failed. '
                  f'\'{e}\' occurs.', 'current')
    return model


def _load_inception_from_url(inception_url: str) -> nn.Module:
    """Load Inception network from the give `inception_url`"""
    inception_url = inception_url if inception_url else TERO_INCEPTION_URL
    print_log(f'Try to download Inception Model from {inception_url}...',
              'current')
    try:
        path = download_from_url(inception_url, dest_dir=MMGEN_CACHE_DIR)
        print_log('Download Finished.', 'current')
        return _load_inception_from_path(path)
    except Exception as e:
        print_log(f'Download Failed. {e} occurs.', 'current')
        return None


def _load_inception_torch(inception_args, metric) -> nn.Module:
    """Load Inception network from PyTorch's model zoo."""
    assert metric in ['FID', 'IS']
    if metric == 'FID':
        inception_model = InceptionV3([3], **inception_args)
    elif metric == 'IS':
        inception_model = inception_v3(pretrained=True, transform_input=False)
        print_log(
            'Load Inception V3 Network from Pytorch Model Zoo '
            'for IS calculation. The results can only used '
            'for monitoring purposes. To get more accuracy IS, '
            'please use Tero\'s Inception V3 checkpoints '
            'and use Bicubic Interpolation with Pillow backend '
            'for image resizing. More details may refer to '
            'https://github.com/open-mmlab/mmgeneration/blob/master/docs/en/quick_run.md#is.',  # noqa
            'current')
    return inception_model


def get_inception_feat_cache_name_and_args(
        dataloader: DataLoader, metric: BaseMetric) -> Tuple[str, dict]:
    """Get the name and meta info of the inception feature cache file
    corresponding to the input dataloader and metric. The meta info includs
    'data_root', 'data_prefix', 'meta_info' and 'pipeline' of the dataset, and
    'inception_style' and 'inception_args' of the metric. Then we calculate the
    hash value of the meta info dict with md5, and the name of the inception
    feature cache will be 'inception_feat_{HASH}.pkl'.

    Args:
        datalaoder (Dataloader): The dataloader of real images.
        metric (BaseMetric): The metric which needs inception features.

    Returns:
        Tuple[str, dict]: Filename and meta info dict of the inception feature
            cache.
    """

    dataset: BaseDataset = dataloader.dataset
    assert isinstance(dataset, Dataset), (
        f'Only support normal dataset, but receive {type(dataset)}.')

    # get dataset info
    data_root = deepcopy(dataset.data_root)
    data_prefix = deepcopy(dataset.data_prefix)
    metainfo = dataset.metainfo
    pipeline = dataset.pipeline
    if isinstance(pipeline, Compose):
        pipeline_str = pipeline.__repr__

    # get metric info
    inception_style = metric.inception_style
    inception_args = getattr(metric, 'inception_args', None)

    args = dict(
        data_root=data_root,
        data_prefix=data_prefix,
        metainfo=metainfo,
        pipeline=pipeline_str,
        inception_style=inception_style,
        inception_args=inception_args)

    md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
    cache_tag = f'inception_state-{md5.hexdigest()}.pkl'
    return cache_tag, args


def prepare_inception_feat(
        dataloader: DataLoader,
        metric: BaseMetric,
        data_preprocessor: Optional[nn.Module] = None) -> np.ndarray:
    """Prepare inception feature for the input metric.

    - If `metric.inception_pkl` is an online path, try to download and load
      it. If cannot download or load, corresponding error will be raised.
    - If `metric.inception_pkl` is local path and file exists, try to load the
      file. If cannot load, corresponding error will be raised.
    - If `metric.inception_pkl` is local path and file not exists, we will
      extract the inception feature manually and save to 'inception_pkl'.
    - If `metric.inception_pkl` is not defined, we will extrace the inception
      feature and save it to default cache dir with default name.

    Args:
        datalaoder (Dataloader): The dataloader of real images.
        metric (BaseMetric): The metric which needs inception features.
        data_preprocessor (Optional[nn.Module]): Data preprocessor of the
            module. Used to preprocess the real images. If not passed, real
            images will automatically normalized to [-1, 1]. Defaults to None.

        Returns:
            np.ndarray: Loaded inception feature.
    """
    if not hasattr(metric, 'inception_pkl'):
        return
    inception_pkl: Optional[str] = metric.inception_pkl

    if isinstance(inception_pkl, str):
        if is_filepath(inception_pkl) and osp.exists(inception_pkl):
            with open(inception_pkl, 'rb') as file:
                inception_state = pickle.load(file)
            print_log(
                f'\'{metric.prefix}\' successful load inception feature '
                f'from \'{inception_pkl}\'', 'currnet')
            return inception_state['inception_feat']
        elif inception_pkl.startswith('s3'):
            try:
                raise NotImplementedError(
                    'Not support download from Ceph currently')
            except Exception as exp:
                raise exp('Not support download from Ceph currently')
        elif inception_pkl.startswith('http'):
            try:
                raise NotImplementedError(
                    'Not support download from url currently')
            except Exception as exp:
                # cannot download, raise error
                raise exp('Not support download from url currently')

    # cannot load or download from file, extract manually
    if inception_pkl is None:
        inception_pkl, args = get_inception_feat_cache_name_and_args(
            dataloader, metric)
        inception_pkl = osp.join(MMGEN_CACHE_DIR, inception_pkl)
    else:
        args = dict()
    if osp.exists(inception_pkl):
        with open(inception_pkl, 'rb') as file:
            real_feat = pickle.load(file)['inception_feat']
        print(f'load preprocessed feat from {inception_pkl}')
        return real_feat

    assert hasattr(metric, 'inception'), (
        'Metric must have a inception network to extract inception features.')

    real_feat = []
    mean = getattr(data_preprocessor, 'mean', None)
    std = getattr(data_preprocessor, 'std', None)

    print_log(
        f'Inception pkl \'{inception_pkl}\' is not found, extract '
        'manually.', 'current')

    import rich.progress

    # init rich pbar for the main process
    if is_main_process():
        columns = [
            rich.progress.TextColumn('[bold blue]{task.description}'),
            rich.progress.BarColumn(bar_width=40),
            rich.progress.TaskProgressColumn(),
            rich.progress.TimeRemainingColumn(),
        ]
        pbar = rich.progress.Progress(*columns)
        pbar.start()
        task = pbar.add_task(
            'Calculate Inception Feature.',
            total=len(dataloader.dataset),
            visible=True)

    for data in dataloader:
        inputs, _ = data_preprocessor(data)

        if isinstance(inputs, dict):
            real_key = 'img' if metric.real_key is None else metric.real_key
            img = inputs[real_key]
        else:
            img = inputs

        # make sure the input image is in [-1, 1]
        if mean is None and std is None:
            # rescale to [-1, 1]
            img = img / 127.5 - 1
        else:
            assert mean is not None and std is not None, (
                '\'mean\' and \'std\' must be None or not None at the '
                f'same time. But receive \'{mean}\' and \'{std}\' '
                'respectively.')

        real_feat_ = metric.forward_inception(img)
        real_feat.append(real_feat_)
        # real_feat += torch.tensor_split(real_feat_, real_feat_.shape[0])
        if is_main_process():
            pbar.update(task, advance=len(real_feat_) * get_world_size())

    # stop the pbar
    if is_main_process():
        pbar.stop()

    # collect results
    real_feat = torch.cat(real_feat)
    # use `all_gather` here, gather tensor is much quicker than gather object.
    real_feat = all_gather(real_feat)

    # only cat on the main process
    if is_main_process():
        real_feat = torch.cat(
            real_feat, dim=0)[:len(dataloader.dataset)].cpu().numpy()
        inception_state = dict(inception_feat=real_feat, **args)
        with open(inception_pkl, 'wb') as file:
            pickle.dump(inception_state, file)
        return real_feat
