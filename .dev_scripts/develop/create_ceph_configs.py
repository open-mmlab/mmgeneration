import glob
import os.path as osp
import shutil
from argparse import ArgumentParser
from copy import deepcopy
from importlib.machinery import SourceFileLoader

from mmengine import Config


def update_data(data_cfg, data_root_mapping):

    local_dataroot_prefix = ['data', './data']
    ceph_dataroot_prefix_temp = 'openmmlab:s3://openmmlab/datasets/{}/'

    # val_dataloader may None
    if data_cfg is None:
        return None
    dataset: dict = data_cfg['dataset']
    dataset_updated = deepcopy(dataset)

    dataset_type: str = dataset['type']
    if 'mmcls' in dataset_type:
        repo_name = 'classification'
    else:
        repo_name = 'editing'
    ceph_dataroot_prefix = ceph_dataroot_prefix_temp.format(repo_name)

    if 'data_root' in dataset:
        data_root: str = dataset['data_root']

        for dataroot_prefix in local_dataroot_prefix:
            if data_root.startswith(dataroot_prefix):
                # avoid cvt './data/imagenet' ->
                # openmmlab:s3://openmmlab/datasets/classification//imagenet
                if data_root.startswith(dataroot_prefix + '/'):
                    dataroot_prefix = dataroot_prefix + '/'
                data_root = data_root.replace(dataroot_prefix,
                                              ceph_dataroot_prefix)
                # add '/' at the end
                if not data_root.endswith('/'):
                    data_root = data_root + '/'
                data_root = data_root_mapping.get(data_root, data_root)
                # NOTE: a dirty solution because high-level and
                # low-level datasets are saved on different clustre
                if 'classification' in data_root:
                    data_root = data_root.replace('openmmlab', 'openmmlab_cls',
                                                  1)
                dataset_updated['data_root'] = data_root

    elif 'data_roots' in dataset:
        # specific for pggan dataset, which need a dict of data_roots
        data_roots: dict = dataset['data_roots']
        for k, data_root in data_roots.items():
            for dataroot_prefix in local_dataroot_prefix:
                if data_root.startswith(dataroot_prefix):
                    # avoid cvt './data/imagenet' ->
                    # openmmlab:s3://openmmlab/datasets/classification//imagenet
                    if data_root.startswith(dataroot_prefix + '/'):
                        dataroot_prefix = dataroot_prefix + '/'
                    data_root = data_root.replace(dataroot_prefix,
                                                  ceph_dataroot_prefix)
                    # add '/' at the end
                    if not data_root.endswith('/'):
                        data_root = data_root + '/'
                    data_root = data_root_mapping.get(data_root, data_root)
                    data_roots[k] = data_root
        dataset_updated['data_roots'] = data_roots

    else:
        raise KeyError

    # update pipeline in dataset_updated inplace
    pipelines = dataset_updated['pipeline']
    for pipeline in pipelines:
        type_ = pipeline['type']
        # only change mmcv(mmcls)'s loading config
        if type_ == 'mmcls.LoadImageFromFile':
            pipeline['file_client_args'] = dict(backend='petrel')
            break
    return dataset_updated


def update_intervals(config):
    # 1. change max-iters and val-interval
    if 'train_cfg' in config and config['train_cfg']:
        train_cfg = config['train_cfg']
        train_cfg['max_iters'] = 500
        train_cfg['val_interval'] = 100
    # 2. change num samples
    if 'val_evaluator' in config and config['val_evaluator']:
        val_metrics = config['val_evaluator']['metrics']
    else:
        val_metrics = []
    if 'test_evaluator' in config and config['test_evaluator']:
        test_metrics = config['test_evaluator']['metrics']
    else:
        test_metrics = []
    # test_metrics = config['test_evaluator']['metrics'] \
    #     if 'test_evaluator' in config else []
    for metric in val_metrics + test_metrics:
        if 'fake_nums' in metric:
            metric['fake_nums'] = min(500, metric['fake_nums'])
        if 'real_nums' in metric:
            metric['real_nums'] = min(500, metric['real_nums'])
    # 3. change vis interval
    if 'custom_hooks' in config and config['custom_hooks']:
        for hook in config['custom_hooks']:
            if hook['type'] == 'GenVisualizationHook':
                hook['interval'] = 100
    # 4. change logging interval
    if 'default_hooks' in config and config['default_hooks']:
        config['default_hooks']['logger'] = dict(
            type='LoggerHook', interval=10)
    return config


def update_ceph_config(filename,
                       args,
                       dry_run=False,
                       data_root_mapping=dict()):
    if filename.startswith(osp.join('configs_ceph', '_base_')):
        # Skip base configs
        return None

    if args.ceph_path is not None:
        if args.ceph_path.endswith('/'):
            args.ceph_path = args.ceph_path[:-1]
        ceph_path = f'{args.ceph_path}/{args.work_dir_prefix}'
        if not ceph_path.endswith('/'):
            ceph_path = ceph_path + '/'
    else:
        # disable save local results to ceph
        ceph_path = None

    try:
        # 0. load config
        config = Config.fromfile(filename)
        # 1. change dataloader
        dataloader_prefix = [
            f'{p}_dataloader' for p in ['train', 'val', 'test']
        ]
        for prefix in dataloader_prefix:
            if not hasattr(config, prefix) or config[prefix] is None:
                continue
            data_cfg = config[prefix]

            dataset_updated = update_data(data_cfg, data_root_mapping)
            config[prefix]['dataset'] = dataset_updated

        # 2. change visualizer
        _, project, name = filename.split('.')[0].split('/')
        for vis_cfg in config['vis_backends']:
            if vis_cfg['type'] == 'GenVisBackend':
                if ceph_path is not None:
                    vis_cfg['ceph_path'] = ceph_path
                    if args.not_delete_local:
                        vis_cfg['delete_local_image'] = False

        # add pavi config
        if args.add_pavi:
            # check if pavi config is inheritance from _base_
            find_inherit = False
            for vis_cfg in config['vis_backends']:
                if vis_cfg['type'] == 'PaviGenVisBackend':
                    vis_cfg['exp_name'] = name
                    vis_cfg['project'] = project
                    find_inherit = True
                    break

            if not find_inherit:
                pavi_cfg = dict(
                    type='PaviGenVisBackend', exp_name=name, project=project)
                config['vis_backends'].append(pavi_cfg)

        # add wandb config
        if args.add_wandb:
            # check if wandb config is inheritance from _base_
            find_inherit = False
            for vis_cfg in config['vis_backends']:
                if vis_cfg['type'] == 'WandbGenVisBackend':
                    vis_cfg['name'] = name  # name of config
                    vis_cfg['project'] = project  # name of model
                    find_inherit = True
                    break

            if not find_inherit:
                pavi_cfg = dict(
                    type='WandbGenVisBackend',
                    init_kwargs=dict(name=name, project=project))
                config['vis_backends'].append(pavi_cfg)

        # add tensorboard config
        if args.add_tensorboard:
            find_inherit = False
            for vis_cfg in config['vis_backends']:
                if vis_cfg['type'] == 'TensorboardGenVisBackend':
                    find_inherit = True
                    break

            if not find_inherit:
                tensorboard_cfg = dict(type='TensorboardGenVisBackend')
                config['vis_backends'].append(tensorboard_cfg)

        config['visualizer']['vis_backends'] = config['vis_backends']

        # 3. change logger hook and checkpoint hook
        file_client_args = dict(backend='petrel')

        for name, hooks in config['default_hooks'].items():
            # ignore ceph path
            if ceph_path is None:
                continue
            if name == 'logger':
                hooks['out_dir'] = ceph_path
                hooks['file_client_args'] = file_client_args
            elif name == 'checkpoint':
                hooks['out_dir'] = ceph_path
                hooks['file_client_args'] = file_client_args

        # 4. change interval and num_samples for quick run
        if args.quick_run:
            update_intervals(config)

        # 5. save
        config.dump(config.filename)
        return True

    except Exception as e:  # noqa
        if dry_run:
            print(e)
            raise
        else:
            return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--ceph-path', type=str, default=None)
    parser.add_argument(
        '--work-dir-prefix',
        type=str,
        default='work_dirs',
        help='Default prefix of the work dirs in the bucket')
    parser.add_argument(
        '--test-file', type=str, default=None, help='Dry-run on a test file.')
    parser.add_argument(
        '--add-pavi', action='store_true', help='Add pavi config or not.')
    parser.add_argument(
        '--add-wandb', action='store_true', help='Add wandb config or not.')
    parser.add_argument(
        '--add-tensorboard',
        action='store_true',
        help='Add Tensorboard config or not.')
    parser.add_argument(
        '--quick-run',
        action='store_true',
        help=('Whether start a quick run to detect bugs. This is usefully in '
              'train/test-benchmark.'))
    parser.add_argument(
        '--not-delete-local',
        action='store_true',
        help='Do not delete local image')
    parser.add_argument(
        '--data-remapping',
        type=str,
        default='',
        help='Path of dataroot remapping file which contain a dict'
        ' of root mapping named ``ROOT_MAP``.')

    args = parser.parse_args()

    ROOT_MAP = dict()
    if len(args.data_remapping) > 0:
        ROOT_MAP = SourceFileLoader('ROOT_MAP',
                                    args.data_remapping).load_module().ROOT_MAP

    if args.test_file is None:

        print('Copying config files to "config_ceph" ...')
        shutil.copytree('configs', 'configs_ceph', dirs_exist_ok=True)

        print('Updating ceph configuration ...')
        files = glob.glob(
            osp.join('configs_ceph', '**', '*.py'), recursive=True)
        res = [
            update_ceph_config(f, args, data_root_mapping=ROOT_MAP)
            for f in files
        ]

        count_skip = res.count(None)
        count_done = res.count(True)
        count_fail = res.count(False)
        fail_list = [fn for status, fn in zip(res, files) if status is False]

        print(f'Successfully update {count_done} configs.')
        print(f'Skip {count_skip} configs.')
        print(f'Fail {count_fail} configs:')
        for fn in fail_list:
            print(fn)

    else:
        update_ceph_config(
            args.test_file, args, dry_run=True, data_root_mapping=ROOT_MAP)
