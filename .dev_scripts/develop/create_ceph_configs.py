import glob
import os.path as osp
import shutil
from argparse import ArgumentParser

from mmengine import Config


def update_ceph_config(filename, args, dry_run=False):
    if filename.startswith(osp.join('configs_ceph', '_base_')):
        # Skip base configs
        return None

    if args.ceph is not None:
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
        local_dataroot_prefix = ['data', './data']
        ceph_dataroot_prefix_temp = 'openmmlab:s3://openmmlab/datasets/{}/'

        for prefix in dataloader_prefix:
            if not hasattr(config, prefix):
                continue
            data_cfg = config[prefix]

            # val_dataloader may None
            if data_cfg is None:
                continue
            dataset: dict = data_cfg['dataset']

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
                        dataset['data_root'] = data_root

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
                            data_root = data_root.replace(
                                dataroot_prefix, ceph_dataroot_prefix)
                            # add '/' at the end
                            if not data_root.endswith('/'):
                                data_root = data_root + '/'
                            data_roots[k] = data_root
                dataset['data_roots'] = data_roots

            else:
                raise KeyError

            pipelines = dataset['pipeline']
            for pipeline in pipelines:
                type_ = pipeline['type']
                # only change mmcv(mmcls)'s loading config
                if type_ == 'mmcls.LoadImageFromFile':
                    pipeline['file_client_args'] = dict(backend='petrel')
                    break
            config[prefix]['dataset'] = dataset

        # 2. change visualizer
        for vis_cfg in config['vis_backends']:
            if vis_cfg['type'] == 'GenVisBackend':
                if ceph_path is not None:
                    vis_cfg['ceph_path'] = ceph_path

        # add pavi config
        if args.add_pavi:
            _, project, name = filename.split('/')
            name = name[:-2]
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

        # 4. save
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

    args = parser.parse_args()

    if args.test_file is None:

        print('Copying config files to "config_ceph" ...')
        shutil.copytree('configs', 'configs_ceph', dirs_exist_ok=True)

        print('Updating ceph configuration ...')
        files = glob.glob(
            osp.join('configs_ceph', '**', '*.py'), recursive=True)
        res = [update_ceph_config(f, args) for f in files]

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
        update_ceph_config(args.test_file, args, dry_run=True)
