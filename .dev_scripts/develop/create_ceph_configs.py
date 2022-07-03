import glob
import os.path as osp
import shutil
from argparse import ArgumentParser

from mmengine import Config


def update_ceph_config(filename, args, dry_run=False):
    if filename.startswith(osp.join('configs_ceph', '_base_')):
        # Skip base configs
        return None

    if args.ceph_path.endswith('/'):
        args.ceph_path = args.ceph_path[:-1]
    ceph_path = f'{args.ceph_path}/{args.work_dir_prefix}'

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
            dataset: dict = data_cfg['dataset']
            data_root: str = dataset['data_root']

            dataset_type: str = dataset['type']
            if 'mmcls' in dataset_type:
                repo_name = 'classification'
            else:
                repo_name = 'editing'
            ceph_dataroot_prefix = ceph_dataroot_prefix_temp.format(repo_name)

            for dataroot_prefix in local_dataroot_prefix:
                if data_root.startswith(dataroot_prefix):
                    # avoid cvt './data/imagenet' ->
                    # openmmlab:s3://openmmlab/datasets/classification//imagenet
                    if data_root.startswith(dataroot_prefix + '/'):
                        dataroot_prefix = dataroot_prefix + '/'
                    dataset['data_root'] = data_root.replace(
                        dataroot_prefix, ceph_dataroot_prefix)

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
                vis_cfg['ceph_path'] = ceph_path

        # 3. change logger hook and checkpoint hook
        file_client_args = dict(backend='petrel')

        for name, hooks in config['default_hooks'].items():
            if name == 'logger':
                hooks['out_dir'] = ceph_path
                hooks['file_client_args'] = file_client_args
            elif name == 'checkpoint':
                hooks['out_dir'] = ceph_path
                hooks['file_client_args'] = file_client_args

        # 4. save
        config.dump(config.filename)
        return True

    except:  # noqa
        if dry_run:
            raise
        else:
            return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--ceph-path', type=str, required=True)
    parser.add_argument(
        '--work-dir-prefix',
        type=str,
        default='work_dirs',
        help='Default prefix of the work dirs in the bucket')
    parser.add_argument(
        '--test-file', type=str, default=None, help='Dry-run on a test file.')

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
        update_ceph_config(args.test_file, args)
