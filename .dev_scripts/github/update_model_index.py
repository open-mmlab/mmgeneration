#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

# This tool is used to update model-index.yml which is required by MIM, and
# will be automatically called as a pre-commit hook. The updating will be
# triggered if any change of model information (.md files in configs/) has been
# detected before a commit.

import glob
import os
import os.path as osp
import re
import sys
import warnings
from functools import reduce

import mmcv

MMGeneration_ROOT = osp.dirname(osp.dirname(osp.dirname(__file__)))

all_training_data = [
    'cifar', 'ffhq', 'celeba', 'imagenet', 'lsun', 'reds', 'ffhq', 'cat',
    'facades', 'summer2winter', 'horse2zebra', 'maps', 'edges2shoes'
]


def dump_yaml_and_check_difference(obj, file):
    """Dump object to a yaml file, and check if the file content is different
    from the original.

    Args:
        obj (any): The python object to be dumped.
        file (str): YAML filename to dump the object to.
    Returns:
        Bool: If the target YAML file is different from the original.
    """

    str_dump = mmcv.dump(obj, None, file_format='yaml', sort_keys=True)

    if osp.isfile(file):
        file_exists = True
        print(f'    exist {file}')
        with open(file, 'r', encoding='utf-8') as f:
            str_orig = f.read()
    else:
        file_exists = False
        str_orig = None

    if file_exists and str_orig == str_dump:
        is_different = False
    else:
        is_different = True
        print(f'    update {file}')
        with open(file, 'w', encoding='utf-8') as f:
            f.write(str_dump)

    return is_different


def collate_metrics(keys):
    """Collect metrics from the first row of the table.

    Args:
        keys (List): Elements in the first row of the table.

    Returns:
        dict: A dict of metrics.
    """
    used_metrics = dict()
    for idx, key in enumerate(keys):
        if key in [
                'Model', 'Models', 'Download', 'Config',
                'Original Download link', 'Data', 'Dataset'
        ]:
            continue
        # process metric such as Best FID and Best IS
        if '(Iter)' in key:
            key = key.split('(')[0].strip()
        used_metrics[key] = idx
    return used_metrics


def get_task_dict(md_file):
    """Get task dict from repo's README.md".

    Args:
        md_file: Path to repo's README.md file.

    Returns:
        dict: Task name of each method.
    """
    with open(md_file, 'r') as md:
        lines = md.readlines()
    i = 0
    task_dict = dict()
    while i < len(lines):
        if '<details open>' in lines[i] and '<summary>' in lines[i + 1]:
            task = re.findall(r'<summary>(.*) \(click to collapse\)</summary>',
                              lines[i + 1])[0].strip()
            j = i + 2
            while j < len(lines):
                if '</details>' in lines[j]:
                    i = j
                    break
                if '-' in lines[j]:
                    path = re.findall(r'-.*\[.*\]\((.*)\).*\)', lines[j])[0]
                    task_dict[path] = task
                j += 1
        i += 1

    return task_dict


def generate_unique_name(md_file):
    """Search config files and return the unique name of them. For Confin.Name.

    Args:
        md_file (str): Path to .md file.
    Returns:
        dict: dict of unique name for each config file.
    """
    # add configs from _base_/models/MODEL_NAME
    md_root = md_file.split('/')[:2]
    model_name = md_file.split('/')[1]
    base_path = osp.join(*md_root, '..', '_base_', 'models', model_name)
    base_files = os.listdir(base_path) if osp.exists(base_path) else []

    files = os.listdir(osp.dirname(md_file)) + base_files
    config_files = [f[:-3] for f in files if f[-3:] == '.py']
    config_files.sort()
    config_files.sort(key=lambda x: len(x))
    split_names = [f.split('_') for f in config_files]
    config_sets = [set(f.split('_')) for f in config_files]
    common_set = reduce(lambda x, y: x & y, config_sets)
    unique_lists = [[n for n in name if n not in common_set]
                    for name in split_names]

    unique_dict = dict()
    name_list = []
    for i, f in enumerate(config_files):
        base = split_names[i][0]
        unique_dict[f] = base
        if len(unique_lists[i]) > 0:
            for unique in unique_lists[i]:
                candidate_name = f'{base}_{unique}'
                if candidate_name not in name_list and base != unique:
                    unique_dict[f] = candidate_name
                    name_list.append(candidate_name)
                    break
    return unique_dict


def parse_md(md_file, task):
    """Parse .md file and convert it to a .yml file which can be used for MIM.

    Args:
        md_file: Path to .md file.
        task (str): Task type of the method.
    Returns:
        Bool: If the target YAML file is different from the original.
    """
    # unique_dict = generate_unique_name(md_file)

    collection_name = osp.splitext(osp.basename(md_file))[0]
    collection = dict(
        Name=collection_name,
        Metadata={'Architecture': []},
        README=osp.relpath(md_file, MMGeneration_ROOT),
        Paper=[])
    models = []
    with open(md_file, 'r') as md:
        lines = md.readlines()
        i = 0
        while i < len(lines):
            # parse method name
            if i == 0:
                name = lines[i][2:].strip()
                collection['Metadata']['Architecture'].append(name)
                collection['Name'] = name
                collection_name = name
                i += 1
            # parse url from '> ['
            elif lines[i][:3] == '> [':
                url = re.findall(r'\(.*\)', lines[i])[0]
                collection['Paper'].append(url[1:-1])
                i += 1

            # parse table
            elif lines[i][0] == '|' and i + 1 < len(lines) and \
                    (lines[i + 1][:3] == '| :' or lines[i + 1][:2] == '|:'):
                cols = [col.strip() for col in lines[i].split('|')][1:-1]
                if 'Config' not in cols or 'Download' not in cols:
                    warnings.warn(f"Lack 'Config' or 'Download' in line {i+1}")
                    i += 1
                    continue
                # config_idx = cols.index('Model')
                config_idx = cols.index('Config')
                checkpoint_idx = cols.index('Download')
                try:
                    flops_idx = cols.index('FLOPs')
                except ValueError:
                    flops_idx = -1
                try:
                    params_idx = cols.index('Params')
                except ValueError:
                    params_idx = -1
                used_metrics = collate_metrics(cols)

                j = i + 2
                while j < len(lines) and lines[j][0] == '|':
                    line = lines[j].split('|')[1:-1]

                    if line[config_idx].find('](') >= 0:
                        left = line[config_idx].index('](') + 2
                        right = line[config_idx].index(')', left)
                        config = line[config_idx][left:right].strip('./')
                    elif line[config_idx].find('△') == -1:
                        j += 1
                        continue

                    if line[checkpoint_idx].find('](') >= 0:
                        if line[checkpoint_idx].find('model](') >= 0:
                            left = line[checkpoint_idx].index('model](') + 7
                        else:
                            left = line[checkpoint_idx].index('ckpt](') + 6
                        right = line[checkpoint_idx].index(')', left)
                        checkpoint = line[checkpoint_idx][left:right]
                    name_key = osp.splitext(osp.basename(config))[0]
                    model_name = name_key

                    # find dataset in config file
                    dataset = 'Others'
                    config_low = config.lower()
                    for d in all_training_data:
                        if d in config_low:
                            dataset = d.upper()
                            break
                    metadata = {'Training Data': dataset}
                    if flops_idx != -1:
                        metadata['FLOPs'] = float(line[flops_idx])
                    if params_idx != -1:
                        metadata['Parameters'] = float(line[params_idx])

                    metrics = {}

                    for key in used_metrics:
                        metrics_data = line[used_metrics[key]]
                        metrics_data = metrics_data.replace('*', '')
                        if '(' in metrics_data and ')' in metrics_data:
                            metrics_data = metrics_data.split('(')[0].strip()
                        try:
                            metrics[key] = float(metrics_data)
                        except ValueError:
                            metrics[key] = metrics_data.strip()

                    model = {
                        'Name':
                        model_name,
                        'In Collection':
                        collection_name,
                        'Config':
                        config,
                        'Metadata':
                        metadata,
                        'Results': [{
                            'Task': task,
                            'Dataset': dataset,
                            'Metrics': metrics
                        }],
                        'Weights':
                        checkpoint
                    }
                    models.append(model)
                    j += 1
                i = j

            else:
                i += 1

    if len(models) == 0:
        warnings.warn('no model is found in this md file')

    result = {'Collections': [collection], 'Models': models}
    yml_file = md_file.replace('README.md', 'metafile.yml')

    is_different = dump_yaml_and_check_difference(result, yml_file)
    return is_different


def update_model_index():
    """Update model-index.yml according to model .md files.

    Returns:
        Bool: If the updated model-index.yml is different from the original.
    """
    configs_dir = osp.join(MMGeneration_ROOT, 'configs')
    yml_files = glob.glob(osp.join(configs_dir, '**', '*.yml'), recursive=True)
    yml_files.sort()

    model_index = {
        'Import':
        [osp.relpath(yml_file, MMGeneration_ROOT) for yml_file in yml_files]
    }
    model_index_file = osp.join(MMGeneration_ROOT, 'model-index.yml')
    is_different = dump_yaml_and_check_difference(model_index,
                                                  model_index_file)

    return is_different


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        configs_root = osp.join(MMGeneration_ROOT, 'configs')
        file_list = glob.glob(
            osp.join(configs_root, '**', '*README.md'), recursive=True)
        file_list.sort()
    else:
        file_list = [
            fn for fn in sys.argv[1:] if osp.basename(fn) == 'README.md'
        ]

    if not file_list:
        sys.exit(0)

    # get task name of each method
    task_dict = get_task_dict(osp.join(MMGeneration_ROOT, 'README.md'))

    file_modified = False
    for fn in file_list:
        print(f'process {fn}')
        task = task_dict[fn]
        file_modified |= parse_md(fn, task)

    file_modified |= update_model_index()

    sys.exit(0)
