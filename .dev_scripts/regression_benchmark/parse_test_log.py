import argparse
import os

import mmcv
import yaml

# --------------------------------------------------------------------------- #
#
# Tips for adopting the test_benchmark.sh and parse_test_log.py:
#
# 1. The directory for pre-trained checkpoints should follow the structure
#    of ``./configs`` folder, especially the name of each method.
# 2. For the regression report, you can quickly check the ``regress_status``
#    a) ``Missing Eval Info``: Cannot read the information table from log file
#       correctly. Please further check the log file and testing output.
#    b) ``Failed``: The evaluation metric cannot match the value in our
#       metafile.
#    c) ``Pass``: Successfully pass the regression test.
# 3. We should check the representation of the result value in metafile and the
#    generated log table. (Related to convert_str_metric_value() function)
# 4. The matric name should be mapped to the standard one. Please check
#    ``metric_name_mapping`` to ensure the metric name in the keys of the dict.
#
# --------------------------------------------------------------------------- #

# used to store log infos: The key indicates the path of each log file, while
# the value contains meta information (method category, ckpt name) and the
# parsed log info
data_dict = {}

metric_name_mapping = {
    'FID50k': 'FID',
    'FID': 'FID',
    'P&R': 'PR',
    'P&R50k': 'PR',
    'PR': 'PR',
    'PR50k': 'PR'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('logdir', help='evaluation log path')
    parser.add_argument(
        '--out',
        type=str,
        default='./work_dirs/',
        help='output directory for benchmark information')

    args = parser.parse_args()
    return args


def read_metafile(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    return data


def read_table_from_log(filepath):
    """Read the evaluation table from the log file.

    .. note:: We assume the log file only records one evaluation procedure.
    """
    # table_info will contain 2 elements. The first element is the head of the
    # table indicating the meaning of each column. The second element is the
    # content of each column, e.g., training configuration, and fid value.
    table_info = []

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if 'mmgen' in line or 'INFO' in line:
                # running message
                continue
            if line[0] == '|' and '+' in line:
                # table split line
                continue
            if line[0] == '|':
                # useful content
                line = line.strip()
                line_split = line.split('|')
                line_split = [x.strip() for x in line_split]
                table_info.append(line_split)

    if len(table_info) < 2:
        print(f'Please check the log file: {filepath}. Cannot get the eval'
              ' information correctly.')
    elif len(table_info) > 3:
        print(f'Got more than 2 lines from the eval table in {filepath}')

    return table_info


def convert_str_metric_value(value):
    if isinstance(value, float):
        return value

    if isinstance(value, int):
        return float(value)

    # otherwise, we assume the value will be string format

    # Case: 3.42 (1.xxx/2.xxx) -- used in FID
    if '(' in value and ')' in value:
        split = value.split('(')
        res = split[0].strip()
        return float(res)

    # Case: 60.xx/40.xx -- used in PR metric
    elif '/' in value:
        split = value.split('/')

        return [float(x.strip()) for x in split]
    else:
        try:
            res = float(value)
            return res
        except Exception as err:
            print(f'Cannot convert str value {value} to float')
            print(f'Unexpected {err}, {type(err)}')
            raise err


def check_info_from_metafile(meta_info):
    """Check whether eval information matches the description from metafile.

    TODO: Set a dict containing the tolerance for different configurations.
    """
    method_cat = meta_info['method']
    ckpt_name = meta_info['ckpt_name']
    metafile_path = os.path.join('./configs', method_cat, 'metafile.yml')

    meta_data_orig = read_metafile(metafile_path)['Models']
    results_orig = None

    for info in meta_data_orig:
        if ckpt_name in info['Weights']:
            results_orig = info['Results']
            break

    if results_orig is None:
        print(f'Cannot find related models for {ckpt_name}')
        return False

    metric_value_orig = {}
    # get the original metric value
    for k, v in results_orig['Metrics']:
        if k in metric_name_mapping:
            metric_value_orig[
                metric_name_mapping[k]] = convert_str_metric_value(v)

    assert len(metric_value_orig
               ) > 0, f'Cannot get metric value in metafile for {ckpt_name}'

    # get the metric value from evaluation table
    eval_info = meta_info['eval_table']
    metric_value_eval = {}
    for i, name in enumerate(eval_info[0]):
        if name in metric_name_mapping:
            metric_name_mapping[
                metric_value_eval[name]] = convert_str_metric_value(
                    eval_info[1])
    assert len(metric_value_eval
               ) > 0, f'Cannot get metric value in eval table: {eval_info}'

    # compare eval info and the original info from metafile


def get_log_files(args):
    """Got all of the log files from the given args.logdir.

    This function is used to initialize ``data_dict``.
    """
    log_paths = mmcv.scandir(args.logdir, '.txt', recursive=True)
    log_paths = [os.path.join(args.logdir, x) for x in log_paths]

    # construct data dict
    for log in log_paths:
        splits = log.split('/')
        method = splits[-2]
        log_name = splits[-1]
        ckpt_name = log_name.replace('_eval_log.txt', '.pth')

        data_dict[log] = dict(
            method=method, log_name=log_name, ckpt_name=ckpt_name)

    print(f'Got {len(data_dict)} logs from {args.logdir}')

    return data_dict


def main():
    args = parse_args()
    get_log_files(args)

    # process log files
    for log_path, meta_info in data_dict.items():
        table_info = read_table_from_log(log_path)

        # only deal with valid table_info
        if len(table_info) == 2:
            meta_info['eval_table'] = table_info
            meta_info['regress_status'] = 'Pass' if check_info_from_metafile(
                meta_info) else 'Failed'
        else:
            meta_info['regress_status'] = 'Missing Eval Info'


if __name__ == '__main__':
    data = read_metafile('configs/styleganv2/metafile.yml')
