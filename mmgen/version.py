# Copyright (c) OpenMMLab. All rights reserved.
__version__ = '0.7.2'


def parse_version_info(version_str):
    """Parse version information.

    Args:
        version_str (str): Version string.

    Returns:
        tuple: Version information in tuple.
    """
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)
