# Copyright (c) OpenMMLab. All rights reserved.
def infer_io_backend(data_root: str) -> str:
    """Infer the io backend from the given data_root.

    Args:
        data_root (str): The path of data root.

    Returns:
        str: The io backend.
    """
    if (data_root.upper().startswith('HTTP')
            or data_root.upper().startswith('HTTPS')):
        backend = 'http'
    elif data_root.upper().startswith('S3') or (
            len(data_root.split(':')) > 2
            and data_root.split(':')[1].upper() == 'S3'):
        # two case:
        # 1. s3://xxxxx (raw petrel path)
        # 2. CONFIG:s3://xxx  (petrel path with specific config)
        backend = 'petrel'
    else:
        # use default one
        backend = 'disk'
    return backend
