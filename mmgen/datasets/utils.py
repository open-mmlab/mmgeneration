# Copyright (c) OpenMMLab. All rights reserved.

import os
import hashlib
import errno
from torch.utils.model_zoo import  import tqdm


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


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    # Python2 support for os.makedirs(.., exist_ok=True)
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and palce it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    #downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(
                url,
                fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == "https":
                print("Failed download. Trying https -> http instead."
                      " Downloading " + url + " to " + fpath)
                urllib.request.urlretrieve(
                    url,
                    fpath,
                    reporthook=gen_bar_updater()
                )


def list_dir(root, prefix=False):
    """List all directories at a given root
        Args:
            root (str): Path to directory whose folders need to be listed
            prefix (bool, optional): If true, prepends the path to each result, otherwise
                only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def download_file_from_google_drive(file_id, root, filename=None, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """