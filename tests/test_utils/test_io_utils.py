import os
from tempfile import TemporaryDirectory

from mmgen.utils import download_from_url


def test_download_from_file():
    img_url = 'https://user-images.githubusercontent.com/12726765/114528756-de55af80-9c7b-11eb-94d7-d3224ada1585.png'  # noqa
    with TemporaryDirectory() as temp_dir:
        local_file = download_from_url(url=img_url, dest_dir=temp_dir)
        assert os.path.exists(local_file)
