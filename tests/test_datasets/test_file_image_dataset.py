import os.path as osp

from mmgen.datasets import FileDataset


class TestFileDataset(object):

    @classmethod
    def setup_class(cls):
        cls.file_path = osp.join(
            osp.dirname(__file__), '..', 'data/file/test.npz')
        cls.default_pipeline = [
            dict(type='Resize', scale=(32, 32), keys=['fake_img']),
            dict(type='ToTensor', keys=['label']),
            dict(type='ImageToTensor', keys=['fake_img']),
            dict(type='Collect', keys=['fake_img', 'label'])
        ]

    def test_unconditional_imgs_dataset(self):
        dataset = FileDataset(self.file_path, pipeline=self.default_pipeline)

        assert len(dataset) == 2
        data_dict = dataset[0]
        img = data_dict['fake_img']
        lab = data_dict['label']
        assert img.shape == (3, 32, 32)
        assert lab == 1
        print(repr(dataset))
        assert repr(dataset) == (
            f'dataset_name: {dataset.__class__}, '
            f'total {2} images in file_path: {self.file_path}')
