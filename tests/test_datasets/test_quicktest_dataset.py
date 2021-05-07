from mmgen.datasets.quick_test_dataset import QuickTestImageDataset


class TestQuickTest:

    @classmethod
    def setup_class(cls):
        cls.dataset = QuickTestImageDataset(size=(256, 256))

    def test_quicktest_dataset(self):
        assert len(self.dataset) == 10000
        img = self.dataset[2]
        assert img['real_img'].shape == (3, 256, 256)
