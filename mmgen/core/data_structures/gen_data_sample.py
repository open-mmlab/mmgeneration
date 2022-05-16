# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.data import BaseDataElement


class GenDataSample(BaseDataElement):

    @property
    def gt_label(self):
        """Class label of an image."""
        return self.get_field('gt_label')

    @gt_label.setter
    def gt_label(self, **kwargs):
        self.set_field('gt_label', dtype=torch.Tensor)

    @gt_label.deleter
    def gt_label(self):
        self.del_field('gt_label')
