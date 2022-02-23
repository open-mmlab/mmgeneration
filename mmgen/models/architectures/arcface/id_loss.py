# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from torch import nn

from mmgen.models.builder import MODULES
from .model_irse import Backbone


@MODULES.register_module('ArcFace')
class IDLossModel(nn.Module):
    # ir se50 weight download link
    _ir_se50_url = 'https://gg0ltg.by.files.1drv.com/y4m3fNNszG03z9n8JQ7EhdtQKW8tQVQMFBisPVRgoXi_UfP8pKSSqv8RJNmHy2JampcPmEazo_Mx6NTFSqBpZmhPniROm9uNoghnzaavvYpxkCfiNmDH9YyIF3g-0nwt6bsjk2X80JDdL5z88OAblSDmB-kuQkWSWvA9BM3Xt8DHMCY8lO4HOQCZ5YWUtFyPAVwEyzTGDM-JRA5EJoN2bF1cg'  # noqa

    def __init__(self, ir_se50_weights=None, device='cuda'):
        super(IDLossModel, self).__init__()
        mmcv.print_log('Loading ResNet ArcFace', 'mmgen')
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        if ir_se50_weights is None:
            ir_se50_weights = self._ir_se50_url
        self.facenet.load_state_dict(
            torch.hub.load_state_dict_from_url(ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet = self.facenet.eval().to(device)

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, pred=None, gt=None):
        n_samples = gt.shape[0]
        y_feats = self.extract_feats(
            gt)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(pred)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count
