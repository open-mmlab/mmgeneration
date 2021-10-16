# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler

from mmgen.apis import init_model


class MMGenUnconditionalHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_model(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data, *args, **kwargs):
        data_decode = dict()
        # `data` type is `list[dict]`
        for k, v in data[0].items():
            # decode strings
            if isinstance(v, bytearray):
                data_decode[k] = v.decode()
        return data_decode

    def inference(self, data, *args, **kwargs):
        sample_model = data['sample_model']
        print(sample_model)
        results = self.model.sample_from_noise(
            None, num_batches=1, sample_model=sample_model, **kwargs)
        return results

    def postprocess(self, data):
        # convert torch tensor to numpy and then convert to bytes
        output_list = []
        for data_ in data:
            data_ = (data_ + 1) / 2
            data_ = data_[[2, 1, 0], ...]
            data_ = data_.clamp_(0, 1)
            data_ = (data_ * 255).permute(1, 2, 0)
            data_np = data_.detach().cpu().numpy().astype(np.uint8)
            data_byte = data_np.tobytes()
            output_list.append(data_byte)

        return output_list
