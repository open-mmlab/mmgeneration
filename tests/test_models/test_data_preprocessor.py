# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.testing import assert_allclose

from mmgen.models.gans import GANDataPreprocessor


class TestBaseDataPreprocessor(TestCase):

    def test_init(self):
        data_preprocessor = GANDataPreprocessor(
            bgr_to_rgb=True,
            mean=[0, 0, 0],
            std=[255, 255, 255],
            pad_size_divisor=16,
            pad_value=10)

        self.assertEqual(data_preprocessor._device.type, 'cpu')
        self.assertTrue(data_preprocessor.channel_conversion, True)
        assert_allclose(data_preprocessor.mean,
                        torch.tensor([0, 0, 0]).view(-1, 1, 1))
        assert_allclose(data_preprocessor.std,
                        torch.tensor([255, 255, 255]).view(-1, 1, 1))
        assert_allclose(data_preprocessor.pad_value, torch.tensor(10))
        self.assertEqual(data_preprocessor.pad_size_divisor, 16)

    def test_forward(self):
        data_preprocessor = GANDataPreprocessor()
        input1 = torch.randn(3, 3, 5)
        input2 = torch.randn(3, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)

        data = [
            dict(inputs=input1, data_sample=label1),
            dict(inputs=input2, data_sample=label2)
        ]

        batch_inputs, batch_labels = data_preprocessor(data)

        self.assertEqual(batch_inputs.shape, (2, 3, 3, 5))

        target_input1 = (input1.clone() - 127.5) / 127.5
        target_input2 = (input2.clone() - 127.5) / 127.5
        assert_allclose(target_input1, batch_inputs[0])
        assert_allclose(target_input2, batch_inputs[1])
        assert_allclose(label1, batch_labels[0])
        assert_allclose(label2, batch_labels[1])

        # if torch.cuda.is_available():
        #     base_data_preprocessor = base_data_preprocessor.cuda()
        #     batch_inputs, batch_labels = base_data_preprocessor(data)
        #     self.assertEqual(batch_inputs.device.type, 'cuda')

        #     base_data_preprocessor = base_data_preprocessor.cpu()
        #     batch_inputs, batch_labels = base_data_preprocessor(data)
        #     self.assertEqual(batch_inputs.device.type, 'cpu')

        #     base_data_preprocessor = base_data_preprocessor.to('cuda:0')
        #     batch_inputs, batch_labels = base_data_preprocessor(data)
        #     self.assertEqual(batch_inputs.device.type, 'cuda')

        imgA1 = torch.randn(3, 3, 5)
        imgA2 = torch.randn(3, 3, 5)
        imgB1 = torch.randn(3, 3, 5)
        imgB2 = torch.randn(3, 3, 5)
        label1 = torch.randn(1)
        label2 = torch.randn(1)
        data = [
            dict(inputs=dict(imgA=imgA1, imgB=imgB1), data_sample=label1),
            dict(inputs=dict(imgA=imgA2, imgB=imgB2), data_sample=label2)
        ]
        batch_inputs, batch_labels = data_preprocessor(data)
        self.assertEqual(list(batch_inputs.keys()), ['imgA', 'imgB'])

        img1 = torch.randn(3, 4, 4)
        img2 = torch.randn(3, 4, 4)
        noise1 = torch.randn(3, 4, 4)
        noise2 = torch.randn(3, 4, 4)
        data = [
            dict(
                inputs=dict(noise=noise1, img=img1, num_batches=2,
                            mode='ema')),
            dict(
                inputs=dict(noise=noise2, img=img2, num_batches=2, mode='ema'))
        ]
        data_preprocessor = GANDataPreprocessor(rgb_to_bgr=True)
        batch_inputs, batch_labels = data_preprocessor(data)
        target_input1 = (img1[[2, 1, 0], ...].clone() - 127.5) / 127.5
        target_input2 = (img2[[2, 1, 0], ...].clone() - 127.5) / 127.5
        self.assertEqual(
            list(batch_inputs.keys()), ['noise', 'img', 'num_batches', 'mode'])
        assert_allclose(batch_inputs['noise'][0], noise1)
        assert_allclose(batch_inputs['noise'][1], noise2)
        assert_allclose(batch_inputs['img'][0], target_input1)
        assert_allclose(batch_inputs['img'][1], target_input2)
        self.assertEqual(batch_inputs['num_batches'], 2)
        self.assertEqual(batch_inputs['mode'], 'ema')

        # test dict input
        sampler_results = dict(num_batches=2, mode='ema')
        batch_inputs, batch_labels = data_preprocessor(sampler_results)
        self.assertEqual(batch_inputs, sampler_results)
        self.assertEqual(batch_labels, [])
