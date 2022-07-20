from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine import MessageHub
from mmengine.testing import assert_allclose
from mmengine.visualization import Visualizer
from torch.utils.data.dataset import Dataset

from mmgen.core import GenDataSample, GenVisualizationHook, PixelData
from mmgen.utils import register_all_modules

register_all_modules()

from mmgen.registry import MODELS  # isort:skip  # noqa


class TestGenVisualizationHook(TestCase):

    Visualizer.get_instance('mmgen')
    MessageHub.get_instance('mmgen')

    def test_init(self):
        hook = GenVisualizationHook(
            interval=10, vis_kwargs_list=dict(type='Noise'))
        self.assertEqual(hook.interval, 10)
        self.assertEqual(hook.vis_kwargs_list, [dict(type='Noise')])
        self.assertEqual(hook.n_samples, 64)
        self.assertFalse(hook.show)

        hook = GenVisualizationHook(
            interval=10,
            vis_kwargs_list=[dict(type='Noise'),
                             dict(type='Translation')])
        self.assertEqual(len(hook.vis_kwargs_list), 2)

        hook = GenVisualizationHook(
            interval=10, vis_kwargs_list=dict(type='GAN'), show=True)
        self.assertEqual(hook._visualizer._vis_backends, {})

    def test_vis_sample_with_gan_alias(self):
        gan_model_cfg = dict(
            type='DCGAN',
            noise_size=10,
            data_preprocessor=dict(type='GANDataPreprocessor'),
            generator=dict(
                type='DCGANGenerator', output_scale=32, base_channels=32))
        model = MODELS.build(gan_model_cfg)
        runner = MagicMock()
        runner.model = model

        hook = GenVisualizationHook(
            interval=10, vis_kwargs_list=dict(type='GAN'), n_samples=9)
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_sample=GenDataSample()) for idx in range(10)
        ]
        hook.vis_sample(runner, 0, data_batch, None)
        called_kwargs = mock_visualuzer.add_datasample.call_args.kwargs
        self.assertEqual(called_kwargs['name'], 'gan')
        self.assertEqual(called_kwargs['target_keys'], None)
        self.assertEqual(called_kwargs['vis_mode'], None)
        gen_batches = called_kwargs['gen_samples']
        self.assertEqual(len(gen_batches), 9)
        noise_in_gen_batches = torch.stack(
            [gen_batches[idx].noise for idx in range(9)], 0)
        noise_in_buffer = torch.cat(
            [buffer['noise'] for buffer in hook.inputs_buffer['GAN']],
            dim=0)[:9]
        self.assertTrue((noise_in_gen_batches == noise_in_buffer).all())

        hook.vis_sample(runner, 1, data_batch, None)
        called_kwargs = mock_visualuzer.add_datasample.call_args.kwargs
        gen_batches = called_kwargs['gen_samples']
        noise_in_gen_batches_new = torch.stack(
            [gen_batches[idx].noise for idx in range(9)], 0)
        self.assertTrue((noise_in_gen_batches_new == noise_in_buffer).all())

    def test_vis_sample_with_translation_alias(self):
        translation_cfg = dict(
            type='CycleGAN',
            data_preprocessor=dict(type='GANDataPreprocessor'),
            generator=dict(
                type='ResnetGenerator',
                in_channels=3,
                out_channels=3,
                base_channels=8,
                norm_cfg=dict(type='IN'),
                use_dropout=False,
                num_blocks=4,
                padding_mode='reflect',
                init_cfg=dict(type='normal', gain=0.02)),
            discriminator=dict(
                type='PatchDiscriminator',
                in_channels=3,
                base_channels=8,
                num_conv=3,
                norm_cfg=dict(type='IN'),
                init_cfg=dict(type='normal', gain=0.02)),
            default_domain='photo',
            reachable_domains=['photo', 'mask'],
            related_domains=['photo', 'mask'])
        model = MODELS.build(translation_cfg)

        class naive_dataset(Dataset):

            def __init__(self, max_len, train=False):
                self.max_len = max_len
                self.train = train

            def __len__(self):
                return self.max_len

            def __getitem__(self, index):
                weight = index if self.train else -index
                return dict(
                    inputs=dict(
                        img_photo=torch.ones(3, 32, 32) * weight,
                        img_mask=torch.ones(3, 32, 32) * (weight + 1)),
                    data_sample=GenDataSample())

        train_dataloader = MagicMock()
        train_dataloader.batch_size = 4
        train_dataloader.dataset = naive_dataset(max_len=15, train=True)
        val_dataloader = MagicMock()
        val_dataloader.batch_size = 4
        val_dataloader.dataset = naive_dataset(max_len=17)

        runner = MagicMock()
        runner.model = model
        runner.train_dataloader = train_dataloader
        runner.val_dataloader = val_dataloader

        hook = GenVisualizationHook(
            interval=10,
            vis_kwargs_list=[
                dict(type='Translation'),
                dict(type='TranslationVal', name='cyclegan_val')
            ],
            n_samples=9)
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_sample=GenDataSample()) for idx in range(4)
        ]
        hook.vis_sample(runner, 0, data_batch, None)
        called_kwargs_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_kwargs_list), 2)
        # trans_called_kwargs, trans_val_called_kwargs = called_kwargs_list
        trans_called_kwargs = called_kwargs_list[0].kwargs
        trans_val_called_kwargs = called_kwargs_list[1].kwargs
        self.assertEqual(trans_called_kwargs['name'], 'translation')
        self.assertEqual(trans_val_called_kwargs['name'], 'cyclegan_val')

        # test train gen samples
        trans_gen_sample = trans_called_kwargs['gen_samples']
        trans_gt_mask_list = [samp.gt_mask for samp in trans_gen_sample]
        trans_gt_photo_list = [samp.gt_photo for samp in trans_gen_sample]

        self.assertEqual(len(trans_gen_sample), 9)
        for idx, (mask, photo) in enumerate(
                zip(trans_gt_mask_list, trans_gt_photo_list)):
            sample_from_dataset = train_dataloader.dataset[idx]['inputs']
            assert_allclose(mask.data * 127.5 + 127.5,
                            sample_from_dataset['img_mask'])
            assert_allclose(photo.data * 127.5 + 127.5,
                            sample_from_dataset['img_photo'])

        # test val gen samples
        trans_gen_sample = trans_val_called_kwargs['gen_samples']
        trans_gt_mask_list = [samp.gt_mask for samp in trans_gen_sample]
        trans_gt_photo_list = [samp.gt_photo for samp in trans_gen_sample]

        self.assertEqual(len(trans_gen_sample), 9)
        for idx, (mask, photo) in enumerate(
                zip(trans_gt_mask_list, trans_gt_photo_list)):
            sample_from_dataset = val_dataloader.dataset[idx]['inputs']
            assert_allclose(mask.data * 127.5 + 127.5,
                            sample_from_dataset['img_mask'])
            assert_allclose(photo.data * 127.5 + 127.5,
                            sample_from_dataset['img_photo'])

        # check input buffer
        input_buffer = hook.inputs_buffer
        input_buffer['translation']

    def test_vis_ddpm_alias_with_user_defined_args(self):
        ddpm_cfg = dict(
            type='BasicGaussianDiffusion',
            num_timesteps=4,
            data_preprocessor=dict(type='GANDataPreprocessor'),
            betas_cfg=dict(type='cosine'),
            denoising=dict(
                type='DenoisingUnet',
                image_size=32,
                in_channels=3,
                base_channels=128,
                resblocks_per_downsample=3,
                attention_res=[16, 8],
                use_scale_shift_norm=True,
                dropout=0.3,
                num_heads=4,
                use_rescale_timesteps=True,
                output_cfg=dict(mean='eps', var='learned_range')),
            timestep_sampler=dict(type='UniformTimeStepSampler'))
        model = MODELS.build(ddpm_cfg)
        runner = MagicMock()
        runner.model = model

        hook = GenVisualizationHook(
            interval=10,
            n_samples=2,
            vis_kwargs_list=dict(
                type='DDPMDenoising', vis_mode='gif', name='ddpm',
                n_samples=3))
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_sample=GenDataSample()) for idx in range(10)
        ]
        hook.vis_sample(runner, 0, data_batch, None)
        called_kwargs = mock_visualuzer.add_datasample.call_args.kwargs
        gen_samples = called_kwargs['gen_samples']
        self.assertEqual(len(gen_samples), 3)
        self.assertEqual(called_kwargs['n_rows'], min(hook.n_rows, 3))

        # test user defined vis kwargs
        hook.vis_kwargs_list = [
            dict(
                type='Arguments',
                forward_mode='sampling',
                name='ddpm_sample',
                n_samples=2,
                n_rows=4,
                vis_mode='gif',
                n_skip=1,
                forward_kwargs=dict(
                    forward_mode='sampling',
                    sample_kwargs=dict(show_pbar=True, save_intermedia=True)))
        ]
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_sample=GenDataSample()) for idx in range(10)
        ]
        hook.vis_sample(runner, 0, data_batch, None)
        called_kwargs = mock_visualuzer.add_datasample.call_args.kwargs
        gen_samples = called_kwargs['gen_samples']
        self.assertEqual(len(gen_samples), 2)
        self.assertEqual(called_kwargs['n_rows'], min(hook.n_rows, 2))

    def test_after_val_iter(self):
        model = MagicMock()
        hook = GenVisualizationHook(
            interval=10, n_samples=2, vis_kwargs_list=dict(type='GAN'))
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        runner = MagicMock()
        runner.model = model

        hook.after_val_iter(runner, 0, [dict()], [GenDataSample()])
        mock_visualuzer.assert_not_called()

    def test_after_train_iter(self):
        gan_model_cfg = dict(
            type='DCGAN',
            noise_size=10,
            data_preprocessor=dict(type='GANDataPreprocessor'),
            generator=dict(
                type='DCGANGenerator', output_scale=32, base_channels=32))
        model = MODELS.build(gan_model_cfg)
        runner = MagicMock()
        runner.model = model

        hook = GenVisualizationHook(
            interval=2, vis_kwargs_list=dict(type='GAN'), n_samples=9)
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        # build a empty data sample
        data_batch = [
            dict(inputs=None, data_sample=GenDataSample()) for idx in range(10)
        ]
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        self.assertEqual(mock_visualuzer.add_datasample.call_count, 1)

        # test vis with messagehub info --> str
        mock_visualuzer.add_datasample.reset_mock()
        message_hub = MessageHub.get_current_instance()

        feat_map = torch.randn(4, 16, 4, 4)
        vis_results = dict(feat_map=feat_map)
        message_hub.update_info('vis_results', vis_results)

        hook.message_vis_kwargs = 'feat_map'
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 2)  # outputs + messageHub
        messageHub_vis_args = called_args_list[1].kwargs
        self.assertEqual(messageHub_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(messageHub_vis_args['gen_samples']), 4)
        self.assertEqual(messageHub_vis_args['vis_mode'], None)
        self.assertEqual(messageHub_vis_args['n_rows'], 4)

        # test vis with messagehub info --> list[str]
        mock_visualuzer.add_datasample.reset_mock()

        hook.message_vis_kwargs = ['feat_map']
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 2)  # outputs + messageHub
        messageHub_vis_args = called_args_list[1].kwargs
        self.assertEqual(messageHub_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(messageHub_vis_args['gen_samples']), 4)
        self.assertEqual(messageHub_vis_args['vis_mode'], None)
        self.assertEqual(messageHub_vis_args['n_rows'], 4)

        # test vis with messagehub info --> dict
        mock_visualuzer.add_datasample.reset_mock()

        hook.message_vis_kwargs = dict(key='feat_map', vis_mode='feature_map')
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 2)  # outputs + messageHub
        messageHub_vis_args = called_args_list[1].kwargs
        self.assertEqual(messageHub_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(messageHub_vis_args['gen_samples']), 4)
        self.assertEqual(messageHub_vis_args['vis_mode'], 'feature_map')
        self.assertEqual(messageHub_vis_args['n_rows'], 4)

        # test vis with messagehub info --> list[dict]
        mock_visualuzer.add_datasample.reset_mock()

        feat_map = torch.randn(4, 16, 4, 4)
        x_t = [GenDataSample(info='x_t')]
        vis_results = dict(feat_map=feat_map, x_t=x_t)
        message_hub.update_info('vis_results', vis_results)

        hook.message_vis_kwargs = [
            dict(key='feat_map', vis_mode='feature_map'),
            dict(key='x_t')
        ]
        for idx in range(3):
            hook.after_train_iter(runner, idx, data_batch, None)
        called_args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(len(called_args_list), 3)  # outputs + messageHub
        # output_vis_args = called_args_list[0].kwargs
        feat_map_vis_args = called_args_list[1].kwargs
        self.assertEqual(feat_map_vis_args['name'], 'train_feat_map')
        self.assertEqual(len(feat_map_vis_args['gen_samples']), 4)
        self.assertEqual(feat_map_vis_args['vis_mode'], 'feature_map')
        self.assertEqual(feat_map_vis_args['n_rows'], 4)

        x_t_vis_args = called_args_list[2].kwargs
        self.assertEqual(x_t_vis_args['name'], 'train_x_t')
        self.assertEqual(len(x_t_vis_args['gen_samples']), 1)
        self.assertEqual(x_t_vis_args['vis_mode'], None)
        self.assertEqual(x_t_vis_args['n_rows'], 1)

        # test vis messageHub info --> errors
        hook.message_vis_kwargs = 'error'
        with self.assertRaises(RuntimeError):
            hook.after_train_iter(runner, 1, data_batch, None)

        message_hub.runtime_info.clear()
        with self.assertRaises(RuntimeError):
            hook.after_train_iter(runner, 1, data_batch, None)

        hook.message_vis_kwargs = dict(key='feat_map', vis_mode='feature_map')
        message_hub.update_info('vis_results', dict(feat_map='feat_map'))
        with self.assertRaises(TypeError):
            hook.after_train_iter(runner, 1, data_batch, None)

    def test_after_test_iter(self):
        model = MagicMock()
        hook = GenVisualizationHook(
            interval=10,
            n_samples=2,
            test_vis_keys=['ema', 'orig', 'new_model.x_t', 'gt_img'],
            vis_kwargs_list=dict(type='GAN'))
        mock_visualuzer = MagicMock()
        mock_visualuzer.add_datasample = MagicMock()
        hook._visualizer = mock_visualuzer

        runner = MagicMock()
        runner.model = model

        gt_list = [torch.randn(3, 6, 6) for _ in range(4)]
        ema_list = [torch.randn(3, 6, 6) for _ in range(4)]
        orig_list = [torch.randn(3, 6, 6) for _ in range(4)]
        x_t_list = [torch.randn(3, 6, 6) for _ in range(4)]

        outputs = []
        for gt, ema, orig, x_t in zip(gt_list, ema_list, orig_list, x_t_list):
            gen_sample = GenDataSample(
                gt_img=PixelData(data=gt),
                ema=GenDataSample(fake_img=PixelData(data=ema)),
                orig=GenDataSample(fake_img=PixelData(data=orig)),
                new_model=GenDataSample(x_t=PixelData(data=x_t)))
            outputs.append(gen_sample)

        hook.after_test_iter(runner, 42, [], outputs)
        args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertEqual(
            len(args_list),
            len(hook.test_vis_keys_list) * len(gt_list))
        # check target consistency
        for idx, args in enumerate(args_list):
            called_kwargs = args.kwargs
            gen_samples = called_kwargs['gen_samples']
            name = called_kwargs['name']
            batch_idx = called_kwargs['batch_idx']
            target_keys = called_kwargs['target_keys']

            self.assertEqual(len(gen_samples), 1)
            idx_in_outputs = idx // 4
            self.assertEqual(batch_idx, idx_in_outputs + 42 * len(outputs))
            self.assertEqual(outputs[idx_in_outputs], gen_samples[0])

            # check ema
            if idx % 4 == 0:
                self.assertEqual(target_keys, 'ema')
                self.assertEqual(name, 'ema')
            # check orig
            elif idx % 4 == 1:
                self.assertEqual(target_keys, 'orig')
                self.assertEqual(name, 'orig')
            # check x_t
            elif idx % 4 == 2:
                self.assertEqual(target_keys, 'new_model.x_t')
                self.assertEqual(name, 'new_model_x_t')
            # check gt
            else:
                self.assertEqual(target_keys, 'gt_img')
                self.assertEqual(name, 'gt_img')

        # test get target key automatically
        hook.test_vis_keys_list = None
        mock_visualuzer.add_datasample.reset_mock()
        hook.after_test_iter(runner, 42, [], outputs)

        args_list = mock_visualuzer.add_datasample.call_args_list
        self.assertTrue(
            all([args.kwargs['target_keys'] for args in args_list]))

        # test get target key automatically with error
        outputs = [
            GenDataSample(
                ema=GenDataSample(
                    fake_img=PixelData(data=torch.randn(3, 6, 6))))
        ]
        with self.assertRaises(AssertionError):
            hook.after_test_iter(runner, 42, [], outputs)
