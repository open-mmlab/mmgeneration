# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only
from torchvision.utils import save_image


@HOOKS.register_module()
class VisualizeUnconditionalSamples(Hook):
    """Visualization hook for unconditional GANs.

    In this hook, we use the official api `save_image` in torchvision to save
    the visualization results.

    Args:
        output_dir (str): The file path to store visualizations.
        fixed_noise (bool, optional): Whether to use fixed noises in sampling.
            Defaults to True.
        num_samples (int, optional): The number of samples to show in
            visualization. Defaults to 16.
        interval (int): The interval of calling this hook. If set to -1,
            the visualization hook will not be called. Default: -1.
        filename_tmpl (str): Format string used to save images. The output file
            name will be formatted as this args. Default: 'iter_{}.png'.
        rerange (bool): Whether to rerange the output value from [-1, 1] to
            [0, 1]. We highly recommend users should preprocess the
            visualization results on their own. Here, we just provide a simple
            interface. Default: True.
        bgr2rgb (bool): Whether to reformat the channel dimension from BGR to
            RGB. The final image we will save is following RGB style.
            Default: True.
        nrow (int): The number of samples in a row. Default: 1.
        padding (int): The number of padding pixels between each samples.
            Default: 4.
        kwargs (dict | None, optional): Key-word arguments for sampling
            function. Defaults to None.
    """

    def __init__(self,
                 output_dir,
                 fixed_noise=True,
                 num_samples=16,
                 interval=-1,
                 filename_tmpl='iter_{}.png',
                 rerange=True,
                 bgr2rgb=True,
                 nrow=4,
                 padding=0,
                 kwargs=None):
        self.output_dir = output_dir
        self.fixed_noise = fixed_noise
        self.num_samples = num_samples
        self.interval = interval
        self.filename_tmpl = filename_tmpl
        self.bgr2rgb = bgr2rgb
        self.rerange = rerange
        self.nrow = nrow
        self.padding = padding

        # the sampling noise will be initialized by the first sampling.
        self.sampling_noise = None

        self.kwargs = kwargs if kwargs is not None else dict()

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return
        # eval mode
        runner.model.eval()
        # no grad in sampling
        with torch.no_grad():
            outputs_dict = runner.model(
                self.sampling_noise,
                return_loss=False,
                num_batches=self.num_samples,
                return_noise=True,
                **self.kwargs)
            imgs = outputs_dict['fake_img']
            noise_ = outputs_dict['noise_batch']
        # initialize samling noise with the first returned noise
        if self.sampling_noise is None and self.fixed_noise:
            self.sampling_noise = noise_

        # train mode
        runner.model.train()

        filename = self.filename_tmpl.format(runner.iter + 1)
        if self.rerange:
            imgs = ((imgs + 1) / 2)
        if self.bgr2rgb and imgs.size(1) == 3:
            imgs = imgs[:, [2, 1, 0], ...]
        if imgs.size(1) == 1:
            imgs = torch.cat([imgs, imgs, imgs], dim=1)
        imgs = imgs.clamp_(0, 1)

        mmcv.mkdir_or_exist(osp.join(runner.work_dir, self.output_dir))
        save_image(
            imgs,
            osp.join(runner.work_dir, self.output_dir, filename),
            nrow=self.nrow,
            padding=self.padding)
