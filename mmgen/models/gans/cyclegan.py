from mmcv.parallel import MMDistributedDataParallel
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models.builder import MODELS
from ..common import GANImageBuffer, set_requires_grad
from .base_translation_model import BaseTranslationModel


@MODELS.register_module()
class CycleGAN(BaseTranslationModel):
    """CycleGAN model for unpaired image-to-image translation.

    Ref:
    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
    Networks

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator.
        gan_loss (dict): Config for the gan loss.
        cycle_loss (dict): Config for the cycle-consistency loss.
        id_loss (dict): Config for the identity loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generator
            update.
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
            `direction`: image-to-image translation direction (the model
            training direction): a2b | b2a.
            `buffer_size`: GAN image buffer size.
        test_cfg (dict): Config for testing. Default: None.
            You may change the testing of gan by setting:
            `direction`: image-to-image translation direction (the model
            training direction): a2b | b2a.
            `show_input`: whether to show input real images.
            `test_direction`: direction in the test mode (the model testing
            direction). CycleGAN has two generators. It decides whether
            to perform forward or backward translation with respect to
            `direction` during testing: a2b | b2a.
        pretrained (str): Path for pretrained model. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 default_style,
                 reachable_styles=[],
                 related_styles=[],
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(
            generator,
            discriminator,
            gan_loss,
            default_style,
            reachable_styles=reachable_styles,
            related_styles=related_styles,
            disc_auxiliary_loss=disc_auxiliary_loss,
            gen_auxiliary_loss=gen_auxiliary_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        # GAN image buffers
        self.image_buffers = dict()
        self.buffer_size = (50 if self.train_cfg is None else
                            self.train_cfg.get('buffer_size', 50))
        for style in self._reachable_styles:
            self.image_buffers[style] = GANImageBuffer(self.buffer_size)

        self.init_weights(pretrained)

        self.use_ema = False

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        for style in self._reachable_styles:
            self.generators[style].init_weights(pretrained=pretrained)
            self.discriminators[style].init_weights(pretrained=pretrained)

    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel`
        interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if isinstance(module, MMDistributedDataParallel):
            return module.module

        return module

    def _get_disc_loss(self, outputs):
        """Backward function for the discriminators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        discriminators = self.get_module(self.discriminators)

        log_vars_d = dict()
        loss_d = 0

        # GAN loss for discriminators['a']
        for style in self._reachable_styles:
            losses = dict()
            styled_image = self.image_buffers[style].query(
                outputs[f'style_{style}'])
            fake_pred = discriminators[style](styled_image.detach())
            losses[f'loss_gan_d_{style}_fake'] = self.gan_loss(
                fake_pred, target_is_real=False, is_disc=True)
            real_pred = discriminators[style](outputs[f'src_{style}'])
            losses[f'loss_gan_d_{style}_real'] = self.gan_loss(
                real_pred, target_is_real=True, is_disc=True)

            _loss_d, _log_vars_d = self._parse_losses(losses)
            _loss_d *= 0.5
            loss_d += _loss_d
            log_vars_d[f'loss_gan_d_{style}'] = _log_vars_d['loss'] * 0.5

        return loss_d, log_vars_d

    def _get_gen_loss(self, outputs):
        """Backward function for the generators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        generators = self.get_module(self.generators)
        discriminators = self.get_module(self.discriminators)

        losses = dict()
        for style in self._reachable_styles:
            # Identity reconstruction for generators
            outputs[f'identity_{style}'] = generators[style](
                outputs[f'src_{style}'])
            # GAN loss for generators
            fake_pred = discriminators[style](outputs[f'style_{style}'])
            losses[f'loss_gan_g_{style}'] = self.gan_loss(
                fake_pred, target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs)
                if loss_ is None:
                    continue

                # mmcv.print_log(f'get loss for {loss_module.name()}')
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses:
                    losses[loss_module.loss_name(
                    )] = losses[loss_module.loss_name()] + loss_
                else:
                    losses[loss_module.loss_name()] = loss_

        loss_g, log_vars_g = self._parse_losses(losses)

        return loss_g, log_vars_g

    def _get_opposite_style(self, style):
        for item in self._reachable_styles:
            if item != style:
                return item
        return None

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generators and discriminators.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # forward generators
        outputs = dict()
        for style in self._reachable_styles:
            # fetch data by style
            another_style = self._get_opposite_style(style)
            img = data_batch[f'img_{another_style}']
            # stylized process
            results = self.forward(img, test_mode=False, style=style)
            outputs[f'src_{another_style}'] = results['orig_img']
            outputs[f'style_{style}'] = results['styled_img']
            # cycle process
            results = self.forward(
                results['styled_img'], test_mode=False, style=another_style)
            outputs[f'cycle_{another_style}'] = results['styled_img']

        log_vars = dict()

        # discriminators
        set_requires_grad(self.discriminators, True)
        # optimize
        optimizer['discriminators'].zero_grad()
        loss_d, log_vars_d = self._get_disc_loss(outputs)
        log_vars.update(log_vars_d)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_d))
        loss_d.backward()
        optimizer['discriminators'].step()

        # generators, no updates to discriminator parameters.
        if (curr_iter % self.disc_steps == 0
                and curr_iter >= self.disc_init_steps):
            set_requires_grad(self.discriminators, False)
            # optimize
            optimizer['generators'].zero_grad()
            loss_g, log_vars_g = self._get_gen_loss(outputs)
            log_vars.update(log_vars_g)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_g))
            loss_g.backward()
            optimizer['generators'].step()

        self.iteration += 1

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        image_results = dict()
        for style in self._reachable_styles:
            image_results[f'src_{style}'] = outputs[f'src_{style}'].cpu()
            image_results[f'style_{style}'] = outputs[f'style_{style}'].cpu()
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs[f'src_{style}']),
            results=image_results)

        return results
