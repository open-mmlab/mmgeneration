import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmgen.datasets.pipelines import Compose
from mmgen.models import CycleGAN, Pix2Pix, build_model


def init_model(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed unconditional model.
    """

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = build_model(
        config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')

    model._cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def sample_uncoditional_model(model,
                              num_samples=16,
                              num_batches=4,
                              sample_model='ema',
                              **kwargs):
    """Sampling from unconditional models.

    Args:
        model (nn.Module): Unconditional models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.

    Returns:
        Tensor: Generated image tensor.
    """
    # set eval mode
    model.eval()
    # constrcut sampling list for batches
    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)
    res_list = []

    # inference
    for batches in batches_list:
        res = model.sample_from_noise(
            None, num_batches=batches, sample_model=sample_model, **kwargs)
        res_list.append(res.cpu())

    results = torch.cat(res_list, dim=0)

    return results


_supported_img2img_model = (Pix2Pix, CycleGAN)


def sample_img2img_model(model, image_path, **kwargs):
    """Sampling from translation models.

    Args:
        model (nn.Module): The loaded model.
        image_path (str): File path of input image.
        img_unpaired (str, optional): File path of the unpaired image.
            If not None, perform unpaired image generation. Default: None.
    Returns:
        Tensor: Translated image tensor.
    """
    assert isinstance(model, _supported_img2img_model)
    cfg = model._cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    if isinstance(model, Pix2Pix):
        data = dict(pair_path=image_path)
    else:
        data = dict(img_a_path=image_path, img_b_path=image_path)
    data = test_pipeline(data)
    if device.type == 'cpu':
        data = collate([data], samples_per_gpu=1)
        data['meta'] = []
    else:
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        results = model(test_mode=True, **data, **kwargs)
    if isinstance(model, Pix2Pix):
        output = results['fake_b']
    else:
        if model.test_direction == 'a2b':
            output = results['fake_b']
        else:
            output = results['fake_a']
    return output
