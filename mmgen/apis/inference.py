# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import torch
from mmengine import is_list_of
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint

from mmgen.models import BaseTranslationModel
from mmgen.registry import MODELS
from mmgen.utils import register_all_modules

register_all_modules()


def set_random_seed(seed, deterministic=False, use_rank_shift=True):
    """Set random seed.

    In this function, we just modify the default behavior of the similar
    function defined in MMCV.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: True.
    """
    # TODO: refactor this  when refactor dir
    # set_random_seed_mmcv(
    #     seed, deterministic=deterministic, use_rank_shift=use_rank_shift)
    return


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
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model = MODELS.build(config.model)

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')

    model._cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def sample_unconditional_model(model,
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
    # construct sampling list for batches
    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)
    res_list = []

    # inference
    for batches in batches_list:
        res = model(
            dict(num_batches=batches, sample_model=sample_model), **kwargs)
        res_list.extend([item.fake_img.data.cpu() for item in res])

    results = torch.stack(res_list, dim=0)
    return results


@torch.no_grad()
def sample_conditional_model(model,
                             num_samples=16,
                             num_batches=4,
                             sample_model='ema',
                             label=None,
                             **kwargs):
    """Sampling from conditional models.

    Args:
        model (nn.Module): Conditional models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.
        label (int | torch.Tensor | list[int], optional): Labels used to
            generate images. Default to None.,

    Returns:
        Tensor: Generated image tensor.
    """
    # set eval mode
    model.eval()
    # construct sampling list for batches
    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    # check and convert the input labels
    if isinstance(label, int):
        label = torch.LongTensor([label] * num_samples)
    elif isinstance(label, torch.Tensor):
        label = label.type(torch.int64)
        if label.numel() == 1:
            # repeat single tensor
            # call view(-1) to avoid nested tensor like [[[1]]]
            label = label.view(-1).repeat(num_samples)
        else:
            # flatten multi tensors
            label = label.view(-1)
    elif isinstance(label, list):
        if is_list_of(label, int):
            label = torch.LongTensor(label)
            # `nargs='+'` parse single integer as list
            if label.numel() == 1:
                # repeat single tensor
                label = label.repeat(num_samples)
        else:
            raise TypeError('Only support `int` for label list elements, '
                            f'but receive {type(label[0])}')
    elif label is None:
        pass
    else:
        raise TypeError('Only support `int`, `torch.Tensor`, `list[int]` or '
                        f'None as label, but receive {type(label)}.')

    # check the length of the (converted) label
    if label is not None and label.size(0) != num_samples:
        raise ValueError('Number of elements in the label list should be ONE '
                         'or the length of `num_samples`. Requires '
                         f'{num_samples}, but receive {label.size(0)}.')

    # make label list
    label_list = []
    for n in range(n_repeat):
        if label is None:
            label_list.append(None)
        else:
            label_list.append(label[n * num_batches:(n + 1) * num_batches])

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)
        if label is None:
            label_list.append(None)
        else:
            label_list.append(label[(n + 1) * num_batches:])

    res_list = []

    # inference
    for batches, labels in zip(batches_list, label_list):
        res = model(
            dict(
                num_batches=batches, labels=labels, sample_model=sample_model),
            **kwargs)
        res_list.extend([item.fake_img.data.cpu() for item in res])
    results = torch.stack(res_list, dim=0)
    return results


def sample_img2img_model(model, image_path, target_domain=None, **kwargs):
    """Sampling from translation models.

    Args:
        model (nn.Module): The loaded model.
        image_path (str): File path of input image.
        style (str): Target style of output image.
    Returns:
        Tensor: Translated image tensor.
    """
    assert isinstance(model, BaseTranslationModel)

    # get source domain and target domain
    if target_domain is None:
        target_domain = model._default_domain
    source_domain = model.get_other_domains(target_domain)[0]

    cfg = model._cfg
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)

    # prepare data
    data = dict()
    # dirty code to deal with test data pipeline
    data['pair_path'] = image_path
    data[f'img_{source_domain}_path'] = image_path
    data[f'img_{target_domain}_path'] = image_path

    data = [test_pipeline(data)]

    data = model.data_preprocessor(data, False)
    inputs_dict = data['inputs']

    source_image = inputs_dict[f'img_{source_domain}']

    # forward the model
    with torch.no_grad():
        results = model(
            source_image,
            test_mode=True,
            target_domain=target_domain,
            **kwargs)
    output = results['target']
    return output


@torch.no_grad()
def sample_ddpm_model(model,
                      num_samples=16,
                      num_batches=4,
                      sample_model='ema',
                      same_noise=False,
                      **kwargs):
    """Sampling from ddpm models.

    Args:
        model (nn.Module): DDPM models in MMGeneration.
        num_samples (int, optional): The total number of samples.
            Defaults to 16.
        num_batches (int, optional): The number of batch size for inference.
            Defaults to 4.
        sample_model (str, optional): Which model you want to use. ['ema',
            'orig']. Defaults to 'ema'.
        noise_batch (torch.Tensor): Noise batch used as denoising starting up.
            Defaults to None.

    Returns:
        list[Tensor | dict]: Generated image tensor.
    """
    model.eval()

    n_repeat = num_samples // num_batches
    batches_list = [num_batches] * n_repeat

    if num_samples % num_batches > 0:
        batches_list.append(num_samples % num_batches)

    noise_batch = torch.randn(model.image_shape) if same_noise else None

    res_list = []
    # inference
    for idx, batches in enumerate(batches_list):
        mmengine.print_log(f'Start to sample batch [{idx+1} / '
                           f'{len(batches_list)}]')
        noise_batch_ = noise_batch[None, ...].expand(batches, -1, -1, -1) \
            if same_noise else None

        batch_input = dict(
            noise=noise_batch_,
            num_batches=batches,
            sample_model=sample_model,
            show_pbar=True,
            **kwargs)
        res = model(batch_input)
        for idx in range(len(res)):
            if res[idx].sample_model == 'ema/orig':
                res_ = {
                    'ema': res[idx].ema.fake_img.data.cpu(),
                    'orig': res[idx].orig.fake_img.data.cpu()
                }
            else:
                res_ = res[idx].fake_img.data.cpu()
            res_list.append(res_)

    # gather the res_list
    if isinstance(res_list[0], dict):
        res_dict = dict()
        for k in res_list[0].keys():
            res_dict[k] = [res[k] for res in res_list]
        return res_dict
    return res_list
