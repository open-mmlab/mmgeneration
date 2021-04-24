try:
    from apex import amp
except ImportError:
    amp = None


def apex_amp_initialize(models, optimizers, init_args=None, mode='gan'):
    init_args = init_args or dict()

    if mode == 'gan':
        _optmizers = [optimizers['generator'], optimizers['discriminator']]

        models, _optmizers = amp.initialize(models, _optmizers, **init_args)
        optimizers['generator'] = _optmizers[0]
        optimizers['discriminator'] = _optmizers[1]

        return models, optimizers

    else:
        raise NotImplementedError(
            f'Cannot initialize apex.amp with mode {mode}')
