# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmgen into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmgen default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmgen`, and all registries will build modules from mmgen's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmgen.datasets  # noqa: F401,F403
    import mmgen.engine  # noqa: F401,F403
    import mmgen.evaluation  # noqa: F401,F403
    import mmgen.models  # noqa: F401,F403
    import mmgen.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmgen')
        if never_created:
            DefaultScope.get_instance('mmgen', scope_name='mmgen')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmgen':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmgen", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmgen". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmgen-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmgen')
