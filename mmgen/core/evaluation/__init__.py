# Copyright (c) OpenMMLab. All rights reserved.
from .eval_hooks import GenerativeEvalHook, TranslationEvalHook
from .evaluation import (make_metrics_table, make_vanilla_dataloader,
                         offline_evaluation, online_evaluation)
# >>> new code
from .evaluator import GenEvaluator
from .metric_utils import slerp
from .metrics import (FrechetInceptionDistance, InceptionScore,
                      MultiScaleStructureSimilarity, SlicedWassersteinDistance)

# <<< new code

__all__ = [
    'offline_evaluation',
    'online_evaluation',
    'GenerativeEvalHook',
    'make_metrics_table',
    'make_vanilla_dataloader',
    'TranslationEvalHook',
    'slerp',
    # >>> new code
    'InceptionScore',
    'FrechetInceptionDistance',
    'MultiScaleStructureSimilarity',
    'SlicedWassersteinDistance',
    'GenEvaluator'
    # <<< new code
]
