# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Sequence, Tuple, Union

from mmengine.structures import BaseDataElement
from torch import Tensor

ForwardInputs = Tuple[Dict[str, Union[Tensor, str, int]], Tensor]
ForwardOutputs = Union[Dict[str, Tensor], Tensor]
SampleList = Sequence[BaseDataElement]

DataSetOutputs = Sequence[Dict[str, Union[Tensor, BaseDataElement]]]
ValSamplerOutputs = Dict[str, Union[Tensor, str, int]]
PreprocessInputs = Union[DataSetOutputs, ValSamplerOutputs]
PreprocessOutputs = Tuple[ForwardInputs, list]

TrainStepInputs = DataSetOutputs
ValTestStepInputs = PreprocessInputs

NoiseVar = Union[Tensor, Callable, None]
LabelVar = Union[Tensor, Callable, List[int], None]
