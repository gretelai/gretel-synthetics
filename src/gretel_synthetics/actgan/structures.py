"""
Complex datastructures for ACTGAN
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from rdt.transformers.base import BaseTransformer


class ActivationFn(str, Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    TANH = "tanh"


class ColumnType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class SpanInfo:
    dim: int
    activation_fn: ActivationFn


@dataclass
class ColumnTransformInfo:
    column_name: str
    column_type: ColumnType
    transform: BaseTransformer
    output_info: List[SpanInfo]
    output_dimensions: int


@dataclass
class ColumnIdInfo:
    discrete_column_id: int
    column_id: int
    value_id: np.ndarray


@dataclass
class EpochInfo:
    """
    When creating a model such as ACTGAN if the ``epoch_callback`` attribute is set to
    a callable, then after each epoch the provided callable will be called with
    an instance of this class as the only argument.
    """

    epoch: int
    loss_g: float
    loss_d: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
