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

    from gretel_synthetics.actgan.column_encodings import ColumnEncoding


class ColumnType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class ConditionalVectorType(str, Enum):
    SINGLE_DISCRETE = "single_discrete"
    ANYWAY = "anyway"
    # TODO: add ANYWAY_WITH_MASK type so the generator input includes a 0/1 mask
    # of which columns are being conditioned on.


@dataclass
class ColumnTransformInfo:
    column_name: str
    column_type: ColumnType
    transform: BaseTransformer
    encodings: List[ColumnEncoding]

    @property
    def output_dimensions(self) -> int:
        return sum(enc.encoded_dim for enc in self.encodings)


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
    loss_r: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
