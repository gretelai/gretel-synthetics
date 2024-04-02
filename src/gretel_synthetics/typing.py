"""
Unified types for Gretel Synthetics
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
import pandas as pd

DFLike = Union[pd.DataFrame, np.ndarray, List[List[Any]]]
SeriesOrDFLike = Union[pd.Series, DFLike]
ListOrSeriesOrDF = Union[List[Any], List[List[Any]], SeriesOrDFLike]
