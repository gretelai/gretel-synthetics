"""
Helpers for interacting with the SDV package.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from rdt.transformers import BaseTransformer
from rdt.transformers.datetime import UnixTimestampEncoder

from gretel_synthetics.detectors.dates import detect_datetimes

if TYPE_CHECKING:
    from gretel_synthetics.detectors.dates import DateTimeColumn

# Matches the expected dictionary schemas for SDV metadata
FieldTypesT = Dict[str, Dict[str, str]]
FieldTransformersT = Dict[str, Union[str, BaseTransformer]]


def datetime_column_to_sdv(
    dt_column: DateTimeColumn,
) -> Tuple[Dict[str, str], UnixTimestampEncoder]:
    """
    Given a ``DateTimeColumn`` object, extract and return SDV compatible field types and field transformer objects
    """
    types = {"type": "datetime", "format": dt_column.inferred_format}
    transformer = UnixTimestampEncoder(
        missing_value_replacement="mean",
        model_missing_values=True,
        datetime_format=dt_column.inferred_format,
    )
    return types, transformer


class EmptyFieldTransformer(BaseTransformer):
    """
    Transformer that handles columns that are all NaN.

    The SDV models (GC, CTGAN) cannot have any NaN values in it, however before
    the model is fit, SDV will transform the dataset using the RDT package
    and for numerical values (to include all NaN columns) all NaN values
    will be replaced with mean of the column - so for all NaN columns the
    mean is NaN, so we replace empty columns with 0 to prepare for the downstream
    model usage.

    Important:
        This transformer should only be run on a column that have all values set to NaN.
        If it runs on column that has some non-NaN values, they will be lost.
    """

    # If all values in a column are NaN, then the type of that column is numerical
    INPUT_SDTYPE = "numerical"
    OUTPUT_SDTYPES = {"value": "numerical"}

    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True

    replacement_value: float

    def __init__(self, replacement_value: float = 0):
        self.replacement_value = replacement_value

    def _fit(self, columns_data: pd.Series):
        pass

    def _transform(self, columns_data: pd.Series) -> pd.Series:
        return pd.Series(
            self.replacement_value, index=columns_data.index, dtype=columns_data.dtype
        )

    def _reverse_transform(self, columns_data: pd.Series) -> pd.Series:
        """
        Reverses transform on a given column, i.e. replaces all values with NaN.

        Important:
            It's possible that data in the column is not all equal to `replacement_value`,
            as the model could have generated values that are close to, but not exactly
            equal that value.
            That's why we replace the whole series with NaNs.

        Args:
            columns_data: Data to be reverse-transformed.

        Returns:
            New series with the same index, but all values replaced with NaN.
        """
        return pd.Series(np.nan, index=columns_data.index, dtype=columns_data.dtype)


class SDVTableMetadata:
    """
    This class serves as a helper that can dynamically update
    certain SDV `Table` metadata objects. Specifically, this class can
    be init'd with some optional `field_types` and `field_transformers` dicts.

    By default, we will save off the field names that *already exist* on the metadata
    and not overwrite them with new settings. If you want to allow these fields to be
    potentially overwritten, you can init this class with the `overwrite` kwarg to True.

    Once this class is init'd you can use the `fit_*` methods to analyze training data
    and potentially update the metadata automatically.

    At any point you can access the `field_types` and `field_transformers` attributes, which
    will have the learned field types and transformers on them.

    Args:
        field_types: An optional existing `field_types` dict that conforms to SDV's metadata schema.
            After calling a `fit_*` method, this field may be updated based on learnt conditions.
        field_transformers: An optional `field_transformers` dict that conforms to SDV's metadata schema.
            After calling a `fit_*` method, this field may be updated based on learnt conditions.
        overwrite: Defaults to `False` - but if set to `True` then any fields that already existed in
            the SDV metadata dicts may be overwritten by new learnings. If this remains `False` then
            no existing fields will be overwritten.
    """

    field_types: FieldTypesT
    field_transformers: FieldTransformersT
    _overwrite: bool

    # These sets will hold the original keys of both mappings, which we can use
    # when determining to replace an already existing setting from a user
    _field_type_keys: FrozenSet[str]
    _field_transformer_keys: FrozenSet[str]

    def __init__(
        self,
        field_types: Optional[FieldTypesT] = None,
        field_transformers: Optional[FieldTransformersT] = None,
        overwrite: bool = False,
    ):
        if field_types is not None:
            self.field_types = field_types.copy()
        else:
            self.field_types = {}

        if field_transformers is not None:
            self.field_transformers = field_transformers.copy()
        else:
            self.field_transformers = {}

        self._overwrite = overwrite

        # save off the original keys for both mappings
        self._field_type_keys = frozenset(self.field_types.keys())
        self._field_transformer_keys = frozenset(self.field_transformers.keys())

    def _set_field_type(self, field_name: str, data: dict) -> None:
        if field_name in self._field_type_keys and not self._overwrite:
            return

        self.field_types[field_name] = data

    def _set_field_transformer(self, field_name: str, data: str) -> None:
        if field_name in self._field_transformer_keys and not self._overwrite:
            return

        self.field_transformers[field_name] = data

    def fit_datetime(
        self,
        data: pd.DataFrame,
        sample_size: Optional[int] = None,
        with_suffix: bool = False,
        must_match_all: bool = False,
    ) -> None:
        detections = detect_datetimes(
            data,
            sample_size=sample_size,
            with_suffix=with_suffix,
            must_match_all=must_match_all,
        )
        for _, column_info in detections.columns.items():
            type_, transformer = datetime_column_to_sdv(column_info)
            self._set_field_type(column_info.name, type_)
            self._set_field_transformer(column_info.name, transformer)

    def fit_empty_columns(self, df: pd.DataFrame) -> None:
        for column in df.columns:
            if df[column].isna().all():
                df[column] = df[column].astype(np.float64)
                self._set_field_transformer(
                    column, EmptyFieldTransformer(replacement_value=0)
                )
