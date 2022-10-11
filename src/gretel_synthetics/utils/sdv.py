"""
Helpers for interacting with the SDV package.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional, TYPE_CHECKING

from gretel_synthetics.detectors.dates import detect_datetimes

if TYPE_CHECKING:
    import pandas as pd

# Matches the expected dictionary schemas for SDV metadata
FieldTypesT = Dict[str, str]
FieldTransformersT = Dict[str, Dict[str, str]]


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
        self, data: pd.DataFrame, sample_size: Optional[int] = None
    ) -> None:
        detections = detect_datetimes(data, sample_size=sample_size)
        for _, column_info in detections.columns.items():
            self._set_field_type(column_info.name, column_info.to_sdv_field_type())
            self._set_field_transformer(
                column_info.name, column_info.to_sdv_transformer()
            )
