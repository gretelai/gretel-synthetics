import random

from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.transformers.datetime import UnixTimestampEncoder
from sdv import Table

from gretel_synthetics.detectors.dates import DateTimeColumn, DateTimeColumns
from gretel_synthetics.detectors.sdv import EmptyFieldTransformer, SDVTableMetadata


def _create_info() -> DateTimeColumns:
    return DateTimeColumns(columns={"footime": DateTimeColumn("footime", "%Y-%m-%d")})


def _create_meta() -> Tuple[dict, dict]:
    """
    Create some user specified metadata for tests

    Returns tuple of field_types, field_transformers
    """
    return {"footime": {"type": "FINDME"}}, {"footime": "FINDME"}


@pytest.mark.parametrize("overwrite", [True, False])
def test_does_update_metadata(overwrite: bool):
    """
    The cases here should set the metaata no matter what because both
    of the input metadata mappings are empty
    """
    meta = SDVTableMetadata(overwrite=overwrite)
    dt_info = _create_info()
    with patch("gretel_synthetics.detectors.sdv.detect_datetimes") as mock_detect:
        mock_detect.return_value = dt_info
        meta.fit_datetime(Mock())
        assert meta.field_types == {
            "footime": {"type": "datetime", "format": "%Y-%m-%d"}
        }
        assert isinstance(meta.field_transformers["footime"], UnixTimestampEncoder)
        assert meta.field_transformers["footime"].datetime_format == "%Y-%m-%d"


def test_does_not_update_metadata():
    """
    Fitting for datetime here will not result in any change
    as we are not allowing for overwriting of user provided values
    """
    curr_field_types, curr_field_transformers = _create_meta()
    meta = SDVTableMetadata(
        field_types=curr_field_types, field_transformers=curr_field_transformers
    )
    dt_info = _create_info()
    with patch("gretel_synthetics.detectors.sdv.detect_datetimes") as mock_detect:
        mock_detect.return_value = dt_info
        meta.fit_datetime(Mock())

        # The original metadata should remain un-changed
        assert meta.field_types == curr_field_types
        assert meta.field_transformers == curr_field_transformers


# Empty Field Tests


def test_transform_with_hypertransformer():
    df = _sample_data()[["empty"]]
    xf = HyperTransformer()
    xf.set_config(
        {
            "sdtypes": {"empty": "numerical"},
            "transformers": {"empty": EmptyFieldTransformer(replacement_value=33)},
        }
    )
    xf.fit(df)

    transformed = xf.transform(df)
    reversed = xf.reverse_transform(transformed)

    assert reversed.equals(df)


def test_transform_sdv_table():
    df = _sample_data()
    table = Table(
        name="test-data",
        field_transformers={"col": EmptyFieldTransformer(replacement_value=22)},
    )
    table.fit(df)

    transformed = table.transform(df)
    reversed = table.reverse_transform(transformed)

    assert reversed.equals(df)


def test_autodetects_empty_columns():
    df = _sample_data()
    detector = SDVTableMetadata()
    detector.fit_empty_columns(df)
    table = Table(name="test-data", field_transformers=detector.field_transformers)
    table.fit(df)

    transformed = table.transform(df)
    reversed = table.reverse_transform(transformed)

    assert reversed.equals(df)


def test_autodetects_no_empty_columns():
    df = _sample_data()[["col_2", "col_3"]]
    detector = SDVTableMetadata()
    detector.fit_empty_columns(df)
    assert detector.field_transformers == {}

    table = Table(name="test-data", field_transformers=detector.field_transformers)
    table.fit(df)

    transformed = table.transform(df)
    reversed = table.reverse_transform(transformed)

    assert reversed.equals(df)


def test_empty_column_transform_replaces_any_value():
    df = _sample_data()

    detector = SDVTableMetadata()
    detector.fit_empty_columns(df)
    table = Table(name="test-data", field_transformers=detector.field_transformers)
    table.fit(df)

    transformed = table.transform(df)

    # now we update transformed data, this mimics a behaviour when transformed data
    #  is used to train a model, but then that model generates small noise in column
    #  that was previously all-NaN
    transformed["empty.value"] = pd.Series(
        0.001, index=transformed["empty.value"].index
    )
    reversed = table.reverse_transform(transformed)

    assert reversed.equals(df)


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "empty": [np.nan] * 1000,
            "col_2": ["abc"] * 1000,
            "col_3": [random.choice([1.2, 0.5, 2, np.nan]) for _ in range(1000)],
        }
    )
