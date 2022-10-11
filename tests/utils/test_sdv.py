from typing import Tuple
from unittest.mock import Mock, patch

import pytest

from gretel_synthetics.detectors.dates import DateTimeColumn, DateTimeColumns
from gretel_synthetics.utils.sdv import SDVTableMetadata


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
    with patch("gretel_synthetics.utils.sdv.detect_datetimes") as mock_detect:
        mock_detect.return_value = dt_info
        meta.fit_datetime(Mock())
        assert meta.field_types == {
            "footime": {"type": "datetime", "format": "%Y-%m-%d"}
        }
        assert meta.field_transformers == {"footime": "UnixTimestampEncoder"}


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
    with patch("gretel_synthetics.utils.sdv.detect_datetimes") as mock_detect:
        mock_detect.return_value = dt_info
        meta.fit_datetime(Mock())

        # The original metadata should remain un-changed
        assert meta.field_types == curr_field_types
        assert meta.field_transformers == curr_field_transformers
