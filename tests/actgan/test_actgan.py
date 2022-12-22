import datetime

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from gretel_synthetics.actgan import ACTGAN
from gretel_synthetics.actgan.data_transformer import BinaryEncodingTransformer
from pandas.api.types import is_number


@pytest.fixture
def test_df() -> pd.DataFrame:
    record_count = 10
    datetimes = [datetime.datetime.utcnow().isoformat() for _ in range(record_count)]
    values = list(range(record_count))
    df = pd.DataFrame(zip(datetimes, values), columns=["timestamp", "value"])
    return df


def test_auto_transform_datetimes(test_df):
    """
    Validate that the auto transform detection converts datetime
    columns to integers prior to the model being fit.
    """

    model = ACTGAN(auto_transform_datetimes=True)
    model._fit = Mock()
    model.fit(test_df)

    _, args, _ = model._fit.mock_calls[0]
    transformed_df = args[0]
    assert is_number(transformed_df[transformed_df.columns[0]][0])


def test_binary_encoder_cutoff(test_df):
    with patch("gretel_synthetics.actgan.actgan.ACTGANSynthesizer._actual_fit"):
        model = ACTGAN(binary_encoder_cutoff=5)
        model.fit(test_df)

        encoder = model._model._transformer._column_transform_info_list[0].transform
        assert isinstance(encoder, BinaryEncodingTransformer)
