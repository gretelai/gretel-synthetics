import numpy as np
import pandas as pd
import pytest

from gretel_synthetics.utils import data as data_utils


def _create_df(size: int) -> pd.DataFrame:
    return pd.DataFrame([{"foo": value} for value in np.random.randint(1, 10000, size)])


@pytest.mark.parametrize(
    "orig_size,target,count",
    [(1000, 800, 0), (1000, 1001, 1), (1000, 2000, 1000), (1000, 2042, 1042)],
)
def test_upsample_df(orig_size, target, count):
    """
    Test cases:
        - Do nothing
        - Only add singular rows
        - Only repeat the dataset evenly
        - Repeat and add singular rows
    """
    df = _create_df(orig_size)
    check = data_utils.upsample_df(df, target)
    assert len(check.df) == max(orig_size, target)
    assert check.original_size == orig_size
    assert check.upsample_count == count
