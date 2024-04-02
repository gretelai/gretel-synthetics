"""
General utilities for data munging and more!
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class UpsampledDataFrame:
    df: pd.DataFrame
    """The new upsampled DataFrame
    """

    original_size: int
    """The number of records that were originally
    in the DataFrame
    """

    upsample_count: int
    """The number of additional records that
    were added
    """


def upsample_df(df: pd.DataFrame, target_size: int) -> UpsampledDataFrame:
    """
    Given a DataFrame, ensure it has a minimum number of records in it.
    If the number of rows is less than ``target_size`` then the data will
    be repeated until reaching exactly ``target_size.``

    If the DataFrame already has more rows than ``target_size``, do nothing.

    Args:
        df: A Pandas DataFrame
        target_size: The target number of rows for the DataFrame

    Returns:
        An instance of ``UpsampledDataFrame``
    """

    original_size = len(df)

    # If we already have enough records, we don't
    # need to do anything so just return what we have
    if original_size >= target_size:
        return UpsampledDataFrame(df=df, original_size=original_size, upsample_count=0)

    # Find how many times we should repeat the dataset
    repeat_count = target_size // original_size
    new_df = df
    if repeat_count > 0:
        new_df = pd.concat([df] * repeat_count)

    final_repeat_count = target_size - len(new_df)

    if final_repeat_count > 0:
        new_df = pd.concat([new_df, new_df.sample(final_repeat_count)])

    new_df.reset_index(inplace=True, drop=True)

    return UpsampledDataFrame(
        df=new_df,
        original_size=original_size,
        upsample_count=len(new_df) - original_size,
    )
