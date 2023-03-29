import random

import numpy as np
import pandas as pd

from gretel_synthetics.actgan.data_transformer import (
    BinaryEncodingTransformer,
    DataTransformer,
)


def test_basic_enc_dec():
    df = pd.DataFrame(data=["A", "B", "C", "D"], columns=["foo"])
    encoder = BinaryEncodingTransformer()
    transformed = encoder.fit_transform(df, "foo")
    columns = transformed.columns
    df = pd.DataFrame(data=[[0, 0, 1]], columns=columns)
    check = encoder.reverse_transform(df)
    assert check["foo"].equals(pd.Series(["A"]))

    # Simulate a "near miss" for the reverse transform
    df = pd.DataFrame(data=[[1, 1, 0]], columns=columns)
    check = encoder.reverse_transform(df)
    assert check.dropna().empty


def test_fill_na_from_list():
    df = pd.DataFrame(
        data=["A", "B", np.nan, "C", "D", np.nan, np.nan], columns=["foo"]
    )
    series = df["foo"]
    options = frozenset(["Y", "Z"])
    check = BinaryEncodingTransformer.fill_na_from_list(series, list(options))
    filtered = check.loc[lambda x: (x == "Y") | (x == "Z")]
    assert len(filtered) == 3


def test_enc_dec_with_mode():
    df = pd.DataFrame(data=["A", "A", "B", "C", "D"], columns=["foo"])
    encoder = BinaryEncodingTransformer(handle_rounding_nan="mode")
    transformed = encoder.fit_transform(df, "foo")
    assert encoder._mode_values == frozenset(["A"])
    assert encoder.handle_rounding_nan == "mode"
    columns = transformed.columns
    df = pd.DataFrame(data=[[0, 0, 1]], columns=columns)
    check = encoder.reverse_transform(df)
    assert check["foo"].equals(pd.Series(["A"]))

    # Simulate a "near miss" for the reverse transform, this NaN should
    # get replaced with the mode value, which is "A"
    df = pd.DataFrame(data=[[1, 1, 0]], columns=columns)
    check = encoder.reverse_transform(df)
    assert list(check["foo"])[0] == "A"


def test_encoder_with_regex_metachars():
    """
    FIXME(PROD-309): This is a test covering this `category_encoders` issue:
    https://github.com/scikit-learn-contrib/category_encoders/issues/392

    We need our own fix for now, which can be removed once we migrate to
    version with the fix upstream.
    """
    col_name = "column*+{} (keep it secret!) [ab12-x]"

    df = pd.DataFrame(data={col_name: ["A", "A", "B", "C", "D"]})
    encoder = BinaryEncodingTransformer()
    transformed = encoder.fit_transform(df, col_name)

    transformed_sample = transformed.head(1)
    check = encoder.reverse_transform(transformed_sample)

    assert check[col_name].equals(pd.Series(["A"]))


def test_transform_large():
    df = pd.DataFrame(
        data=[random.choice(["A", "B", "C"]) for _ in range(700)], columns=["foo"]
    )
    transformer = DataTransformer(binary_encoder_cutoff=2)
    transformer.fit(df, discrete_columns=["foo"])

    transformed = transformer.transform(df)

    result_df = transformer.inverse_transform(transformed)

    pd.testing.assert_frame_equal(df, result_df)
