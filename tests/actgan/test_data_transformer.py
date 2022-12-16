import numpy as np
import pandas as pd

from gretel_synthetics.actgan.data_transformer import BinaryEncodingTransformer


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
