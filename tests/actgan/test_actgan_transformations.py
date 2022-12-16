import numpy as np
import pandas as pd
import pytest

from gretel_synthetics.actgan.data_transformer import BinaryEncodingTransformer


def test_binary_encoding_transformer_init():
    bet = BinaryEncodingTransformer()

    assert bet.INPUT_SDTYPE == "categorical"


def test_binary_encoding_transformer_prep_data():
    bet = BinaryEncodingTransformer()
    data = list([1, 2, 3])
    data = bet._prepare_data(data)
    assert type(data) == np.ndarray

    data = np.array([[[1, 2, 3]]])
    with pytest.raises(ValueError):
        data = bet._prepare_data(data)

    data = np.random.random((2, 2))
    with pytest.raises(ValueError):
        data = bet._prepare_data(data)

    data = np.random.random((4, 1))
    data = bet._prepare_data(data)
    assert len(data.shape) == 1


def test_binary_encoding_transformer_fit():
    bet = BinaryEncodingTransformer()
    df = pd.DataFrame(np.array([[0, 1], [0, 3], [1, 1], [2, 3]]), columns=["a", "b"])
    bet.fit(df, "a")
    assert bet.encoder.get_feature_names() == ["a"]
    bet.fit(df, "b")
    assert bet.encoder.get_feature_names() == ["b"]


def test_binary_encoding_transformer_transform():
    bet = BinaryEncodingTransformer()
    df = pd.DataFrame(
        np.array([["m", "f"], ["m", "k"], ["f", "m"], ["g", "k"]]), columns=["a", "b"]
    )
    bet.fit(df, "a")
    transformed = bet.transform(df)
    assert np.allclose(
        transformed.values[:, 1:4].astype(int),
        [[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
    )


def test_binary_encoding_transformer_reverse_transform():
    bet = BinaryEncodingTransformer()
    df = pd.DataFrame(
        np.array([["m", "f"], ["m", "k"], ["f", "m"], ["g", "k"]]), columns=["a", "b"]
    )
    bet.fit(df, "a")
    transformed = bet.transform(df)
    data = bet.reverse_transform(transformed)
    assert ((df.sort_index(axis=1) == data.sort_index(axis=1)).all()).all()
