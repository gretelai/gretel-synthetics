import numpy as np

from gretel_synthetics.actgan.column_encodings import (
    BinaryColumnEncoding,
    OneHotColumnEncoding,
)


def test_binary_column_encoding():
    decoded_data = np.array([0, 1, 7, 13], dtype=np.uint8)

    encoding = BinaryColumnEncoding(num_bits=4)
    encoded = encoding.encode(decoded_data).astype(np.float32)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 1],
        ]
    ).astype(np.float32)

    np.testing.assert_array_equal(encoded, expected)

    decoded = encoding.decode(encoded)
    np.testing.assert_array_equal(decoded_data, decoded)


def test_one_hot_column_encoding():
    decoded_data = np.array([0, 1, 6, 2], dtype=np.uint8)

    encoding = OneHotColumnEncoding(num_values=7)
    encoded = encoding.encode(decoded_data).astype(np.float32)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0],
        ],
    ).astype(np.float32)

    assert np.array_equal(encoded, expected)

    decoded = encoding.decode(encoded)
    assert np.array_equal(decoded_data, decoded)
