import numpy as np

from gretel_synthetics.actgan.column_encodings import (
    BinaryColumnEncoding,
    FloatColumnEncoding,
    OneHotColumnEncoding,
)
from gretel_synthetics.actgan.structures import ColumnTransformInfo, ColumnType
from gretel_synthetics.actgan.train_data import TrainData


def test_train_data_encoding():
    column_infos = [
        ColumnTransformInfo(
            column_name="discrete-binary",
            column_type=ColumnType.DISCRETE,
            transform=None,
            encodings=[BinaryColumnEncoding(num_bits=3)],
        ),
        ColumnTransformInfo(
            column_name="continuous",
            column_type=ColumnType.CONTINUOUS,
            transform=None,
            encodings=[
                FloatColumnEncoding(),
                OneHotColumnEncoding(10),
            ],
        ),
        ColumnTransformInfo(
            column_name="discrete-onehot",
            column_type=ColumnType.DISCRETE,
            transform=None,
            encodings=[OneHotColumnEncoding(6)],
        ),
    ]

    column_data_lists = [
        [np.array([3, 1, 4, 2])],  # binary-encoded discrete column
        [
            np.array([0.1, 0.4, 0.9, 0.6]),  # continuous mode value
            np.array([9, 0, 8, 2]),  # continuous mode index (one-hot)
        ],
        [np.array([5, 1, 4, 0])],  # one-hot encoded discrete column
    ]

    expected_encoded = np.array(
        [
            [0, 1, 1]  # 3
            + [0.1]  # 0.1
            + [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 9
            + [0, 0, 0, 0, 0, 1],  # 5
            [0, 0, 1]  # 1
            + [0.4]  # 0.4
            + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0
            + [0, 1, 0, 0, 0, 0],  # 1
            [1, 0, 0]  # 4
            + [0.9]  # 0.9
            + [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # 8
            + [0, 0, 0, 0, 1, 0],  # 4
            [0, 1, 0]  # 2
            + [0.6]  # 0.6
            + [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 2
            + [1, 0, 0, 0, 0, 0],  # 0
        ],
        dtype=np.float32,
    )

    td = TrainData(column_infos, column_data_lists)

    encoded = td.to_numpy_encoded()
    np.testing.assert_array_equal(encoded, expected_encoded)

    encoded_sel = td.to_numpy_encoded(row_indices=[3, 1, 0])
    np.testing.assert_array_equal(encoded_sel, expected_encoded[[3, 1, 0]])
