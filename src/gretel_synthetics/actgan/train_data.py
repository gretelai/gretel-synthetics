from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from gretel_synthetics.actgan.structures import ColumnTransformInfo


class TrainData:
    """Memory-efficient representation of training data.

    TrainData stores the data to be fed to the network, using separate arrays for each
    column. Categorical data, both one-hot and binary encoded, is stored in integer-valued
    columns, resulting in substantial space savings over storing the encoded form.

    The encoded form of a set of rows can be retrieved via the method to_numpy_encoded.

    Args:
        column_infos:
            Metadata about all input columns.
        decoded_data:
            List of lists of NumPy arrays. This must be the same length as column_infos, and
            each inner list must be the same length as the encodings field in the respective
            column info. Each array stores the value of the transformed column in decoded form.
    """

    _num_rows: int
    _columns_data: List[Tuple[ColumnTransformInfo, List[np.ndarray]]]

    def __init__(
        self,
        column_infos: List[ColumnTransformInfo],
        decoded_data: List[List[np.ndarray]],
    ):
        if len(column_infos) != len(decoded_data):
            raise ValueError("column_infos must match decoded_data in length")
        self._columns_data = list(zip(column_infos, decoded_data))

        if any(
            len(col_info.encodings) != len(data_list)
            for col_info, data_list in self._columns_data
        ):
            raise ValueError(
                "data lists length must match number of encodings for each column"
            )

        self._num_rows = next(
            (len(data) for data_list in decoded_data for data in data_list), 0
        )
        if any(
            len(data.shape) != 1 or len(data) != self._num_rows
            for data_list in decoded_data
            for data in data_list
        ):
            raise ValueError(
                "all data lists must be 1D arrays with an equal number of elements"
            )

    def __len__(self) -> int:
        return self._num_rows

    @property
    def columns_and_data(self) -> List[Tuple[ColumnTransformInfo, List[np.ndarray]]]:
        return self._columns_data

    @property
    def column_infos(self) -> List[ColumnTransformInfo]:
        return [col_info for col_info, _ in self._columns_data]

    @property
    def encoded_dim(self) -> int:
        """Returns the dimension (number of columns) of the encoded data."""
        return sum(
            enc.encoded_dim
            for column_info, _ in self._columns_data
            for enc in column_info.encodings
        )

    def to_numpy_encoded(
        self,
        row_indices: Optional[Union[np.ndarray, List[int]]] = None,
        dtype=np.float32,
    ) -> np.ndarray:
        """Returns the encoded form of the data, or a subset thereof.

        Args:
            row_indices:
                If None, returns all rows. Otherwise, this can be a list of ints
                or an integer-typed NumPy array that contains the indices of the rows
                to return.
            dtype:
                The NumPy dtype of the resulting array.

        Returns:
            2D array of encoded row data.
        """

        row_sel = slice(None)
        if row_indices is not None:
            if not isinstance(row_indices, np.ndarray):
                row_indices = np.array(row_indices)
            row_sel = row_indices

        return np.concatenate(
            [
                enc.encode(data[row_sel])
                for col_info, data_list in self._columns_data
                for enc, data in zip(col_info.encodings, data_list)
            ],
            axis=1,
            dtype=dtype,
        )
