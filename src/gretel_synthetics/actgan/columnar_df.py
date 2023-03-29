from __future__ import annotations

import copy

from typing import Dict, List, Union

import numpy as np
import pandas as pd


class ColumnarDF:
    """A DataFrame-like type that stores columns individually.

    This class supports a subset of the operations supported on a Pandas
    DataFrame. In contrast to a DataFrame, however, it is optimized to
    allow efficient addition and removal (dropping) of columns.

    Because a DataFrame stores data of the same type in a single numpy array
    such that elements in the same row are adjacent, dropping a column requires
    reallocating the entire array. This class, in contrast, stores the Series
    of each column individually, making dropping a column a lightweight O(1)
    operation.
    """

    _col_data: Dict[str, pd.Series]
    """The data of the individual columns, as a dict mapping column name to values.

    We rely on the stable (insertion) order of Python dicts to avoid storing the
    ordered list of columns; this property is guaranteed for Python as of 3.7+
    (see https://mail.python.org/pipermail/python-dev/2017-December/151283.html).
    """

    def __init__(self, col_data: Union[List[pd.Series], Dict[str, pd.Series]]):
        if isinstance(col_data, list):
            col_data = {s.name: s for s in col_data}
        # Ensure all series have the default index
        if not all(_has_default_index(s) for s in col_data.values()):
            raise ValueError("series for columnar dataframe must have default index")
        # Shallow copy the series to ensure ownership
        self._col_data = {c: s.copy(deep=False) for c, s in col_data.items()}

    @staticmethod
    def from_df(df: pd.DataFrame) -> ColumnarDF:
        """Converts a Pandas DataFrame to a ColumnarDF.

        This method does not allocate any new data storage. The backing
        storage of each column is the same as in the original dataframe.

        Args:
            df: the DataFrame to convert.

        Returns:
            the columnar dataframe.
        """
        if not _has_default_index(df):
            raise ValueError("dataframe must have default index")
        cols = {c: s for c, s in df.items()}
        return ColumnarDF(cols)

    def copy(self, deep: bool = False) -> ColumnarDF:
        """Copies a ColumnarDF.

        Args:
            deep: only provided for compatibility. deep copies are not supported.

        Returns:
            a shallow copy of this ColumnarDF.
        """
        if deep:
            raise ValueError("explicit deep copies are not supported")
        return ColumnarDF(col_data=self._col_data)

    @property
    def columns(self) -> List[str]:
        return list(self._col_data.keys())

    @columns.setter
    def columns(self, new_names: List[str]):
        if len(self._col_data) != len(new_names):
            raise ValueError(
                f"expected {len(self._col_data)} new column names, got {len(new_names)}"
            )

        new_cols = {
            new_name: s.rename(new_name, inplace=True)
            for new_name, s in zip(new_names, self._col_data.values())
        }
        self._col_data = new_cols

    @property
    def empty(self) -> bool:
        return all(s.empty for s in self._col_data.values())

    def __getitem__(self, key) -> Union[pd.Series, ColumnarDF]:
        if not isinstance(key, list):
            # Single key -> lookup single column series
            return self._col_data[key]

        # List key -> projection
        invalid_col_names = [
            col_name for col_name in key if col_name not in self._col_data
        ]
        if invalid_col_names:
            raise KeyError(f"{invalid_col_names} not in index")
        col_data = {col_name: self._col_data[col_name] for col_name in key}
        return ColumnarDF(col_data=col_data)

    def __contains__(self, key: str) -> bool:
        return key in self._col_data

    def __setitem__(
        self,
        key: Union[str, List[str]],
        value: Union[pd.Series, np.ndarray, pd.DataFrame, ColumnarDF],
    ):
        if isinstance(key, list):
            if not isinstance(value, pd.DataFrame) and not isinstance(
                value, ColumnarDF
            ):
                raise ValueError(
                    "right-hand side must be a (columnar) dataframe when using a list key"
                )
            if len(key) != len(value.columns):
                raise ValueError(
                    "number of columns of right-hand side dataframe must equal key length"
                )
            if not _has_default_index(value):
                raise ValueError("right-hand side dataframe must have default index")
            for col_name, rhs_col in zip(key, value.columns):
                self._col_data[col_name] = value[rhs_col]
            return

        if isinstance(value, np.ndarray):
            # Implictly convert a NumPy array to a pd.Series.
            # Note that using a multi-dimensional array here is invalid,
            # but we let the pd.Series constructor catch this.
            value = pd.Series(name=key, data=value)
        if not _has_default_index(value):
            raise ValueError("right-hand side series must have default index")
        self._col_data[key] = value

    def drop(self, labels=None, *, axis=0, columns=None, inplace: bool = False):
        """Drops columns from this ColumnarDF.

        Dropping of rows is not supported.

        Args:
            labels:
                Can be used to specify the columns to drop. If this is used,
                axis=1 must be specified as well.
            axis:
                The axis along which to drop. Must be set to 1, unless columns
                is specified.
            columns:
                The columns to drop. If columns are specified via this parameter
                instead of labels, axis does not need to be specified.
            inplace:
                If True, this ColumnarDF will be modified. Otherwise, the returned
                ColumnarDF will be a new object sharing the data for the remaining
                columns.

        Returns:
            ColumnarDF with the given columns removed.
        """
        # We only support dropping columns. That can be accomplished by specifying
        # the columns as labels and axis=1, or by just specifying columns.
        if labels is not None:
            if columns is not None:
                raise ValueError("cannot specify both labels and columns")
            if axis != 1:
                raise ValueError("only axis 1 is supported")
            columns = labels
        if columns is None:
            raise ValueError("must specify either columns or labels with an axis of 1")

        if not isinstance(columns, list):
            columns = [columns]

        new_cols = copy.copy(self._col_data)
        for col in columns:
            del new_cols[col]

        if inplace:
            self._col_data = new_cols
            return self

        return ColumnarDF(new_cols)

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of this ColumnarDF.

        For simplicity, we assume that all series are of equal length,
        and the resulting shape is based on the shape of the first column
        series.

        Returns:
            tuple specifying the shape.
        """
        if not self._col_data:
            return (0, 0)

        sshape = next(s for s in self._col_data.values()).shape
        return (*sshape, len(self._col_data))

    @property
    def dtypes(self) -> pd.Series:
        """Get the data types for this ColumnarDF.

        Returns:
            a series specifying the data type for each column.
        """
        return pd.Series({c: s.dtype for c, s in self._col_data.items()})

    def __len__(self) -> int:
        if not self._col_data:
            return 0
        return len(next(s for s in self._col_data.values()))

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._col_data)

    def reindex(self, columns: List[str]) -> ColumnarDF:
        """Reindexes this ColumnarDF.

        Because this type doesn't support index operations, this method does not
        perform actual reindexing. It merely acts as a projection.

        Args:
            columns:
                The columns that should be present in the resulting data frame.

        Returns:
            A ColumnarDF with only the given columns.
        """
        return self[columns]

    @property
    def index(self) -> pd.Index:
        return pd.RangeIndex(len(self))

    @index.setter
    def set_index(self, new_index: pd.Index):
        if not new_index.equals(pd.RangeIndex(len(self))):
            raise ValueError("ColumnarDF only supports default index")


def _has_default_index(d: Union[pd.Series, pd.DataFrame]) -> bool:
    return d.index.equals(pd.RangeIndex(len(d)))
