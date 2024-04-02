from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ColumnEncoding(ABC):
    """ColumnEncoding specifies how a logical value in a column is encoded.

    Note that this is different from the RDT transformer for a column: an RDT
    transformer operates on a column on the input data, whereas a ColumnEncoding
    applies to a column in the transformed data.

    Using a ``ClusterBasedNormalizer``, for example, a floating-point column in the
    input is transformed into two columns in the transformed data: a normalized
    value within a mode, and the index of the mode distribution that this value
    is drawn from. For the purpose of training and sampling, the latter mode index
    is then one-hot encoded to form the actual training data fed to the network.
    A ``ColumnEncoding`` is responsible for the latter transformation (encoding) step
    only.
    """

    def encode(self, decoded: np.ndarray) -> np.ndarray:
        """Encodes column values.

        Args:
            decoded: 1D array of column values.

        Returns:
            2D array of shape `(len(decoded), self.encoded_dim)`, with the i-th row
            representing an encoding of decoded[i].
        """
        if len(decoded.shape) != 1:
            raise ValueError("argument to encode must be a 1D array")
        return self._encode(decoded)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decodes column values.

        Args:
            encoded: 2D array with self.encoded_dim columns.

        Returns:
            1D array with `len(encoded)` elements, where the i-th element is the decoding
            of `decoded[i]`.
        """
        if len(encoded.shape) != 2 or encoded.shape[1] != self.encoded_dim:
            raise ValueError(
                f"argument to decode must be a 2D array with {self.encoded_dim} columns"
            )
        return self._decode(encoded)

    @abstractmethod
    def _encode(self, decoded: np.ndarray) -> np.ndarray:
        """Implementation of the encoding logic.

        Verification of the input shape has been taken care of.
        """
        ...

    @abstractmethod
    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        """Implementation of the encoding logic.

        Verification of the input shape has been taken care of.
        """
        ...

    @property
    @abstractmethod
    def encoded_dim(self) -> int: ...

    @property
    @abstractmethod
    def decoded_dtype(self) -> np.dtype: ...


class FloatColumnEncoding(ColumnEncoding):
    """No-op encoding for floating-point values.

    This 'encoding' merely transposes the input vector.
    """

    def _encode(self, decoded: np.ndarray) -> np.ndarray:
        return decoded.reshape(-1, 1)

    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        return encoded.reshape(-1)

    @property
    def encoded_dim(self) -> int:
        return 1

    @property
    def decoded_dtype(self) -> np.dtype:
        return np.float32


_DTYPES_WITH_BITS = [(np.uint8, 8), (np.uint16, 16), (np.uint32, 32), (np.uint64, 64)]


class OneHotColumnEncoding(ColumnEncoding):
    """One-hot encoding for integer values.

    Args:
        num_values: the number of distinct values that can be encoded.
    """

    _num_values: int
    _decoded_dtype: np.dtype

    def __init__(self, num_values: int):
        self._num_values = num_values
        selected_dtype = None
        for dtype, bits in _DTYPES_WITH_BITS:
            if num_values <= 2**bits:
                selected_dtype = dtype
                break
        if selected_dtype is None:
            raise ValueError(f"no numpy integer type can hold a value of {num_values}")
        self._decoded_dtype = selected_dtype

    def _encode(self, decoded: np.ndarray) -> np.ndarray:
        encoded = np.zeros((decoded.shape[0], self._num_values))
        # Ensure decoded array is integer-valued
        if not np.issubdtype(decoded.dtype, np.integer):
            decoded = decoded.astype(self.decoded_dtype)
        encoded[np.arange(decoded.shape[0]), decoded] = 1
        return encoded

    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        decoded = np.argmax(encoded, axis=1).astype(self.decoded_dtype)
        return decoded

    @property
    def decoded_dtype(self) -> np.dtype:
        return self._decoded_dtype

    @property
    def encoded_dim(self) -> int:
        return self._num_values

    @property
    def num_values(self) -> int:
        return self._num_values


class BinaryColumnEncoding(ColumnEncoding):
    """Binary encoding for column values.

    Args:
        num_bits: number of bits to use for representation of values.
    """

    _num_bits: int
    _decoded_dtype: np.dtype

    def __init__(self, num_bits: int):
        self._num_bits = num_bits
        selected_dtype = None
        for dtype, bits in _DTYPES_WITH_BITS:
            if num_bits <= bits:
                selected_dtype = dtype
                break
        if selected_dtype is None:
            raise ValueError(f"no numpy integer type can fit {num_bits} bits")
        self._decoded_dtype = selected_dtype

    def _encode(self, decoded: np.ndarray) -> np.ndarray:
        encoded = np.zeros((decoded.shape[0], self._num_bits))
        # Ensure decoded array is integer-valued
        if not np.issubdtype(decoded.dtype, np.integer):
            decoded = decoded.astype(self.decoded_dtype)
        encoded = (decoded[:, None] & (1 << np.arange(self._num_bits)[::-1])) != 0
        return encoded

    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        decoded = encoded.astype(self.decoded_dtype).dot(
            1 << np.arange(self._num_bits)[::-1]
        )
        return decoded

    @property
    def decoded_dtype(self) -> np.dtype:
        return self._decoded_dtype

    @property
    def encoded_dim(self) -> int:
        return self._num_bits
