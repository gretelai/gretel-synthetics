from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from rdt.transformers.numerical import FloatFormatter, INTEGER_BOUNDS, MAX_DECIMALS


@contextmanager
def patch_float_formatter_rounding_bug():
    """Returns a contextmanager object that temporarily patches FloatFormatter.

    A bug in RDT's FloatFormatter in versions <=1.2.1 might result in floating-point
    numbers being rounded to integers. Enclose code using FloatFormatter in a `with`
    block with this object to ensure a patched version not suffering from the bug is
    used.
    """
    orig_reverse_transform = FloatFormatter._reverse_transform
    orig_learn_rounding_digits = FloatFormatter._learn_rounding_digits
    try:
        FloatFormatter._reverse_transform = _patched_float_formatter_reverse_transform
        FloatFormatter._learn_rounding_digits = staticmethod(
            _patched_float_formatter_learn_rounding_digits
        )
        yield
    finally:
        FloatFormatter._learn_rounding_digits = staticmethod(orig_learn_rounding_digits)
        FloatFormatter._reverse_transform = orig_reverse_transform


# The below function is mostly copied from
# https://github.com/sdv-dev/RDT/blob/v1.2.1/rdt/transformers/numerical.py#L188
# which is MIT-licensed, fixing a bug as detailed below.
def _patched_float_formatter_reverse_transform(self, data):
    """Convert data back into the original format.

    Args:
        data (pd.Series or numpy.ndarray):
            Data to transform.

    Returns:
        numpy.ndarray
    """

    if not isinstance(data, np.ndarray):
        data = data.to_numpy()

    if self.missing_value_replacement is not None:
        data = self.null_transformer.reverse_transform(data)

    if self.enforce_min_max_values:
        data = data.clip(self._min_value, self._max_value)
    elif self.computer_representation != "Float":
        min_bound, max_bound = INTEGER_BOUNDS[self.computer_representation]
        data = data.clip(min_bound, max_bound)

    is_integer = np.dtype(self._dtype).kind == "i"
    # BUGFIX: Instead of checking for self._learn_rounding_scheme, check if
    # self._rounding_digits is not None. This implies self._learn_rounding_scheme,
    # but self._rounding_digits MAY actually be None if the data cannot be rounded
    # to any number of decimal digits (consider, e.g., that 0.9... and 0.1.... use
    # a different exponent in the IEEE754 representation and thus have different
    # numbers of bits available for decimal places). The idea that there may be
    # a "maximum" number of decimal digits that suffices is a pure heuristic that
    # only works for some types of input data (basically, when all values are in the
    # range [1.0, 2.0) ).
    if self._rounding_digits is not None:
        data = data.round(self._rounding_digits)
    elif is_integer:
        data = data.round(0)
    # END BUGFIX

    if pd.isna(data).any() and is_integer:
        return data

    return data.astype(self._dtype)


def _patched_float_formatter_learn_rounding_digits(data):
    # check if data has any decimals
    data = np.array(data)
    roundable_data = data[~(np.isinf(data) | pd.isna(data))]
    if not ((roundable_data % 1) != 0).any():
        # BUGFIX: if the above evaluates to true, that means none of the
        # non-NaN input values have any non-zero decimals.
        return 0
    if (roundable_data == roundable_data.round(MAX_DECIMALS)).all():
        for decimal in range(MAX_DECIMALS + 1):
            if (roundable_data == roundable_data.round(decimal)).all():
                return decimal

    return None
