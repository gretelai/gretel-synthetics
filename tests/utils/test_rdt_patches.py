from contextlib import contextmanager

import numpy as np
import pandas as pd

from rdt.transformers.numerical import FloatFormatter

from gretel_synthetics.utils.rdt_patches import (
    _patched_float_formatter_reverse_transform,
    patch_float_formatter_rounding_bug,
)


def test_original_rounding_bug_upstream():
    # RDT float formatter assumes that maximum number of decimals for a float64
    # is 15. However, that only applies to values in the [1.0, 2.0) range due to
    # the IEEE754 representation. Since values with an exponent lower than 0 cannot
    # be losslessly rounded to <= 15 digits, the formatter gives up, but a bug causes
    # the reverse_transform logic to assume this couldn't have happened, and thus
    # rounds values to 0 digits.

    # 15 decimal digits with a value < 1 are sufficient to trigger the bug.
    vals = np.array([0.123456789e-6], dtype=np.float64)
    df = pd.DataFrame(vals.reshape(-1, 1), columns=["A"])

    formatter = FloatFormatter(learn_rounding_scheme=True)
    transformed_df = formatter.fit_transform(df, "A")
    assert transformed_df.iloc[0].to_numpy() == vals

    # reverse_transform will round to 0
    roundtripped_df = formatter.reverse_transform(transformed_df)
    assert roundtripped_df["A"][0] == 0.0


def test_original_rounding_bug_fixed():
    vals = np.array([0.123456789e-6], dtype=np.float64)
    df = pd.DataFrame(vals.reshape(-1, 1), columns=["A"])

    with patch_float_formatter_rounding_bug():
        formatter = FloatFormatter(learn_rounding_scheme=True)
        transformed_df = formatter.fit_transform(df, "A")
        assert transformed_df.iloc[0].to_numpy() == vals

        # reverse_transform will round to 0
        roundtripped_df = formatter.reverse_transform(transformed_df)
        assert roundtripped_df["A"].to_numpy() == vals


def test_integer_rounding_bug_regression():
    # This is a secondary bug that was introduced as a result of the first attempt
    # at fixing the previous bug. If all values are integer-valued (but float-typed),
    # FloatFormatter sets its "rounding digits" value to `None`. The first fix for the
    # above bug took this to mean "values cannot be rounded", so it would never perform
    # any rounding, leading to float values _not_ being rounded to integer in the reverse
    # transform.

    # Replicating the old, incorrect patch
    @contextmanager
    def patch_float_formatter_rounding_bug_old():
        orig_reverse_transform = FloatFormatter._reverse_transform
        try:
            FloatFormatter._reverse_transform = (
                _patched_float_formatter_reverse_transform
            )
            yield
        finally:
            FloatFormatter._reverse_transform = orig_reverse_transform

    vals = np.array([1.0], dtype=np.float64)
    df = pd.DataFrame(vals.reshape(-1, 1), columns=["A"])

    with patch_float_formatter_rounding_bug_old():
        formatter = FloatFormatter(learn_rounding_scheme=True)
        transformed_df = formatter.fit_transform(df, "A")
        assert transformed_df.iloc[0].to_numpy() == vals

        # reverse_transform should round for integer values (doesn't with the bug)
        frac_df_rounded = formatter.reverse_transform(transformed_df / 3.0)
        assert frac_df_rounded.iloc[0, 0] == 1.0 / 3.0


def test_integer_rounding_bug_fixed():
    # This tests that the above bug is now fixed. The fix was simply to set
    # "rounding digits" to 0 for integer values.
    vals = np.array([1.0], dtype=np.float64)
    df = pd.DataFrame(vals.reshape(-1, 1), columns=["A"])

    with patch_float_formatter_rounding_bug():
        formatter = FloatFormatter(learn_rounding_scheme=True)
        transformed_df = formatter.fit_transform(df, "A")
        assert transformed_df.iloc[0].to_numpy() == vals

        # reverse_transform should round for integer values (but used to not do so)
        frac_df_rounded = formatter.reverse_transform(transformed_df / 3.0)
        assert frac_df_rounded.iloc[0, 0] == 0.0  # rounding 0.33.. should yield 0.0
