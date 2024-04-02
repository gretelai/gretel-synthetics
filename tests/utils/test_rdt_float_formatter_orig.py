"""Tests that the original RDT FloatFormatter test suite passes with our patches applied."""

import re

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.null import NullTransformer
from rdt.transformers.numerical import FloatFormatter

from gretel_synthetics.utils.rdt_patches import patch_float_formatter_rounding_bug

with patch_float_formatter_rounding_bug():
    # This is the original suite of tests for the FloatFormatter from rdt.
    # Source code is copied from
    # https://github.com/sdv-dev/RDT/blob/v1.2.1/tests/unit/transformers/test_numerical.py
    # from the 1.2.1 version of RDT.
    # Copyright (c) 2018, MIT Data To AI Lab
    # Copyright (c) 2022 DataCebo, Inc.
    # Licensed under the MIT License.
    class TestFloatFormatter(TestCase):
        def test___init__super_attrs(self):
            """super() arguments are properly passed and set as attributes."""
            nt = FloatFormatter(
                missing_value_replacement="mode", model_missing_values=False
            )

            assert nt.missing_value_replacement == "mode"
            assert nt.model_missing_values is False

        def test_get_output_sdtypes(self):
            """Test the ``get_output_sdtypes`` method when a null column is created.
            When a null column is created, this method should apply the ``_add_prefix``
            method to the following dictionary of output sdtypes:
            output_sdtypes = {
                'value': 'float',
                'is_null': 'float'
            }
            Setup:
                - initialize a ``FloatFormatter`` transformer which:
                    - sets ``self.null_transformer`` to a ``NullTransformer`` where
                    ``self.model_missing_values`` is True.
                    - sets ``self.column_prefix`` to a string.
            Output:
                - the ``output_sdtypes`` dictionary, but with the ``self.column_prefix``
                added to the beginning of the keys.
            """
            # Setup
            transformer = FloatFormatter()
            transformer.null_transformer = NullTransformer(
                missing_value_replacement="fill"
            )
            transformer.null_transformer._model_missing_values = True
            transformer.column_prefix = "a#b"

            # Run
            output = transformer.get_output_sdtypes()

            # Assert
            expected = {"a#b.value": "float", "a#b.is_null": "float"}
            assert output == expected

        def test_is_composition_identity_null_transformer_true(self):
            """Test the ``is_composition_identity`` method with a ``null_transformer``.
            When the attribute ``null_transformer`` is not None and a null column is not created,
            this method should simply return False.
            Setup:
                - initialize a ``FloatFormatter`` transformer which sets
                ``self.null_transformer`` to a ``NullTransformer`` where
                ``self.model_missing_values`` is False.
            Output:
                - False
            """
            # Setup
            transformer = FloatFormatter()
            transformer.null_transformer = NullTransformer(
                missing_value_replacement="fill"
            )

            # Run
            output = transformer.is_composition_identity()

            # Assert
            assert output is False

        def test_is_composition_identity_null_transformer_false(self):
            """Test the ``is_composition_identity`` method without a ``null_transformer``.
            When the attribute ``null_transformer`` is None, this method should return
            the value stored in the ``COMPOSITION_IS_IDENTITY`` attribute.
            Setup:
                - initialize a ``FloatFormatter`` transformer which sets
                ``self.null_transformer`` to None.
            Output:
                - the value stored in ``self.COMPOSITION_IS_IDENTITY``.
            """
            # Setup
            transformer = FloatFormatter()
            transformer.null_transformer = None

            # Run
            output = transformer.is_composition_identity()

            # Assert
            assert output is True

        def test__learn_rounding_digits_more_than_15_decimals(self):
            """Test the _learn_rounding_digits method with more than 15 decimals.
            If the data has more than 15 decimals, None should be returned.
            Input:
            - An array that contains floats with more than 15 decimals.
            Output:
            - None
            """
            data = np.random.random(size=10).round(20)

            output = FloatFormatter._learn_rounding_digits(data)

            assert output is None

        def test__learn_rounding_digits_less_than_15_decimals(self):
            """Test the _learn_rounding_digits method with less than 15 decimals.
            If the data has less than 15 decimals, the maximum number of decimals
            should be returned.
            Input:
            - An array that contains floats with a maximum of 3 decimals and a
            NaN.
            Output:
            - 3
            """
            data = np.array([10, 0.0, 0.1, 0.12, 0.123, np.nan])

            output = FloatFormatter._learn_rounding_digits(data)

            assert output == 3

        def test__learn_rounding_digits_negative_decimals_float(self):
            """Test the _learn_rounding_digits method with floats multiples of powers of 10.
            If the data has all multiples of 10 the output should be None.
            Input:
            - An array that contains floats that are multiples of powers of 10, 100 and 1000
            and a NaN.
            Output:
            - None
            """
            data = np.array([1230.0, 12300.0, 123000.0, np.nan])

            output = FloatFormatter._learn_rounding_digits(data)

            assert output is None

        def test__learn_rounding_digits_negative_decimals_integer(self):
            """Test the _learn_rounding_digits method with integers multiples of powers of 10.
            If the data has all multiples of 10 the output should be None.
            Input:
            - An array that contains integers that are multiples of powers of 10, 100 and 1000
            and a NaN.
            Output:
            - None
            """
            data = np.array([1230, 12300, 123000, np.nan])

            output = FloatFormatter._learn_rounding_digits(data)

            assert output is None

        def test__learn_rounding_digits_all_missing_value_replacements(self):
            """Test the _learn_rounding_digits method with data that is all NaNs.
            If the data is all NaNs, expect that the output is None.
            Input:
            - An array of NaN.
            Output:
            - None
            """
            data = np.array([np.nan, np.nan, np.nan, np.nan])

            output = FloatFormatter._learn_rounding_digits(data)

            assert output is None

        def test__validate_values_within_bounds(self):
            """Test the ``_validate_values_within_bounds`` method.
            If all values are correctly bounded, it shouldn't do anything.
            Setup:
                - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
            Input:
                - a Dataframe.
            """
            # Setup
            data = pd.Series([15, None, 25])
            transformer = FloatFormatter()
            transformer.computer_representation = "UInt8"

            # Run
            transformer._validate_values_within_bounds(data)

        def test__validate_values_within_bounds_under_minimum(self):
            """Test the ``_validate_values_within_bounds`` method.
            Expected to crash if a value is under the bound.
            Setup:
                - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
            Input:
                - a Dataframe.
            Side Effect:
                - raise ``ValueError``.
            """
            # Setup
            data = pd.Series([-15, None, 0], name="a")
            transformer = FloatFormatter()
            transformer.computer_representation = "UInt8"

            # Run / Assert
            err_msg = re.escape(
                "The minimum value in column 'a' is -15.0. All values represented by 'UInt8'"
                " must be in the range [0, 255]."
            )
            with pytest.raises(ValueError, match=err_msg):
                transformer._validate_values_within_bounds(data)

        def test__validate_values_within_bounds_over_maximum(self):
            """Test the ``_validate_values_within_bounds`` method.
            Expected to crash if a value is over the bound.
            Setup:
                - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
            Input:
                - a Dataframe.
            """
            # Setup
            data = pd.Series([255, None, 256], name="a")
            transformer = FloatFormatter()
            transformer.computer_representation = "UInt8"

            # Run / Assert
            err_msg = re.escape(
                "The maximum value in column 'a' is 256.0. All values represented by 'UInt8'"
                " must be in the range [0, 255]."
            )
            with pytest.raises(ValueError, match=err_msg):
                transformer._validate_values_within_bounds(data)

        def test__validate_values_within_bounds_floats(self):
            """Test the ``_validate_values_within_bounds`` method.
            Expected to crash if float values are passed when ``computer_representation`` is an int.
            Setup:
                - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
            Input:
                - a Dataframe.
            """
            # Setup
            data = pd.Series([249.2, None, 250.0, 10.2], name="a")
            transformer = FloatFormatter()
            transformer.computer_representation = "UInt8"

            # Run / Assert
            err_msg = re.escape(
                "The column 'a' contains float values [249.2, 10.2]."
                " All values represented by 'UInt8' must be integers."
            )
            with pytest.raises(ValueError, match=err_msg):
                transformer._validate_values_within_bounds(data)

        def test__fit(self):
            """Test the ``_fit`` method.
            Validate that the ``_dtype`` and ``.null_transformer.missing_value_replacement`` attributes
            are set correctly.
            Setup:
                - initialize a ``FloatFormatter`` with the ``missing_value_replacement``
                parameter set to ``'missing_value_replacement'``.
            Input:
                - a pandas series containing a None.
            Side effect:
                - it sets the ``null_transformer.missing_value_replacement``.
                - it sets the ``_dtype``.
                - it calls ``_validate_values_within_bounds``.
            """
            # Setup
            data = pd.Series([1.5, None, 2.5])
            transformer = FloatFormatter(
                missing_value_replacement="missing_value_replacement"
            )
            transformer._validate_values_within_bounds = Mock()

            # Run
            transformer._fit(data)

            # Asserts
            expected = "missing_value_replacement"
            assert transformer.null_transformer._missing_value_replacement == expected
            assert transformer._dtype == float
            transformer._validate_values_within_bounds.assert_called_once_with(data)

        def test__fit_learn_rounding_scheme_false(self):
            """Test ``_fit`` with ``learn_rounding_scheme`` set to ``False``.
            If the ``learn_rounding_scheme`` is set to ``False``, the ``_fit`` method
            should not set its ``_rounding_digits`` instance variable.
            Input:
            - An array with floats rounded to one decimal and a None value
            Side Effect:
            - ``_rounding_digits`` should be ``None``
            """
            # Setup
            data = pd.Series([1.5, None, 2.5])

            # Run
            transformer = FloatFormatter(
                missing_value_replacement="missing_value_replacement",
                learn_rounding_scheme=False,
            )
            transformer._fit(data)

            # Asserts
            assert transformer._rounding_digits is None

        def test__fit_learn_rounding_scheme_true(self):
            """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.
            If ``learn_rounding_scheme`` is set to ``True``, the ``_fit`` method
            should set its ``_rounding_digits`` instance variable to what is learned
            in the data.
            Input:
            - A Series with floats up to 4 decimals and a None value
            Side Effect:
            - ``_rounding_digits`` is set to 4
            """
            # Setup
            data = pd.Series([1, 2.1, 3.12, 4.123, 5.1234, 6.123, 7.12, 8.1, 9, None])

            # Run
            transformer = FloatFormatter(
                missing_value_replacement="mean", learn_rounding_scheme=True
            )
            transformer._fit(data)

            # Asserts
            assert transformer._rounding_digits == 4

        def test__fit_learn_rounding_scheme_true_max_decimals(self):
            """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.
            If the ``learn_rounding_scheme`` parameter is set to ``True``, ``_fit`` should learn
            the ``_rounding_digits`` to be the max number of decimal places seen in the data.
            The max amount of decimals that floats can be accurately compared with is 15.
            If the input data has values with more than 14 decimals, we will not be able to
            accurately learn the number of decimal places required, so we do not round.
            Input:
            - Series with a value that has 15 decimals
            Side Effect:
            - ``_rounding_digits`` is set to ``None``
            """
            # Setup
            data = pd.Series([0.000000000000001])

            # Run
            transformer = FloatFormatter(
                missing_value_replacement="mean", learn_rounding_scheme=True
            )
            transformer._fit(data)

            # Asserts
            assert transformer._rounding_digits is None

        def test__fit_learn_rounding_scheme_true_inf(self):
            """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.
            If the ``learn_rounding_scheme`` parameter is set to ``True``, and the data
            contains only integers or infinite values, ``_fit`` should learn
            ``_rounding_digits`` to be None.
            Input:
            - Series with ``np.inf`` as a value
            Side Effect:
            - ``_rounding_digits`` is set to None
            """
            # Setup
            data = pd.Series([15000, 4000, 60000, np.inf])

            # Run
            transformer = FloatFormatter(
                missing_value_replacement="mean", learn_rounding_scheme=True
            )
            transformer._fit(data)

            # Asserts
            assert transformer._rounding_digits is None

        def test__fit_learn_rounding_scheme_true_max_zero(self):
            """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.
            If the ``learn_rounding_scheme`` parameter is set to ``True``, and the max
            in the data is 0, ``_fit`` should learn the ``_rounding_digits`` to be None.
            Input:
            - Series with 0 as max value
            Side Effect:
            - ``_rounding_digits`` is set to None
            """
            # Setup
            data = pd.Series([0, 0, 0])

            # Run
            transformer = FloatFormatter(
                missing_value_replacement="mean", learn_rounding_scheme=True
            )
            transformer._fit(data)

            # Asserts
            assert transformer._rounding_digits is None

        def test__fit_enforce_min_max_values_false(self):
            """Test ``_fit`` with ``enforce_min_max_values`` set to ``False``.
            If the ``enforce_min_max_values`` parameter is set to ``False``,
            the ``_fit`` method should not set its ``min`` or ``max``
            instance variables.
            Input:
            - Series of floats and null values
            Side Effect:
            - ``_min_value`` and ``_max_value`` stay ``None``
            """
            # Setup
            data = pd.Series([1.5, None, 2.5])

            # Run
            transformer = FloatFormatter(
                missing_value_replacement="mean", enforce_min_max_values=False
            )
            transformer._fit(data)

            # Asserts
            assert transformer._min_value is None
            assert transformer._max_value is None

        def test__fit_enforce_min_max_values_true(self):
            """Test ``_fit`` with ``enforce_min_max_values`` set to ``True``.
            If the ``enforce_min_max_values`` parameter is set to ``True``,
            the ``_fit`` method should learn the min and max values from the _fitted data.
            Input:
            - Series of floats and null values
            Side Effect:
            - ``_min_value`` and ``_max_value`` are learned
            """
            # Setup
            data = pd.Series([-100, -5000, 0, None, 100, 4000])

            # Run
            transformer = FloatFormatter(
                missing_value_replacement="mean", enforce_min_max_values=True
            )
            transformer._fit(data)

            # Asserts
            assert transformer._min_value == -5000
            assert transformer._max_value == 4000

        def test__transform(self):
            """Test the ``_transform`` method.
            Validate that this method calls the ``self.null_transformer.transform`` method once.
            Setup:
                - create an instance of a ``FloatFormatter`` and set ``self.null_transformer``
                to a ``NullTransformer``.
            Input:
                - a pandas series.
            Output:
                - the transformed numpy array.
            """
            # Setup
            data = pd.Series([1, 2, 3])
            transformer = FloatFormatter()
            transformer._validate_values_within_bounds = Mock()
            transformer.null_transformer = Mock()

            # Run
            transformer._transform(data)

            # Assert
            transformer._validate_values_within_bounds.assert_called_once_with(data)
            assert transformer.null_transformer.transform.call_count == 1

        def test__reverse_transform_learn_rounding_scheme_false(self):
            """Test ``_reverse_transform`` when ``learn_rounding_scheme`` is ``False``.
            The data should not be rounded at all.
            Input:
            - Random array of floats between 0 and 1
            Output:
            - Input array
            """
            # Setup
            data = np.random.random(10)

            # Run
            transformer = FloatFormatter(missing_value_replacement=None)
            transformer.learn_rounding_scheme = False
            transformer._rounding_digits = None
            result = transformer._reverse_transform(data)

            # Assert
            np.testing.assert_array_equal(result, data)

        def test__reverse_transform_rounding_none_dtype_int(self):
            """Test ``_reverse_transform`` with ``_dtype`` as ``np.int64`` and no rounding.
            The data should be rounded to 0 decimals and returned as integer values if the ``_dtype``
            is ``np.int64`` even if ``_rounding_digits`` is ``None``.
            Input:
            - Array of multiple float values with decimals.
            Output:
            - Input array rounded an converted to integers.
            """
            # Setup
            data = np.array([0.0, 1.2, 3.45, 6.789])

            # Run
            transformer = FloatFormatter(missing_value_replacement=None)
            transformer._rounding_digits = None
            transformer._dtype = np.int64
            result = transformer._reverse_transform(data)

            # Assert
            expected = np.array([0, 1, 3, 7])
            np.testing.assert_array_equal(result, expected)

        def test__reverse_transform_rounding_none_with_nulls(self):
            """Test ``_reverse_transform`` when ``_rounding_digits`` is ``None`` and there are nulls.
            The data should not be rounded at all.
            Input:
            - 2d Array of multiple float values with decimals and a column setting at least 1 null.
            Output:
            - First column of the input array as entered, replacing the indicated value with a
            missing_value_replacement.
            """
            # Setup
            data = [
                [0.0, 0.0],
                [1.2, 0.0],
                [3.45, 1.0],
                [6.789, 0.0],
            ]

            data = pd.DataFrame(data, columns=["a", "b"])

            # Run
            transformer = FloatFormatter(missing_value_replacement="mean")
            null_transformer = Mock()
            null_transformer.reverse_transform.return_value = np.array(
                [0.0, 1.2, np.nan, 6.789]
            )
            transformer.null_transformer = null_transformer
            transformer.learn_rounding_scheme = False
            transformer._rounding_digits = None
            transformer._dtype = float
            result = transformer._reverse_transform(data)

            # Assert
            expected = np.array([0.0, 1.2, np.nan, 6.789])
            np.testing.assert_array_equal(result, expected)

        def test__reverse_transform_rounding_none_with_nulls_dtype_int(self):
            """Test ``_reverse_transform`` rounding when dtype is int and there are nulls.
            The data should be rounded to 0 decimals and returned as float values with
            nulls in the right place.
            Input:
            - 2d Array of multiple float values with decimals and a column setting at least 1 null.
            Output:
            - First column of the input array rounded, replacing the indicated value with a
            ``NaN``, and kept as float values.
            """
            # Setup
            data = np.array(
                [
                    [0.0, 0.0],
                    [1.2, 0.0],
                    [3.45, 1.0],
                    [6.789, 0.0],
                ]
            )

            # Run
            transformer = FloatFormatter(missing_value_replacement="mean")
            null_transformer = Mock()
            null_transformer.reverse_transform.return_value = np.array(
                [0.0, 1.2, np.nan, 6.789]
            )
            transformer.null_transformer = null_transformer
            transformer.learn_rounding_digits = False
            transformer._rounding_digits = None
            transformer._dtype = int
            result = transformer._reverse_transform(data)

            # Assert
            expected = np.array([0.0, 1.0, np.nan, 7.0])
            np.testing.assert_array_equal(result, expected)

        def test__reverse_transform_rounding_small_numbers(self):
            """Test ``_reverse_transform`` when ``_rounding_digits`` is positive.
            The data should round to the maximum number of decimal places
            set in the ``_rounding_digits`` value.
            Input:
            - Array with decimals
            Output:
            - Same array rounded to the provided number of decimal places
            """
            # Setup
            data = np.array([1.1111, 2.2222, 3.3333, 4.44444, 5.555555])

            # Run
            transformer = FloatFormatter(missing_value_replacement=None)
            transformer.learn_rounding_scheme = True
            transformer._rounding_digits = 2
            result = transformer._reverse_transform(data)

            # Assert
            expected_data = np.array([1.11, 2.22, 3.33, 4.44, 5.56])
            np.testing.assert_array_equal(result, expected_data)

        def test__reverse_transform_rounding_big_numbers_type_int(self):
            """Test ``_reverse_transform`` when ``_rounding_digits`` is negative.
            The data should round to the number set in the ``_rounding_digits``
            attribute and remain ints.
            Input:
            - Array with with floats above 100
            Output:
            - Same array rounded to the provided number of 0s
            - Array should be of type int
            """
            # Setup
            data = np.array([2000.0, 120.0, 3100.0, 40100.0])

            # Run
            transformer = FloatFormatter(missing_value_replacement=None)
            transformer._dtype = int
            transformer.learn_rounding_scheme = True
            transformer._rounding_digits = -3
            result = transformer._reverse_transform(data)

            # Assert
            expected_data = np.array([2000, 0, 3000, 40000])
            np.testing.assert_array_equal(result, expected_data)
            assert result.dtype == int

        def test__reverse_transform_rounding_negative_type_float(self):
            """Test ``_reverse_transform`` when ``_rounding_digits`` is negative.
            The data should round to the number set in the ``_rounding_digits``
            attribute and remain floats.
            Input:
            - Array with with larger numbers
            Output:
            - Same array rounded to the provided number of 0s
            - Array should be of type float
            """
            # Setup
            data = np.array([2000.0, 120.0, 3100.0, 40100.0])

            # Run
            transformer = FloatFormatter(missing_value_replacement=None)
            transformer.learn_rounding_scheme = True
            transformer._rounding_digits = -3
            result = transformer._reverse_transform(data)

            # Assert
            expected_data = np.array([2000.0, 0.0, 3000.0, 40000.0])
            np.testing.assert_array_equal(result, expected_data)
            assert result.dtype == float

        def test__reverse_transform_rounding_zero_decimal_places(self):
            """Test ``_reverse_transform`` when ``_rounding_digits`` is 0.
            The data should round to the number set in the ``_rounding_digits``
            attribute.
            Input:
            - Array with with larger numbers
            Output:
            - Same array rounded to the 0s place
            """
            # Setup
            data = np.array([2000.554, 120.2, 3101, 4010])

            # Run
            transformer = FloatFormatter(missing_value_replacement=None)
            transformer.learn_rounding_scheme = True
            transformer._rounding_digits = 0
            result = transformer._reverse_transform(data)

            # Assert
            expected_data = np.array([2001, 120, 3101, 4010])
            np.testing.assert_array_equal(result, expected_data)

        def test__reverse_transform_enforce_min_max_values(self):
            """Test ``_reverse_transform`` with ``enforce_min_max_values`` set to ``True``.
            The ``_reverse_transform`` method should clip any values above
            the ``max_value`` and any values below the ``min_value``.
            Input:
            - Array with values above the max and below the min
            Output:
            - Array with out of bound values clipped to min and max
            """
            # Setup
            data = np.array([-np.inf, -5000, -301, -250, 0, 125, 401, np.inf])

            # Run
            transformer = FloatFormatter(missing_value_replacement=None)
            transformer.enforce_min_max_values = True
            transformer._max_value = 400
            transformer._min_value = -300
            result = transformer._reverse_transform(data)

            # Asserts
            np.testing.assert_array_equal(
                result, np.array([-300, -300, -300, -250, 0, 125, 400, 400])
            )

        def test__reverse_transform_enforce_min_max_values_with_nulls(self):
            """Test ``_reverse_transform`` with nulls and ``enforce_min_max_values`` set to ``True``.
            The ``_reverse_transform`` method should clip any values above
            the ``max_value`` and any values below the ``min_value``. Null values
            should be replaced with ``np.nan``.
            Input:
            - 2d array where second column has some values over 0.5 representing null values
            Output:
            - Array with out of bounds values clipped and null values injected
            """
            # Setup
            data = np.array(
                [
                    [-np.inf, 0],
                    [-5000, 0.1],
                    [-301, 0.8],
                    [-250, 0.4],
                    [0, 0],
                    [125, 1],
                    [401, 0.2],
                    [np.inf, 0.5],
                ]
            )
            expected_data = np.array([-300, -300, np.nan, -250, 0, np.nan, 400, 400])

            # Run
            transformer = FloatFormatter(missing_value_replacement="mean")
            transformer._max_value = 400
            transformer._min_value = -300
            transformer.enforce_min_max_values = True
            transformer.null_transformer = Mock()
            transformer.null_transformer.reverse_transform.return_value = expected_data
            result = transformer._reverse_transform(data)

            # Asserts
            null_transformer_calls = (
                transformer.null_transformer.reverse_transform.mock_calls
            )
            np.testing.assert_array_equal(null_transformer_calls[0][1][0], data)
            np.testing.assert_array_equal(result, expected_data)

        def test__reverse_transform_enforce_computer_representation(self):
            """Test ``_reverse_transform`` with ``computer_representation`` set to ``Int8``.
            The ``_reverse_transform`` method should clip any values out of bounds.
            Input:
            - Array with values above the max and below the min
            Output:
            - Array with out of bound values clipped to min and max
            """
            # Setup
            data = np.array([-np.inf, np.nan, -5000, -301, -100, 0, 125, 401, np.inf])

            # Run
            transformer = FloatFormatter(computer_representation="Int8")
            result = transformer._reverse_transform(data)

            # Asserts
            np.testing.assert_array_equal(
                result, np.array([-128, np.nan, -128, -128, -100, 0, 125, 127, 127])
            )
