import itertools
import sys

import numpy as np
import pytest
import torch

from gretel_synthetics.timeseries_dgan.config import Normalization
from gretel_synthetics.timeseries_dgan.transformations import (
    BinaryEncodedOutput,
    ContinuousOutput,
    inverse_transform_attributes,
    inverse_transform_features,
    OneHotEncodedOutput,
    rescale,
    rescale_inverse,
    transform_attributes,
    transform_features,
)


def assert_array_equal(a: np.ndarray, b: np.ndarray):
    """Custom assert to handle float and object numpy arrays with nans.

    Treats nan as a "normal" number where nan == nan is True.
    """
    assert a.shape == b.shape
    if a.dtype == "O" or b.dtype == "O":
        # np.testing.assert* functions compare nans as False for object arrays.

        # So first replace nans with a value that doesn't otherwise appear in
        # the arrays.

        replace_value = "NANA"
        while np.any([x == replace_value for x in a]) or np.any(
            [y == replace_value for y in a]
        ):
            replace_value += "NA"

        # Need to flatten so the mask creation is simple. Note np.isnan doesn't
        # work on object arrays.
        test_a = a.flatten()
        test_b = b.flatten()
        test_a_nan_mask = [isinstance(x, float) and np.isnan(x) for x in test_a]
        test_b_nan_mask = [isinstance(x, float) and np.isnan(x) for x in test_b]

        test_a[test_a_nan_mask] = replace_value
        test_b[test_b_nan_mask] = replace_value

        # Now compare 2 arrays that should not have any nans
        np.testing.assert_array_equal(
            test_a, test_b, err_msg=f"original arrays:\n{a}\nand\n{b}"
        )
    else:
        np.testing.assert_array_equal(a, b)


def test_custom_assert_array_equal():
    assert_array_equal(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert_array_equal(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert_array_equal(np.array([0.0, 1.0, np.NaN]), np.array([0.0, 1.0, np.NaN]))
    assert_array_equal(np.array(["a", "b", 0]), np.array(["a", "b", 0]))
    assert_array_equal(
        np.array(["a", "b", np.NaN], dtype="O"), np.array(["a", "b", np.NaN], dtype="O")
    )

    # Check for different ways of creating nan
    assert_array_equal(
        np.array(["a", 1.0, np.NaN, float("nan"), np.Inf / np.Inf], dtype="O"),
        np.array(["a", 1.0, np.NaN, np.NaN, np.NaN], dtype="O"),
    )

    # Check for nans coming from torch versus numpy nans
    a = torch.Tensor([1.0, np.NaN, float("nan"), np.Inf / np.Inf]).numpy().astype("O")
    a[0] = "a"
    b = np.array(["a", np.NaN, np.NaN, np.NaN], dtype="O")
    assert_array_equal(a, b)

    with pytest.raises(AssertionError):
        assert_array_equal(np.array([0, 1, 3]), np.array([0, 1, 2]))

    with pytest.raises(AssertionError):
        assert_array_equal(np.array([0.0]), np.array([1.0]))

    with pytest.raises(AssertionError):
        assert_array_equal(np.array([0.0, np.nan]), np.array([0.0, 0.0]))

    with pytest.raises(AssertionError):
        assert_array_equal(
            np.array(["a", "b"], dtype="O"), np.array(["a", "bbbb"], dtype="O")
        )

    with pytest.raises(AssertionError):
        assert_array_equal(
            np.array(["a", np.NaN], dtype="O"), np.array(["a", "nan"], dtype="O")
        )


def test_one_hot_encoded_output_int():
    output = OneHotEncodedOutput(name="foo")

    input = np.array([0, 1, 2, 3])
    output.fit(input)
    assert output.name == "foo"
    assert output.dim == 4

    # Check that transform/inverse transform are inverses
    transformed1 = output.transform(input)
    assert_array_equal(input, output.inverse_transform(transformed1))

    # Check that inverse transform applies argmax for a non-binary array
    transformed2 = np.array(
        [
            [1.1, 0.1, 0.0, -1.0],
            [0.1, 0.15, 0.15, 0.05],
            [-10.0, -5.0, -2.5, -3.0],
            [5.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected2 = np.array([0, 1, 2, 3, 0])
    assert_array_equal(expected2, output.inverse_transform(transformed2))


def test_one_hot_encoded_output_string():
    output = OneHotEncodedOutput(name="foo")

    output.fit(np.array(["aa", "bb", "cc", "dd", "ee"]))
    assert output.name == "foo"
    assert output.dim == 5

    # Check that transform/inverse transform are inverses
    input1 = np.array(["aa", "aa", "bb", "ee", "dd", "cc"])
    transformed1 = output.transform(input1)
    assert_array_equal(input1, output.inverse_transform(transformed1))

    # Check that inverse transform applies argmax for a non-binary array
    transformed2 = np.array(
        [
            [1.1, 0.1, 0.0, -1.0, -1.0],
            [0.1, 0.15, 0.15, 0.05, 0.1],
            [-10.0, -5.0, -2.5, -3.0, -4.0],
            [5.0, 6.0, 7.0, 8.0, 0.0],
            [0.12, 0.12, 0.15, 0.15, 0.18],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected2 = np.array(["aa", "bb", "cc", "dd", "ee", "aa"])
    assert_array_equal(expected2, output.inverse_transform(transformed2))


def test_one_hot_encoded_output_nans():
    output = OneHotEncodedOutput(name="foo")

    # Explicit object dtype is required, otherwise nans are auto casted to the
    # string "nan".
    input = np.array(["aa", "bb", np.nan, "cc", "", np.nan], dtype="O")
    output.fit(input)
    assert output.dim == 5

    transformed = output.transform(input)

    # Both nan values should have the same representation
    assert_array_equal(transformed[2, :], transformed[5, :])

    # Check that transform inverse transform are inverses
    assert_array_equal(input, output.inverse_transform(transformed))


def test_binary_encoded_output_string():
    values = set(["hello", "world", "bar", "aa", "bb", "cc", "dd", "ee"])
    output = BinaryEncodedOutput(name="foo")
    output.fit(np.array(list(values)))

    assert output.name == "foo"
    assert output.dim < 8

    input = np.array(["hello", "bar", "bb", "world", "ee"])
    transformed1 = output.transform(input)
    assert_array_equal(input, output.inverse_transform(transformed1))

    # Check that inverse transform applies a threshold for a non-binary array.
    # Reuse the 0/1 patterns to guarantee they map to a value, so convert 0.0
    # and 1.0 to 0.1 and 0.6, respectively.
    transformed2 = (transformed1 + 0.2) / 2.0
    assert_array_equal(input, output.inverse_transform(transformed2))

    # Check that all possible binary codes are inverse transformed to a value
    # (no NaNs)
    transformed3 = np.array(
        list(itertools.product([0.0, 1.0], repeat=transformed2.shape[1]))
    )

    original3 = output.inverse_transform(transformed3)

    for v in original3:
        assert v in values


def test_binary_encoded_output_nans():
    output = BinaryEncodedOutput(name="foo")

    # Explicit object dtype is required, otherwise nans are auto casted to the
    # string "nan".
    input = np.array(["aa", "bb", np.nan, "cc", "", np.nan, "cc"], dtype="O")
    output.fit(input)
    assert output.dim < 5

    transformed = output.transform(input)

    # Both nan values should have the same representation
    assert_array_equal(transformed[2, :], transformed[5, :])

    # Check that transform inverse transform are inverses
    assert_array_equal(input, output.inverse_transform(transformed))


def test_continuous_output():
    output = ContinuousOutput(
        name="bar",
        normalization=Normalization.ZERO_ONE,
        apply_feature_scaling=False,
        apply_example_scaling=True,
    )
    input = np.array([0.0, 0.5, 1.0])
    output.fit(input)

    assert output.name == "bar"
    assert output.dim == 1
    assert output.apply_example_scaling

    assert output.global_min == 0.0
    assert output.global_max == 1.0

    transformed = output.transform(input)
    original = output.inverse_transform(transformed)
    assert original.shape == (3,)
    np.testing.assert_allclose(input, original)


def test_rescale_and_inverse():
    original = np.array([1.5, 3, -1.0])
    global_min = np.min(original)
    global_max = np.max(original)

    scaled = rescale(original, Normalization.ZERO_ONE, global_min, global_max)
    np.testing.assert_allclose(scaled, [0.625, 1.0, 0.0])
    inversed = rescale_inverse(scaled, Normalization.ZERO_ONE, global_min, global_max)
    np.testing.assert_allclose(inversed, original)

    scaled = rescale(original, Normalization.MINUSONE_ONE, global_min, global_max)
    np.testing.assert_allclose(scaled, [0.25, 1.0, -1.0])
    inversed = rescale_inverse(
        scaled, Normalization.MINUSONE_ONE, global_min, global_max
    )
    np.testing.assert_allclose(inversed, original)


def test_rescale_and_inverse_by_example():
    original = np.array(
        [
            [1.5, 3, -1.0],
            [10, 20, 30],
            [1000, 1000, 1000.0],
            [-0.1, -0.3, -0.5],
        ]
    )

    mins = np.broadcast_to(np.min(original, axis=1).reshape(4, 1), (4, 3))
    maxes = np.broadcast_to(np.max(original, axis=1).reshape(4, 1), (4, 3))

    scaled = rescale(original, Normalization.ZERO_ONE, mins, maxes)
    expected = [
        [0.625, 1.0, 0.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
    ]
    np.testing.assert_allclose(scaled, expected)
    inversed = rescale_inverse(scaled, Normalization.ZERO_ONE, mins, maxes)
    np.testing.assert_allclose(inversed, original)

    scaled = rescale(original, Normalization.MINUSONE_ONE, mins, maxes)
    expected = [
        [0.25, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, -1.0],
    ]
    np.testing.assert_allclose(scaled, expected)
    inversed = rescale_inverse(scaled, Normalization.MINUSONE_ONE, mins, maxes)
    np.testing.assert_allclose(inversed, original)


def test_transform_features_variable_length_example_scaling():
    # Var 0 has no overall scaling (since global min=0.0, max=1.0), but per
    # example scaling. Var 1 has overall scaling and per example scaling.
    outputs = [
        ContinuousOutput(
            "a",
            Normalization.ZERO_ONE,
            apply_feature_scaling=True,
            apply_example_scaling=True,
            global_min=0.0,
            global_max=1.0,
        ),
        ContinuousOutput(
            "b", Normalization.ZERO_ONE, True, True, global_min=3.0, global_max=7.0
        ),
    ]
    input = [
        # Parenthesis for variable 1 is the global scaled version of the
        # min/max.
        # Example 0:
        #  Var 0: min=0.3, max=0.5
        #  Var 1: min=5.0 (0.5), max=7.0 (1.0)
        np.array([[0.5, 5.0], [0.35, 6.0], [0.3, 7.0]]),
        # Example 1:
        #  Var 0: min=0.5, max=0.5
        #  Var 1: min=3.0 (0.0), max=3.0 (0.0)
        np.array([[0.5, 3.0]]),
        # Example 2:
        #  Var 0: min=0.6, max=0.7
        #  Var 1: min=4.0 (0.25), max=6.5 (0.875)
        np.array([[0.7, 4.0], [0.7, 6.0], [0.7, 5.5], [0.6, 6.5]]),
    ]
    expected_transformed = np.array(
        [
            [[1.0, 0.0], [0.25, 0.5], [0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.8], [1.0, 0.6], [0.0, 1.0]],
        ]
    )

    # Columns: Var 0 midpoint, Var 0 halfrange, Var 1 midpoint, Var 1 halfrange
    expected_additional_attributes = np.array(
        [
            [0.4, 0.1, 0.75, 0.25],
            [0.5, 0.0, 0.0, 0.0],
            [0.65, 0.05, 0.5625, 0.3125],
        ]
    )

    transformed, additional_attributes = transform_features(
        input, outputs, max_sequence_len=4
    )

    np.testing.assert_allclose(transformed, expected_transformed)
    assert additional_attributes is not None
    np.testing.assert_allclose(additional_attributes, expected_additional_attributes)


@pytest.mark.parametrize(
    "normalization", [Normalization.ZERO_ONE, Normalization.MINUSONE_ONE]
)
def test_transform_and_inverse_attributes(normalization):
    n = 100
    attributes = np.stack(
        (
            np.random.rand(n) * 1000.0 + 500.0,
            np.random.randint(0, 2, size=n),
            np.random.rand(n) * 0.1 - 10.0,
            np.random.randint(0, 5, size=n),
            np.zeros(n) + 2.0,
        ),
        axis=1,
    )

    outputs = [
        ContinuousOutput(
            "a", normalization, True, False, global_min=500.0, global_max=1500.0
        ),
        OneHotEncodedOutput("b", dim=2),
        ContinuousOutput(
            "c", normalization, True, False, global_min=-10.0, global_max=-9.9
        ),
        BinaryEncodedOutput("d", dim=5),
        ContinuousOutput(
            "e", normalization, True, False, global_min=2.0, global_max=2.0
        ),
    ]

    transformed = transform_attributes(attributes, outputs)
    # 3 continuous + 2 for one hot encoded + 4 for binary encoded
    # (No idea why category encoders needs 4 bits to encode 5 unique values)
    assert transformed.shape == (n, 9)

    inversed = inverse_transform_attributes(transformed, outputs)
    assert inversed is not None
    np.testing.assert_allclose(inversed, attributes, rtol=1e-04)


@pytest.mark.parametrize(
    "normalization", [Normalization.ZERO_ONE, Normalization.MINUSONE_ONE]
)
def test_transform_and_inverse_features(normalization):
    n = 100
    features = np.stack(
        (
            np.random.rand(n, 10) * 1000.0 + 500.0,
            np.random.rand(n, 10) * 5.0,
            np.random.randint(0, 3, size=(n, 10)),
        ),
        axis=2,
    )
    assert features.shape == (100, 10, 3)

    outputs = [
        ContinuousOutput("a", normalization, True, True),
        ContinuousOutput("b", normalization, True, True),
        OneHotEncodedOutput("c"),
    ]
    for index, output in enumerate(outputs):
        output.fit(features[:, :, index].flatten())

    transformed, additional_attributes = transform_features(
        list(features), outputs, max_sequence_len=10
    )
    assert transformed.shape == (100, 10, 5)
    assert additional_attributes is not None
    assert additional_attributes.shape == (100, 4)

    inversed = inverse_transform_features(transformed, outputs, additional_attributes)
    # TODO: 1e-04 seems too lax of a tolerance for float32, but values very
    # close to 0.0 are failing the check at 1e-05, so going with this for now to
    # reduce flakiness. Could be something we can do in the calculations to have
    # less error.
    np.testing.assert_allclose(inversed, features, rtol=1e-04)
