from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gretel_synthetics.timeseries_dgan.config import Normalization
from gretel_synthetics.timeseries_dgan.transformations import (
    ContinuousOutput,
    DiscreteOutput,
    inverse_transform,
    rescale,
    rescale_inverse,
    transform,
)


def test_output():
    o1 = DiscreteOutput(name="foo", dim=4)
    assert o1.name == "foo"
    assert o1.dim == 4

    o2 = ContinuousOutput(
        name="bar",
        normalization=Normalization.ZERO_ONE,
        global_min=0.0,
        global_max=1.0,
        apply_feature_scaling=False,
        apply_example_scaling=True,
    )
    assert o2.name == "bar"
    assert o2.dim == 1
    assert o2.apply_example_scaling == True

    with pytest.raises(FrozenInstanceError):
        o2.dim = 2

    with pytest.raises(TypeError):
        ContinuousOutput(
            name="baz",
            dim=3,
            normalization=Normalization.ZERO_ONE,
            global_min=0.0,
            global_max=1.0,
            apply_feature_scaling=True,
            apply_example_scaling=True,
        )


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
        ContinuousOutput("a", normalization, 500.0, 1500.0, True, False),
        DiscreteOutput("b", 2),
        ContinuousOutput("c", normalization, -10.0, -9.9, True, False),
        DiscreteOutput("d", 5),
        ContinuousOutput("e", normalization, 2.0, 2.0, True, False),
    ]
    transformed = transform(attributes, outputs, 1)
    assert transformed.shape == (n, 10)

    inversed = inverse_transform(transformed, outputs, 1)
    np.testing.assert_allclose(inversed, attributes)


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
        ContinuousOutput("a", normalization, 500.0, 1500.0, True, True),
        ContinuousOutput("b", normalization, 0.0, 5.0, True, True),
        DiscreteOutput("c", 3),
    ]

    transformed, additional_attributes = transform(features, outputs, 2)
    assert transformed.shape == (100, 10, 5)
    assert additional_attributes.shape == (100, 4)

    inversed = inverse_transform(transformed, outputs, 2, additional_attributes)
    np.testing.assert_allclose(inversed, features)
