from dataclasses import FrozenInstanceError

import pytest

from gretel_synthetics.timeseries_dgan.config import Normalization
from gretel_synthetics.timeseries_dgan.transformations import (
    ContinuousOutput,
    DiscreteOutput,
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
