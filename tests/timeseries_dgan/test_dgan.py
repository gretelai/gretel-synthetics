import itertools

import numpy as np
import pandas as pd
import pytest

from gretel_synthetics.timeseries_dgan.config import (
    DGANConfig,
    Normalization,
    OutputType,
)
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.transformations import (
    ContinuousOutput,
    DiscreteOutput,
)


@pytest.fixture
def attribute_data():
    n = 100
    attributes = np.concatenate(
        (
            np.random.rand(n, 1),
            np.random.randint(0, 3, size=(n, 1)),
        ),
        axis=1,
    )
    return (attributes, [OutputType.CONTINUOUS, OutputType.DISCRETE])


@pytest.fixture
def feature_data():
    n = 100
    features = np.concatenate(
        (
            np.random.randint(0, 4, size=(n, 20, 1)),
            np.random.rand(n, 20, 1),
        ),
        axis=2,
    )
    return (features, [OutputType.DISCRETE, OutputType.CONTINUOUS])


@pytest.fixture
def config() -> DGANConfig:
    return DGANConfig(
        max_sequence_len=20,
        sample_len=5,
        batch_size=10,
        epochs=2,
    )


def test_generate():
    attribute_outputs = [
        ContinuousOutput(
            name="a",
            normalization=Normalization.ZERO_ONE,
            global_min=0,
            global_max=1,
            apply_feature_scaling=False,
            apply_example_scaling=False,
        ),
        DiscreteOutput(name="b", dim=3),
        DiscreteOutput(name="c", dim=4),
    ]
    feature_outputs = [
        DiscreteOutput(name="d", dim=4),
        ContinuousOutput(
            name="e",
            normalization=Normalization.ZERO_ONE,
            global_min=0,
            global_max=1,
            apply_feature_scaling=False,
            apply_example_scaling=False,
        ),
    ]
    config = DGANConfig(max_sequence_len=20, sample_len=5, batch_size=25)

    dg = DGAN(
        config=config,
        attribute_outputs=attribute_outputs,
        feature_outputs=feature_outputs,
    )

    # Check requesting various number of examples
    attributes, features = dg.generate_numpy(8)

    assert attributes.shape == (8, 3)
    assert features.shape == (8, 20, 2)

    attributes, features = dg.generate_numpy(64)

    assert attributes.shape == (64, 3)
    assert features.shape == (64, 20, 2)

    attributes, features = dg.generate_numpy(200)

    assert attributes.shape == (200, 3)
    assert features.shape == (200, 20, 2)

    attributes, features = dg.generate_numpy(1)

    assert attributes.shape == (1, 3)
    assert features.shape == (1, 20, 2)

    # Check passing noise vectors

    attributes, features = dg.generate_numpy(
        attribute_noise=dg.attribute_noise_func(20),
        feature_noise=dg.feature_noise_func(20),
    )

    assert attributes.shape == (20, 3)
    assert features.shape == (20, 20, 2)


def test_generate_example_normalized():
    attribute_outputs = [
        ContinuousOutput(
            name="a",
            normalization=Normalization.ZERO_ONE,
            global_min=0,
            global_max=1,
            apply_feature_scaling=False,
            apply_example_scaling=False,
        ),
        DiscreteOutput(name="b", dim=3),
        DiscreteOutput(name="c", dim=4),
    ]
    feature_outputs = [
        DiscreteOutput(name="d", dim=4),
        ContinuousOutput(
            name="e",
            normalization=Normalization.ZERO_ONE,
            global_min=0,
            global_max=1,
            apply_feature_scaling=True,
            apply_example_scaling=True,
        ),
    ]

    config = DGANConfig(max_sequence_len=20, sample_len=5)

    dg = DGAN(
        config=config,
        attribute_outputs=attribute_outputs,
        feature_outputs=feature_outputs,
    )
    attributes, features = dg.generate_numpy(8)

    assert attributes.shape == (8, 3)
    assert features.shape == (8, 20, 2)

    attributes, features = dg.generate_numpy(64)

    assert attributes.shape == (64, 3)
    assert features.shape == (64, 20, 2)


@pytest.mark.parametrize(
    "use_attribute_discriminator,is_normalized",
    itertools.product([False, True], [False, True]),
)
def test_train(
    attribute_data,
    feature_data,
    config: DGANConfig,
    use_attribute_discriminator,
    is_normalized,
):
    attributes, attribute_types = attribute_data
    features, feature_types = feature_data

    config.use_attribute_discriminator = use_attribute_discriminator
    config.apply_example_scaling = is_normalized
    config.apply_feature_scaling = is_normalized

    dg = DGAN(config=config)

    dg.train_numpy(
        attributes=attributes,
        features=features,
        attribute_types=attribute_types,
        feature_types=feature_types,
    )

    attributes, features = dg.generate_numpy(18)

    assert attributes.shape == (18, 2)
    assert features.shape == (18, 20, 2)


def test_train_dataframe(config: DGANConfig):
    n = 500
    df = pd.DataFrame(
        {
            "a1": np.random.randint(0, 3, size=n),
            "a2": np.random.rand(n),
            "2022-01-01": np.random.rand(n),
            "2022-02-01": np.random.rand(n),
            "2022-03-01": np.random.rand(n),
            "2022-04-01": np.random.rand(n),
        }
    )

    config.max_sequence_len = 4
    config.sample_len = 1

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_attribute_columns=["a1", "a2"],
        attribute_types=[OutputType.DISCRETE, OutputType.CONTINUOUS],
    )

    synthetic_df = dg.generate_dataframe(5)

    assert synthetic_df.shape == (5, 6)
    assert list(synthetic_df.columns) == list(df.columns)


def test_extract_from_dataframe(config):
    df = pd.DataFrame(
        {
            "a1": np.random.randint(0, 3, size=6),
            "a2": np.random.rand(6),
            "2022-01-01": np.random.rand(6),
            "2022-02-01": np.random.rand(6),
            "2022-03-01": np.random.rand(6),
            "2022-04-01": np.random.rand(6),
        }
    )

    dg = DGAN(config=config)

    attributes, features = dg._extract_from_dataframe(
        df, attribute_columns=["a1", "a2"]
    )

    assert attributes.shape == (6, 2)
    assert features.shape == (6, 4, 1)

    attributes, features = dg._extract_from_dataframe(
        df,
        attribute_columns=["a1"],
        feature_columns=["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"],
    )

    assert attributes.shape == (6, 1)
    assert features.shape == (6, 4, 1)

    attributes, features = dg._extract_from_dataframe(
        df,
        attribute_columns=[],
        feature_columns=["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"],
    )

    assert attributes.shape == (6, 0)
    assert features.shape == (6, 4, 1)


@pytest.mark.parametrize(
    "use_attribute_discriminator,apply_example_scaling,noise_dim,sample_len",
    itertools.product([False, True], [False, True], [10, 25], [2, 5]),
)
def test_save_and_load(
    attribute_data,
    feature_data,
    config: DGANConfig,
    tmp_path,
    use_attribute_discriminator,
    apply_example_scaling,
    noise_dim,
    sample_len,
):
    attributes, attribute_types = attribute_data
    features, feature_types = feature_data

    config.epochs = 1
    config.use_attribute_discriminator = use_attribute_discriminator
    config.apply_example_scaling = apply_example_scaling
    config.attribute_noise_dim = noise_dim
    config.feature_noise_dim = noise_dim
    config.sample_len = sample_len

    dg = DGAN(config=config)

    dg.train_numpy(
        attributes=attributes,
        features=features,
        attribute_types=attribute_types,
        feature_types=feature_types,
    )

    n = 25
    attribute_noise = dg.attribute_noise_func(n)
    feature_noise = dg.feature_noise_func(n)

    expected_attributes, expected_features = dg.generate_numpy(
        attribute_noise=attribute_noise, feature_noise=feature_noise
    )

    file_name = str(tmp_path / "model.pt")
    dg.save(file_name)

    loaded_dg = DGAN.load(file_name)

    attributes, features = loaded_dg.generate_numpy(
        attribute_noise=attribute_noise, feature_noise=feature_noise
    )

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)
