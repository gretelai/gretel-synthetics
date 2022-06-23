import itertools
import os.path

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
        epochs=10,
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
def test_train_numpy(
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


@pytest.mark.parametrize(
    "use_attribute_discriminator,is_normalized",
    itertools.product([False, True], [False, True]),
)
def test_train_numpy_no_attributes_1(
    feature_data,
    config: DGANConfig,
    use_attribute_discriminator,
    is_normalized,
):
    features, feature_types = feature_data

    config.use_attribute_discriminator = use_attribute_discriminator
    config.apply_example_scaling = is_normalized
    config.apply_feature_scaling = is_normalized

    dg = DGAN(config=config)

    dg.train_numpy(
        features=features,
        feature_types=feature_types,
    )

    attributes, features = dg.generate_numpy(18)

    assert attributes == None
    assert features.shape == (18, 20, 2)


def test_train_dataframe(config: DGANConfig):
    n = 50
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


def test_train_dataframe_batch_size_larger_than_dataset(config: DGANConfig):
    n = 50
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
    config.batch_size = 1000

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_attribute_columns=["a1", "a2"],
        attribute_types=[OutputType.DISCRETE, OutputType.CONTINUOUS],
    )

    synthetic_df = dg.generate_dataframe(5)
    assert synthetic_df.shape == (5, 6)
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_dataframe_batch_size_not_divisible_by_dataset_length(config: DGANConfig):
    n = 1000
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
    config.sample_len = 2
    config.batch_size = 300
    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_attribute_columns=["a1", "a2"],
        attribute_types=[OutputType.DISCRETE, OutputType.CONTINUOUS],
    )

    synthetic_df = dg.generate_dataframe(5)
    assert synthetic_df.shape == (5, 6)
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_dataframe_no_attributes(config: DGANConfig):
    n = 50
    df = pd.DataFrame(
        {
            "2022-01-01": np.random.rand(n),
            "2022-02-01": np.random.rand(n),
            "2022-03-01": np.random.rand(n),
            "2022-04-01": np.random.rand(n),
        }
    )

    config.max_sequence_len = 4
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)
    dg.train_dataframe(df=df)

    assert type(dg) == DGAN

    n_samples = 5
    synthetic_df = dg.generate_dataframe(n_samples)

    assert synthetic_df.shape == (n_samples, len(df.columns))
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_numpy_no_attributes_2(config: DGANConfig):
    features = np.random.rand(100, 20, 2)
    n_samples = 10
    config.epochs = 1
    model_attributes_blank = DGAN(config=config)
    model_attributes_blank.train_numpy(features=features)
    synthetic_attributes, synthetic_features = model_attributes_blank.generate_numpy(
        n_samples
    )

    assert type(model_attributes_blank) == DGAN
    assert synthetic_attributes == None
    assert synthetic_features.shape == (n_samples, features.shape[1], features.shape[2])

    model_attributes_none = DGAN(config)
    model_attributes_none.train_numpy(attributes=None, features=features)
    synthetic_attributes, synthetic_features = model_attributes_none.generate_numpy(
        n_samples
    )

    assert type(model_attributes_none) == DGAN
    assert synthetic_attributes == None
    assert synthetic_features.shape == (n_samples, features.shape[1], features.shape[2])


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


def test_extract_from_dataframe_no_attributes(config):
    dg = DGAN(config=config)
    df = pd.DataFrame(
        {
            "2022-01-01": np.random.rand(6),
            "2022-02-01": np.random.rand(6),
            "2022-03-01": np.random.rand(6),
        }
    )
    attributes, features = dg._extract_from_dataframe(
        df,
        feature_columns=["2022-01-01", "2022-02-01", "2022-03-01"],
    )

    assert attributes == None
    assert features.shape == (6, 3, 1)


def test_extract_from_dataframe_no_attributes_no_column_name(config):
    dg = DGAN(config=config)
    config.max_sequence_len = 3
    config.sample_len = 1
    config.epochs = 1

    df = pd.DataFrame(
        {
            "2022-01-01": np.random.rand(6),
            "2022-02-01": np.random.rand(6),
            "2022-03-01": np.random.rand(6),
        }
    )
    attributes, features = dg._extract_from_dataframe(
        df,
    )

    assert attributes == None
    assert features.shape == (6, 3, 1)


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


@pytest.mark.parametrize(
    "use_attribute_discriminator,apply_example_scaling,noise_dim,sample_len",
    itertools.product([False, True], [False, True], [10, 25], [2, 5]),
)
def test_save_and_load_no_attributes(
    feature_data,
    config: DGANConfig,
    tmp_path,
    use_attribute_discriminator,
    apply_example_scaling,
    noise_dim,
    sample_len,
):
    features, feature_types = feature_data

    config.epochs = 1
    config.use_attribute_discriminator = use_attribute_discriminator
    config.apply_example_scaling = apply_example_scaling
    config.attribute_noise_dim = noise_dim
    config.feature_noise_dim = noise_dim
    config.sample_len = sample_len

    dg = DGAN(config=config)

    dg.train_numpy(
        features=features,
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

    assert attributes == expected_attributes == None
    assert features.shape == expected_features.shape
    np.testing.assert_allclose(features, expected_features)


def test_save_and_load_dataframe_with_attributes(config: DGANConfig, tmp_path):
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
    config.max_sequence_len = 4
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_attribute_columns=["a1", "a2"],
        attribute_types=[OutputType.DISCRETE, OutputType.CONTINUOUS],
    )
    file_name = str(tmp_path / "model.pt")
    dg.save(file_name)

    assert os.path.isfile(file_name)

    loaded_dg = dg.load(file_name)
    synthetic_df = loaded_dg.generate_dataframe(10)

    assert type(loaded_dg) == DGAN
    assert list(synthetic_df.columns) == list(df.columns)


def test_save_and_load_dataframe_no_attributes(config: DGANConfig, tmp_path):
    df = pd.DataFrame(
        {
            "2022-01-01": np.random.rand(6),
            "2022-02-01": np.random.rand(6),
            "2022-03-01": np.random.rand(6),
        }
    )

    config.max_sequence_len = 3
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
    )
    file_name = str(tmp_path / "model.pt")
    dg.save(file_name)

    assert os.path.isfile(file_name)

    loaded_dg = dg.load(file_name)
    synthetic_df = loaded_dg.generate_dataframe(10)

    assert type(loaded_dg) == DGAN
    assert list(synthetic_df.columns) == list(df.columns)
