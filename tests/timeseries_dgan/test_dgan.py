import itertools
import os.path

import numpy as np
import pandas as pd
import pytest

from gretel_synthetics.timeseries_dgan.config import (
    DfStyle,
    DGANConfig,
    Normalization,
    OutputType,
)
from gretel_synthetics.timeseries_dgan.dgan import (
    _DataFrameConverter,
    _LongDataFrameConverter,
    _WideDataFrameConverter,
    DGAN,
)
from gretel_synthetics.timeseries_dgan.transformations import (
    ContinuousOutput,
    DiscreteOutput,
)
from pandas.testing import assert_frame_equal


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


def test_train_numpy_batch_size_of_1(config: DGANConfig):
    # Check model trains when (# of examples) % batch_size == 1.

    config.batch_size = 10
    config.epochs = 1

    features = np.random.rand(91, 20, 2)
    attributes = np.random.randint(0, 3, (91, 1))

    model = DGAN(config=config)
    model.train_numpy(
        features=features,
        attributes=attributes,
        feature_types=[OutputType.CONTINUOUS] * 2,
        attribute_types=[OutputType.DISCRETE],
    )

    synthetic_attributes, synthetic_features = model.generate_numpy(11)
    assert synthetic_attributes is not None
    assert synthetic_attributes.shape == (11, 1)
    assert synthetic_features.shape == (11, 20, 2)


def test_train_dataframe_wide(config: DGANConfig):
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
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
        df_style=DfStyle.WIDE,
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
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
        df_style=DfStyle.WIDE,
    )

    # We want to confirm the training does update the model params, so we create
    # some fixed noise inputs and check if they produce different outputs before
    # and after some more training.
    attribute_noise = dg.attribute_noise_func(50)
    feature_noise = dg.feature_noise_func(50)
    before_attributes, before_features = dg.generate_numpy(
        attribute_noise=attribute_noise, feature_noise=feature_noise
    )

    dg.train_dataframe(
        df=df,
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
        df_style=DfStyle.WIDE,
    )

    after_attributes, after_features = dg.generate_numpy(
        attribute_noise=attribute_noise, feature_noise=feature_noise
    )
    # Generated data should be different.
    assert np.any(np.not_equal(before_attributes, after_attributes))
    assert np.any(np.not_equal(before_features, after_features))

    synthetic_df = dg.generate_dataframe(5)
    assert synthetic_df.shape == (5, 6)
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_1_example(config: DGANConfig, feature_data):
    features, feature_types = feature_data
    # Keep 1 example
    features = features[0:1, :]

    dg = DGAN(config=config)

    with pytest.raises(ValueError, match="multiple examples to train"):
        dg.train_numpy(features=features, feature_types=feature_types)


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
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
        df_style=DfStyle.WIDE,
    )

    synthetic_df = dg.generate_dataframe(5)
    assert synthetic_df.shape == (5, 6)
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_dataframe_wide_no_attributes(config: DGANConfig):
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
    dg.train_dataframe(df=df, df_style=DfStyle.WIDE)

    assert type(dg) == DGAN

    n_samples = 5
    synthetic_df = dg.generate_dataframe(n_samples)

    assert synthetic_df.shape == (n_samples, len(df.columns))
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_dataframe_long(config: DGANConfig):
    n = 500
    df = pd.DataFrame(
        {
            "example_id": np.repeat(range(n), 4),
            "a1": np.repeat(np.random.randint(0, 3, size=n), 4),
            "a2": np.repeat(np.random.rand(n), 4),
            "f1": np.random.rand(4 * n),
            "f2": np.random.rand(4 * n),
        }
    )

    config.max_sequence_len = 4
    config.sample_len = 2

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        attribute_columns=["a1", "a2"],
        example_id_column="example_id",
        discrete_columns=["a1"],
        df_style=DfStyle.LONG,
    )

    synthetic_df = dg.generate_dataframe(5)

    assert synthetic_df.shape == (5 * 4, 5)
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_dataframe_long_no_attributes(config: DGANConfig):
    n = 500
    df = pd.DataFrame(
        {
            "example_id": np.repeat(range(n), 4),
            "f1": np.random.rand(4 * n),
            "f2": np.random.rand(4 * n),
        }
    )

    config.max_sequence_len = 4
    config.sample_len = 4

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        example_id_column="example_id",
        df_style=DfStyle.LONG,
    )

    synthetic_df = dg.generate_dataframe(5)

    assert synthetic_df.shape == (5 * 4, 3)
    assert list(synthetic_df.columns) == list(df.columns)


def test_train_numpy_nans(config: DGANConfig, feature_data):
    features, feature_types = feature_data
    # Insert a NaN
    features[11, 3, 1] = np.NaN

    dg = DGAN(config=config)

    with pytest.raises(ValueError, match="NaN"):
        dg.train_numpy(features=features, feature_types=feature_types)


def test_train_dataframe_nans(config: DGANConfig):
    n = 50
    df = pd.DataFrame(
        {
            "2022-01-01": np.random.rand(n),
            "2022-02-01": np.NaN,
            "2022-03-01": np.random.rand(n),
            "2022-04-01": np.random.rand(n),
        }
    )

    config.max_sequence_len = 4
    config.sample_len = 1

    dg = DGAN(config=config)
    with pytest.raises(ValueError, match="NaN"):
        dg.train_dataframe(df=df, df_style=DfStyle.WIDE)


@pytest.fixture
def df_wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a1": [1, 2, 2, 0, 0, 1],
            "a2": [5.0, 2.5, -1.0, 3.0, 2.0, 1.0],
            "2022-01-01": [1.0, 1.0, 1.0, 2.0, 2.0, 5.0],
            "2022-02-01": [2.0, 3.0, 4.0, 3.0, 2.0, -1.0],
            "2022-03-01": [2.5, 4.0, 5.0, 1.5, 2.0, 0.0],
        }
    )


def test_wide_data_frame_converter1(df_wide):
    expected_attributes = [
        [1, 5.0],
        [2, 2.5],
        [2, -1.0],
        [0, 3.0],
        [0, 2.0],
        [1, 1.0],
    ]
    expected_features = [
        [[1.0], [2.0], [2.5]],
        [[1.0], [3.0], [4.0]],
        [[1.0], [4.0], [5.0]],
        [[2.0], [3.0], [1.5]],
        [[2.0], [2.0], [2.0]],
        [[5.0], [-1.0], [0.0]],
    ]
    converter = _WideDataFrameConverter.create(
        df_wide, attribute_columns=["a1", "a2"], discrete_columns=["a1"]
    )
    attributes, features = converter.convert(df_wide)

    assert attributes.shape == (6, 2)
    assert features.shape == (6, 3, 1)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check invert produces original df
    df_out = converter.invert(attributes, features)

    assert_frame_equal(df_out, df_wide)


def test_wide_data_frame_converter2(df_wide):
    expected_attributes = [
        [1],
        [2],
        [2],
        [0],
        [0],
        [1],
    ]
    expected_features = [
        [[1.0], [2.0], [2.5]],
        [[1.0], [3.0], [4.0]],
        [[1.0], [4.0], [5.0]],
        [[2.0], [3.0], [1.5]],
        [[2.0], [2.0], [2.0]],
        [[5.0], [-1.0], [0.0]],
    ]

    converter = _WideDataFrameConverter.create(
        df_wide,
        attribute_columns=["a1"],
        feature_columns=["2022-01-01", "2022-02-01", "2022-03-01"],
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_wide)

    assert attributes.shape == (6, 1)
    assert features.shape == (6, 3, 1)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check invert produces original df
    df_out = converter.invert(attributes, features)

    assert_frame_equal(df_out, df_wide.drop(columns=["a2"]))


def test_wide_data_frame_converter_no_attributes(df_wide):
    expected_features = [
        [[1.0], [2.0], [2.5]],
        [[1.0], [3.0], [4.0]],
        [[1.0], [4.0], [5.0]],
        [[2.0], [3.0], [1.5]],
        [[2.0], [2.0], [2.0]],
        [[5.0], [-1.0], [0.0]],
    ]

    converter = _WideDataFrameConverter.create(
        df_wide,
        attribute_columns=[],
        feature_columns=["2022-01-01", "2022-02-01", "2022-03-01"],
    )
    attributes, features = converter.convert(df_wide)

    assert attributes == None
    assert features.shape == (6, 3, 1)

    np.testing.assert_allclose(features, expected_features)

    # Check invert produces original df
    df_out = converter.invert(attributes, features)

    assert_frame_equal(df_out, df_wide.drop(columns=["a1", "a2"]))


def test_wide_data_frame_converter_no_attributes_no_column_name(df_wide):
    df_wide.drop(columns=["a1", "a2"], inplace=True)

    expected_features = [
        [[1.0], [2.0], [2.5]],
        [[1.0], [3.0], [4.0]],
        [[1.0], [4.0], [5.0]],
        [[2.0], [3.0], [1.5]],
        [[2.0], [2.0], [2.0]],
        [[5.0], [-1.0], [0.0]],
    ]

    converter = _WideDataFrameConverter.create(df_wide)
    attributes, features = converter.convert(df_wide)

    assert attributes == None
    assert features.shape == (6, 3, 1)

    np.testing.assert_allclose(features, expected_features)

    # Check invert produces original df
    df_out = converter.invert(attributes, features)

    assert_frame_equal(df_out, df_wide)


def test_wide_data_frame_converter_save_and_load(df_wide):
    converter = _WideDataFrameConverter.create(
        df_wide,
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
    )

    expected_attributes, expected_features = converter.convert(df_wide)

    expected_df = converter.invert(expected_attributes, expected_features)

    state = converter.state_dict()

    loaded_converter = _DataFrameConverter.load_from_state_dict(state)

    attributes, features = loaded_converter.convert(df_wide)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    df = loaded_converter.invert(attributes, features)

    assert_frame_equal(df, expected_df)


@pytest.fixture
def df_long() -> pd.DataFrame:
    # Add an hour to the times so sorting just by the time column has a total
    # ordering and is deterministic.
    return pd.DataFrame(
        {
            "example_id": [0, 0, 0, 1, 1, 1],
            "time": [
                "2022-01-01 00",
                "2022-01-03 00",
                "2022-01-02 00",
                "2022-01-01 01",
                "2022-01-02 01",
                "2022-01-03 01",
            ],
            "a1": [10, 10, 10, 11, 11, 11],
            "a2": [1.5, 1.5, 1.5, 3.3, 3.3, 3.3],
            "f1": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "f2": [51.0, 50.0, 49.0, 48.0, 47.0, 46.0],
            "f3": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_long_data_frame_converter1(df_long):
    # All column params are given
    # Time column reorders some rows

    expected_attributes = [
        [10, 1.5],
        [11, 3.3],
    ]
    expected_features = [
        [[101.0, 51.0, 0.0], [103.0, 49.0, 0.0], [102.0, 50.0, 0.0]],
        [[104.0, 48.0, 0.0], [105.0, 47.0, 0.0], [106.0, 46.0, 0.0]],
    ]

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (2, 2)
    assert features.shape == (2, 3, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        example_id_column="example_id",
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)
    assert attributes.shape == (2, 2)
    assert features.shape == (2, 3, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check the inverse returns the original df
    df_out = converter.invert(attributes, features)

    # We intentionally reordered rows based on time, so need to sort original
    # before comparing. Time column is not preserved, so remove it before
    # comparing.
    assert_frame_equal(
        df_out.drop(columns=["time"]),
        df_long.sort_values(by=["example_id", "time"])
        .drop(columns=["time"])
        .reset_index(drop=True),
    )


def test_long_data_frame_converter2(df_long):
    # No time column, use order in input dataframe
    df_long = df_long.drop(columns=["time"])

    # Ignore time column, use original ordering in dataframe
    expected_attributes = [
        [10, 1.5],
        [11, 3.3],
    ]
    expected_features = [
        [[101.0, 51.0, 0.0], [102.0, 50.0, 0.0], [103.0, 49.0, 0.0]],
        [[104.0, 48.0, 0.0], [105.0, 47.0, 0.0], [106.0, 46.0, 0.0]],
    ]

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (2, 2)
    assert features.shape == (2, 3, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (2, 2)
    assert features.shape == (2, 3, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check the inverse returns the original df
    df_out = converter.invert(attributes, features)

    assert_frame_equal(df_out, df_long)


def test_long_data_frame_converter3(df_long):
    # No example id column, extract as 1 example
    # Time column reorders some rows
    df_long = df_long.drop(columns=["example_id"])
    # Make attributes constant since we have 1 example
    df_long["a1"] = 10
    df_long["a2"] = 1.5

    expected_attributes = [
        [10, 1.5],
    ]
    expected_features = [
        [
            [101.0, 51.0, 0.0],
            [104.0, 48.0, 0.0],
            [103.0, 49.0, 0.0],
            [105.0, 47.0, 0.0],
            [102.0, 50.0, 0.0],
            [106.0, 46.0, 0.0],
        ],
    ]

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (1, 2)
    assert features.shape == (1, 6, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (1, 2)
    assert features.shape == (1, 6, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check the inverse returns the original df
    df_out = converter.invert(attributes, features)

    # We intentionally reordered rows based on time, so need to sort original
    # before comparing. We can keep and compare the time column though since we
    # should preserve values exactly for the "1 example".
    assert_frame_equal(df_out, df_long.sort_values(by="time").reset_index(drop=True))


def test_long_data_frame_converter4(config, df_long):
    # No example id or time column, extract as 1 example with original order

    df_long = df_long.drop(columns=["example_id", "time"])
    # Make attributes constant since we have 1 example
    df_long["a1"] = 10
    df_long["a2"] = 1.5

    expected_attributes = [
        [10, 1.5],
    ]
    expected_features = [
        [
            [101.0, 51.0, 0.0],
            [102.0, 50.0, 0.0],
            [103.0, 49.0, 0.0],
            [104.0, 48.0, 0.0],
            [105.0, 47.0, 0.0],
            [106.0, 46.0, 0.0],
        ],
    ]

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (1, 2)
    assert features.shape == (1, 6, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (1, 2)
    assert features.shape == (1, 6, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check the inverse returns the original df
    df_out = converter.invert(attributes, features)

    assert_frame_equal(df_out, df_long)


def test_long_data_frame_converter_extra_cols(df_long):
    # Check if converter works if there are extra columns
    # in the input df
    df_long["random1"] = "foo"
    df_long["random2"] = range(len(df_long))

    expected_attributes = [
        [10, 1.5],
        [11, 3.3],
    ]
    expected_features = [
        [[101.0, 51.0, 0.0], [102.0, 50.0, 0.0], [103.0, 49.0, 0.0]],
        [[104.0, 48.0, 0.0], [105.0, 47.0, 0.0], [106.0, 46.0, 0.0]],
    ]

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert attributes.shape == (2, 2)
    assert features.shape == (2, 3, 3)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    # Check the inverse returns the original df
    df_out = converter.invert(attributes, features)

    # Remove time and random columns from original
    assert_frame_equal(
        df_out,
        df_long.drop(columns=["time", "random1", "random2"]),
    )


def test_long_data_frame_converter_attribute_errors(df_long):
    df_long = df_long.drop(columns="time")  # Don't need time for these tests

    # Insert attribute value that is not equal for all rows with same example id
    df_long.loc[2, "a1"] = 42.0

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    with pytest.raises(ValueError):
        converter.convert(df_long)

    # Same if we don't use example_id where attributes should be constant
    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        discrete_columns=["a1"],
    )
    with pytest.raises(ValueError):
        converter.convert(df_long)


def test_long_data_frame_converter_mixed_feature_types(df_long):
    # Check int and float types for dataframe work and produce
    # features array with float64 type

    df_long = df_long.drop(columns="time")

    # Replace f2 with another discrete column
    df_long["f2"] = df_long["a1"]
    assert df_long["f2"].dtype == "int64"

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
    )

    _, features = converter.convert(df_long)
    assert features.dtype == "float64"


def test_long_data_frame_converter_example_id_object(df_long):
    # Check dtype of features and attributes is float when example id column is
    # of object type like strings

    df_long = df_long.drop(columns="time")

    # Make example id column a string
    df_long["example_id"] = df_long["example_id"].astype("str")
    assert df_long["example_id"].dtype == "object"

    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
    )

    attributes, features = converter.convert(df_long)
    assert attributes is not None
    assert attributes.dtype == "float64"
    assert features.dtype == "float64"


def test_long_data_frame_converter_save_and_load(df_long):
    converter = _LongDataFrameConverter.create(
        df_long,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        time_column="time",
        discrete_columns=["a1"],
    )

    expected_attributes, expected_features = converter.convert(df_long)

    expected_df = converter.invert(expected_attributes, expected_features)

    state = converter.state_dict()

    loaded_converter = _DataFrameConverter.load_from_state_dict(state)

    attributes, features = loaded_converter.convert(df_long)

    np.testing.assert_allclose(attributes, expected_attributes)
    np.testing.assert_allclose(features, expected_features)

    df = loaded_converter.invert(attributes, features)

    assert_frame_equal(df, expected_df)


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
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
        df_style=DfStyle.WIDE,
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
        df_style=DfStyle.WIDE,
    )
    file_name = str(tmp_path / "model.pt")
    dg.save(file_name)

    assert os.path.isfile(file_name)

    loaded_dg = dg.load(file_name)
    synthetic_df = loaded_dg.generate_dataframe(10)

    assert type(loaded_dg) == DGAN
    assert list(synthetic_df.columns) == list(df.columns)
