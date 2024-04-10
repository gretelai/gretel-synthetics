import itertools
import os.path

from collections import Counter
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytest

from pandas.api.types import is_numeric_dtype, is_object_dtype
from pandas.testing import assert_frame_equal

from gretel_synthetics.errors import DataError, ParameterError
from gretel_synthetics.timeseries_dgan.config import (
    DfStyle,
    DGANConfig,
    Normalization,
    OutputType,
)
from gretel_synthetics.timeseries_dgan.dgan import (
    _DataFrameConverter,
    _discrete_cols_to_int,
    _LongDataFrameConverter,
    _WideDataFrameConverter,
    DGAN,
    find_max_consecutive_nans,
    nan_linear_interpolation,
    validation_check,
)
from gretel_synthetics.timeseries_dgan.transformations import (
    BinaryEncodedOutput,
    ContinuousOutput,
    OneHotEncodedOutput,
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


def assert_attributes_features_shape(
    attributes: Optional[np.ndarray],
    features: list[np.ndarray],
    attributes_shape: Optional[tuple[int, int]],
    features_shape: tuple[int, int, int],
):

    if attributes_shape:
        assert attributes is not None
        assert attributes.shape == attributes_shape

    assert len(features) == features_shape[0]
    assert all(seq.shape == features_shape[1:] for seq in features)


def assert_attributes_features(
    attributes: Optional[np.ndarray],
    features: list[np.ndarray],
    expected_attributes: Optional[Union[np.ndarray, Sequence[Sequence[Any]]]],
    expected_features: Union[
        np.ndarray, list[np.ndarray], Sequence[Sequence[Sequence[Any]]]
    ],
):
    if expected_attributes is not None:
        assert attributes is not None
        np.testing.assert_allclose(attributes, expected_attributes)

    for f, ef in zip(features, expected_features):
        np.testing.assert_allclose(f, ef)


def test_discrete_cols_to_int():
    df = pd.DataFrame(
        data=zip(["1", "2", "3", "4"], ["one", "two", "three", "four"]),
        columns=["value", "str"],
    )

    assert df.equals(_discrete_cols_to_int(df.copy(), None))
    assert df.equals(_discrete_cols_to_int(df.copy(), ["missing"]))

    check_df = _discrete_cols_to_int(df.copy(), ["value", "str", "missing"])
    assert is_numeric_dtype(check_df["value"])
    assert is_object_dtype(check_df["str"])


def test_generate():
    attribute_outputs = [
        ContinuousOutput(
            name="a",
            normalization=Normalization.ZERO_ONE,
            apply_feature_scaling=False,
            apply_example_scaling=False,
            global_min=0.0,
            global_max=1.0,
        ),
        OneHotEncodedOutput(name="b", dim=3),
        BinaryEncodedOutput(name="c", dim=4),
    ]
    feature_outputs = [
        OneHotEncodedOutput(name="d", dim=4),
        ContinuousOutput(
            name="e",
            normalization=Normalization.ZERO_ONE,
            apply_feature_scaling=False,
            apply_example_scaling=False,
            global_min=0.0,
            global_max=1.0,
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

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(8, 3), features_shape=(8, 20, 2)
    )

    attributes, features = dg.generate_numpy(64)

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(64, 3), features_shape=(64, 20, 2)
    )

    attributes, features = dg.generate_numpy(200)

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(200, 3), features_shape=(200, 20, 2)
    )

    attributes, features = dg.generate_numpy(1)

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(1, 3), features_shape=(1, 20, 2)
    )

    # Check passing noise vectors

    attributes, features = dg.generate_numpy(
        attribute_noise=dg.attribute_noise_func(20),
        feature_noise=dg.feature_noise_func(20),
    )
    assert_attributes_features_shape(
        attributes, features, attributes_shape=(20, 3), features_shape=(20, 20, 2)
    )


def test_generate_example_normalized():
    attribute_outputs = [
        ContinuousOutput(
            name="a",
            normalization=Normalization.ZERO_ONE,
            apply_feature_scaling=False,
            apply_example_scaling=False,
            global_min=0.0,
            global_max=1.0,
        ),
        OneHotEncodedOutput(name="b", dim=3),
        BinaryEncodedOutput(name="c", dim=4),
    ]
    feature_outputs = [
        OneHotEncodedOutput(name="d", dim=4),
        ContinuousOutput(
            name="e",
            normalization=Normalization.ZERO_ONE,
            apply_feature_scaling=True,
            apply_example_scaling=True,
            global_min=0.0,
            global_max=1.0,
        ),
    ]

    config = DGANConfig(max_sequence_len=20, sample_len=5)

    dg = DGAN(
        config=config,
        attribute_outputs=attribute_outputs,
        feature_outputs=feature_outputs,
    )
    attributes, features = dg.generate_numpy(8)

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(8, 3), features_shape=(8, 20, 2)
    )

    attributes, features = dg.generate_numpy(64)

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(64, 3), features_shape=(64, 20, 2)
    )


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

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(18, 2), features_shape=(18, 20, 2)
    )


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

    assert_attributes_features_shape(
        attributes, features, attributes_shape=None, features_shape=(18, 20, 2)
    )


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
    assert_attributes_features_shape(
        synthetic_attributes,
        synthetic_features,
        attributes_shape=None,
        features_shape=(n_samples, features.shape[1], features.shape[2]),
    )

    model_attributes_none = DGAN(config)
    model_attributes_none.train_numpy(attributes=None, features=features)
    synthetic_attributes, synthetic_features = model_attributes_none.generate_numpy(
        n_samples
    )

    assert type(model_attributes_none) == DGAN
    assert_attributes_features_shape(
        synthetic_attributes,
        synthetic_features,
        attributes_shape=None,
        features_shape=(n_samples, features.shape[1], features.shape[2]),
    )


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

    assert_attributes_features_shape(
        synthetic_attributes,
        synthetic_features,
        attributes_shape=(11, 1),
        features_shape=(11, 20, 2),
    )


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


def test_build(config: DGANConfig):
    dg = DGAN(config=config)

    assert dg.is_built == False
    with pytest.raises(
        RuntimeError, match="Must build DGAN model prior to generating samples."
    ):
        dg.generate_numpy(1)


def test_train_1_example(config: DGANConfig, feature_data):
    features, feature_types = feature_data
    # Keep 1 example
    features = features[0:1, :]

    dg = DGAN(config=config)

    with pytest.raises(DataError, match="multiple examples to train") as e:
        dg.train_numpy(features=features, feature_types=feature_types)

    # ensure that the exception can be caught using Python build-in ones as well
    assert isinstance(e.value, ValueError)


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


def test_example_id_must_be_unique(config: DGANConfig):
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

    with pytest.raises(ParameterError) as err:
        dg.train_dataframe(
            df=df,
            attribute_columns=["a1", "a2"],
            example_id_column="example_id",
            discrete_columns=["a1", "example_id"],
            df_style=DfStyle.LONG,
        )
    assert "any other column lists" in str(err.value)


def test_time_col_example_id_col_not_equal(config: DGANConfig):
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

    # The `time_column` here is nonsense but just to validate that
    # both that and the example ID column can't be the same
    with pytest.raises(ParameterError) as err:
        dg.train_dataframe(
            df=df,
            attribute_columns=["a1", "a2"],
            example_id_column="example_id",
            time_column="example_id",
            discrete_columns=["a1"],
            df_style=DfStyle.LONG,
        )
    assert "values cannot be the same" in str(err.value)
    # ensure that the exception can be caught using Python build-in ones as well
    assert isinstance(err.value, ValueError)


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


def test_train_dataframe_long_no_attributes_no_example_id(config: DGANConfig):
    # Checking functionality of autosplit when no example id, but attributes are provided
    # by the user. This test should catch the exception thrown by the function that tells
    # the user that autosplitting is not available and that they need to provide an example
    # id column.

    n = 250
    df = pd.DataFrame(
        {
            "a1": np.repeat(np.random.randint(0, 10, size=n), 6),
            "a2": np.repeat(np.random.rand(n), 6),
            "a3": np.repeat(np.random.rand(n), 6),
            "f1": np.random.rand(6 * n),
            "f2": np.random.rand(6 * n),
            "f3": np.random.rand(6 * n),
        }
    )

    config.max_sequence_len = 6
    config.sample_len = 2

    dg = DGAN(config=config)

    with pytest.raises(ParameterError) as exc_info:
        dg.train_dataframe(
            df=df,
            attribute_columns=["a1", "a2"],
            discrete_columns=["a1"],
            df_style=DfStyle.LONG,
        )

    assert "auto-splitting not available" in str(exc_info.value)


def test_train_dataframe_long_no_attributes_no_example_id_with_time(config: DGANConfig):
    # Checking functionality of autosplit when no example id and no attributes, but the
    # time column is provided by the user

    n = 250
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="1/1/2018", periods=3 * n).tolist(),
            "f1": np.random.rand(3 * n),
            "f2": np.random.rand(3 * n),
        }
    )

    config.max_sequence_len = 6
    config.sample_len = 2

    dg = DGAN(config=config)

    dg.train_dataframe(df=df, df_style=DfStyle.LONG, time_column="date")

    synthetic_df = dg.generate_dataframe(5)

    assert synthetic_df.shape == (6 * 5, 4)
    assert list(synthetic_df.columns)[-1] == "example_id"
    assert list(synthetic_df.columns)[0] == "date"
    assert synthetic_df["example_id"].value_counts()[0] == config.max_sequence_len


def test_train_dataframe_long_no_attributes_no_example_id_auto_split(
    config: DGANConfig,
):
    # Checking functionality of autosplit when no example id and no attributes are provided
    # by the user

    n = 500
    df = pd.DataFrame(
        {
            "f1": np.random.rand(4 * n),
            "f2": np.random.rand(4 * n),
        }
    )

    config.max_sequence_len = 6
    config.sample_len = 2

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_style=DfStyle.LONG,
    )

    synthetic_df = dg.generate_dataframe(5)

    assert synthetic_df.shape == (6 * 5, 3)
    assert list(synthetic_df.columns)[-1] == "example_id"
    assert synthetic_df["example_id"].value_counts()[0] == config.max_sequence_len


def test_find_max_consecutive_nans():
    # Checking the output of the "find_max_consecutive_nans" function.
    # We create a 1-d random array, insert nans in different locations and lengths,
    # testing the maximum consecutive nans in the data.
    n = 50
    features = np.random.rand(n)
    features[0:5] = features[7:21] = features[30:40] = features[-2:] = np.nan

    assert find_max_consecutive_nans(features) == 14

    features = np.random.rand(n)
    features[0:12] = features[20:22] = features[-3:] = np.nan

    assert find_max_consecutive_nans(features) == 12

    features = np.random.rand(n)
    features[0:8] = features[20:22] = features[-17:] = np.nan

    assert find_max_consecutive_nans(features) == 17


def test_nan_linear_interpolation():
    # Checks the functionality and output of the "nan_linear_interpolation" function.
    # Inserting nans in different length and locations of a 3-d array.
    # np interpolation uses padding for values in the begining and the end of an array.

    features = [
        np.array(
            [[0.0, 1.0, 2.0], [np.nan, 7, 5.0], [np.nan, 4, 8.0], [8.0, 10.0, np.nan]]
        ),
        np.array(
            [
                [np.nan, 13.0, 14.0],
                [np.nan, 16.0, 17.0],
                [18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0],
            ]
        ),
        np.array([[5.0, np.nan, 85.0], [np.nan, 10.0, 80.0], [5.0, 10.0, np.nan]]),
    ]
    # Note, the interpolation is linear if there is a value before and after the
    # section of nans. For nans at the beginning or end of a sequence, the
    # interpolation assumes a diff of 0 and uses the first/last non-nan value as
    # a constant to replace the nans.
    expected_features = [
        np.array(
            [
                [0.0, 1.0, 2.0],
                [8.0 / 3.0, 7.0, 5.0],
                [16.0 / 3.0, 4.0, 8.0],
                [8.0, 10.0, 8.0],
            ]
        ),
        np.array(
            [
                [18.0, 13.0, 14.0],
                [18.0, 16.0, 17.0],
                [18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0],
            ]
        ),
        np.array(
            [[5.0, 10.0, 85.0], [5.0, 10.0, 80.0], [5.0, 10.0, 80.0]],
        ),
    ]

    nan_linear_interpolation(list(features), continuous_features_ind=[0, 1, 2])

    assert all(np.isnan(seq).sum() == 0 for seq in features)

    for f, ef in zip(features, expected_features):
        np.testing.assert_allclose(f, ef)


def test_validation_check():

    # Checking the functionality and output of the validation check for 3
    # scenarios of:
    # 1. Erroring out when invalid records are too high.
    # 2. Dropping invalid records with lower ratio.
    # 3. keeping the fixable valid examples.

    n = 50
    # Set nans for feature 2 , time points 2 and 3, and the first 26 examples. All
    # the examples are considered invalid. The check will raise an error since there are
    # too many invalid examples.
    invalid_examples = np.random.rand(n, 20, 3)
    invalid_examples[0:26, 2:4, 2] = np.nan
    with pytest.raises(DataError, match="NaN"):
        validation_check(list(invalid_examples), continuous_features_ind=[0, 1, 2])

    # Set nans for various features. Features 1 and 2 have fixable invalid examples,
    # while feature 0 has 10 invalid examples which should be dropped (high consecutive nans)
    invalid_examples_dropped = np.random.rand(n, 20, 3)
    invalid_examples_dropped[0:2, 2:3, 2] = np.nan
    invalid_examples_dropped[20:30, 10:20, 0] = np.nan
    invalid_examples_dropped[30:40, 15, 1] = np.nan

    expected = np.array([True] * n)
    expected[20:30] = False
    np.testing.assert_equal(
        validation_check(
            list(invalid_examples_dropped), continuous_features_ind=[0, 1, 2]
        ),
        expected,
    )

    # inserting small number of nans for each feature, non should be dropped during
    # the check.
    valid_examples = np.random.rand(n, 20, 3)
    valid_examples[5:7, 2, 2] = np.nan
    valid_examples[15:20, 15, 0] = np.nan
    valid_examples[-5:, 8, 1] = np.nan
    assert validation_check(
        list(valid_examples), continuous_features_ind=[0, 1, 2]
    ).all()


def test_train_numpy_nans(config: DGANConfig):
    # checking the functionality of the "train_numpy" when including continuous NaNs.
    # Since the interpolation is done before the transformation, we check if no NaNs are
    # generated.

    n = 100
    features = np.concatenate(
        (
            np.random.randint(0, 4, size=(n, 20, 1)),
            np.random.rand(n, 20, 1),
            np.random.rand(n, 20, 1),
        ),
        axis=2,
    )
    feature_types = [OutputType.DISCRETE, OutputType.CONTINUOUS, OutputType.CONTINUOUS]
    # insert sparse NaNs in continuous feature #1.
    features[11, 3, 1] = features[65:73, 17, 1] = np.NaN
    # insert cosecutive NaNs in continuous feature #2.
    features[5:10, 2:4, 2] = features[80:90, 4:10, 2] = np.NaN

    dg = DGAN(config=config)
    dg.train_numpy(features=features, feature_types=feature_types)
    synthetic_attributes, synthetic_features = dg.generate_numpy(50)

    assert synthetic_attributes is None
    assert np.isnan(synthetic_features).sum() == 0


def test_train_dataframe_wide_nans_all_invalid_examples(config: DGANConfig):
    # check the functionality of "train_dataframe_wide" when all examples are invalid.
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
    with pytest.raises(DataError, match="NaN"):
        dg.train_dataframe(df=df, df_style=DfStyle.WIDE)


def test_train_dataframe_wide_nans_some_valid_examples(config: DGANConfig):
    # check the functionality of "train_dataframe_wide" when some examples are NaNs.
    n = 50
    df = pd.DataFrame(
        {
            "2022-01-01": np.random.rand(n),
            "2022-02-01": np.random.rand(n),
            "2022-03-01": np.random.rand(n),
            "2022-04-01": np.random.rand(n),
            "2022-05-01": np.random.rand(n),
            "2022-06-01": np.random.rand(n),
            "2022-07-01": np.random.rand(n),
            "2022-08-01": np.random.rand(n),
            "2022-09-01": np.random.rand(n),
            "2022-10-01": np.random.rand(n),
            "2022-11-01": np.random.rand(n),
            "2022-12-01": np.nan,
        }
    )

    config.max_sequence_len = 12
    config.sample_len = 1

    dg = DGAN(config=config)
    dg.train_dataframe(df=df, df_style=DfStyle.WIDE)
    synthetic_df = dg.generate_dataframe(30)

    assert not pd.isna(synthetic_df).any().any()


def test_train_dataframe_long_nans(config: DGANConfig):
    n = 50
    df = pd.DataFrame(
        {
            "example_id": np.repeat(range(n), 20),
            "f1": np.random.rand(20 * n),
            "f2": np.random.rand(20 * n),
        }
    )

    df.iloc[0, 2] = df.iloc[500, 1] = df.iloc[900, 1] = np.nan
    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        example_id_column="example_id",
        df_style=DfStyle.LONG,
    )

    synthetic_df = dg.generate_dataframe(n)
    assert not pd.isna(synthetic_df).any().any()


@pytest.mark.parametrize("binary_encoder_cutoff", [150, 1])
def test_train_dataframe_with_strings(config: DGANConfig, binary_encoder_cutoff):
    n = 50
    expected_categories = set(["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", ""])
    df = pd.DataFrame(
        {
            "c": np.random.choice(sorted(list(expected_categories)), n),
            "n1": np.random.rand(n),
            "n2": np.random.rand(n),
            "example_id": np.repeat(range(10), 5),
        }
    )

    config.max_sequence_len = 5
    config.binary_encoder_cutoff = binary_encoder_cutoff
    dg = DGAN(config=config)
    dg.train_dataframe(df=df, example_id_column="example_id", df_style=DfStyle.LONG)

    synthetic_df = dg.generate_dataframe(5)

    assert len(synthetic_df) == 5 * 5

    for i in range(len(synthetic_df)):
        value = synthetic_df.loc[i, "c"]
        assert (
            value in expected_categories
        ), f"row {i} contained unexpected category='{value}'"


@pytest.mark.parametrize("binary_encoder_cutoff", [150, 1])
def test_train_dataframe_wide_with_strings(config: DGANConfig, binary_encoder_cutoff):

    expected_categories = set(["aa", "bb", "cc", "dd", "ee"])
    n = 50
    df = pd.DataFrame(np.random.rand(n, 5))
    df["attribute"] = np.random.choice(sorted(list(expected_categories)), n)

    config.max_sequence_len = 5
    config.binary_encoder_cutoff = binary_encoder_cutoff
    dg = DGAN(config=config)
    dg.train_dataframe(df=df, attribute_columns=["attribute"], df_style=DfStyle.WIDE)

    synthetic_df = dg.generate_dataframe(100)

    assert synthetic_df.shape == (100, 6)

    for i in range(len(synthetic_df)):
        value = synthetic_df.loc[i, "attribute"]
        assert (
            value in expected_categories
        ), f"row {i} contained unexpected category='{value}'"


def test_train_dataframe_long_attribute_mismatch_nans(config: DGANConfig):
    # Reproduce error found in internal testing.
    n = 50
    df = pd.DataFrame(
        {
            "example_id": np.repeat(np.arange(10), repeats=5),
            "a": "foo",
            "f": np.random.rand(n),
        }
    )

    # We tried to replace nans for the first example, but accidentally had an
    # off-by-one index issue since DataFrame.loc slicing is inclusive on both
    # endpoints.
    df.loc[0:5, "a"] = np.nan
    # And then the pandas groupby min() did not produce a result for all columns
    # so we would get a key error instead of the expected error mesasge. Pandas
    # appears to drop groupby results if there's any error, and taking min of a
    # slice with a mix of nan and string fails. This only occurs when a example
    # has a mix of nan and string attribute values in the different rows, so in
    # standard usage this never happens because attribute values are constant
    # for an example.

    config.max_sequence_len = 5
    dg = DGAN(config=config)

    with pytest.raises(DataError, match="not constant within each example"):
        dg.train_dataframe(
            df,
            example_id_column="example_id",
            attribute_columns=["a"],
            df_style=DfStyle.LONG,
        )


def test_train_dataframe_long_float_example_id(config: DGANConfig):
    # Reproduce error from production where example_id_column is float and
    # there's no 0.0 value. Should train with no errors.
    n = 50
    df = pd.DataFrame(
        {
            "example_id": np.repeat(np.arange(10.0, 15.0, 0.5), repeats=5),
            "time": [str(x) for x in pd.date_range("2022-01-01", periods=n)],
            "f": np.random.rand(n),
        }
    )

    config.max_sequence_len = 5
    dg = DGAN(config=config)

    dg.train_dataframe(
        df,
        example_id_column="example_id",
        time_column="time",
        df_style=DfStyle.LONG,
    )


def test_train_numpy_with_strings(config: DGANConfig):
    n = 50
    features = np.stack(
        [
            np.random.choice(["aa", "bb", "cc"], n),
            np.random.rand(n),
            np.random.rand(n),
        ],
        axis=1,
    ).reshape(
        -1, 5, 3
    )  # convert to 3-d features array

    config.max_sequence_len = 5
    dg = DGAN(config=config)
    dg.train_numpy(features=features)

    synthetic_attributes, synthetic_features = dg.generate_numpy(5)

    assert_attributes_features_shape(
        synthetic_attributes,
        synthetic_features,
        attributes_shape=None,
        features_shape=(5, 5, 3),
    )

    expected_categories = set(["aa", "bb", "cc"])

    for seq in synthetic_features:
        assert all([x in expected_categories for x in seq[:, 0]])


def test_train_numpy_max_sequence_len_error(config: DGANConfig):
    n = 50

    features = np.random.random((n, 25, 5))

    # Set max_sequence_len to the number of features, instead of number of time
    # points (25) as it should be.
    config.max_sequence_len = 5

    dg = DGAN(config=config)

    with pytest.raises(ParameterError, match="max_sequence_len"):
        dg.train_numpy(features=features)


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

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(6, 2), features_shape=(6, 3, 1)
    )

    assert_attributes_features(
        attributes, features, expected_attributes, expected_features
    )

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

    assert_attributes_features_shape(
        attributes, features, attributes_shape=(6, 1), features_shape=(6, 3, 1)
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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

    assert_attributes_features_shape(
        attributes, features, attributes_shape=None, features_shape=(6, 3, 1)
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes=None,
        expected_features=expected_features,
    )

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

    assert_attributes_features_shape(
        attributes, features, attributes_shape=None, features_shape=(6, 3, 1)
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes=None,
        expected_features=expected_features,
    )

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

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(2, 2),
        features_shape=(2, 3, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        example_id_column="example_id",
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(2, 2),
        features_shape=(2, 3, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(2, 2),
        features_shape=(2, 3, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(2, 2),
        features_shape=(2, 3, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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
        max_sequence_len=6,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(1, 2),
        features_shape=(1, 6, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=6,
        attribute_columns=["a1", "a2"],
        time_column="time",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)
    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(1, 2),
        features_shape=(1, 6, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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
        max_sequence_len=6,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(1, 2),
        features_shape=(1, 6, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

    # Check works the same if feature column param is omitted
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=6,
        attribute_columns=["a1", "a2"],
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(1, 2),
        features_shape=(1, 6, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    attributes, features = converter.convert(df_long)

    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=(2, 2),
        features_shape=(2, 3, 3),
    )

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
        discrete_columns=["a1"],
    )
    with pytest.raises(DataError):
        converter.convert(df_long)

    # Same if we don't use example_id where attributes should be constant
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=6,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        discrete_columns=["a1"],
    )
    with pytest.raises(DataError):
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
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
    )

    _, features = converter.convert(df_long)
    assert all(seq.dtype == "float64" for seq in features)


def test_long_data_frame_converter_example_id_object(df_long):
    # Check dtype of features and attributes is float when example id column is
    # of object type like strings

    df_long = df_long.drop(columns="time")

    # Make example id column a string
    df_long["example_id"] = df_long["example_id"].astype("str")
    assert df_long["example_id"].dtype == "object"

    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=3,
        attribute_columns=["a1", "a2"],
        feature_columns=["f1", "f2", "f3"],
        example_id_column="example_id",
    )

    attributes, features = converter.convert(df_long)
    assert attributes is not None
    assert attributes.dtype == "float64"
    assert all(seq.dtype == "float64" for seq in features)


def test_long_data_frame_converter_example_id_float():
    # Check converter creation with a float example id column that has no values
    # of 0.0.

    df_long = pd.DataFrame(
        {
            "example_id": [1.0, 1.0, 2.0, 2.0],
            "time": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
            ],
            "f": [2.0, 3.0, 4.0, 5.0],
        }
    )

    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=2,
        example_id_column="example_id",
        time_column="time",
    )

    attributes, features = converter.convert(df_long)
    assert_attributes_features_shape(
        attributes,
        features,
        attributes_shape=None,
        features_shape=(2, 2, 1),
    )

    assert all(seq.dtype == "float64" for seq in features)


def test_long_data_frame_converter_variable_length():
    df_long = pd.DataFrame(
        {
            "example_id": ["a", "b", "b", "c", "c", "c"],
            "f": [2.0, 2.5, 3.0, 1.0, 1.5, 4.0],
        }
    )

    expected_features = [
        np.array([[2.0, 0.0]]),
        np.array([[2.5, 1.0], [3.0, 0.0]]),
        np.array([[1.0, 1.0], [1.5, 1.0], [4.0, 0.0]]),
    ]
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=3,
        example_id_column="example_id",
    )

    attributes, features = converter.convert(df_long)
    assert converter._feature_types == [OutputType.CONTINUOUS, OutputType.DISCRETE]

    assert_attributes_features(
        attributes,
        features,
        expected_attributes=None,
        expected_features=expected_features,
    )


def test_long_data_frame_converter_variable_length_error(df_long):
    with pytest.raises(DataError):
        _LongDataFrameConverter.create(
            df_long,
            max_sequence_len=1,
            example_id_column="example_id",
        )


def test_long_data_frame_converter_save_and_load(df_long):
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=3,
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

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

    df = loaded_converter.invert(attributes, features)

    assert_frame_equal(df, expected_df)


def test_long_data_frame_converter_save_and_load_variable_length(df_long):
    # Remove first row so the first example has 2 time points and the second
    # example has 3 time points
    df_long = df_long[1:]
    converter = _LongDataFrameConverter.create(
        df_long,
        max_sequence_len=5,
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

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )

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

    assert_attributes_features(
        attributes,
        features,
        expected_attributes,
        expected_features,
    )


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

    assert_attributes_features(
        attributes,
        features,
        expected_attributes=expected_attributes,
        expected_features=expected_features,
    )


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


def test_attribute_and_feature_overlap(config: DGANConfig):
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

    with pytest.raises(ParameterError) as err:
        dg.train_dataframe(
            df=df,
            attribute_columns=["a1", "a2"],
            feature_columns=["a1"],
            discrete_columns=["a1"],
            df_style=DfStyle.WIDE,
        )
    assert "not have overlapping" in str(err.value)


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


def test_dataframe_long_no_continuous_features(config: DGANConfig):
    # Model should train with only discrete/categorical features
    df = pd.DataFrame(
        {
            "a": np.random.choice(["foo", "bar", "baz"], size=9),
            "b": np.random.choice(["yes", "no"], size=9),
        }
    )

    config.max_sequence_len = 3
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_style=DfStyle.LONG,
        discrete_columns=["a", "b"],
    )


def test_dataframe_wide_no_continuous_features(config: DGANConfig):
    # Model should train with only discrete/categorical features
    df = pd.DataFrame(
        {
            "2023-01-01": np.random.choice(["yes", "no"], size=6),
            "2023-01-02": np.random.choice(["yes", "no"], size=6),
            "2023-01-03": np.random.choice(["yes", "no"], size=6),
        }
    )

    config.max_sequence_len = 3
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_style=DfStyle.WIDE,
        discrete_columns=["2023-01-01", "2023-01-02", "2023-01-03"],
    )


def test_dataframe_long_partial_example(config: DGANConfig):
    # Not enough rows to create a single example.
    df = pd.DataFrame(
        {
            "a": np.random.choice(["foo", "bar", "baz"], size=9),
            "b": np.random.random(size=9),
        }
    )

    config.max_sequence_len = 10
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)

    with pytest.raises(DataError, match="requires max_sequence_len"):
        dg.train_dataframe(
            df=df,
            df_style=DfStyle.LONG,
            discrete_columns=["a"],
        )


def test_dataframe_long_one_and_partial_example(config: DGANConfig):
    # Using auto split with more than max_sequence_len rows, but not enough to
    # make 2 examples, which are required for training.
    df = pd.DataFrame(
        {
            "a": np.random.choice(["foo", "bar", "baz"], size=9),
            "b": np.random.random(size=9),
        }
    )

    config.max_sequence_len = 5
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)

    with pytest.raises(DataError, match="multiple examples to train"):
        dg.train_dataframe(
            df=df,
            df_style=DfStyle.LONG,
            discrete_columns=["a"],
        )


def test_dataframe_variable_sequences(config: DGANConfig):
    # Variable length sequences that dgan should automatically pad to
    # max_sequence_len

    # Build dataframe of variable length sequences
    rows = []
    for id, seq_length in enumerate([3, 6, 5, 1, 1, 8, 8, 3]):
        a1 = np.random.choice(["x", "y", "z"])
        for _ in range(seq_length):
            rows.append(
                (
                    id,
                    a1,
                    np.random.random(),
                    np.random.choice(["foo", "bar"]),
                )
            )
    df = pd.DataFrame(rows, columns=["example_id", "a1", "f1", "f2"])

    config.max_sequence_len = 8
    config.sample_len = 1
    config.epochs = 1

    dg = DGAN(config=config)

    dg.train_dataframe(
        df=df,
        df_style=DfStyle.LONG,
        example_id_column="example_id",
        attribute_columns=["a1"],
    )

    df_synth = dg.generate_dataframe(3)
    assert df.shape[1] == df_synth.shape[1]
    assert all(str(x) == str(y) for x, y in zip(df_synth.columns, df.columns))

    for count in Counter(df_synth["example_id"]).most_common():
        assert 1 <= count[1] <= 8
