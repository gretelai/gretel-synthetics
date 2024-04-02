import datetime
import itertools

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pandas.api.types import is_number

from gretel_synthetics.actgan import ACTGAN
from gretel_synthetics.actgan.data_transformer import BinaryEncodingTransformer
from gretel_synthetics.actgan.structures import ConditionalVectorType


@pytest.fixture
def test_df() -> pd.DataFrame:
    record_count = 10
    datetimes = [datetime.datetime.utcnow().isoformat() for _ in range(record_count)]
    values = list(range(record_count))
    df = pd.DataFrame(zip(datetimes, values), columns=["timestamp", "value"])
    return df


def test_auto_transform_datetimes(test_df):
    """
    Validate that the auto transform detection converts datetime
    columns to integers prior to the model being fit.
    """

    model = ACTGAN(auto_transform_datetimes=True)
    model._fit = Mock()
    model.fit(test_df)

    _, args, _ = model._fit.mock_calls[0]
    transformed_df = args[0]
    assert is_number(transformed_df[transformed_df.columns[0]][0])


def test_binary_encoder_cutoff(test_df):
    with patch("gretel_synthetics.actgan.actgan.ACTGANSynthesizer._actual_fit"):
        model = ACTGAN(binary_encoder_cutoff=5)
        model.fit(test_df)

        encoder = model._model._transformer._column_transform_info_list[0].transform
        assert isinstance(encoder, BinaryEncodingTransformer)


@pytest.mark.parametrize(
    "log_frequency,conditional_vector_type,force_conditioning,binary_encoder_cutoff",
    itertools.product(
        [False, True],
        [
            ConditionalVectorType.SINGLE_DISCRETE,
            ConditionalVectorType.ANYWAY,
        ],
        [False, True],
        [1, 500],
    ),
)
def test_actgan_implementation(
    log_frequency,
    conditional_vector_type,
    force_conditioning,
    binary_encoder_cutoff,
):
    # Test basic actgan setup with various parameters and to confirm training
    # and synthesize does not crash, i.e., all the tensor shapes match. Use a
    # small model and small dataset to keep tests quick.
    n = 100
    df = pd.DataFrame(
        {
            "int_column": np.random.randint(0, 200, size=n),
            "float_column": np.random.random(size=n),
            "categorical_column": np.random.choice(["a", "b", "c"], size=n),
            "high_cardinality_column": np.random.choice(
                [f"x{i}" for i in range(n * 3)], size=n
            ),
        }
    )

    conditional_select_mean_columns = None
    if conditional_vector_type != ConditionalVectorType.SINGLE_DISCRETE:
        conditional_select_mean_columns = 2

    model = ACTGAN(
        epochs=1,
        batch_size=20,
        generator_dim=[32, 32],
        discriminator_dim=[32, 32],
        log_frequency=log_frequency,
        conditional_vector_type=conditional_vector_type,
        force_conditioning=force_conditioning,
        conditional_select_mean_columns=conditional_select_mean_columns,
        binary_encoder_cutoff=binary_encoder_cutoff,
    )

    # Check training
    model.fit(df)

    # Check unconditional generation
    df_synth = model.sample(num_rows=100)

    assert df_synth.shape == (100, len(df.columns))
    assert list(df.columns) == list(df_synth.columns)

    if conditional_vector_type != ConditionalVectorType.SINGLE_DISCRETE:
        # Check conditional generation from numeric column
        df_synth = model.sample_remaining_columns(
            pd.DataFrame(
                {
                    "int_column": [10] * 10,
                }
            )
        )
        assert list(df.columns) == list(df_synth.columns)

    # Check conditional generation from discrete column
    df_synth = model.sample_remaining_columns(
        pd.DataFrame(
            {
                "categorical_column": ["b"] * 10,
            }
        )
    )
    assert list(df.columns) == list(df_synth.columns)


@pytest.mark.parametrize(
    "log_frequency,conditional_vector_type,force_conditioning",
    itertools.product(
        [False, True],
        [
            ConditionalVectorType.SINGLE_DISCRETE,
            ConditionalVectorType.ANYWAY,
        ],
        [False, True],
    ),
)
def test_actgan_implementation_all_numeric(
    log_frequency, conditional_vector_type, force_conditioning
):
    # Test basic actgan setup with various parameters and to confirm training
    # and synthesize does not crash, i.e., all the tensor shapes match. Use a
    # small model and small dataset to keep tests quick.
    n = 100
    df = pd.DataFrame(
        {
            "int_column": np.random.randint(0, 200, size=n),
            "float_column": np.random.random(size=n),
        }
    )

    conditional_select_mean_columns = None
    if conditional_vector_type != ConditionalVectorType.SINGLE_DISCRETE:
        conditional_select_mean_columns = 2

    model = ACTGAN(
        epochs=1,
        batch_size=20,
        generator_dim=[32, 32],
        discriminator_dim=[32, 32],
        log_frequency=log_frequency,
        conditional_vector_type=conditional_vector_type,
        force_conditioning=force_conditioning,
        conditional_select_mean_columns=conditional_select_mean_columns,
    )

    # Check training
    model.fit(df)

    # Check unconditional generation
    df_synth = model.sample(num_rows=100)

    assert df_synth.shape == (100, len(df.columns))
    assert list(df.columns) == list(df_synth.columns)

    if conditional_vector_type != ConditionalVectorType.SINGLE_DISCRETE:
        # Check conditional generation from numeric column
        df_synth = model.sample_remaining_columns(
            pd.DataFrame(
                {
                    "int_column": [10] * 10,
                }
            )
        )
        assert list(df.columns) == list(df_synth.columns)
