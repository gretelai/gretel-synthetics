"""
E2E Tests for training and generating data
"""
import json

import pytest
import pandas as pd

from gretel_synthetics.tokenizers import SentencePieceTokenizerTrainer, CharTokenizerTrainer, BaseTokenizerTrainer
from gretel_synthetics.batch import PATH_HOLDER, DataFrameBatch
from gretel_synthetics.config import TensorFlowConfig
import gretel_synthetics.const as const


DATA = "https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/data/USAdultIncome14K.csv"


@pytest.fixture(scope="module")
def train_df():
    df = pd.read_csv(DATA)
    return df.sample(n=5000)


#######################################################
# Basic training only, we don't care about generation
########################################################

def test_train_batch_sp_regression(train_df, tmp_path):
    """Batch mode with default SentencePiece tokenizer. Using the backwards
    compat mode for <= 0.14.0.
    """
    config = {
        "epochs": 1,
        "field_delimiter": ",",
        "checkpoint_dir": tmp_path
    }
    batcher = DataFrameBatch(
        df=train_df,
        config=config
    )
    batcher.create_training_data()
    batcher.train_all_batches()

    model_params = json.loads(open(tmp_path / "batch_0" / const.MODEL_PARAMS).read())
    assert model_params[const.MODEL_TYPE] == TensorFlowConfig.__name__

    tok_params = json.loads(open(tmp_path / "batch_0" / BaseTokenizerTrainer.settings_fname).read())
    assert tok_params["tokenizer_type"] == SentencePieceTokenizerTrainer.__name__


def test_train_batch_sp(train_df, tmp_path):
    config = TensorFlowConfig(
        epochs=1,
        field_delimiter=",",
        checkpoint_dir=tmp_path,
        input_data_path=PATH_HOLDER
    )
    batcher = DataFrameBatch(
        df=train_df,
        config=config
    )
    batcher.create_training_data()
    batcher.train_all_batches()

    model_params = json.loads(open(tmp_path / "batch_0" / const.MODEL_PARAMS).read())
    assert model_params[const.MODEL_TYPE] == TensorFlowConfig.__name__

    tok_params = json.loads(open(tmp_path / "batch_0" / BaseTokenizerTrainer.settings_fname).read())
    assert tok_params["tokenizer_type"] == SentencePieceTokenizerTrainer.__name__


#########################
# Longer train + gen run
#########################

def test_train_batch_char_tok(train_df, tmp_path):
    config = TensorFlowConfig(
        epochs=5,
        field_delimiter=",",
        checkpoint_dir=tmp_path,
        input_data_path=PATH_HOLDER,
        learning_rate=.01
    )
    batcher = DataFrameBatch(
        df=train_df,
        config=config,
        tokenizer=CharTokenizerTrainer(config=config)
    )
    batcher.create_training_data()
    batcher.train_all_batches()

    tok_params = json.loads(open(tmp_path / "batch_0" / BaseTokenizerTrainer.settings_fname).read())
    assert tok_params["tokenizer_type"] == CharTokenizerTrainer.__name__

    batcher.generate_all_batch_lines(num_lines=100, max_invalid=5000)
    syn_df = batcher.batches_to_df()
    assert syn_df.shape[0] == 100


def test_train_batch_sp_tok(train_df, tmp_path):
    config = TensorFlowConfig(
        epochs=5,
        field_delimiter=",",
        checkpoint_dir=tmp_path,
        input_data_path=PATH_HOLDER,
        learning_rate=.01
    )
    tokenizer = SentencePieceTokenizerTrainer(
        vocab_size=10000,
        config=config
    )
    batcher = DataFrameBatch(
        df=train_df,
        config=config,
        tokenizer=tokenizer
    )
    batcher.create_training_data()
    batcher.train_all_batches()

    batcher.generate_all_batch_lines(num_lines=100, max_invalid=5000)
    syn_df = batcher.batches_to_df()
    assert syn_df.shape[0] == 100
