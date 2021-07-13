"""
E2E Tests for training and generating data
"""
import json
import os

import pytest
import pandas as pd

from gretel_synthetics.tokenizers import SentencePieceTokenizerTrainer, CharTokenizerTrainer, BaseTokenizerTrainer
from gretel_synthetics.batch import PATH_HOLDER, DataFrameBatch
from gretel_synthetics.config import TensorFlowConfig
import gretel_synthetics.const as const
from gretel_synthetics.train import EpochState


DATA = "https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/data/USAdultIncome14K.csv"


@pytest.fixture(scope="module")
def train_df():
    df = pd.read_csv(DATA)
    return df.sample(n=5000)


#######################################################
# Basic training only, we don't care about generation
########################################################


def test_bad_microbatch_size(tmp_path):
    with pytest.raises(ValueError) as err:
        config = TensorFlowConfig(
            epochs=1,
            field_delimiter=",",
            checkpoint_dir=tmp_path,
            input_data_path=PATH_HOLDER,
            batch_size=64,
            dp=True,
            dp_microbatches=65000
        )
    assert "Number of dp_microbatches should divide evenly into batch_size" in str(err)


def test_bad_epoch_callback(tmp_path):
    with pytest.raises(ValueError) as err:
        config = TensorFlowConfig(
            epochs=1,
            field_delimiter=",",
            checkpoint_dir=tmp_path,
            input_data_path=PATH_HOLDER,
            epoch_callback=1
        )
    assert "must be a callable" in str(err)
    

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

_tok_gen_count = 50


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

    batcher.generate_all_batch_lines(num_lines=_tok_gen_count, max_invalid=5000)
    syn_df = batcher.batches_to_df()
    assert syn_df.shape[0] == _tok_gen_count


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

    batcher.generate_all_batch_lines(num_lines=_tok_gen_count, max_invalid=5000)
    syn_df = batcher.batches_to_df()
    assert syn_df.shape[0] == _tok_gen_count

    # Generate with a RecordFactory
    factory = batcher.create_record_factory(num_lines=_tok_gen_count, max_invalid=5000)
    syn_df = factory.generate_all(output="df")
    assert syn_df.shape[0] == _tok_gen_count
    assert list(syn_df.columns) == list(train_df.columns)
    assert factory.summary["valid_count"] == _tok_gen_count


def test_train_small_df(train_df, tmp_path):
    small_df = train_df.sample(n=50)
    config = TensorFlowConfig(
        epochs=5,
        field_delimiter=",",
        checkpoint_dir=tmp_path,
        input_data_path=PATH_HOLDER
    )
    batcher = DataFrameBatch(
        df=small_df,
        config=config
    )
    batcher.create_training_data()
    with pytest.raises(RuntimeError) as excinfo:
        batcher.train_all_batches()
    assert "Model training failed" in str(excinfo.value)


def test_epoch_callback(train_df, tmp_path):
    def epoch_callback(s: EpochState):
        with open(tmp_path / 'callback_dump.txt', 'a') as f:
            f.write(f'{s.epoch},{s.accuracy},{s.loss},{s.val_accuracy},{s.val_loss},{s.epsilon},{s.delta},{s.batch}\n')
    config = TensorFlowConfig(
        epochs=5,
        field_delimiter=",",
        checkpoint_dir=tmp_path,
        input_data_path=PATH_HOLDER,
        learning_rate=.01,
        epoch_callback=epoch_callback,
        dp=True,
        dp_microbatches=1,
    )
    tokenizer = SentencePieceTokenizerTrainer(
        vocab_size=10000,
        config=config
    )
    batcher = DataFrameBatch(
        batch_size=4,
        df=train_df,
        config=config,
        tokenizer=tokenizer
    )
    batcher.create_training_data()
    batcher.train_all_batches()
    with open(tmp_path / 'callback_dump.txt', 'r') as f:
        lines = f.readlines()
        assert len(lines) == 20
        for i, line in enumerate(lines):
            fields = line.strip().split(',')
            assert len(fields) == 8
            assert int(fields[0]) == i % 5
            assert(float(fields[1]))
            assert(float(fields[2]))
            assert(float(fields[3]))
            assert(float(fields[4]))
            assert(float(fields[5]))
            assert(float(fields[6]))
            assert int(fields[7]) == i // 5
    os.remove(tmp_path / 'callback_dump.txt')
