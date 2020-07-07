from unittest.mock import patch
from pathlib import Path
import uuid
import json
import shutil

import pytest

from gretel_synthetics.config import LocalConfig, TOKENIZER_PREFIX


@patch("gretel_synthetics.config.Path.mkdir")
def test_local_config(mkdir):
    target = uuid.uuid4().hex
    test_data_dir = Path(__file__).parent
    test_data_file = test_data_dir / "data" / "smol.txt"
    lc = LocalConfig(checkpoint_dir=target, input_data_path=test_data_file.as_posix())

    mkdir.assert_called
    assert lc.epochs == 15
    assert lc.input_data_path == test_data_file.as_posix()
    assert lc.tokenizer_prefix == TOKENIZER_PREFIX
    assert lc.training_data == Path(target, "training_data.txt").as_posix()
    assert lc.tokenizer_model == Path(target, "m.model").as_posix()


@patch("gretel_synthetics.config.Path.mkdir")
def test_local_config_settings(mkdir):
    lc = LocalConfig(checkpoint_dir="foo", input_data_path="bar")
    check = lc.as_dict()
    assert check == {
        "max_lines": 0,
        "epochs": 15,
        "batch_size": 64,
        "buffer_size": 10000,
        "seq_length": 100,
        "embedding_dim": 256,
        "rnn_units": 256,
        "dropout_rate": 0.2,
        "rnn_initializer": "glorot_uniform",
        "vocab_size": 20000,
        "character_coverage": 1.0,
        "dp": False,
        "dp_learning_rate": 0.015,
        "dp_noise_multiplier": 1.1,
        "dp_l2_norm_clip": 1.0,
        "dp_microbatches": 256,
        "gen_temp": 1.0,
        "gen_chars": 0,
        "gen_lines": 1000,
        "max_line_len": 2048,
        "save_all_checkpoints": False,
        "checkpoint_dir": "foo",
        "field_delimiter": None,
        "field_delimiter_token": "<d>",
        "overwrite": False,
        "input_data_path": "bar",
    }


def test_local_config_missing_attrs():
    with pytest.raises(AttributeError):
        LocalConfig()

    with pytest.raises(AttributeError):
        LocalConfig(checkpoint_dir="foo")

    with pytest.raises(AttributeError):
        LocalConfig(input_data_path="foo")


def test_local_config_save_model_params():
    test_data_dir = Path(__file__).parent
    target = test_data_dir / uuid.uuid4().hex
    test_data_file = test_data_dir / "data" / "smol.txt"
    lc = LocalConfig(
        checkpoint_dir=target.as_posix(), input_data_path=test_data_file.as_posix()
    )
    check = lc.save_model_params()
    assert json.loads(open(check).read())
    shutil.rmtree(target)
