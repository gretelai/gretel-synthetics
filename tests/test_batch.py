from pathlib import Path
import shutil
from unittest.mock import patch
import random
from copy import deepcopy

import pytest
import pandas as pd

from gretel_synthetics.batch import DataFrameBatch
from gretel_synthetics.generate import gen_text


checkpoint_dir = str(Path(__file__).parent / "checkpoints")


config_template = {
    "max_lines": 0,
    "epochs": 30,
    "vocab_size": 15000,
    "max_line_len": 2048,
    "character_coverage": 1.0,
    "gen_chars": 0,
    "gen_lines": 5,
    "rnn_units": 256,
    "batch_size": 64,
    "buffer_size": 1000,
    "dropout_rate": 0.2,
    "dp": True,
    "dp_learning_rate": 0.015,
    "dp_noise_multiplier": 1.1,
    "dp_l2_norm_clip": 1.0,
    "dp_microbatches": 256,
    "field_delimiter": "|",
    "overwrite": False,
    "checkpoint_dir": checkpoint_dir,
}


@pytest.fixture(scope="module")
def test_data():
    path = Path(__file__).parent / "data" / "santander.csv.gz"
    yield pd.read_csv(path)
    if Path(checkpoint_dir).is_dir():
        shutil.rmtree(checkpoint_dir)
        pass

def simple_validator(line: str):
    return len(line.split(",")) == 5


def test_missing_delim():
    config = deepcopy(config_template)
    config.pop("field_delimiter")
    with pytest.raises(ValueError):
        DataFrameBatch(df=test_data, config=config)


def test_init(test_data):
    with pytest.raises(ValueError):
        DataFrameBatch(df="nope", config=config_template)

    # should create the dir structure based on auto
    # batch sizing
    batches = DataFrameBatch(df=test_data, config=config_template)
    first_row = [
        "ID_code",
        "target",
        "var_0",
        "var_1",
        "var_2",
        "var_3",
        "var_4",
        "var_5",
        "var_6",
        "var_7",
        "var_8",
        "var_9",
        "var_10",
        "var_11",
        "var_12",
        "var_13",
    ]
    assert batches.batches[0].headers == first_row
    assert len(batches.batches.keys()) == 13
    for i, batch in batches.batches.items():
        assert Path(batch.checkpoint_dir).is_dir()
        assert Path(batch.checkpoint_dir).name == f"batch_{i}"

    batches.create_training_data()
    df = pd.read_csv(batches.batches[0].input_data_path, sep=config_template["field_delimiter"])
    assert len(df.columns) == len(first_row)

    with pytest.raises(ValueError):
        batches.train_batch(99)

    with patch("gretel_synthetics.batch.train_rnn") as mock_train:
        batches.train_batch(5)
        arg = batches.batches[5].config
        mock_train.assert_called_with(arg)

    with patch("gretel_synthetics.batch.train_rnn") as mock_train:
        batches.train_all_batches()
        args = [b.config for b in batches.batches.values()]
        called_args = []
        for _, a, _ in mock_train.mock_calls:
            called_args.append(a[0])
        assert args == called_args

    with pytest.raises(ValueError):
        batches.set_batch_validator(5, "foo")

    with pytest.raises(ValueError):
        batches.set_batch_validator(99, simple_validator)

    batches.set_batch_validator(5, simple_validator)
    assert batches.batches[5].validator("1,2,3,4,5")
    # load validator back from disk
    batches.batches[5].load_validator_from_file()
    assert batches.batches[5].validator("1,2,3,4,5")

    # generate lines, simulating generation the max
    # valid line count
    def good():
        return gen_text(text="1,2,3,4,5", valid=random.choice([None, True]), delimiter=",")
    
    def bad():
        return gen_text(text="1,2,3", valid=False, delimiter=",")

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.return_value = [good(), good(), good(), bad(), bad(), good(), good()]
        assert batches.generate_batch_lines(5)

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.return_value = [good(), good(), good(), bad(), bad(), good()]
        assert not batches.generate_batch_lines(5)

    with patch.object(batches, "generate_batch_lines") as mock_gen:
        batches.generate_all_batch_lines()
        assert mock_gen.call_count == len(batches.batches.keys())
       

    # get synthetic df
    line = gen_text(text="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", valid=True, delimiter=",")
    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.return_value = [line] * len(batches.batches[10].headers)
        batches.generate_batch_lines(10)

    assert len(batches.batches[10].synthetic_df) == len(batches.batches[10].headers)


def test_batches_to_df(test_data):
    batches = DataFrameBatch(df=pd.DataFrame([
        {"foo": "bar", "foo1": "bar1", "foo2": "bar2", "foo3": 3}]), config=config_template, batch_size=1)

    batches.batches[0].add_valid_data(
        gen_text(text="baz", valid=True, delimiter=",")
    )
    batches.batches[1].add_valid_data(
        gen_text(text="baz1", valid=True, delimiter=",")
    )
    batches.batches[2].add_valid_data(
        gen_text(text="baz2", valid=True, delimiter=",")
    )
    batches.batches[3].add_valid_data(
        gen_text(text="5", valid=True, delimiter=",")
    )

    check = batches.batches_to_df()
    assert list(check.columns) == ["foo", "foo1", "foo2", "foo3"]
    assert check.shape == (1, 4)
    assert [t.name for t in list(check.dtypes)] == ['object', 'object', 'object', 'int64']
