from pathlib import Path
import shutil
from unittest.mock import patch, Mock
import random
from copy import deepcopy
from dataclasses import asdict

import pytest
import pandas as pd

from gretel_synthetics.batch import DataFrameBatch, MAX_INVALID, READ
from gretel_synthetics.generate import GenText
from gretel_synthetics.errors import TooManyInvalidError


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
    "dp": False,
    "learning_rate": 0.015,
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


def simple_validator(line: str):
    return len(line.split(",")) == 5


def test_batch_size(test_data):
    test_data = test_data.iloc[:, :60]
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=15)
    assert batches.batch_size == 15
    assert [len(x) for x in batches.batch_headers] == [15, 15, 15, 15]

    test_data = test_data.iloc[:, :59]
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=15)
    assert batches.batch_size == 15
    assert [len(x) for x in batches.batch_headers] == [15, 15, 15, 14]


def test_missing_config(test_data):
    with pytest.raises(ValueError):
        DataFrameBatch(df=test_data)


def test_auto_gen_lines(test_data):
    config = deepcopy(config_template)
    config.pop("gen_lines")
    d = DataFrameBatch(df=test_data, config=config)
    assert d.config["gen_lines"] == test_data.shape[0]


def test_missing_delim(test_data):
    config = deepcopy(config_template)
    config.pop("field_delimiter")
    with pytest.raises(ValueError):
        DataFrameBatch(df=test_data, config=config)

def test_init(test_data):
    with pytest.raises(ValueError):
        DataFrameBatch(df="nope", config=config_template)

    # should create the dir structure based on auto
    # batch sizing
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=15)
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
    ]
    assert batches.batches[0].headers == first_row
    assert len(batches.batches.keys()) == 14
    for i, batch in batches.batches.items():
        assert Path(batch.checkpoint_dir).is_dir()
        assert Path(batch.checkpoint_dir).name == f"batch_{i}"

    batches.create_training_data()
    df = pd.read_csv(batches.batches[0].input_data_path, sep=config_template["field_delimiter"])
    assert len(df.columns) == len(first_row)

    with pytest.raises(ValueError):
        batches.train_batch(99)

    with patch("gretel_synthetics.batch.train") as mock_train:
        batches.train_batch(5)
        arg = batches.batches[5].config
        mock_train.assert_called_with(arg, None)

    with patch("gretel_synthetics.batch.train") as mock_train:
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
        return GenText(text="1,2,3,4,5", valid=random.choice([None, True]), delimiter=",")
    
    def bad():
        return GenText(text="1,2,3", valid=False, delimiter=",")

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.return_value = [good(), good(), good(), bad(), bad(), good(), good()]
        assert batches.generate_batch_lines(5, max_invalid=1)
        check_call = mock_gen.mock_calls[0]
        _, _, kwargs = check_call
        assert kwargs["max_invalid"] == 1

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.return_value = [good(), good(), good(), bad(), bad(), good(), good()]
        assert batches.generate_batch_lines(5)

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.return_value = [good(), good(), good(), bad(), bad(), good()]
        assert not batches.generate_batch_lines(5)

    with patch.object(batches, "generate_batch_lines") as mock_gen:
        batches.generate_all_batch_lines(max_invalid=15)
        assert mock_gen.call_count == len(batches.batches.keys())
        check_call = mock_gen.mock_calls[0]
        _, _, kwargs = check_call
        assert kwargs["max_invalid"] == 15
       

    # get synthetic df
    line = GenText(text="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", valid=True, delimiter=",")
    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.return_value = [line] * len(batches.batches[10].headers)
        batches.generate_batch_lines(10)

    assert len(batches.batches[10].synthetic_df) == len(batches.batches[10].headers)


def test_batches_to_df(test_data):
    batches = DataFrameBatch(df=pd.DataFrame([
        {"foo": "bar", "foo1": "bar1", "foo2": "bar2", "foo3": 3}]), config=config_template, batch_size=2)

    batches.batches[0].add_valid_data(
        GenText(text="baz|baz1", valid=True, delimiter="|")
    )
    batches.batches[1].add_valid_data(
        GenText(text="baz2|5", valid=True, delimiter="|")
    )

    check = batches.batches_to_df()
    assert list(check.columns) == ["foo", "foo1", "foo2", "foo3"]
    assert check.shape == (1, 4)
    assert [t.name for t in list(check.dtypes)] == ['object', 'object', 'object', 'int64']


def test_generate_batch_lines_raise_on_exceed(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template)
    batches.create_training_data()

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.side_effect = TooManyInvalidError()
        assert not batches.generate_batch_lines(0)

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.side_effect = TooManyInvalidError()
        with pytest.raises(TooManyInvalidError):
            assert not batches.generate_batch_lines(0, raise_on_exceed_invalid=True)


def test_generate_batch_lines_always_raise_other_exceptions(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template)
    batches.create_training_data()

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.side_effect = RuntimeError()
        with pytest.raises(RuntimeError):
            assert not batches.generate_batch_lines(0)

    with patch("gretel_synthetics.batch.generate_text") as mock_gen:
        mock_gen.side_effect = RuntimeError()
        with pytest.raises(RuntimeError):
            assert not batches.generate_batch_lines(0, raise_on_exceed_invalid=True)


def test_generate_all_batch_lines_raise_on_failed(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template)
    batches.create_training_data()

    batches.generate_batch_lines = Mock()
    batches.generate_all_batch_lines()
    _, args, kwargs = batches.generate_batch_lines.mock_calls[0]
    assert args == (0,)
    assert kwargs == {
        "max_invalid": MAX_INVALID,
        "raise_on_exceed_invalid": False,
        "num_lines": None,
        "parallelism": 0,
        "seed_fields": None
    }

    batches.generate_batch_lines = Mock()
    batches.generate_all_batch_lines(max_invalid=10, raise_on_failed_batch=True, num_lines=5)
    _, args, kwargs = batches.generate_batch_lines.mock_calls[0]
    assert args == (0,)
    assert kwargs == {
        "max_invalid": 10,
        "raise_on_exceed_invalid": True,
        "num_lines": 5,
        "parallelism": 0,
        "seed_fields": None
    }


def test_read_mode(test_data):
    writer = DataFrameBatch(df=test_data, config=config_template)
    writer.create_training_data()

    # missing checkpoint dir
    with pytest.raises(ValueError):
        DataFrameBatch(mode=READ)

    # bad checkpoint dir
    with pytest.raises(ValueError):
        DataFrameBatch(mode=READ, checkpoint_dir="bad_dir")

    # NOTE: normally saving the params is done during training,
    # but we do it here manually since we won't actually train
    for _, batch in writer.batches.items():
        batch.config.save_model_params()

    # checkpoint dir exists in config
    DataFrameBatch(config=config_template, mode=READ)

    # checkpoint dir as a kwarg
    reader = DataFrameBatch(checkpoint_dir=checkpoint_dir, mode=READ)

    write_batch = writer.batches[0]
    read_batch = reader.batches[0]

    assert write_batch.checkpoint_dir == read_batch.checkpoint_dir
    assert write_batch.headers == read_batch.headers
    assert asdict(write_batch.config) == asdict(read_batch.config)
    assert reader.master_header_list == writer.master_header_list


def test_validate_seed_lines_too_many_fields(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=3)
    
    with pytest.raises(RuntimeError) as err:
        batches._validate_batch_seed_values(
            batches.batches[0],
            {
                "ID_Code": "foo",
                "target": 0,
                "var_0": 33,
                "var_1": 33
            }
        )
    assert "number of seed fields" in str(err.value)



def test_validate_seed_lines_field_not_present(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=3)

    with pytest.raises(RuntimeError) as err:
        batches._validate_batch_seed_values(
            batches.batches[0],
            {
                "ID_code": "foo",
                "target": 0,
                "var_1": 33,
            }
        )
    assert "The header: var_0 is not in the seed" in str(err.value)


def test_validate_seed_lines_ok_full_size(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=3)

    check = batches._validate_batch_seed_values(
        batches.batches[0],
        {
            "ID_code": "foo",
            "target": 0,
            "var_0": 33,
        }
    )
    assert check == "foo|0|33|"


def test_validate_seed_lines_ok_one_field(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=3)

    check = batches._validate_batch_seed_values(
        batches.batches[0],
        {
            "ID_code": "foo",
        }
    )
    assert check == "foo|"


def test_validate_seed_lines_ok_two_field(test_data):
    batches = DataFrameBatch(df=test_data, config=config_template, batch_size=3)

    check = batches._validate_batch_seed_values(
        batches.batches[0],
        {
            "ID_code": "foo",
            "target": 1
        }
    )
    assert check == "foo|1|"
