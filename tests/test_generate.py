from unittest.mock import Mock

import pytest

from gretel_synthetics.generate import Settings
from gretel_synthetics.const import NEWLINE
from gretel_synthetics.config import TensorFlowConfig
from gretel_synthetics.errors import GenerationError


@pytest.fixture
def mock_tokenizer():
    t = Mock()
    t.newline_str = NEWLINE
    t.tokenize_delimiter = lambda s: s.replace(",", "<d>") 
    return t


def test_default_start_string(tf_config, mock_tokenizer):
    check = Settings(config=tf_config, tokenizer=mock_tokenizer)
    assert check.start_string == NEWLINE


def test_no_delim_bad_start_string(tmpdir):
    config = TensorFlowConfig(checkpoint_dir=tmpdir, input_data_path=tmpdir)
    with pytest.raises(GenerationError):
        Settings(config=config, start_string=123, tokenizer=mock_tokenizer)


def test_delim_missing_trailing_delim(tf_config):
    with pytest.raises(GenerationError):
        Settings(config=tf_config, start_string="foo,bar", tokenizer=mock_tokenizer)


def test_delim_multi_field(tf_config, mock_tokenizer):
    check = Settings(config=tf_config, start_string="foo,bar,baz,", tokenizer=mock_tokenizer)
    assert check.start_string == "foo<d>bar<d>baz<d>"


def test_delim_multi_field_multi_starts(tf_config, mock_tokenizer):
    check = Settings(
        config=tf_config,
        start_string=["one,two,three,", "four,five,six,", "seven,eight,nine,"],
        tokenizer=mock_tokenizer
    )
    assert check.start_string == ['one<d>two<d>three<d>', 'four<d>five<d>six<d>', 'seven<d>eight<d>nine<d>']


def test_delim_single_field(tf_config, mock_tokenizer):
    check = Settings(config=tf_config, start_string="onlyonefield,", tokenizer=mock_tokenizer)
    assert check.start_string == "onlyonefield<d>"
