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


def test_delim_multi_field(tf_config):
    check = Settings(config=tf_config, start_string="foo,bar,baz,", tokenizer=mock_tokenizer)
    assert check.start_string == "foo<d>bar<d>baz<d>"


def test_delim_single_field(tf_config):
    check = Settings(config=tf_config, start_string="onlyonefield,", tokenizer=mock_tokenizer)
    assert check.start_string == "onlyonefield<d>"
