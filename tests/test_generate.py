import pytest

from gretel_synthetics.generate import Settings
from gretel_synthetics.tensorflow.generator import _replace_decoded_tokens
from gretel_synthetics.const import NEWLINE
from gretel_synthetics.tensorflow.config import TensorFlowConfig
from gretel_synthetics.errors import GenerationError


def test_default_start_string(tf_config):
    check = Settings(config=tf_config)
    assert check.start_string == NEWLINE


def test_no_delim_bad_start_string(tmpdir):
    config = TensorFlowConfig(checkpoint_dir=tmpdir, input_data_path=tmpdir)
    with pytest.raises(GenerationError):
        Settings(config=config, start_string=123)


def test_delim_missing_trailing_delim(tf_config):
    with pytest.raises(GenerationError):
        Settings(config=tf_config, start_string="foo,bar")


def test_delim_multi_field(tf_config):
    check = Settings(config=tf_config, start_string="foo,bar,baz,")
    assert check.start_string == "foo<d>bar<d>baz<d>"


def test_delim_single_field(tf_config):
    check = Settings(config=tf_config, start_string="onlyonefield,")
    assert check.start_string == "onlyonefield<d>"


@pytest.fixture
def batch_decoded():
    return [
        (0, "foo<d>bar<d>baz"),
        (1, "one<d>two<d>three"),
        (2, "uno<d>dos<d>tres")
    ]


def test_no_prefix_decode_tokens(tmpdir, batch_decoded):
    config = TensorFlowConfig(checkpoint_dir=tmpdir, input_data_path=tmpdir)
    check = _replace_decoded_tokens(batch_decoded, config, prefix=None)
    assert check == [(0, 'foo<d>bar<d>baz'), (1, 'one<d>two<d>three'), (2, 'uno<d>dos<d>tres')]


def test_no_prefix_delim_decode_tokens(tmpdir, batch_decoded):
    config = TensorFlowConfig(checkpoint_dir=tmpdir, input_data_path=tmpdir, field_delimiter=":")
    check = _replace_decoded_tokens(batch_decoded, config, prefix=None)
    assert check == [(0, 'foo:bar:baz'), (1, 'one:two:three'), (2, 'uno:dos:tres')]


def test_prefix_delim_decode_tokens(tmpdir, batch_decoded):
    config = TensorFlowConfig(checkpoint_dir=tmpdir, input_data_path=tmpdir, field_delimiter=":")
    check = _replace_decoded_tokens(batch_decoded, config, prefix="hello:world:")
    assert check == [(0, 'hello:world:foo:bar:baz'), (1, 'hello:world:one:two:three'), (2, 'hello:world:uno:dos:tres')]