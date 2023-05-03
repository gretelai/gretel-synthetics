import threading

from typing import Any, Iterator, List
from unittest.mock import Mock

import pytest

from gretel_synthetics.batch import Batch, RecordFactory
from gretel_synthetics.config import TensorFlowConfig
from gretel_synthetics.const import NEWLINE
from gretel_synthetics.errors import GenerationError
from gretel_synthetics.generate import gen_text, Settings


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
    check = Settings(
        config=tf_config, start_string="foo,bar,baz,", tokenizer=mock_tokenizer
    )
    assert check.start_string == "foo<d>bar<d>baz<d>"


def test_delim_multi_field_multi_starts(tf_config, mock_tokenizer):
    check = Settings(
        config=tf_config,
        start_string=["one,two,three,", "four,five,six,", "seven,eight,nine,"],
        tokenizer=mock_tokenizer,
    )
    assert check.start_string == [
        "one<d>two<d>three<d>",
        "four<d>five<d>six<d>",
        "seven<d>eight<d>nine<d>",
    ]


def test_delim_single_field(tf_config, mock_tokenizer):
    check = Settings(
        config=tf_config, start_string="onlyonefield,", tokenizer=mock_tokenizer
    )
    assert check.start_string == "onlyonefield<d>"


def test_generate_doesnt_return_partial_record_when_stopped(tmp_path):
    """
    Test for a case, when generation is stopped by the _thread_event signal.
    Makes sure that we are not returning a partially generated records.
    """
    dummy_dir = str(tmp_path)
    dummy_config = TensorFlowConfig(
        checkpoint_dir=dummy_dir, input_data_path=dummy_dir, field_delimiter="|"
    )

    rf = RecordFactory(
        num_lines=10, batches={}, header_list=["colA", "colB"], delimiter="|"
    )
    generators = [
        (_dummy_batch(["colA"], dummy_config), _gen_and_set_thread_event(rf)),
        (_dummy_batch(["colB"], dummy_config), _just_gen("123.33")),
    ]

    record = rf._generate_record(generators)
    assert record == {"colA": "world", "colB": "123.33"}
    assert rf._counter.invalid_count == 1

    record = rf._generate_record(generators)
    assert record is None


def test_generate_resets_previous_progress(tmp_path):
    dummy_dir = str(tmp_path)

    rf = RecordFactory(
        num_lines=10,
        batches={},
        header_list=["colA", "colB"],
        delimiter="|",
    )

    reset_called = False

    def callback(_: Any, *, reset=False):
        if reset is True:
            nonlocal reset_called
            reset_called = True

    result = rf.generate_all(
        callback=callback, callback_interval=2, callback_threading=True
    )

    assert reset_called is True
    # all will be empty, as there are no batches configured
    assert result.records == [{}] * 10


def _gen_and_set_thread_event(factory: RecordFactory) -> Iterator[gen_text]:
    """
    This generator:
    - yields an invalid text
    - yields a valid text
    - sets the _thread_event on the factory instance and yields a valid text
    """
    yield gen_text(valid=False, text="hello", delimiter="|")
    yield gen_text(valid=True, text="world", delimiter="|")

    event = threading.Event()
    event.set()
    factory._thread_event = event
    yield gen_text(valid=True, text="world", delimiter="|")


def _just_gen(value: str) -> Iterator[gen_text]:
    while True:
        yield gen_text(valid=True, text=value, delimiter="|")


def _dummy_batch(headers: List[str], conf: TensorFlowConfig):
    return Batch(
        checkpoint_dir="dummy",
        input_data_path="dummy",
        headers=headers,
        config=conf,
    )
