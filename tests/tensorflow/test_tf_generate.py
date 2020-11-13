from unittest.mock import MagicMock, patch, Mock
import json

import pytest
import tensorflow as tf

from gretel_synthetics.tensorflow.generator import _predict_chars
from gretel_synthetics.generate import generate_text, PredString


NEWLINE = "<n>"

@pytest.fixture
def random_cat():
    return tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)


@patch("tensorflow.random.categorical")
def test_predict_chars(mock_cat, tf_config, random_cat):
    config = tf_config

    tf_config.gen_chars = 10
    mock_model = Mock(return_value=tf.constant([[[1.0]]]))
    mock_tensor = MagicMock()
    mock_tensor[-1, 0].numpy.return_value = 1
    mock_cat.return_value = mock_tensor

    tokenizer = Mock()
    tokenizer.newline_str = NEWLINE
    tokenizer.encode_to_ids.return_value = [3]
    tokenizer.decode_from_ids.return_value = f"this is the end{NEWLINE}"
    # sp.DecodeIds.side_effect = ["this", " ", "is", " ", "the", " ", "end", "<n>"]
    line = next(_predict_chars(mock_model, tokenizer, NEWLINE, config))
    assert line == PredString(data="this is the end")
 
    config = tf_config
    mock_tensor = MagicMock()
    mock_tensor[-1, 0].numpy.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mock_cat.return_value = mock_tensor
    tf_config.gen_chars = 3
    tokenizer = Mock()
    tokenizer.newline_str = NEWLINE
    tokenizer.encode_to_ids.return_value = [3]
    ret_data = [partial_rep for partial in ["a", "ab", "abc", "abcd"] for partial_rep in [partial] * config.predict_batch_size]
    tokenizer.decode_from_ids.side_effect = ret_data
    # sp.DecodeIds.side_effect = ["a", "b", "c", "d"]
    line = next(_predict_chars(mock_model, tokenizer, NEWLINE, config))


@patch("gretel_synthetics.tokenizers.SentencePieceTokenizer.load")
@patch("gretel_synthetics.tensorflow.generator._predict_chars")
@patch("gretel_synthetics.tensorflow.model._prepare_model")
@patch("pickle.load")
@patch("gretel_synthetics.generate.open")
def test_generate_text(_open, pickle, prepare, predict, spm, tf_config):
    tf_config.gen_lines = 10
    predict.side_effect = [[PredString(json.dumps({"foo": i}))] for i in range(0, 10)]
    out = []

    tokenizer = Mock()
    spm.return_value = tokenizer

    for rec in generate_text(tf_config, line_validator=json.loads, parallelism=1):
        out.append(rec.as_dict())

    assert len(out) == 10
    assert out[0] == {
        "valid": True,
        "text": '{"foo": 0}',
        "explain": None,
        "delimiter": ",",
    }

    # now with no validator
    predict.side_effect = [[PredString(json.dumps({"foo": i}))] for i in range(0, 10)]
    out = []
    for rec in generate_text(tf_config, parallelism=1):
        out.append(rec.as_dict())
    assert len(out) == 10
    assert out[0] == {
        "valid": None,
        "text": '{"foo": 0}',
        "explain": None,
        "delimiter": ",",
    }

    # add validator back in, with a few bad json strings
    predict.side_effect = [
        [PredString(json.dumps({"foo": i})) for i in range(0, 3)],
        [PredString("nope"), PredString("foo"), PredString("bar")],
        [PredString(json.dumps({"foo": i})) for i in range(6, 10)],
    ]
    out = []
    try:
        for rec in generate_text(tf_config, line_validator=json.loads, parallelism=1):
            out.append(rec.as_dict())
    except RuntimeError:
        pass
    assert len(out) == 10
    assert not out[4]["valid"]

    # assert max invalid
    predict.side_effect = [
        [PredString(json.dumps({"foo": i})) for i in range(0, 3)],
        [PredString("nope"), PredString("foo"), PredString("bar")],
        [PredString(json.dumps({"foo": i})) for i in range(6, 10)],
    ]
    out = []
    try:
        for rec in generate_text(tf_config, line_validator=json.loads, max_invalid=2, parallelism=1):
            out.append(rec.as_dict())
    except RuntimeError as err:
        assert "Maximum number" in str(err)
    assert len(out) == 6
    assert not out[4]["valid"]

    # max invalid, validator returns a bool
    def _val(line):
        try:
            json.loads(line)
        except Exception:
            return False
        else:
            return True

    predict.side_effect = [
        [PredString(json.dumps({"foo": i})) for i in range(0, 3)],
        [PredString("nope"), PredString("foo"), PredString("bar")],
        [PredString(json.dumps({"foo": i})) for i in range(6, 10)],
    ]
    out = []
    try:
        for rec in generate_text(tf_config, line_validator=_val, max_invalid=2, parallelism=1):
            out.append(rec.as_dict())
    except RuntimeError as err:
        assert "Maximum number" in str(err)
    assert len(out) == 6
    assert not out[4]["valid"]
