from unittest.mock import MagicMock, patch, Mock
import json

import pytest
import tensorflow as tf

from gretel_synthetics.generator import _predict_chars
from gretel_synthetics.generate import generate_text, PredString


@pytest.fixture
def random_cat():
    return tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)


@patch("tensorflow.random.categorical")
@patch("tensorflow.expand_dims")
def test_predict_chars(mock_dims, mock_cat, global_local_config, random_cat):
    global_local_config.gen_chars = 0
    mock_model = Mock(return_value=[1.0])
    mock_tensor = MagicMock()
    mock_tensor[-1, 0].numpy.return_value = 1
    mock_cat.return_value = mock_tensor

    sp = Mock()
    sp.DecodeIds.side_effect = ["this", " ", "is", " ", "the", " ", "end", "<n>"]

    line = _predict_chars(mock_model, sp, "\n", global_local_config)
    assert line == PredString(data="this is the end")

    mock_tensor = MagicMock()
    mock_tensor[-1, 0].numpy.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mock_cat.return_value = mock_tensor
    global_local_config.gen_chars = 3
    sp = Mock()
    sp.DecodeIds.side_effect = ["a", "b", "c", "d"]
    line = _predict_chars(mock_model, sp, "\n", global_local_config)
    assert line.data == "abc"


@patch("gretel_synthetics.generator.spm.SentencePieceProcessor")
@patch("gretel_synthetics.generator._predict_chars")
@patch("gretel_synthetics.generator._prepare_model")
@patch("pickle.load")
@patch("gretel_synthetics.generate.open")
def test_generate_text(_open, pickle, prepare, predict, spm, global_local_config):
    global_local_config.gen_lines = 10
    predict.side_effect = [PredString(json.dumps({"foo": i})) for i in range(0, 10)]
    out = []

    sp = Mock()
    spm.return_value = sp

    for rec in generate_text(global_local_config, line_validator=json.loads, parallelism=1):
        out.append(rec.as_dict())

    assert len(out) == 10
    assert out[0] == {
        "valid": True,
        "text": '{"foo": 0}',
        "explain": None,
        "delimiter": ",",
    }

    # now with no validator
    predict.side_effect = [PredString(json.dumps({"foo": i})) for i in range(0, 10)]
    out = []
    for rec in generate_text(global_local_config, parallelism=1):
        out.append(rec.as_dict())
    assert len(out) == 10
    assert out[0] == {
        "valid": None,
        "text": '{"foo": 0}',
        "explain": None,
        "delimiter": ",",
    }

    # add validator back in, with a few bad json strings
    predict.side_effect = (
        [PredString(json.dumps({"foo": i})) for i in range(0, 3)]
        + [PredString("nope"), PredString("foo"), PredString("bar")]
        + [PredString(json.dumps({"foo": i})) for i in range(6, 10)]
    )
    out = []
    try:
        for rec in generate_text(global_local_config, line_validator=json.loads, parallelism=1):
            out.append(rec.as_dict())
    except RuntimeError:
        pass
    assert len(out) == 10
    assert not out[4]["valid"]

    # assert max invalid
    predict.side_effect = (
        [PredString(json.dumps({"foo": i})) for i in range(0, 3)]
        + [PredString("nope"), PredString("foo"), PredString("bar")]
        + [PredString(json.dumps({"foo": i})) for i in range(6, 10)]
    )
    out = []
    try:
        for rec in generate_text(global_local_config, line_validator=json.loads, max_invalid=2, parallelism=1):
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

    predict.side_effect = (
        [PredString(json.dumps({"foo": i})) for i in range(0, 3)]
        + [PredString("nope"), PredString("foo"), PredString("bar")]
        + [PredString(json.dumps({"foo": i})) for i in range(6, 10)]
    )
    out = []
    try:
        for rec in generate_text(global_local_config, line_validator=_val, max_invalid=2, parallelism=1):
            out.append(rec.as_dict())
    except RuntimeError as err:
        assert "Maximum number" in str(err)
    assert len(out) == 6
    assert not out[4]["valid"]
