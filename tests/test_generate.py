from unittest.mock import MagicMock, patch, Mock
import json

import pytest 
import tensorflow as tf

from gretel_synthetics.generate import generate_text, _predict_chars, _pred_string


@pytest.fixture
def random_cat():
    return tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)


@patch('tensorflow.random.categorical')
@patch('tensorflow.expand_dims')
def test_predict_chars(mock_dims, mock_cat, global_local_config, random_cat):
    global_local_config.gen_chars = 10
    mock_model = Mock(return_value=[1.0])
    mock_tensor = MagicMock()
    mock_tensor[-1, 0].numpy.return_value = 1
    mock_cat.return_value = mock_tensor

    sp = Mock()
    sp.DecodeIds.return_value = 'this is the end<n>'

    line = _predict_chars(mock_model, sp, '\n', global_local_config)
    assert line == _pred_string(data='this is the end')

    mock_tensor = MagicMock()
    mock_tensor[-1, 0].numpy.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mock_cat.return_value = mock_tensor
    global_local_config.gen_chars = 3
    sp = Mock()
    sp.DecodeIds.side_effect = ['a', 'ab', 'abc', 'abcd']
    line = _predict_chars(mock_model, sp, '\n', global_local_config)
    assert line.data == 'abc'
    

@patch('gretel_synthetics.generate.spm.SentencePieceProcessor')
@patch('gretel_synthetics.generate._predict_chars')
@patch('gretel_synthetics.generate._prepare_model')
@patch('pickle.load')
@patch('gretel_synthetics.generate.open')
def test_generate_text(_open, pickle, prepare, predict, spm, global_local_config):
    global_local_config.gen_lines = 10
    predict.side_effect = [_pred_string(json.dumps({'foo': i})) for i in range(0, 10)]
    out = []

    sp = Mock()
    spm.return_value = sp

    for rec in generate_text(global_local_config, line_validator=json.loads):
        out.append(rec)

    assert out == [{'valid': True, 'text': '{"foo": 0}', 'explain': None}, {'valid': True, 'text': '{"foo": 1}', 'explain': None}, {'valid': True, 'text': '{"foo": 2}', 'explain': None}, {'valid': True, 'text': '{"foo": 3}', 'explain': None}, {'valid': True, 'text': '{"foo": 4}', 'explain': None}, {'valid': True, 'text': '{"foo": 5}', 'explain': None}, {'valid': True, 'text': '{"foo": 6}', 'explain': None}, {'valid': True, 'text': '{"foo": 7}', 'explain': None}, {'valid': True, 'text': '{"foo": 8}', 'explain': None}, {'valid': True, 'text': '{"foo": 9}', 'explain': None}]
    
    # now with no validator, should be same result
    predict.side_effect = [_pred_string(json.dumps({'foo': i})) for i in range(0, 10)]
    out = []
    for rec in generate_text(global_local_config):
        out.append(rec)
    assert out == [{'valid': None, 'text': '{"foo": 0}', 'explain': None}, {'valid': None, 'text': '{"foo": 1}', 'explain': None}, {'valid': None, 'text': '{"foo": 2}', 'explain': None}, {'valid': None, 'text': '{"foo": 3}', 'explain': None}, {'valid': None, 'text': '{"foo": 4}', 'explain': None}, {'valid': None, 'text': '{"foo": 5}', 'explain': None}, {'valid': None, 'text': '{"foo": 6}', 'explain': None}, {'valid': None, 'text': '{"foo": 7}', 'explain': None}, {'valid': None, 'text': '{"foo": 8}', 'explain': None}, {'valid': None, 'text': '{"foo": 9}', 'explain': None}]    


    # add validator back in, with a few bad json strings
    predict.side_effect = [_pred_string(json.dumps({'foo': i})) for i in range(0, 3)] + [_pred_string('nope'), _pred_string('foo'), _pred_string('bar')] + [_pred_string(json.dumps({'foo': i})) for i in range(6, 10)]
    out = []
    for rec in generate_text(global_local_config, line_validator=json.loads):
        out.append(rec)
    assert out == [{'valid': True, 'text': '{"foo": 0}', 'explain': None}, {'valid': True, 'text': '{"foo": 1}', 'explain': None}, {'valid': True, 'text': '{"foo": 2}', 'explain': None}, {'valid': False, 'text': 'nope', 'explain': 'Expecting value: line 1 column 1 (char 0)'}, {'valid': False, 'text': 'foo', 'explain': 'Expecting value: line 1 column 1 (char 0)'}, {'valid': False, 'text': 'bar', 'explain': 'Expecting value: line 1 column 1 (char 0)'}, {'valid': True, 'text': '{"foo": 6}', 'explain': None}, {'valid': True, 'text': '{"foo": 7}', 'explain': None}, {'valid': True, 'text': '{"foo": 8}', 'explain': None}, {'valid': True, 'text': '{"foo": 9}', 'explain': None}]
