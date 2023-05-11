from unittest.mock import MagicMock, Mock, patch

import pytest
import tensorflow as tf

from gretel_synthetics.generate import PredString
from gretel_synthetics.tensorflow.generator import _predict_chars

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
    ret_data = [
        partial_rep
        for partial in ["a", "ab", "abc", "abcd"]
        for partial_rep in [partial] * config.predict_batch_size
    ]
    tokenizer.decode_from_ids.side_effect = ret_data
    # sp.DecodeIds.side_effect = ["a", "b", "c", "d"]
    line = next(_predict_chars(mock_model, tokenizer, NEWLINE, config))
