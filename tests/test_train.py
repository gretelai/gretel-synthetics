from copy import deepcopy
from unittest.mock import Mock, patch

from gretel_synthetics.tokenizers import (
    CharTokenizerTrainer,
    SentencePieceTokenizerTrainer,
)
from gretel_synthetics.train import _create_default_tokenizer, train


def test_default_tokenizer(tf_config):
    config = deepcopy(tf_config)

    config.vocab_size = 15
    tokenizer = _create_default_tokenizer(config)
    assert isinstance(tokenizer, SentencePieceTokenizerTrainer)

    config.vocab_size = 0
    tokenizer = _create_default_tokenizer(config)
    assert isinstance(tokenizer, CharTokenizerTrainer)


@patch("gretel_synthetics.tensorflow.train.build_model")
@patch("gretel_synthetics.tensorflow.train._save_history_csv")
def test_train_calls_train_rnn(save_history, model, tf_config):
    mock_model = Mock()
    model.return_value = mock_model

    train(tf_config)

    model.assert_called_with(
        vocab_size=71,
        batch_size=tf_config.batch_size,
        store=tf_config,
    )

    mock_model.fit.assert_called()
