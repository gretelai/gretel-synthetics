import os
import pytest
from unittest.mock import patch, Mock, MagicMock

from gretel_synthetics.train import train_rnn, train_tokenizer


def test_create_vocab(global_local_config):
    sp = train_tokenizer(global_local_config)
    assert len(sp) == 72
    assert sp.PieceToId('</s>') == 2
    assert sp.IdToPiece(2) == '</s>'


@patch('gretel_synthetics.train.build_sequential_model')
def test_train_rnn(model, global_local_config):
    mock_model = Mock()
    model.return_value = mock_model

    train_rnn(global_local_config)
    
    model.assert_called_with(
        vocab_size=72,
        batch_size=global_local_config.batch_size,
        store=global_local_config
    )

    mock_model.fit.assert_called

    # let's re-run with a much smaller max_chars value
    """
    mock_model = Mock()
    model.return_value = mock_model
    global_local_config.max_chars = 3
    train_rnn(global_local_config)

    model.assert_called_with(
        vocab_size=3,
        batch_size=global_local_config.batch_size,
        store=global_local_config
    )

    mock_model.fit.assert_called
    """