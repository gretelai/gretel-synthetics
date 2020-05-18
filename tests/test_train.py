from unittest.mock import patch, Mock, MagicMock

from gretel_synthetics.train import train_rnn, _train_tokenizer


def test_create_vocab(global_local_config):
    sp = _train_tokenizer(global_local_config)
    assert len(sp) == 71
    assert sp.PieceToId('</s>') == 2
    assert sp.IdToPiece(2) == '</s>'


@patch('gretel_synthetics.train._build_sequential_model')
def test_train_rnn(model, global_local_config):
    mock_model = Mock()
    model.return_value = mock_model

    train_rnn(global_local_config)
    
    model.assert_called_with(
        vocab_size=71,
        batch_size=global_local_config.batch_size,
        store=global_local_config
    )

    mock_model.fit.assert_called
