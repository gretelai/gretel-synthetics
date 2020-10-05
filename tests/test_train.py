import pytest
from copy import deepcopy

from unittest.mock import patch, Mock, MagicMock
import pandas as pd

from gretel_synthetics.train import (
    train_rnn,
    _train_tokenizer,
    _ModelHistory,
    _save_history_csv,
    VAL_LOSS,
    VAL_ACC,
)


def test_create_vocab(global_local_config):
    sp = _train_tokenizer(global_local_config)
    assert len(sp) == 71
    assert sp.PieceToId("</s>") == 2
    assert sp.IdToPiece(2) == "</s>"


@patch("gretel_synthetics.train.build_sequential_model")
@patch("gretel_synthetics.train._save_history_csv")
def test_train_rnn(save_history, model, global_local_config):
    mock_model = Mock()
    model.return_value = mock_model

    train_rnn(global_local_config)

    model.assert_called_with(
        vocab_size=71,
        batch_size=global_local_config.batch_size,
        store=global_local_config,
    )

    mock_model.fit.assert_called


@pytest.fixture(scope="module")
def history():
    hist = _ModelHistory(None, None)
    values = [1, 2, 3, 4, 5, 6]
    hist.losses = deepcopy(values)
    hist.accuracy = deepcopy(values)
    hist.epsilons = deepcopy(values)
    hist.deltas = deepcopy(values)
    hist.best = [0] * 6
    return hist


def test_save_history_no_best(tmpdir, history):
    _save_history_csv(history, tmpdir, True, "loss", None)
    df = pd.read_csv(tmpdir + "/model_history.csv")
    check = df[df["best"] == 1]
    assert list(check["epoch"])[0] == 5


def test_save_history_best_loss(tmpdir, history):
    _save_history_csv(history, tmpdir, True, VAL_LOSS, 3)
    df = pd.read_csv(tmpdir + "/model_history.csv")
    check = df[df["best"] == 1]
    assert list(check["epoch"])[0] == 2


def test_save_history_best_acc(tmpdir, history):
    _save_history_csv(history, tmpdir, True, VAL_ACC, 1)
    df = pd.read_csv(tmpdir + "/model_history.csv")
    check = df[df["best"] == 1]
    assert list(check["epoch"])[0] == 0


def test_save_history_best_not_valid(tmpdir, history):
    _save_history_csv(history, tmpdir, True, VAL_ACC, 11)
    df = pd.read_csv(tmpdir + "/model_history.csv")
    check = df[df["best"] == 1]
    assert list(check["epoch"])[0] == 5
