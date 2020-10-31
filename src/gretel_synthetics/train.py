"""
Abstract model training module
"""
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gretel_synthetics.base_config import BaseConfig
else:
    BaseConfig = None


def train(store: BaseConfig):
    """Train a Synthetic Model.
    """
    train_fn = store.get_training_callable()
    train_fn(store)


def train_rnn(store: BaseConfig):
    """
    FIXME: Facade pass through to maintain backwards compat. Just call into
    the new train interface
    """
    train(store)
