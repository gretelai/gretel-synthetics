"""
Tensorflow - Keras Sequential RNN (GRU)
"""
from typing import TYPE_CHECKING

import tensorflow as tf

from gretel_synthetics.tensorflow.default_model import build_default_model
from gretel_synthetics.tensorflow.dp_model import build_dp_model

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig
    from gretel_synthetics.tokenizers import BaseTokenizer
else:
    BaseConfig = None
    BaseTokenizer = None


def build_model(vocab_size: int, batch_size: int, store: BaseConfig) -> tf.keras.Sequential:
    """
    Utilizing tf.keras.Sequential model
    """
    model = None

    if store.dp:
        model = build_dp_model(store, batch_size, vocab_size)
    else:
        model = build_default_model(store, batch_size, vocab_size)

    print(model.summary())
    return model


def _prepare_model(
    tokenizer: BaseTokenizer, batch_size: int, store: BaseConfig
) -> tf.keras.Sequential:  # pragma: no cover
    model = build_model(
        vocab_size=tokenizer.total_vocab_size, batch_size=batch_size, store=store
    )

    load_dir = store.checkpoint_dir

    model.load_weights(tf.train.latest_checkpoint(load_dir)).expect_partial()

    model.build(tf.TensorShape([1, None]))
    model.summary()

    return model


def load_model(
    store: BaseConfig,
    tokenizer: BaseTokenizer,
) -> tf.keras.Sequential:
    model = _prepare_model(tokenizer, store.predict_batch_size, store)
    return model
