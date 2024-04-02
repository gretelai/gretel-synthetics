"""
Tensorflow - Keras Sequential RNN (GRU)
"""

import logging

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
from keras import backend as k

logger = logging.getLogger(__name__)


def build_model(
    vocab_size: int, batch_size: int, store: BaseConfig
) -> tf.keras.Sequential:
    """
    Utilizing tf.keras.Sequential model
    """
    model = None

    if store.dp:
        model = build_dp_model(store, batch_size, vocab_size)
    else:
        model = build_default_model(store, batch_size, vocab_size)

    _print_model_summary(model)
    return model


def _prepare_model(
    tokenizer: BaseTokenizer, batch_size: int, store: BaseConfig
) -> tf.keras.Sequential:  # pragma: no cover
    config = k.get_config()

    # Don't pre-allocate memory, allocate as needed
    config.gpu_options.allow_growth = True

    k.set_session(tf.compat.v1.Session(config=config))

    model = build_model(
        vocab_size=tokenizer.total_vocab_size, batch_size=batch_size, store=store
    )

    load_dir = store.checkpoint_dir

    model.load_weights(tf.train.latest_checkpoint(load_dir)).expect_partial()

    model.build(tf.TensorShape([1, None]))

    _print_model_summary(model)
    return model


def _print_model_summary(model: tf.keras.Model) -> None:
    model_summary = ""

    def print_fn(line: str) -> None:
        nonlocal model_summary
        model_summary += "\t" + line + "\n"

    model.summary(print_fn=print_fn)
    logger.info("Model summary: \n%s", model_summary)


def load_model(
    store: BaseConfig,
    tokenizer: BaseTokenizer,
) -> tf.keras.Sequential:
    model = _prepare_model(tokenizer, store.predict_batch_size, store)
    return model
