"""
Tensorflow - Keras Sequential RNN (GRU)
"""
from typing import TYPE_CHECKING

from tensorflow.keras.optimizers import RMSprop  # pylint: disable=import-error
import tensorflow as tf

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig
    from gretel_synthetics.tokenizers import BaseTokenizer
else:
    BaseConfig = None
    BaseTokenizer = None


DEFAULT = "default"


def build_sequential_model(
    vocab_size: int,
    batch_size: int,
    store: BaseConfig
) -> tf.keras.Sequential:
    """
    Utilizing tf.keras.Sequential model (LSTM)
    """
    optimizer = RMSprop(learning_rate=0.01)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, store.embedding_dim, batch_input_shape=[batch_size, None]
            ),
            tf.keras.layers.Dropout(store.dropout_rate),
            tf.keras.layers.LSTM(
                store.rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer=store.rnn_initializer,
            ),
            tf.keras.layers.Dropout(store.dropout_rate),
            tf.keras.layers.LSTM(
                store.rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer=store.rnn_initializer,
            ),
            tf.keras.layers.Dropout(store.dropout_rate),
            tf.keras.layers.Dense(vocab_size),
        ])

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def _prepare_model(
    tokenizer: BaseTokenizer, batch_size: int, store: BaseConfig
) -> tf.keras.Sequential:  # pragma: no cover
    model = build_sequential_model(
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
