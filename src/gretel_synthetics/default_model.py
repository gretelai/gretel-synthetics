import logging
import tensorflow as tf


def build_default_model(optimizer_cls, store, batch_size, vocab_size) -> tf.keras.Sequential:
    """
    Build a RNN-based sequential model

    Args:
        optimizer_cls: tf.keras.optimizer class
        store: LocalConfig
        batch_size: Batch size for training and prediction
        vocab_size: Size of training vocabulary

    Returns:
        tf.keras.Sequential model
    """
    optimizer = optimizer_cls(
        learning_rate=store.learning_rate
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, store.embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(store.dropout_rate),
        tf.keras.layers.GRU(store.rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer=store.rnn_initializer),
        tf.keras.layers.Dropout(store.dropout_rate),
        tf.keras.layers.GRU(store.rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer=store.rnn_initializer),
        tf.keras.layers.Dropout(store.dropout_rate),
        tf.keras.layers.Dense(vocab_size)
    ])

    logging.info(f"Using {optimizer._keras_api_names[0]} optimizer")
    return model
