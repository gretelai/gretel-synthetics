import logging
import tensorflow as tf


def build_dp_model(optimizer_cls, store, batch_size, vocab_size) -> tf.keras.Sequential:
    """
    Build a RNN-based sequential model with differentially private training (Experimental)

    Args:
        optimizer_cls: Differentially private optimizer class
        store: LocalConfig
        batch_size: Batch size for training and prediction
        vocab_size: Size of training vocabulary

    Returns:
        tf.keras.Sequential model
    """
    logging.warning("Experimental: Differentially private training enabled")
    optimizer = optimizer_cls(
        l2_norm_clip=store.dp_l2_norm_clip,
        noise_multiplier=store.dp_noise_multiplier,
        num_microbatches=store.dp_microbatches,
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

    logging.info(f"Using {optimizer._keras_api_names[0]} optimizer in differentially private mode")
    return model
