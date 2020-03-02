"""
Tensorflow - Keras Sequential RNN (GRU)
"""
import logging

import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer import make_gaussian_optimizer_class as make_dp_optimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

from gretel_synthetics.config import BaseConfig


def build_sequential_model(vocab_size: int, batch_size: int, store: BaseConfig) -> tf.keras.Sequential:
    """
    Utilizing tf.keras.Sequential model (LSTM)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, store.embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(store.dropout_rate),
        tf.keras.layers.LSTM(store.rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer=store.rnn_initializer),
        tf.keras.layers.Dropout(store.dropout_rate),
        tf.keras.layers.LSTM(store.rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer=store.rnn_initializer),
        tf.keras.layers.Dropout(store.dropout_rate),
        tf.keras.layers.Dense(vocab_size)
    ])

    if store.dp:
        logging.info("Utilizing differential privacy in optimizer")

        RMSPropOptimizer = tf.compat.v1.train.RMSPropOptimizer
        DPRmsPropGaussianOptimizer = make_dp_optimizer(RMSPropOptimizer)

        optimizer = DPRmsPropGaussianOptimizer(
            l2_norm_clip=store.dp_l2_norm_clip,
            noise_multiplier=store.dp_noise_multiplier,
            num_microbatches=store.dp_microbatches,
            learning_rate=store.dp_learning_rate)

        """
        Compute vector of per-example loss rather than its mean over a minibatch.
        To support gradient manipulation over each training point.
        """
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction=tf.losses.Reduction.NONE)

    else:
        logging.info("Utilizing non-private optimizer")
        optimizer = 'adam'
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    logging.info(model.summary())
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def compute_epsilon(steps: int, store: BaseConfig):
    # Note: inverse of number of training samples recommended for minimum
    # delta in differential privacy
    return compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=steps,
                                                         batch_size=store.batch_size,
                                                         noise_multiplier=store.dp_noise_multiplier,
                                                         epochs=store.epochs,
                                                         delta=1.0 / float(steps))
