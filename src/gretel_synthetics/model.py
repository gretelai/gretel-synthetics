"""
Tensorflow - Keras Sequential RNN (GRU)
"""
import logging
from typing import Tuple

from tensorflow.keras.optimizers import RMSprop  # pylint: disable=import-error
import tensorflow as tf

from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdagradOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

from gretel_synthetics.config import BaseConfig

optimizers = {
    'Adagrad': {'dp': DPKerasAdagradOptimizer, 'default': Adagrad},
    'Adam': {'dp': DPKerasAdamOptimizer, 'default': Adam},
    'SGD': {'dp': DPKerasSGDOptimizer, 'default': SGD},
    'default': {'dp': DPKerasAdamOptimizer, 'default': RMSprop},
}


def select_optimizer(optimizer: str, store: BaseConfig):
    """
    Args:
        optimizer: Select default optimizer. 'Adagrad', 'Adam', 'SGD', 'default' are supported

    Returns:
        optimizer class
    """
    if optimizer in optimizers.keys():
        if store.dp:
            return optimizers[optimizer]['dp']
        else:
            return optimizers[optimizer]['default']
    else:
        logging.error("Invalid optimizer selected in configuration")
        raise NotImplementedError


def build_sequential_model(
    vocab_size: int, batch_size: int, store: BaseConfig
) -> tf.keras.Sequential:
    """
    Utilizing tf.keras.Sequential model (LSTM)
    """
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
        ] )

    if store.dp:
        logging.info("Differentially private training enabled")
        optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=store.dp_l2_norm_clip,
            noise_multiplier=store.dp_noise_multiplier,
            num_microbatches=store.dp_microbatches,
            learning_rate=store.dp_learning_rate
        )

        # Compute vector of per-example loss rather than its mean over a minibatch.
        # To support gradient manipulation over each training point.
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.losses.Reduction.NONE
        )
    else:
        logging.warning("Differentially private training _not_ enabled")
        optimizer = RMSprop(learning_rate=0.01)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    logging.info(f"Using {optimizer._name} optimizer {'in differentially private mode' if store.dp else ''}")
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def compute_epsilon(steps: int, store: BaseConfig, epoch_number: int = None) -> Tuple[float, float]:
    """
    Calculate epsilon and delta values for differential privacy

    Returns:
        Tuple of eps, opt_order
    """
    # Note: inverse of number of training samples recommended for minimum
    # delta in differential privacy
    if epoch_number is None:
        epoch_number = store.epochs - 1
    return compute_dp_sgd_privacy.compute_dp_sgd_privacy(
        n=steps,
        batch_size=store.batch_size,
        noise_multiplier=store.dp_noise_multiplier,
        epochs=epoch_number,
        delta=1.0 / float(steps),
    )
