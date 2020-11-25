from typing import Tuple, TYPE_CHECKING
import logging
import importlib

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_keras_optimizer_class

if TYPE_CHECKING:
    from gretel_synthetics.config import TensorFlowConfig
else:
    TensorFlowConfig = None


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def build_dp_model(store, batch_size, vocab_size) -> tf.keras.Sequential:
    """
    Build a RNN-based sequential model with differentially private training (Experimental)

    Args:
        store: LocalConfig
        batch_size: Batch size for training and prediction
        vocab_size: Size of training vocabulary

    Returns:
        tf.keras.Sequential model
    """
    logging.warning("Experimental: Differentially private training enabled")

    try:
        recurrent_v2 = importlib.import_module("tensorflow.python.keras.layers.recurrent_v2")
        # NOTE: This patches the LSTMs to use the new Keras 2.4.x code paths
        # and will have no effect when the module function is removed
        use_new_code = getattr(recurrent_v2, "_use_new_code", None)
        if use_new_code is not None:
            logging.warning("******* Patching TensorFlow to utilize new Keras code paths, see: %s", "https://github.com/tensorflow/tensorflow/issues/44917 *******")  # noqa
            recurrent_v2._use_new_code = lambda: True  # pylint: disable=protected-access
    except ModuleNotFoundError:
        pass

    optimizer = make_keras_optimizer_class(RMSprop)(
        l2_norm_clip=store.dp_l2_norm_clip,
        noise_multiplier=store.dp_noise_multiplier,
        num_microbatches=store.dp_microbatches,
        learning_rate=store.learning_rate
    )

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

    logging.info(f"Using {optimizer._keras_api_names[0]} optimizer in differentially private mode")
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def compute_epsilon(steps: int, store: TensorFlowConfig, epoch_number: int = None) -> Tuple[float, float]:
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
