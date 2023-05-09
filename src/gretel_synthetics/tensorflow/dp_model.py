import importlib
import logging
import math

from typing import List, Tuple, TYPE_CHECKING, Union

import tensorflow as tf

from packaging import version

# The optimizers package has been moved to optimizers.legacy
# post TF 2.10.
if version.parse(tf.__version__) >= version.parse("2.11"):
    from tensorflow.keras.optimizers.legacy import RMSprop
else:
    from tensorflow.keras.optimizers import RMSprop

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    make_keras_optimizer_class,
)

if TYPE_CHECKING:
    from gretel_synthetics.config import TensorFlowConfig
else:
    TensorFlowConfig = None

ORDERS = [1 + x / 20 for x in range(1, 100)] + list(range(6, 64)) + [128, 256, 512]


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


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
        recurrent_v2 = importlib.import_module("keras.layers.recurrent_v2")
        # NOTE: This patches the LSTMs to use the new Keras 2.4.x code paths
        # and will have no effect when the module function is removed
        use_new_code = getattr(recurrent_v2, "_use_new_code", None)
        if use_new_code is not None:
            logging.warning(
                "******* Patching TensorFlow to utilize new Keras code paths, see: %s",
                "https://github.com/tensorflow/tensorflow/issues/44917 *******",
            )  # noqa
            recurrent_v2._use_new_code = (
                lambda: True
            )  # pylint: disable=protected-access
    except ModuleNotFoundError:
        pass

    optimizer = make_keras_optimizer_class(RMSprop)(
        l2_norm_clip=store.dp_l2_norm_clip,
        noise_multiplier=store.dp_noise_multiplier,
        num_microbatches=store.dp_microbatches,
        learning_rate=store.learning_rate,
    )

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
        ]
    )

    logging.info(
        f"Using {optimizer._keras_api_names[0]} optimizer in differentially private mode"
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def compute_dp_sgd_privacy(
    n: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float,
    orders: List[Union[float, int]] = ORDERS,
) -> Tuple[float, float]:
    """Compute epsilon based on the given hyperparameters.
    Adaptation of tensorflow privacy with expanded rdp orders.

    Args:
        n: Number of examples in the training data
        batch_size: Batch size used in training
        noise_multiplier: Noise multiplier used in training
        epochs: Number of epochs in training
        delta: Value of delta for which to compute epsilon

    Returns:
        Tuple of eps, opt_order
    """
    if n <= 0:
        raise ValueError("Number of examples in the training data must be non-zero.")
    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        raise ValueError(
            "Number of training examples must be larger than the batch size."
        )
    steps = int(math.ceil(epochs * n / batch_size))
    return compute_dp_sgd_privacy_lib.apply_dp_sgd_analysis(
        q, noise_multiplier, steps, orders, delta
    )


def compute_epsilon(
    n: int, store: TensorFlowConfig, epoch_number: int = None
) -> Tuple[float, float]:
    """
    Calculate epsilon and delta values for differential privacy

    Returns:
        Tuple of eps, opt_order
    """
    # Note: inverse of number of training samples recommended for minimum
    # delta in differential privacy
    if epoch_number is None:
        epoch_number = store.epochs - 1

    return compute_dp_sgd_privacy(
        n=n,
        batch_size=store.batch_size,
        noise_multiplier=store.dp_noise_multiplier,
        epochs=epoch_number,
        delta=1.0 / float(n),
    )
