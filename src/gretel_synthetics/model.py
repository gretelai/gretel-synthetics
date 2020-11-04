"""
Tensorflow - Keras Sequential RNN (GRU)
"""
from typing import Tuple, TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  # pylint: disable=import-error
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_keras_optimizer_class
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

from gretel_synthetics.default_model import build_default_model
from gretel_synthetics.dp_model import build_dp_model

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig
else:
    BaseConfig = None

DEFAULT = "default"
OPTIMIZERS = {
    DEFAULT: {'dp': make_keras_optimizer_class(RMSprop), 'default': RMSprop}
}


def select_optimizer(store: BaseConfig):
    if store.dp:
        return OPTIMIZERS[DEFAULT]["dp"]
    else:
        return OPTIMIZERS[DEFAULT][DEFAULT]


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def build_model(vocab_size: int, batch_size: int, store: BaseConfig) -> tf.keras.Sequential:
    """
    Utilizing tf.keras.Sequential model
    """
    optimizer_cls = select_optimizer(store)
    model = None

    if store.dp:
        model = build_dp_model(optimizer_cls, store, batch_size, vocab_size)
    else:
        model = build_default_model(optimizer_cls, store, batch_size, vocab_size)

    print(model.summary())
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