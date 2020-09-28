"""
Tensorflow - Keras Sequential RNN (GRU)
"""
import logging
from typing import Tuple

from tensorflow.keras.optimizers import RMSprop  # pylint: disable=import-error
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer import (
    make_gaussian_optimizer_class as make_dp_optimizer
 )
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow.python.keras.engine import data_adapter


from gretel_synthetics.config import BaseConfig


class DPSequentialModel(tf.keras.Sequential):

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape():
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses
            )

            trainable_variables = self.trainable_variables
            # gradients = tape.gradient(loss, trainable_variables)
            gradients = self.optimizer.optimizer.compute_gradients(
                loss, trainable_variables
            )
            self.optimizer.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


def build_sequential_model(
    vocab_size: int, batch_size: int, store: BaseConfig
) -> tf.keras.Sequential:
    """
    Utilizing tf.keras.Sequential model (LSTM)
    """
    model_cls = tf.keras.Sequential
    if store.dp_custom and store.dp:
        logging.info("Using custom training loop")
        model_cls = DPSequentialModel

    model = model_cls(
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

    if store.dp:
        logging.info("Differentially private training enabled")

        if store.dp_custom:
            logging.info("Building DP RMS optimizer")
            rms_prop_optimizer = tf.compat.v1.train.RMSPropOptimizer
            dp_rms_prop_optimizer = make_dp_optimizer(rms_prop_optimizer)

            optimizer = dp_rms_prop_optimizer(
                l2_norm_clip=store.dp_l2_norm_clip,
                noise_multiplier=store.dp_noise_multiplier,
                num_microbatches=store.dp_microbatches,
                learning_rate=store.dp_learning_rate
            )
        else:
            logging.info("Using DP Keras Adam optimizer")
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
