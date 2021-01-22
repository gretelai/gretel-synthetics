import logging
from dataclasses import dataclass

from gretel_synthetics.config import BaseConfig


@dataclass
class TensorFlowConfig(BaseConfig):
    """TensorFlow config that contains all of the main parameters for
    training a model and generating data.

    Args:
        epochs (optional): Number of epochs to train the model. An epoch is an iteration over the entire
            training set provided. For production use cases, 15-50 epochs are recommended.
            The default is ``100`` and is intentionally set extra high.  By default, ``early_stopping``
            is also enabled and will stop training epochs once the model is no longer improving.
        early_stopping (optional). Defaults to ``True``.  If enabled, regardless of the number of epochs, automatically
            deduce when the model is no longer improving and terminating training.
        early_stopping_patience (optional). Defaults to 5.  Number of epochs to wait for when there is no improvement
            in the model. After this number of epochs, training will terminate.
        best_model_metric (optional). Defaults to "loss". The metric to use to track when a model is no
            longer improving. Defaults to the loss value. An alternative option is "accuracy."
            A error will be raised if either of this values are not used.
        batch_size (optional): Number of samples per gradient update. Using larger batch sizes can help
            make more efficient use of CPU/GPU parallelization, at the cost of memory.
            If unspecified, batch_size will default to ``64``.
        buffer_size (optional): Buffer size which is used to shuffle elements during training.
            Default size is ``10000``.
        seq_length (optional): The maximum length sentence we want for a single training input in
            characters. Note that this setting is different than max_line_length, as seq_length
            simply affects the length of the training examples passed to the neural network to
            predict the next token. Default size is ``100``.
        embedding_dim (optional): Vector size for the lookup table used in the neural network
            Embedding layer that maps the numbers of each character. Default size is ``256``.
        rnn_units (optional): Positive integer, dimensionality of the output space for LSTM layers.
            Default size is ``256``.
        dropout_rate (optional): Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Using a dropout can help to prevent overfitting
            by ignoring randomly selected neurons during training. 0.2 (20%) is often used as a good
            compromise between retaining model accuracy and preventing overfitting. Default is 0.2.
        rnn_initializer (optional): Initializer for the kernal weights matrix, used for the linear
            transformation of the inputs. Default is ``glorot_transform``.
        dp (optional): If ``True``, train model with differential privacy enabled. This setting provides
            assurances that the models will encode general patterns in data rather than facts
            about specific training examples. These additional guarantees can usefully strengthen
            the protections offered for sensitive data and content, at a small loss in model
            accuracy and synthetic data quality. The differential privacy epsilon and delta values
            will be printed when training completes. Default is ``False``.
        learning_rate (optional): The higher the learning rate, the more that each update during
            training matters. Note: When training with differential privacy enabled,
            if the updates are noisy (such as when the additive noise is large
            compared to the clipping threshold), a low learning rate may help with training.
            Default is ``0.01``.
        dp_noise_multiplier (optional): The amount of noise sampled and added to gradients during
            training. Generally, more noise results in better privacy, at the expense of
            model accuracy. Default is ``0.1``.
        dp_l2_norm_clip (optional): The maximum Euclidean (L2) norm of each gradient is applied to
            update model parameters. This hyperparameter bounds the optimizer's sensitivity to
            individual training points. Default is ``3.0``.
        dp_microbatches (optional): Each batch of data is split into smaller units called micro-batches.
            Computational overhead can be reduced by increasing the size of micro-batches to include
            more than one training example. The number of micro-batches should divide evenly into
            the overall ``batch_size``. Default is ``64``.
        gen_temp (optional): Controls the randomness of predictions by scaling the logits before
            applying softmax. Low temperatures result in more predictable text. Higher temperatures
            result in more surprising text. Experiment to find the best setting. Default is ``1.0``.
        gen_chars (optional): Maximum number of characters to generate per line. Default is ``0`` (no limit).
        gen_lines (optional): Maximum number of text lines to generate. This function is used by
            ``generate_text`` and the optional ``line_validator`` to make sure that all lines created
            by the model pass validation. Default is ``1000``.
        predict_batch_size (optional): How many words to generate in parallel. Higher values may result in increased
            throughput. The default of ``64`` should provide reasonable performance for most users.
        reset_states (optional): Reset RNN model states between each record created guarantees more
            consistent record creation over time, at the expense of model accuracy. Default is ``True``.
        save_all_checkpoints (optional). Set to ``True`` to save all model checkpoints as they are created,
            which can be useful for optimal model selection. Set to ``False`` to save only the latest
            checkpoint. Default is ``True``.
        save_best_model (optional). Defaults to ``True``. Track the best version of the model (checkpoint) to be used.
            If ``save_all_checkpoints`` is disabled, then the saved model will be overwritten by newer ones only if they
            are better.
    """

    # Training configurations
    epochs: int = 100
    early_stopping: bool = True
    early_stopping_patience: int = 5
    best_model_metric: str = const.VAL_LOSS
    batch_size: int = 64
    buffer_size: int = 10000
    seq_length: int = 100
    embedding_dim: int = 256
    rnn_units: int = 256
    learning_rate: float = 0.01
    dropout_rate: float = 0.2
    rnn_initializer: str = "glorot_uniform"

    # Diff privacy configs
    dp: bool = False
    dp_noise_multiplier: float = 0.1
    dp_l2_norm_clip: float = 3.0
    dp_microbatches: int = 64

    # Generation settings
    gen_temp: float = 1.0
    gen_chars: int = 0
    gen_lines: int = 1000
    predict_batch_size: int = 64
    reset_states: bool = True

    # Checkpoint storage
    save_all_checkpoints: bool = False
    save_best_model: bool = True

    def __post_init__(self):
        if self.dp:
            major, minor, _ = tf.__version__.split(".")
            if (int(major), int(minor)) < (2, 4):
                raise RuntimeError(
                    "Running in differential privacy mode requires TensorFlow 2.4.x or greater. "
                    "Please see the README for details"
                )

        if self.best_model_metric not in (const.VAL_LOSS, const.VAL_ACC):
            raise AttributeError("Invalid value for best_model_metric")

        super().__post_init__()

    def get_generator_class(self):
        return TensorFlowGenerator

    def get_training_callable(self):
        return train_rnn

    def gpu_check(self):
        device_name = tf.test.gpu_device_name()
        if not device_name.startswith("/device:GPU:"):
            logging.warning("***** GPU not found, CPU will be used instead! *****")