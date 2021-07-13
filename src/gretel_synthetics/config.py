"""
This module provides a set of dataclasses that can be used to hold all necessary
confguration parameters for training a model and generating data.

For example usage please see our Jupyter Notebooks.
"""
from dataclasses import dataclass, asdict
from abc import abstractmethod
from typing import Callable, TYPE_CHECKING, Optional
from pathlib import Path
import json
import logging
import tensorflow as tf


import gretel_synthetics.const as const
from gretel_synthetics.tensorflow.train import train_rnn
from gretel_synthetics.tensorflow.generator import TensorFlowGenerator


logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


if TYPE_CHECKING:
    from gretel_synthetics.generate import BaseGenerator
else:
    BaseGenerator = None


@dataclass
class BaseConfig:
    """This class should not be used directly, engine specific
    classes should derived from this class.
    """

    input_data_path: str = None
    """Path to raw training data, user provided.
    """

    validation_split: bool = True
    """Use a fraction of the training data as validation data.
    Use of a validation set is recommended as it helps prevent
    over-fitting and memorization.
    When enabled, 20% of data will be used for validation."""

    checkpoint_dir: str = None
    """Directory where model data will
    be stored, user provided.
    """

    training_data_path: str = None
    """Where annotated and tokenized training data will be stored. This attr
    will be modified during construction.
    """

    field_delimiter: Optional[str] = None
    """If the input data is structured, you may specify a field delimiter which can
    be used to split the generated text into a list of strings. For more detail
    please see the ``GenText`` class in the ``generate.py`` module.
    """

    field_delimiter_token: str = "<d>"
    """Depending on the tokenizer used, a special token can be used to represent
    characters. For tokenizers, like SentencePiece that support this, we will replace
    the field delimiter char with this token to provide better learning and generation.
    If the tokenizer used does not support custom tokens, this value will be ignored
    """

    model_type: str = None
    """A string version of the model config class. This is used
    to keep track of what underlying engine was used when
    writing the config to a file. This will be automatically updated
    during construction.
    """

    max_lines: int = 0
    """The maximum number of lines to utilize from the
    raw input data."""

    overwrite: bool = False
    """Set to ``True`` to automatically overwrite previously saved model checkpoints.
        If ``False``, the trainer will generate an error if checkpoints exist in the model directory.
        Default is ``False``.
    """

    epoch_callback: Optional[Callable] = None
    """Callback to be invoked at the end of each epoch.  It will be invoked with a EpochState instance 
    as its only parameter.  NOTE that the callback is deleted when save_model_params is called, we do not
    attempt to serialize it to json.
    """

    # Default SP tokenizer settings. This are kept here for
    # backwards compatibility for <= 0.14.x
    vocab_size: int = 20000
    character_coverage: float = 1.0
    pretrain_sentence_count: int = 1000000
    max_line_len: int = 2048

    def as_dict(self):
        """Serialize the config attrs to a dict
        """
        d = asdict(self)
        return d

    def save_model_params(self):
        save_path = Path(self.checkpoint_dir) / const.MODEL_PARAMS
        # logging.info(f"Saving model history to {save_path.name}")
        out_dict = self.as_dict()
        # Do not attempt to serialize Callable to json, just delete it.
        if out_dict.get('epoch_callback') is not None:
            del out_dict['epoch_callback']
        with open(save_path, "w") as f:
            json.dump(out_dict, f, indent=2)
        return save_path

    @abstractmethod
    def get_generator_class(self) -> BaseGenerator:
        """This must be implemented by all specific
        configs. It should return the class that should
        be used as the Generator for creating records.
        """
        pass

    @abstractmethod
    def get_training_callable(self) -> Callable:
        """This must be implemented by all specific
        configs. It should return a callable that
        should be used as the entrypoint for
        training a model.
        """
        pass

    def gpu_check(self):
        """Optionally do a GPU check and warn if
        a GPU is not available, if not overridden,
        do nothing
        """
        pass

    def __post_init__(self):
        if not self.checkpoint_dir or not self.input_data_path:
            raise AttributeError(
                "Must provide checkpoint_dir and input_data_path params!"
            )
        if not Path(self.checkpoint_dir).exists():
            Path(self.checkpoint_dir).resolve().mkdir()
        self.training_data_path = Path(
            self.checkpoint_dir, const.TRAINING_DATA
        ).as_posix()

        # assign the model type for serialization
        self.model_type = self.__class__.__name__


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
        best_model_metric (optional). Defaults to "val_loss" or "loss" if a validation set is not used.
            The metric to use to track when a model is no longer improving. Alternative options are "val_acc"
            or "acc". A error will be raised if a valid value is not specified.
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
    best_model_metric: str = None
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
            if self.batch_size % self.dp_microbatches != 0:
                raise ValueError('Number of dp_microbatches should divide evenly into batch_size')

            # TODO: To enable micro-batch size greater than 1, we need to update the differential privacy
            #  optimizer loss function to compute the vector of per-example losses, rather than the mean
            #  over a mini-batch.
            if self.dp_microbatches != 1:
                logging.warning("***** Currently only a differential privacy micro-batch size of 1 is supported. "
                                "Setting micro-batch size to 1. *****")
                self.dp_microbatches = 1

        if self.best_model_metric is None:
            if self.validation_split:
                self.best_model_metric = const.METRIC_VAL_LOSS
            else:
                self.best_model_metric = const.METRIC_LOSS

        if self.best_model_metric not in (const.METRIC_VAL_LOSS,
                                          const.METRIC_VAL_ACCURACY,
                                          const.METRIC_LOSS,
                                          const.METRIC_ACCURACY):
            raise AttributeError("Invalid value for best_model_metric")

        if self.epoch_callback is not None:
            if not callable(self.epoch_callback):
                raise ValueError("epoch_callback must be a callable!")

        super().__post_init__()

    def get_generator_class(self):
        return TensorFlowGenerator

    def get_training_callable(self):
        return train_rnn

    def gpu_check(self):
        device_name = tf.test.gpu_device_name()
        if not device_name.startswith("/device:GPU:"):
            logging.warning("***** GPU not found, CPU will be used instead! *****")


#################
# Config Factory
#################

CONFIG_MAP = {cls.__name__: cls for cls in BaseConfig.__subclasses__()}
"""A mapping of configuration subclass string names to their actual classes. This
can be used to re-instantiate a config from a serialized state.
"""


def config_from_model_dir(model_dir: str) -> BaseConfig:
    """Factory that will take a known directory of a model
    and return a class instance for that config. We automatically
    try and detect the correct BaseConfig sub-class to use based
    on the saved model params.

    If there is no ``model_type`` param in the saved config, we
    assume that the model was saved using an earlier version of the
    package and will instantiate a TensorFlowConfig
    """
    params_file = Path(model_dir) / const.MODEL_PARAMS
    params_dict = json.loads(open(params_file).read())
    model_type = params_dict.pop(const.MODEL_TYPE, None)

    # swap out the checkpoint dir location for the currently
    # provided checkpoint dir, this allows us to load a model
    # from another file path for data generation when the current
    # location of the model dir does not match the one that was
    # used for training originally
    params_dict["checkpoint_dir"] = model_dir
    old_dp_learning_rate = params_dict.pop("dp_learning_rate", .001)

    # backwards compat with <= 0.14.0
    if model_type is None:
        config = TensorFlowConfig(**params_dict)
        config.learning_rate = old_dp_learning_rate if config.dp else .01
        return config
    cls = CONFIG_MAP[model_type]
    return cls(**params_dict)


# Backwards compat with <= 0.14.0
LocalConfig = TensorFlowConfig
