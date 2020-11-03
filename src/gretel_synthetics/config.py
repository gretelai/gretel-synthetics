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

    input_data_path: str = None
    """Path to raw training data
    """

    checkpoint_dir: str = None
    """Directory where model data will
    be stored
    """

    training_data_path: str = None
    """Where annotated and tokenized training data will be stored
    """

    field_delimiter: Optional[str] = None
    field_delimiter_token: str = "<d>"

    model_type: str = None
    """A string version of the model config class. This is used
    to keep track of what underlying engine was used when
    writing the config to a file.
    """

    max_lines: int = 0
    """The maximum number of lines to utilize from the
    raw input data"""

    # Defaulst SP tokenizer settings
    vocab_size: int = 20000
    character_coverage: float = 1.0
    pretrain_sentence_count: int = 1000000
    max_line_len: int = 2048

    def as_dict(self):
        d = asdict(self)
        return d

    def save_model_params(self):
        save_path = Path(self.checkpoint_dir) / const.MODEL_PARAMS
        # logging.info(f"Saving model history to {save_path.name}")
        out_dict = self.as_dict()
        with open(save_path, "w") as f:
            json.dump(out_dict, f, indent=2)
        return save_path

    @abstractmethod
    def get_generator_class(self) -> BaseGenerator:
        """This must be implemented by all specific
        configs. It should return the class that should
        be used as the Generator for creating records
        """
        pass

    @abstractmethod
    def get_training_callable(self) -> Callable:
        """This must be implemented by all specific
        configs. It should return a callable that
        should be used as the entrypoint for
        training a model
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
    training a model and generating data.  This base config generally
    should not be used directly. Instead you should use one of the
    subclasses which are specific to model and checkpoint storage.

    Args:
        max_lines (optional): Number of rows of file to read. Useful for training on a subset of large files.
            If unspecified, max_lines will default to ``0`` (process all lines).
        max_line_len (optional): Maximum line length for input training data. Any lines longer than
            this length will be ignored. Default is ``2048``.
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
        optimizer (optional): Optimizer used by the neural network to maximize accuracy and reduce
            loss. Currently supported optimizers: ``Adam``, ``SGD``, and ``Adagrad``. Default is ``Adam``.
        field_delimiter (optional): Delimiter to use for training on structured data. When specified,
            the delimiter is passed as a user-specified token to the tokenizer, which can improve
            synthetic data quality. For unstructured text, leave as ``None``. For structured text
            such as comma or tab separated values, specify "," or "\t" respectively. Default is ``None``.
        field_delimiter_token (optional): User specified token to replace ``field_delimiter`` with
            while annotating data for training the model. Default is ``<d>``.
        vocab_size (optional): Pre-determined maximum vocabulary size prior to neural model training, based
            on subword units including byte-pair-encoding (BPE) and unigram language model, with the extension
            of direct training from raw sentences. We generally recommend using a large vocabulary
            size of 20,000 to 50,000. Default is ``20000``.
        character_coverage (optional): The amount of characters covered by the model. Unknown characters
            will be replaced with the <unk> tag. Good defaults are ``0.995`` for languages with rich
            character sets like Japanese or Chinese, and 1.0 for other languages or machine data.
            Default is ``1.0``.
        pretrain_sentence_count (optional): The number of lines spm_train first loads. Remaining lines are simply
            discarded. Since spm_train loads entire corpus into memory, this size will depend on the memory
            size of the machine. It also affects training time.
            Default is ``1000000``.
        dp (optional): If ``True``, train model with differential privacy enabled. This setting provides
            assurances that the models will encode general patterns in data rather than facts
            about specific training examples. These additional guarantees can usefully strengthen
            the protections offered for sensitive data and content, at a small loss in model
            accuracy and synthetic data quality. The differential privacy epsilon and delta values
            will be printed when training completes. Default is ``False``.
        dp_learning_rate (optional): The higher the learning rate, the more that each update during
            training matters. If the updates are noisy (such as when the additive noise is large
            compared to the clipping threshold), a low learning rate may help with training.
            Default is ``0.015``.
        dp_noise_multiplier (optional): The amount of noise sampled and added to gradients during
            training. Generally, more noise results in better privacy, at the expense of
            model accuracy. Default is ``1.1``.
        dp_l2_norm_clip (optional): The maximum Euclidean (L2) norm of each gradient is applied to
            update model parameters. This hyperparameter bounds the optimizer's sensitivity to
            individual training points. Default is ``1.0``.
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
        save_all_checkpoints (optional). Set to ``True`` to save all model checkpoints as they are created,
            which can be useful for optimal model selection. Set to ``False`` to save only the latest
            checkpoint. Default is ``True``.
        save_best_model (optional). Defaults to ``True``. Track the best version of the model (checkpoint) to be used.
            If ``save_all_checkpoints`` is disabled, then the saved model will be overwritten by newer ones only if they
            are better.
        overwrite (optional). Set to ``True`` to automatically overwrite previously saved model checkpoints.
            If ``False``, the trainer will generate an error if checkpoints exist in the model directory.
            Default is ``False``.
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
    dropout_rate: float = 0.2
    rnn_initializer: str = "glorot_uniform"

    # Diff privacy configs
    dp: bool = False
    dp_learning_rate: float = 0.001
    dp_noise_multiplier: float = 1.1
    dp_l2_norm_clip: float = 1.0
    dp_microbatches: int = 256

    # Generation settings
    gen_temp: float = 1.0
    gen_chars: int = 0
    gen_lines: int = 1000
    predict_batch_size: int = 64

    # Checkpoint storage
    save_all_checkpoints: bool = False
    save_best_model: bool = True
    overwrite: bool = False

    def __post_init__(self):
        # FIXME: Remove @ 0.15.X when new optimizers are available for DP
        if self.dp:
            raise RuntimeError(
                "DP mode is disabled in v0.14.X. Please remove or set this value to ``False`` to continue with out DP.  DP will be re-enabled in v0.15.X. Please see the README for more details"  # noqa
            )

        if self.best_model_metric not in (const.VAL_LOSS, const.VAL_ACC):
            raise AttributeError("Invalid value for bset_model_metric")

        super().__post_init__()

    def get_generator_class(self):
        return TensorFlowGenerator

    def get_training_callable(self):
        return train_rnn


#################
# Config Factory
#################

CONFIG_MAP = {cls.__name__: cls for cls in BaseConfig.__subclasses__()}


def config_from_model_dir(model_dir: str):
    params_file = Path(model_dir) / const.MODEL_PARAMS
    params_dict = json.loads(open(params_file).read())
    model_type = params_dict.pop(const.MODEL_TYPE, None)

    # swap out the checkpoint dir location for the currently
    # provided checkpoint dir, this allows us to load a model
    # from another file path for data generation when the current
    # location of the model dir does not match the one that was
    # used for training originally
    params_dict["checkpoint_dir"] = model_dir

    # backwards compat with <= 0.14.0
    if model_type is None:
        return TensorFlowConfig(**params_dict)
    cls = CONFIG_MAP[model_type]
    return cls(**params_dict)


# FIXME: Backwards compat with <= 0.14.0
LocalConfig = TensorFlowConfig
