"""
This module provides a set of dataclasses that can be used to hold all necessary
confguration parameters for training a model and generating data.

For example usage please see our Jupyter Notebooks.
"""
import json
import logging
from pathlib import Path
from abc import abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Optional

logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


TOKENIZER_PREFIX = "m"
MODEL_PARAMS = "model_params.json"


@dataclass
class BaseConfig:
    """Base dataclass that contains all of the main parameters for
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
            Default is ``30``.
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
        save_all_checkpoints (optional). Set to ``True`` to save all model checkpoints as they are created,
            which can be useful for optimal model selection. Set to ``False`` to save only the latest
            checkpoint. Default is ``True``.
        overwrite (optional). Set to ``True`` to automatically overwrite previously saved model checkpoints.
            If ``False``, the trainer will generate an error if checkpoints exist in the model directory.
            Default is ``False``.


    """

    # Training configurations
    max_lines: int = 0
    epochs: int = 15
    batch_size: int = 64
    buffer_size: int = 10000
    seq_length: int = 100
    embedding_dim: int = 256
    rnn_units: int = 256
    dropout_rate: float = 0.2
    rnn_initializer: str = "glorot_uniform"
    max_line_len: int = 2048

    # Input data configs
    field_delimiter: Optional[str] = None
    field_delimiter_token: str = "<d>"

    # Tokenizer settings
    vocab_size: int = 20000
    character_coverage: float = 1.0

    # Diff privacy configs
    dp: bool = False
    dp_learning_rate: float = 0.015
    dp_noise_multiplier: float = 1.1
    dp_l2_norm_clip: float = 1.0
    dp_microbatches: int = 256

    # Generation settings
    gen_temp: float = 1.0
    gen_chars: int = 0
    gen_lines: int = 1000

    # Checkpoint storage
    save_all_checkpoints: bool = False
    overwrite: bool = False

    @abstractmethod
    def _set_tokenizer(self):  # pragma: no cover
        pass


@dataclass
class _PathSettings:
    """This dataclass stores path locations to
    store tokenizer and training data locations. It should not
    be used directly. It will be utilized by any configuration
    classes that need to utilize path-based storage.
    """

    tokenizer_model: str = None
    training_data: str = None
    tokenizer_prefix: str = TOKENIZER_PREFIX


@dataclass
class _PathSettingsMixin:
    """If a specific config needs to make use of
    ``PathSettings``, this dataclass will make an
    attr of ``paths`` available and also bring in
    property methods to allow easy access to the
    various path attributes.

    This makes it possible to easily remove the path
    settings when serializing the configuration.
    """

    paths: _PathSettings = field(default_factory=_PathSettings)

    @property
    def tokenizer_prefix(self):
        return self.paths.tokenizer_prefix

    @property
    def tokenizer_model(self):
        return self.paths.tokenizer_model

    @property
    def training_data(self):
        return self.paths.training_data


@dataclass
class LocalConfig(BaseConfig, _PathSettingsMixin):
    """This configuration will use the local file system
    to store all models, training data, and checkpoints

    Args:
        checkpoint_dir: The local directory where all checkpoints and additional support
            files for training and generation will be stored.
        input_data_path: A path to a file that will be used as initial training input.
            This file will be opened, annotated, and then written out to a path
            that is generated based on the ``checkpoint_dir.``
    """

    checkpoint_dir: str = None
    input_data_path: str = None

    def __post_init__(self):
        if not self.checkpoint_dir or not self.input_data_path:
            raise AttributeError(
                "Must provide checkpoint_dir and input_path_dir params!"
            )
        if not Path(self.checkpoint_dir).exists():
            Path(self.checkpoint_dir).resolve().mkdir()
        self._set_tokenizer()

    def _set_tokenizer(self):
        self.paths.tokenizer_prefix = "m"
        self.paths.tokenizer_model = Path(self.checkpoint_dir, "m.model").as_posix()
        self.paths.training_data = Path(
            self.checkpoint_dir, "training_data.txt"
        ).as_posix()

    def as_dict(self):
        d = asdict(self)
        d.pop("paths")
        return d

    def save_model_params(self):
        save_path = Path(self.checkpoint_dir) / MODEL_PARAMS
        logging.info(f"Saving model history to {save_path.name}")
        with open(save_path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)
        return save_path
