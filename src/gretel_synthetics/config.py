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
class _BaseConfig:
    """Base dataclass that contains all of the main parameters for
    training a model and generating data.  This base config generally
    should not be used directly. Instead you should use one of the
    subclasses which are specific to model and checkpoint storage.
    """

    # Training configurations
    max_lines: int = 0
    epochs: int = 30
    batch_size: int = 64
    buffer_size: int = 10000
    seq_length: int = 100
    embedding_dim: int = 256
    rnn_units: int = 256
    dropout_rate: float = 0.2
    rnn_initializer: str = "glorot_uniform"

    # Input data configs
    field_delimiter: Optional[str] = None
    field_delimiter_token: str = "<d>"

    # Tokenizer settings
    vocab_size: int = 500
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
    gen_lines: int = 500

    # Checkpoint storage
    save_all_checkpoints: bool = True
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
class LocalConfig(_BaseConfig, _PathSettingsMixin):
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
