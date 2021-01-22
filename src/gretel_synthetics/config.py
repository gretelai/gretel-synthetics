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


# Backwards compat with <= 0.14.0 #TODO update
# LocalConfig = TensorFlowConfig
