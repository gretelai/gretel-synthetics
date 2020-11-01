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

from gretel_synthetics.const import MODEL_PARAMS

if TYPE_CHECKING:
    from gretel_synthetics.generate import BaseGenerator
else:
    BaseGenerator = None

TOKENIZER_PREFIX = "m"


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
    """Where annotated training data will be stored
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

    def as_dict(self):
        d = asdict(self)
        return d

    def save_model_params(self):
        save_path = Path(self.checkpoint_dir) / MODEL_PARAMS
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
            self.checkpoint_dir, "training_data.txt"
        ).as_posix()

        # assign the model type for serialization
        self.model_type = self.__class__.__name__
