"""
This module provides a set of dataclasses that can be used to hold all necessary
confguration parameters for training a model and generating data.

For example usage please see our Jupyter Notebooks.
"""
from dataclasses import dataclass
from abc import abstractmethod
from typing import Callable, TYPE_CHECKING, Optional
from pathlib import Path

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

    max_lines: int = 0
    """The maximum number of lines to utilize from the
    raw input data"""

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
                "Must provide checkpoint_dir and input_path_dir params!"
            )
        if not Path(self.checkpoint_dir).exists():
            Path(self.checkpoint_dir).resolve().mkdir()
        self.training_data_path = Path(
            self.checkpoint_dir, "training_data.txt"
        ).as_posix()
