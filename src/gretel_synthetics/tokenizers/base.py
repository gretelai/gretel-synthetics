"""
Interface definitions for tokenizers
"""
from abc import ABC, abstractmethod
from typing import List, Any
import logging
import json
from pathlib import Path

from gretel_synthetics.base_config import BaseConfig

logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


class BaseTokenizerTrainer(ABC):

    vocab_size: int
    config: BaseConfig
    settings_fname = "tokenizer_params.json"
    tokenizer_type = "tokenizer_type"

    def __init__(self, *, config: BaseConfig, vocab_size: int = 20000):
        self.vocab_size = vocab_size
        self.config = config

    def create_annotated_training_data(self) -> str:
        """Read in the configurations raw input data path, and
        create a file I/O pipeline where each line of the input
        data path can optionally route through an annotation
        function and then we will write each raw line out into
        a training data file as specified by the config.
        """
        logging.info(f"Loading training data from {self.config.input_data_path}")
        num_lines = 0
        annotated_lines = []
        with open(self.config.input_data_path, "r", encoding="utf-8", errors="replace") as infile:
            with open(self.config.training_data_path, "w") as fout:
                for line in infile:
                    num_lines += 1
                    if self.config.max_lines and num_lines >= self.config.max_lines:
                        break
              
                    # Tokenizer specific line processing
                    annotated_line = self._annotate_training_line(line)

                    annotated_lines.append(annotated_line)
                    fout.write(annotated_line)
        return "".join(annotated_lines)

    def _annotate_training_line(self, line: str):
        return line

    def train(self):
        self._train()
        settings = self._get_save_settings()
        self._save_settings(settings)

    @abstractmethod
    def _get_save_settings(self) -> dict:
        """Subclasses must create a dict that holds serialized
        params for the tokenizer
        """
        pass

    @abstractmethod
    def _train(self):
        """Subclasses must implement a method that trains a tokenizer
        and then writes it to disk in the checkpoint dir specified
        by the config
        """
        pass

    def _save_settings(self, settings: dict):
        settings[self.tokenizer_type] = self.__class__.__name__
        with open(Path(self.config.checkpoint_dir) / self.settings_fname, "w") as fout:
            fout.write(json.dumps(settings))


class BaseTokenizer(ABC):

    _model: Any
    """This holds the actual model data, which can be any type of object,
    the interfaces implemented by sublcasses should know how to interact
    with it in order to satisfy the interface definitions defined
    here
    """

    def __init__(self, model_data: Any):
        self._model = model_data

    @classmethod
    @abstractmethod
    def load(cls, model_dir: str):
        pass

    @property
    @abstractmethod
    def total_vocab_size(self):
        pass

    @abstractmethod
    def encode_to_ids(self, data: str) -> List[int]:
        pass

    @abstractmethod
    def decode_from_ids(self, ids: List[int]) -> str:
        pass
