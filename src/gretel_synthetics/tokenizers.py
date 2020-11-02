"""
Interface definitions for tokenizers
"""
from abc import ABC, abstractmethod
from typing import List, Any, Dict, TYPE_CHECKING
import logging
import json
from pathlib import Path
import shutil

from smart_open import open as smart_open
import sentencepiece as spm
import cloudpickle
import numpy as np

import gretel_synthetics.const as const

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig
else:
    BaseConfig = None


logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)

spm_logger = logging.getLogger("sentencepiece")
spm_logger.setLevel(logging.INFO)


#########
# ABC
#########


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
        with smart_open(self.config.input_data_path, "r", encoding="utf-8", errors="replace") as infile:
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


##################
# Single Char
##################

class CharTokenizerTrainer(BaseTokenizerTrainer):

    def _train(self):
        text = open(
            self.config.training_data_path, "rb"
        ).read().decode()
        vocab = sorted(set(text))[:self.vocab_size]
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        self._save_model(char2idx, idx2char)

    def _save_model(self, char2idx, idx2char):
        cloudpickle.dump(char2idx, open(Path(self.config.checkpoint_dir) / "char2idx.p", "wb"))
        cloudpickle.dump(idx2char, open(Path(self.config.checkpoint_dir) / "idx2char.p", "wb"))

    def _get_save_settings(self):
        return {
            "vocab_size": self.vocab_size
        }


class CharTokenizer(BaseTokenizer):

    _model: Dict[str, dict]

    @classmethod
    def load(cls, config: BaseConfig):
        model = {
            "char2idx": cloudpickle.load(
                open(Path(config.checkpoint_dir) / "char2idx.p", "rb")
            ),
            "idx2char": cloudpickle.load(
                open(Path(config.checkpoint_dir) / "idx2char.p", "rb")
            )
        }
        return cls(model)

    @property
    def total_vocab_size(self):
        return len(self._model["idx2char"])

    def encode_to_ids(self, data: str) -> List[int]:
        return [self._model["char2idx"][char] for char in data]

    def decode_from_ids(self, ids: List[int]) -> str:
        chars = [self._model["idx2char"][id] for id in ids]
        return "".join(chars)

#################
# Sentence Piece
#################


class SentencepieceTokenizerTrainer(BaseTokenizerTrainer):

    vocab_size: int
    character_coverage: float
    pretrain_sentence_count: int
    max_line_line: int

    def __init__(
        self,
        *,
        character_coverage: float = 1.0,
        pretrain_sentence_count: int = 1000000,
        max_line_len: int = 2048,
        **kwargs
    ):
        self.character_coverage = character_coverage
        self.pretrain_sentence_count = pretrain_sentence_count
        self.max_line_line = max_line_len

        super().__init__(**kwargs)

    def _annotate_training_line(self, line: str):
        if self.config.field_delimiter is not None:
            line = line.strip().replace(
                self.config.field_delimiter, self.config.field_delimiter_token
            )
            line += f"{const.NEWLINE}\n"
        else:
            line = line.strip() + const.NEWLINE + "\n"

        return line

    def _train(self):
        logging.info("Training SentencePiece tokenizer")
        spm.SentencePieceTrainer.Train(
            input=self.config.training_data_path,
            model_prefix=const.MODEL_PREFIX,
            user_defined_symbols=[const.NEWLINE, self.config.field_delimiter_token],
            vocab_size=self.vocab_size,
            hard_vocab_limit=False,
            max_sentence_length=self.max_line_line,
            input_sentence_size=self.pretrain_sentence_count,
            shuffle_input_sentence=True,
            character_coverage=self.character_coverage
        )

        # The training automatically saves to disk,
        # so we have to now load it back in after we move
        # it to our checkpoint directory.
        self._save_model()

    def _save_model(self):
        for part in ("model", "vocab"):
            src = Path.cwd() / f"{const.MODEL_PREFIX}.{part}"
            dst = Path(self.config.checkpoint_dir) / f"{const.MODEL_PREFIX}.{part}"
            shutil.move(src.as_posix(), dst.as_posix())

    def _get_save_settings(self):
        return {
            "vocab_size": self.vocab_size,
            "character_coverage": self.character_coverage,
            "pretrain_sentence_count": self.pretrain_sentence_count,
            "max_line_len": self.max_line_line
        }
            

def _log_sample_data(model_dir: str, sp: spm.SentencePieceProcessor):
    training_data_path = Path(model_dir) / const.TRAINING_DATA
    if not training_data_path.is_file():
        logging.info("Training data not found for SP sampling")
        return

    with open(training_data_path) as fin:
        sample = fin.readline().strip()

    logging.info(f"Tokenizer model vocabulary size: {len(sp)} tokens")
    logging.info(
        "Mapping first line of training data\n\n{}\n ---- sample tokens mapped to pieces ---- > \n{}\n".format(
            repr(sample), ", ".join(sp.SampleEncodeAsPieces(sample, -1, 0.1))
        )
    )
    logging.info(
        "Mapping first line of training data\n\n{}\n ---- sample tokens mapped to int ---- > \n{}\n".format(
            repr(sample), ", ".join([str(idx) for idx in sp.EncodeAsIds(sample)])
        )
    )


class SentencePieceTokenizer(BaseTokenizer):

    _model: spm.SentencePieceProcessor

    @classmethod
    def load(cls, model_dir: str):
        sp = spm.SentencePieceProcessor()
        model_fname = f"{const.MODEL_PREFIX}.model"
        logging.info("Loading tokenizer from: %s", model_fname)
        model_path = Path(model_dir) / model_fname
        sp.Load(str(model_path))
        _log_sample_data(model_dir, sp)
        return cls(sp)

    @property
    def total_vocab_size(self):
        return len(self._model)

    def encode_to_ids(self, data: str) -> List[int]:
        return self._model.EncodeAsIds(data)

    def decode_from_ids(self, ids: List[int]) -> str:
        return self._model.DecodeIds(ids)


##########
# Factory
##########

TOK_MAP = {
    SentencepieceTokenizerTrainer.__name__: SentencePieceTokenizer,
    CharTokenizerTrainer.__name__: CharTokenizer,
}


def tokenizer_from_model_dir(model_dir: str):
    params_file = Path(model_dir) / BaseTokenizerTrainer.settings_fname
    
    # Backwards compat with <= 0.14.0
    if not params_file.is_file():
        tok_cls = SentencePieceTokenizer
    else:
        params_dict = json.loads(open(params_file).read())
        tok_type = params_dict.pop(BaseTokenizerTrainer.tokenizer_type)
        tok_cls = TOK_MAP[tok_type]
    return tok_cls.load(model_dir)
