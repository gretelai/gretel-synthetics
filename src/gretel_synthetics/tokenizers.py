"""
Interface definitions for tokenizers
"""
from abc import ABC, abstractmethod
from typing import List, Any, Dict, TYPE_CHECKING, Iterator, Optional
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


FIELD_DELIM = "field_delimiter"
FIELD_DELIM_TOKEN = "field_delimiter_token"


class TokenizerError(Exception):
    pass


#########
# ABC
#########


class Base(ABC):

    settings_fname: str = "tokenizer_params.json"
    newline_str: str = None

    def __init__(self):
        if self.newline_str is None:
            raise AttributeError("Sublasses should define newline_str as a class attr")


class BaseTokenizerTrainer(Base):

    vocab_size: int
    config: BaseConfig
    num_lines: int = 0
    tokenizer_type: str = "tokenizer_type"

    def __init__(self, *, config: BaseConfig, vocab_size: Optional[int] = None):
        self.vocab_size = vocab_size
        self.config = config

        super().__init__()

    def create_annotated_training_data(self) -> Iterator[str]:
        """Read in the configurations raw input data path, and
        create a file I/O pipeline where each line of the input
        data path can optionally route through an annotation
        function and then we will write each raw line out into
        a training data file as specified by the config.
        """
        logging.info(f"Loading training data from {self.config.input_data_path}")
        self.num_lines = 0
        with smart_open(self.config.input_data_path, "r", encoding="utf-8", errors="replace") as infile:
            with open(self.config.training_data_path, "w") as fout:
                for line in infile:
                    self.num_lines += 1
                    if self.config.max_lines and self.num_lines >= self.config.max_lines:
                        break
              
                    # Tokenizer specific line processing
                    annotated_line = self._annotate_training_line(line)
                    fout.write(annotated_line)
        return self.training_data_iter()

    def training_data_iter(self) -> Iterator[str]:
        """Create a generator that will iterate each line of the training
        data that was created during the annotation step.
        """
        with open(self.config.training_data_path, "r") as fin:
            for line in fin:
                yield line

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

        # We save off the field delimiter and field delimiter token
        # to the settings so that they can be loaded back in later
        # for use in decoding
        settings[FIELD_DELIM] = self.config.field_delimiter
        settings[FIELD_DELIM_TOKEN] = self.config.field_delimiter_token

        with open(Path(self.config.checkpoint_dir) / self.settings_fname, "w") as fout:
            fout.write(json.dumps(settings))


class BaseTokenizer(Base):

    _model: Any
    """This holds the actual model data, which can be any type of object,
    the interfaces implemented by sublcasses should know how to interact
    with it in order to satisfy the interface definitions defined
    here
    """

    field_delimiter: Optional[str] = None
    field_delimiter_token: Optional[str] = None

    def __init__(self, model_data: Any, model_dir: str):
        self._model = model_data
        self._model_dir = model_dir

        self._load_delimiter_data()
        super().__init__()

    def _load_delimiter_data(self):
        params_file = Path(self._model_dir) / self.settings_fname

        if not params_file.is_file():
            model_params_dict = json.loads(
                open(Path(self._model_dir) / const.MODEL_PARAMS).read()
            )
            self.field_delimiter = model_params_dict[FIELD_DELIM]
            self.field_delimiter_token = model_params_dict[FIELD_DELIM_TOKEN]
        else:
            params_dict = json.loads(open(params_file).read())
            self.field_delimiter = params_dict.get(FIELD_DELIM, None)
            self.field_delimiter_token = params_dict.get(FIELD_DELIM_TOKEN, None)

    @classmethod
    @abstractmethod
    def load(cls, model_dir: str):
        """Given a directory to a model, load the specific tokenizer
        model into an instance. Subclasses should implement this logic
        specific to how they need to load a model back in
        """
        pass

    @property
    @abstractmethod
    def total_vocab_size(self):
        """Return the total count of unique tokens in the vocab, specific
        to the underlying tokenizer to be used.
        """
        pass

    def encode_to_ids(self, data: str) -> List[int]:
        return self._encode_to_ids(data)

    @abstractmethod
    def _encode_to_ids(self, data: str) -> List[int]:
        pass

    def decode_from_ids(self, ids: List[int]) -> str:
        decoded_str = self._decode_from_ids(ids)
        return self._replace_decoded_tokens(decoded_str)

    @abstractmethod
    def _decode_from_ids(self, ids: List[int]) -> str:
        pass

    def _replace_decoded_tokens(self, decoded_line: str) -> str:
        """This function will implicitly be called after decoding
        IDs. If there is specific token replacement that needs
        to be done, then subclasses should overload and implement
        otherwise we just pass the decoded line through
        """
        return decoded_line


##################
# Single Char
##################

class CharTokenizerTrainer(BaseTokenizerTrainer):

    newline_str: str = "\n"

    def _train(self):
        text = open(
            self.config.training_data_path, "rb"
        ).read().decode()
        vocab = sorted(set(text))
        if self.vocab_size is not None:
            vocab = vocab[:self.vocab_size]
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
    newline_str: str = "\n"

    @classmethod
    def load(cls, model_dir: str):
        model = {
            "char2idx": cloudpickle.load(
                open(Path(model_dir) / "char2idx.p", "rb")
            ),
            "idx2char": cloudpickle.load(
                open(Path(model_dir) / "idx2char.p", "rb")
            )
        }
        return cls(model, model_dir)

    @property
    def total_vocab_size(self):
        return len(self._model["idx2char"])

    def _encode_to_ids(self, data: str) -> List[int]:
        try:
            return [self._model["char2idx"][char] for char in data]
        except KeyError as err:
            raise TokenizerError("Some characters in the input string are not part of the vocab") from err

    def _decode_from_ids(self, ids: List[int]) -> str:
        try:
            chars = [self._model["idx2char"][id] for id in ids]
            return "".join(chars)
        except IndexError as err:
            raise TokenizerError("Some IDs do not have mappings to chars in the vocab") from err

#################
# Sentence Piece
#################


class SentencePieceTokenizerTrainer(BaseTokenizerTrainer):

    vocab_size: int
    character_coverage: float
    pretrain_sentence_count: int
    max_line_line: int
    newline_str: str = "<n>"

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

        # SP Needs a vocab size int, so we set a default one if
        # not already provided in the kwargs
        if "vocab_size" not in kwargs:
            kwargs["vocab_size"] = 20000

        super().__init__(**kwargs)

    def _annotate_training_line(self, line: str):
        if self.config.field_delimiter is not None:
            line = line.strip().replace(
                self.config.field_delimiter, self.config.field_delimiter_token
            )
            line += f"{self.newline_str}\n"
        else:
            line = line.strip() + self.newline_str + "\n"

        return line

    def _train(self):
        logging.info("Training SentencePiece tokenizer")
        spm.SentencePieceTrainer.Train(
            input=self.config.training_data_path,
            model_prefix=const.MODEL_PREFIX,
            user_defined_symbols=[self.newline_str, self.config.field_delimiter_token],
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
    newline_str: str = "<n>"

    @classmethod
    def load(cls, model_dir: str):
        sp = spm.SentencePieceProcessor()
        model_fname = f"{const.MODEL_PREFIX}.model"
        logging.info("Loading tokenizer from: %s", model_fname)
        model_path = Path(model_dir) / model_fname
        sp.Load(str(model_path))
        _log_sample_data(model_dir, sp)
        return cls(sp, model_dir)

    @property
    def total_vocab_size(self):
        return len(self._model)

    def _encode_to_ids(self, data: str) -> List[int]:
        return self._model.EncodeAsIds(data)

    def _decode_from_ids(self, ids: List[int]) -> str:
        return self._model.DecodeIds(ids)

    def _replace_decoded_tokens(self, decoded_line: str) -> str:
        if self.field_delimiter is not None:
            decoded_line = decoded_line.replace(
                self.field_delimiter_token,
                self.field_delimiter
            )
        return decoded_line


##########
# Factory
##########

TOK_MAP = {
    SentencePieceTokenizerTrainer.__name__: SentencePieceTokenizer,
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
