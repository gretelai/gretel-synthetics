"""
Interface definitions for tokenizers.  The classes in the module are segmented into two abstract types:
Trainers and Tokenizers.  They are kept separate because the parameters used to train a tokenizer
are not necessarily loaded back in and utilized by a trained tokenizer.  While its more explicit
to utilize two types of classes, it also removes any ambiguity in which methods are able to be used
based on training or tokenizing.

Trainers require a specific configuration to be provided. Based on the configuration received, the
tokenizer trainers will create the actual training data file that will be used by the downstream
training process. In this respect, utilizing at least one of these tokenizers is required for
training since it is the tokenizers responsbility to create the final training data to be used.

The general process that is followed when using these tokenizers is:

Create a trainer instance, with desired parameters, including providing the config as a required param.

Call the ``annotate_data`` for your tokenizer trainer. What is important to note here
is that this method actually iterates the input data line by line, and does any special processing, then
writes a new data file that will be used for actual training. This new data file is written to the
model directory.

Call the ``train`` method, which will create your tokenization model and save it to the model
directory.

Now you will use the ``load()`` class method from an actual tokenizer class to load that
trained model in and now you can use it on input data.

"""

import json
import logging
import re
import shutil

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    TYPE_CHECKING,
)

import cloudpickle
import numpy as np
import sentencepiece as spm

from smart_open import open as smart_open

import gretel_synthetics.const as const

from gretel_synthetics.errors import ParameterError

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig
else:
    BaseConfig = None


logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
    """High level base class for shared class attrs and validation. Should not
    be used directly.
    """

    settings_fname: str = "tokenizer_params.json"
    newline_str: str = None

    def __init__(self):
        if self.newline_str is None:
            raise AttributeError("Sublasses should define newline_str as a class attr")


class BaseTokenizerTrainer(Base):
    """Base class for training tokenizers. Should not be used directly."""

    vocab_size: int
    """The max size of the vocab (tokens) to be extracted from
    the input dataset.
    """

    config: BaseConfig
    """A subclass instace of ``BaseConfig``.  This will be used to find the input
    data for tokenization
    """

    num_lines: int = 0
    """The number of lines that were processed after ``create_annotated_training_data``
    is called
    """

    tokenizer_type: str = "tokenizer_type"

    def __init__(self, *, config: BaseConfig, vocab_size: Optional[int] = None):
        self.vocab_size = vocab_size
        self.config = config

        super().__init__()

    def annotate_data(self) -> Iterator[str]:
        """
        This should be called _before_ training as it is required
        to have the annotated training data created in the model
        directory.

        Read in the configurations raw input data path, and
        create a file I/O pipeline where each line of the input
        data path can optionally route through an annotation
        function and then we will write each raw line out into
        a training data file as specified by the config.
        """
        input_path = self.config.input_data_path
        output_path = self.config.training_data_path
        self.num_lines = 0

        logging.info(f"Loading input data from {input_path}")
        with smart_open(input_path, "r", encoding="utf-8", errors="replace") as infile:
            with open(output_path, "w") as fout:
                for line in infile:
                    self.num_lines += 1
                    if (
                        self.config.max_lines
                        and self.num_lines >= self.config.max_lines
                    ):
                        break

                    # Tokenizer specific line processing
                    annotated_line = self._annotate_training_line(line)
                    fout.write(annotated_line)
        return self.data_iterator()

    def data_iterator(self) -> Iterator[str]:
        """Create a generator that will iterate each line of the training
        data that was created during the annotation step.  Synthetic model trainers
        will most likely need to iterate this to process each line of the annotated
        training data.
        """
        with open(self.config.training_data_path, "r") as fin:
            for line in fin:
                yield line

    def _annotate_training_line(self, line: str):
        """Implicitly gets called during the annotation process. Subclasses
        can optionally override this to do any actual custom tokenization
        steps per training line.
        """
        return line

    def train(self):
        """Train a tokenizer and save the tokenizer settings to a file
        located in the model directory specified by the ``config`` object
        """
        self._train()
        settings = self._get_save_settings()
        self._save_settings(settings)

    @abstractmethod
    def _get_save_settings(self) -> dict:
        """Subclasses must create a dict that holds serialized
        params for the tokenizer.
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
    """Base class for loading a tokenizer from disk. Should not be
    used directly.
    """

    _model: Any
    """This holds the actual model data, which can be any type of object,
    the interfaces implemented by sublcasses should know how to interact
    with it in order to satisfy the interface definitions defined
    here
    """

    field_delimiter: Optional[str] = None
    field_delimiter_token: Optional[str] = None
    settings: Optional[dict] = None

    def __init__(self, model_data: Any, model_dir: str):
        self._model = model_data
        self._model_dir = model_dir

        self._load_delimiter_data()
        self._load_settings_dict()
        super().__init__()

    def _load_delimiter_data(self):
        params_file = Path(self._model_dir) / self.settings_fname

        if not params_file.is_file():
            with open(Path(self._model_dir) / const.MODEL_PARAMS) as f:
                model_params_dict = json.load(f)
            self.field_delimiter = model_params_dict[FIELD_DELIM]
            self.field_delimiter_token = model_params_dict[FIELD_DELIM_TOKEN]
        else:
            with open(params_file) as f:
                params_dict = json.load(f)
            self.field_delimiter = params_dict.get(FIELD_DELIM, None)
            self.field_delimiter_token = params_dict.get(FIELD_DELIM_TOKEN, None)

    def _load_settings_dict(self):
        """Load the dictionary of custom settings that were saved out
        when creating the tokenizer
        """
        params_file = Path(self._model_dir) / self.settings_fname
        if not params_file.is_file():
            self.settings = {}
        else:
            self.settings = json.loads(params_file.read_text())

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
        """Given an input string, convert it to a list of
        token IDs
        """
        return self._encode_to_ids(data)

    @abstractmethod
    def _encode_to_ids(self, data: str) -> List[int]:
        pass

    def decode_from_ids(self, ids: List[int]) -> str:
        """Given a list of token IDs, convert it to
        a single string that would be the original string
        it was.

        NOTE:
            We automatically call a method that can optionally
            restore any special reserved tokens back to their
            original values (such as field delimiter values, etc)
        """
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

    def tokenize_delimiter(self, line: str) -> str:
        return line

    def detokenize_delimiter(self, line: str) -> str:
        return line


##################
# Single Char
##################


class CharTokenizerTrainer(BaseTokenizerTrainer):
    """Train a simple tokenizer that maps every single character
    to a unique ID.  If ``vocab_size`` is not specified, the learned
    vocab size will be the number of unique characters in the training
    dataset.

    Args:
        vocab_size: Max number of tokens (chars) to map to tokens.
    """

    newline_str: str = "\n"

    def _train(self):
        vocab = set()
        with open(self.config.training_data_path, "rb") as fin:
            for line in fin:
                line = line.decode()
                _vocab = set(line)
                vocab = vocab.union(_vocab)
        vocab = sorted(vocab)
        if self.vocab_size is not None:
            vocab = vocab[: self.vocab_size]
        char2idx = {u: i for i, u in enumerate(vocab)}
        idx2char = np.array(vocab)
        self._save_model(char2idx, idx2char)

    def _save_model(self, char2idx, idx2char):
        cloudpickle.dump(
            char2idx, open(Path(self.config.checkpoint_dir) / "char2idx.p", "wb")
        )
        cloudpickle.dump(
            idx2char, open(Path(self.config.checkpoint_dir) / "idx2char.p", "wb")
        )

    def _get_save_settings(self):
        return {"vocab_size": self.vocab_size}


class CharTokenizer(BaseTokenizer):
    """Load a simple character tokenizer from disk to conduct
    encoding an decoding operations
    """

    _model: Dict[str, dict]
    newline_str: str = "\n"

    @classmethod
    def load(cls, model_dir: str):
        """Create an instance of this tokenizer.

        Args:
            model_dir: The path to the model directory
        """
        model = {
            "char2idx": cloudpickle.load(open(Path(model_dir) / "char2idx.p", "rb")),
            "idx2char": cloudpickle.load(open(Path(model_dir) / "idx2char.p", "rb")),
        }
        return cls(model, model_dir)

    @property
    def total_vocab_size(self):
        """Get the number of unique characters (tokens)"""
        return len(self._model["idx2char"])

    def _encode_to_ids(self, data: str) -> List[int]:
        try:
            return [self._model["char2idx"][char] for char in data]
        except KeyError as err:
            raise TokenizerError(
                "Some characters in the input string are not part of the vocab"
            ) from err

    def _decode_from_ids(self, ids: List[int]) -> str:
        try:
            chars = [self._model["idx2char"][id] for id in ids]
            return "".join(chars)
        except IndexError as err:
            raise TokenizerError(
                "Some IDs do not have mappings to chars in the vocab"
            ) from err


#################
# Sentence Piece
#################


class VocabSizeTooSmall(ParameterError):
    """
    Error that is raised when the `vocab_size` is too small for the given data.
    This happens, when the `vocab_size` is set to a value that is smaller than the
    number of required characters.
    """

    ...


class SentencePieceTokenizerTrainer(BaseTokenizerTrainer):
    """Train a tokenizer using Google SentencePiece."""

    vocab_size: int
    """Pre-determined maximum vocabulary size prior to neural model training, based
    on subword units including byte-pair-encoding (BPE) and unigram language model,
    with the extension of direct training from raw sentences.
    We generally recommend using a large vocabulary size of
    20,000 to 50,000. Default is ``20000``.
    """

    character_coverage: float
    """The amount of characters covered by the model. Unknown characters
    will be replaced with the <unk> tag. Good defaults are ``0.995`` for languages with rich
    character sets like Japanese or Chinese, and 1.0 for other languages or machine data.
    Default is ``1.0``.
    """

    pretrain_sentence_count: int
    """The number of lines spm_train first loads. Remaining lines are simply
    discarded. Since spm_train loads entire corpus into memory, this size will depend on the memory
    size of the machine. It also affects training time. Default is ``1000000``.
    """

    max_line_line: int
    """Maximum line length for input training data. Any lines longer than
    this length will be ignored. Default is ``2048``.
    """

    newline_str: str = "<n>"

    def __init__(
        self,
        *,
        character_coverage: float = 1.0,
        pretrain_sentence_count: int = 1000000,
        max_line_len: int = 2048,
        **kwargs,
    ):
        self.character_coverage = character_coverage
        self.pretrain_sentence_count = pretrain_sentence_count
        self.max_line_line = max_line_len

        # SP Needs a vocab size int, so we set a default one if
        # not already provided in the kwargs
        if "vocab_size" not in kwargs:
            kwargs["vocab_size"] = 20000

        super().__init__(**kwargs)

    def _no_delim_line(self, line: str) -> str:
        return line.strip() + self.newline_str + "\n"

    def _annotate_training_line(self, line: str):
        if self.config.field_delimiter is not None:
            line = line.strip().replace(
                self.config.field_delimiter, self.config.field_delimiter_token
            )
            line += f"{self.newline_str}\n"
        else:
            line = self._no_delim_line(line)

        return line

    def _train(self, extra_symbols: Optional[List[str]] = None):
        if extra_symbols is None:
            extra_symbols = []
        user_defined_symbols = [
            self.newline_str,
            self.config.field_delimiter_token,
        ] + extra_symbols
        logger.info("Training SentencePiece tokenizer")
        try:
            spm.SentencePieceTrainer.Train(
                input=self.config.training_data_path,
                model_prefix=const.MODEL_PREFIX,
                user_defined_symbols=user_defined_symbols,
                vocab_size=self.vocab_size,
                hard_vocab_limit=False,
                max_sentence_length=self.max_line_line,
                input_sentence_size=self.pretrain_sentence_count,
                shuffle_input_sentence=True,
                character_coverage=self.character_coverage,
            )
        except RuntimeError as e:
            if "Vocabulary size is smaller than required_chars" in str(e):
                raise VocabSizeTooSmall(
                    "The value for the `vocab_size` parameter is too small for the "
                    "provided dataset. Please increase it or set it to `0` to use "
                    "character-based tokenization."
                ) from e

            raise e

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
            "max_line_len": self.max_line_line,
        }


_DEFAULT_COL_PATTERN = "<col{}>"
_COL_PATTERN_RE = re.compile(r"^<[a-zA-z]{3,5}\{\}>$")
COL_PATTERN = "col_pattern"


def _add_column_markers(
    delim: str,
    col_pattern: str,
    newline_str: str,
    line: str,
    ignore_newline: bool = False,
) -> Tuple[str, Set[str]]:
    symbols = set()  # track every column token we create i.e. <colN>
    line = line.strip()
    parts = line.split(delim)
    new_parts = []
    for idx, value in enumerate(parts):
        symbol = col_pattern.format(idx)
        new_parts.append(symbol)
        new_parts.append(value)
        symbols.add(symbol)
    if not ignore_newline:
        new_parts.extend([newline_str, "\n"])
    return "".join(new_parts), symbols


class SentencePieceColumnTokenizerTrainer(SentencePieceTokenizerTrainer):
    _col_pattern: str
    _col_symbols: Set[str]

    def __init__(self, col_pattern: str = _DEFAULT_COL_PATTERN, **kwargs):
        if not _COL_PATTERN_RE.match(col_pattern):
            raise ValueError(
                f"col_pattern must satisfy the following pattern: {_COL_PATTERN_RE.pattern}"
            )
        self._col_pattern = col_pattern
        self._col_symbols = set()
        super().__init__(**kwargs)

    def _annotate_training_line(self, line: str) -> str:
        if self.config.field_delimiter is not None:
            new_line, symbols = _add_column_markers(
                self.config.field_delimiter, self._col_pattern, self.newline_str, line
            )
            self._col_symbols = self._col_symbols.union(symbols)
            return new_line
        else:
            return self._no_delim_line(line)

    def _get_save_settings(self) -> dict:
        curr_dict = super()._get_save_settings()
        curr_dict[COL_PATTERN] = self._col_pattern
        return curr_dict

    def _train(self):
        super()._train(extra_symbols=sorted(list(self._col_symbols)))


def _log_sample_data(model_dir: str, sp: spm.SentencePieceProcessor):
    training_data_path = Path(model_dir) / const.TRAINING_DATA
    if not training_data_path.is_file():
        logger.info("Training data not found for SP sampling")
        return

    with open(training_data_path) as fin:
        sample = fin.readline().strip()

    logger.info(f"Tokenizer model vocabulary size: {len(sp)} tokens")
    logger.info(
        "Mapping first line of training data\n\n{}\n ---- sample tokens mapped to pieces ---- > \n{}\n".format(
            repr(sample), ", ".join(sp.SampleEncodeAsPieces(sample, -1, 0.1))
        ),
        extra={"maybe_sensitive": True},
    )
    logger.info(
        "Mapping first line of training data\n\n{}\n ---- sample tokens mapped to int ---- > \n{}\n".format(
            repr(sample), ", ".join([str(idx) for idx in sp.EncodeAsIds(sample)])
        ),
        extra={"maybe_sensitive": True},
    )


class SentencePieceTokenizer(BaseTokenizer):
    """Load a SentencePiece tokenizer from disk so encoding / decoding
    can be done
    """

    _model: spm.SentencePieceProcessor
    newline_str: str = "<n>"

    @classmethod
    def load(cls, model_dir: str):
        """Load a SentencePiece tokenizer from a model directory.

        Args:
            model_dir: The model directory.
        """
        sp = spm.SentencePieceProcessor()
        model_fname = f"{const.MODEL_PREFIX}.model"
        logger.info("Loading tokenizer from: %s", model_fname)
        model_path = Path(model_dir) / model_fname
        sp.Load(str(model_path))
        _log_sample_data(model_dir, sp)
        return cls(sp, model_dir)

    @property
    def total_vocab_size(self):
        """The number of unique tokens in the model"""
        return len(self._model)

    def _encode_to_ids(self, data: str) -> List[int]:
        return self._model.EncodeAsIds(data)

    def _decode_from_ids(self, ids: List[int]) -> str:
        return self._model.DecodeIds(ids)

    def _replace_decoded_tokens(self, decoded_line: str) -> str:
        if self.field_delimiter is not None:
            decoded_line = decoded_line.replace(
                self.field_delimiter_token, self.field_delimiter
            )
        return decoded_line

    def tokenize_delimiter(self, line: str) -> str:
        return line.replace(self.field_delimiter, self.field_delimiter_token)

    def detokenize_delimiter(self, line: str) -> str:
        return line.replace(self.field_delimiter_token, self.field_delimiter)


class SentencePieceColumnTokenizer(SentencePieceTokenizer):
    _col_pattern: str
    _col_pattern_re: Pattern

    def __init__(self, sp: spm.SentencePieceProcessor, model_dir: str):
        super().__init__(sp, model_dir)
        self._col_pattern = self.settings.get(COL_PATTERN)
        self._col_pattern_re = re.compile(self._col_pattern.replace("{}", "\\d+"))

    def _restore_delims(self, line_with_tokens: str) -> str:
        # NOTE: Since with this tokenizer the <colN> values are _before_
        # the actual values, we skip the first string in the split parts
        # because it will just be an empty string, so something like
        # <col0>foo<col1>bar will split into ['', 'foo', 'bar]
        parts = self._col_pattern_re.split(line_with_tokens)[1:]
        return self.field_delimiter.join(parts)

    def _replace_decoded_tokens(self, decoded_line: str) -> str:
        return self._restore_delims(decoded_line)

    def tokenize_delimiter(self, line: str) -> str:
        new_line, _ = _add_column_markers(
            self.field_delimiter,
            self._col_pattern,
            self.newline_str,  # ignored
            line,
            ignore_newline=True,
        )
        return new_line

    def detokenize_delimiter(self, line: str) -> str:
        return self._replace_decoded_tokens(line)


##########
# Factory
##########

TOK_MAP = {
    SentencePieceTokenizerTrainer.__name__: SentencePieceTokenizer,
    SentencePieceColumnTokenizerTrainer.__name__: SentencePieceColumnTokenizer,
    CharTokenizerTrainer.__name__: CharTokenizer,
}


def tokenizer_from_model_dir(model_dir: str) -> BaseTokenizer:
    """A factory function that will return a tokenizer instance that
    can be used for encoding / decoding data.  It will try to automatically
    infer what type of class to use based on the stored tokenizer params
    in the provided model directory.

    If no specific tokenizer type is found, we assume that we are restoring
    a SentencePiece tokenizer because the model is from a version <=
    0.14.x

    Args:
        model_dir: A directory that holds synthetic model data.
    """
    params_file = Path(model_dir) / BaseTokenizerTrainer.settings_fname

    # Backwards compat with <= 0.14.0
    if not params_file.is_file():
        tok_cls = SentencePieceTokenizer
    else:
        params_dict = json.loads(open(params_file).read())
        tok_type = params_dict.pop(BaseTokenizerTrainer.tokenizer_type)
        tok_cls = TOK_MAP[tok_type]
    return tok_cls.load(model_dir)
