from typing import TYPE_CHECKING, Callable, List, Iterable, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import namedtuple

import cloudpickle

import sentencepiece as spm
import tensorflow as tf

from gretel_synthetics.model import _build_sequential_model

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig, LocalConfig
else:
    BaseConfig = None
    LocalConfig = None


def _load_tokenizer(store: LocalConfig) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(store.tokenizer_model)
    return sp


def _prepare_model(
    sp: spm.SentencePieceProcessor, batch_size: int, store: LocalConfig
) -> tf.keras.Sequential:  # pragma: no cover
    model = _build_sequential_model(
        vocab_size=len(sp), batch_size=batch_size, store=store
    )

    load_dir = store.checkpoint_dir

    model.load_weights(tf.train.latest_checkpoint(load_dir)).expect_partial()

    model.build(tf.TensorShape([1, None]))
    model.summary()

    return model


def _load_model(
    store: LocalConfig,
) -> Tuple[spm.SentencePieceProcessor, tf.keras.Sequential]:
    sp = _load_tokenizer(store)
    model = _prepare_model(sp, 1, store)
    return sp, model


PredString = namedtuple("pred_string", ["data"])


@dataclass
class Settings:
    """
    Arguments for a generator generating lines of text.

    This class contains basic settings for a generation process. It is separated from the Generator class
    for ensuring reliable serializability without an excess amount of code tied to it.
    """

    config: LocalConfig
    start_string: str = "<n>"
    line_validator: Optional[Callable] = None
    max_invalid: int = 1000

    def serialize(self) -> bytes:
        return cloudpickle.dumps(self)


def deserialize_settings(serialized: bytes) -> Settings:
    """
    Deserializes a serialized ``Settings`` instance.

    Args:
        serialized: the bytes of the serialized ``Settings`` instance.

    Returns:
        The deserialized ``Settings`` instance.

    Raises:
        A ``TypeError`` if the deserialized object is not a ``Settings`` instance.
    """
    obj = cloudpickle.loads(serialized)
    if not isinstance(obj, Settings):
        raise TypeError("deserialized object is of type {}, not Settings".format(type(obj).__name__))
    return obj


@dataclass
class gen_text:
    """
    A record that is yielded from the ``Generator.generate_next`` generator.

    Attributes:
        valid: True, False, or None. If the line passed a validation function,
            then this will be ``True``. If the validation function raised an exception
            then this will be automatically set to ``False``. If no validation function
            is used, then this value will be ``None.``
        text: The actual record as a string
        explain: A string that describes why a record failed validation. This is the
            string representation of the ``Exception`` that is thrown in a validation
            function. This will only be set if validation fails, otherwise will be ``None.``
        delimiter: If the generated text are column/field based records. This will hold the delimiter
            used to separate the fields from each other.
    """

    valid: bool = None
    text: str = None
    explain: str = None
    delimiter: str = None

    def as_dict(self) -> dict:
        """Serialize the generated record to a dictionary
        """
        return asdict(self)

    def values_as_list(self) -> Optional[List[str]]:
        """Attempt to split the generated text on the provided delimiter

        Returns:
            A list of values that are separated by the object's delimiter or None is there
            is no delimiter in the text
        """
        if self.delimiter is not None:
            tmp = self.text.rstrip(self.delimiter)
            return tmp.split(self.delimiter)
        return None


class Generator:
    """
    A class for generating synthetic lines of text.

    Text generation is initiated via the ``generate_next`` method, which returns an iterable that yields values
    until the given number of _valid_ lines is returned. Each Generator also has a maximum budget of the number
    of invalid lines that can be generated across _all_ calls to ``generate_next``.

    Args:
            settings: the generator settings to use.
    """
    settings: Settings
    model: tf.keras.Sequential
    sp: spm.SentencePieceProcessor
    delim: str
    total_invalid: int = 0
    total_generated: int = 0

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sp, self.model = _load_model(settings.config)
        self.delim = settings.config.field_delimiter

    def generate_next(self, num_lines: int) -> Iterable[gen_text]:
        """
        Returns a sequence of lines.

        Args:
            num_lines: the number of _valid_ lines that should be generated during this call. The actual
                number of lines returned may be higher, in case of invalid lines in the generation output.

        Yields:
            A ``gen_text`` object for every line (valid or invalid) that is generated.
        """
        valid_lines_generated = 0
        while valid_lines_generated < num_lines:
            rec = _predict_chars(self.model, self.sp, self.settings.start_string, self.settings.config).data
            _valid = None
            try:
                if not self.settings.line_validator:
                    yield gen_text(text=rec, valid=None, explain=None, delimiter=self.delim)
                else:
                    check = self.settings.line_validator(rec)
                    if check is False:
                        _valid = False
                        self.total_invalid += 1
                    else:
                        _valid = True
                    yield gen_text(text=rec, valid=_valid, explain=None, delimiter=self.delim)
            except Exception as err:
                # NOTE: this catches any exception raised by the line validator, which
                # also creates an invalid record
                self.total_invalid += 1
                yield gen_text(text=rec, valid=False, explain=str(err), delimiter=self.delim)
            else:
                if self.settings.line_validator and _valid:
                    valid_lines_generated += 1
                elif not self.settings.line_validator:
                    valid_lines_generated += 1
                else:
                    ...

            if self.total_invalid > self.settings.max_invalid:
                raise RuntimeError("Maximum number of invalid lines reached!")


def _predict_chars(
    model: tf.keras.Sequential,
    sp: spm.SentencePieceProcessor,
    start_string: str,
    store: BaseConfig,
) -> PredString:
    """
    Evaluation step (generating text using the learned model).

    Args:
        model: tf.keras.Sequential model
        sp: SentencePiece tokenizer
        start_string: string to bootstrap model
        store: our config object
    Returns:
        Yields line of text per iteration
    """

    # Converting our start string to numbers (vectorizing)
    input_eval = sp.EncodeAsIds(start_string)
    input_eval = tf.expand_dims(input_eval, 0)

    # Here batch size == 1
    model.reset_states()

    sentence_ids = []

    while True:
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to
        # predict the word returned by the model
        predictions = predictions / store.gen_temp
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        sentence_ids.append(int(predicted_id))

        decoded = sp.DecodeIds(sentence_ids)
        if store.field_delimiter is not None:
            decoded = decoded.replace(
                store.field_delimiter_token, store.field_delimiter
            )

        if "<n>" in decoded:
            return PredString(decoded.replace("<n>", ""))
        elif 0 < store.gen_chars <= len(decoded):
            return PredString(decoded)
