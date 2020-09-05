from typing import TYPE_CHECKING, Callable, List, Iterable, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import namedtuple

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
    model = _prepare_model(sp, store.predict_batch_size, store)
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


class TooManyInvalidError(RuntimeError):
    pass


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
    _compiled_predict_and_sample: Callable

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sp, self.model = _load_model(settings.config)
        self.delim = settings.config.field_delimiter

        @tf.function
        def compiled_predict_and_sample(input_eval):
            return _predict_and_sample(self.model, input_eval, self.settings.config.gen_temp)

        self._compiled_predict_and_sample = compiled_predict_and_sample

    def generate_next(self, num_lines: int, hard_limit: Optional[int] = None) -> Iterable[gen_text]:
        """
        Returns a sequence of lines.

        Args:
            num_lines: the number of _valid_ lines that should be generated during this call. The actual
                number of lines returned may be higher, in case of invalid lines in the generation output.
            hard_limit: if set, imposes a hard limit on the overall number of lines that are generated during
                this call, regardless of whether the requested number of valid lines was hit.

        Yields:
            A ``gen_text`` object for every line (valid or invalid) that is generated.
        """
        valid_lines_generated = 0
        total_lines_generated = 0

        predictions = self._predict_forever()
        while valid_lines_generated < num_lines and (hard_limit is None or total_lines_generated < hard_limit):
            rec = next(predictions).data
            total_lines_generated += 1
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
                raise TooManyInvalidError("Maximum number of invalid lines reached!")

    def _predict_forever(self):
        while True:
            yield from _predict_chars(
                self.model, self.sp, self.settings.start_string, self.settings.config,
                self._compiled_predict_and_sample)


def _predict_chars(
    model: tf.keras.Sequential,
    sp: spm.SentencePieceProcessor,
    start_string: str,
    store: BaseConfig,
    predict_and_sample: Optional[Callable] = None,
) -> Iterable[PredString]:
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
    start_vec = sp.EncodeAsIds(start_string)
    input_eval = tf.constant([start_vec for _ in range(store.predict_batch_size)])

    if predict_and_sample is None:
        def predict_and_sample(this_input):
            return _predict_and_sample(model, this_input, store.gen_temp)

    # Batch prediction
    batch_sentence_ids = [[] for _ in range(store.predict_batch_size)]
    not_done = set(i for i in range(store.predict_batch_size))

    model.reset_states()

    while not_done:
        input_eval = predict_and_sample(input_eval)
        for i in not_done:
            batch_sentence_ids[i].append(int(input_eval[i, 0].numpy()))

        batch_decoded = [(i, sp.DecodeIds(batch_sentence_ids[i])) for i in not_done]
        if store.field_delimiter is not None:
            batch_decoded = [(i, decoded.replace(
                store.field_delimiter_token, store.field_delimiter
            )) for i, decoded in batch_decoded]

        for i, decoded in batch_decoded:
            end_idx = decoded.find("<n>")
            if end_idx >= 0:
                decoded = decoded[:end_idx]
                yield PredString(decoded)
                not_done.remove(i)
            elif 0 < store.gen_chars <= len(decoded):
                yield PredString(decoded)
                not_done.remove(i)


def _predict_and_sample(model, input_eval, gen_temp):
    predictions = model(input_eval)[:, -1, :]

    # using a categorical distribution to
    # predict the word returned by the model
    predictions = predictions / gen_temp
    predicted_ids = tf.random.categorical(predictions, num_samples=1)

    return predicted_ids
