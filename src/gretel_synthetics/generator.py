from typing import TYPE_CHECKING, Callable, Generator as GeneratorType, List, Iterable, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import namedtuple

import sentencepiece as spm
import tensorflow as tf

from gretel_synthetics.model import build_sequential_model

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig, LocalConfig
else:
    BaseConfig = None
    LocalConfig = None

NEWLINE = "<n>"


class GenerationError(Exception):
    pass


def _load_tokenizer(store: LocalConfig) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.Load(store.tokenizer_model)
    return sp


def _prepare_model(
    sp: spm.SentencePieceProcessor, batch_size: int, store: LocalConfig
) -> tf.keras.Sequential:  # pragma: no cover
    model = build_sequential_model(
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

    This class also will take a provided start string and validate that it can be utilized for text
    generation. If the ``start_string`` is something other than the default, we have to do a couple things:
        - If the config utilizes a field delimiter, the ``start_string`` MUST end with that delimiter
        - Convert the user-facing delim char into the special delim token specified in the config
    """

    config: LocalConfig
    start_string: str = NEWLINE
    line_validator: Optional[Callable] = None
    max_invalid: int = 1000

    def __post_init__(self):
        if self.start_string != NEWLINE:
            self._process_start_string()

    def _process_start_string(self):
        if not isinstance(self.start_string, str):
            raise GenerationError("Seed start_string must be a str!")
        if self.config.field_delimiter is not None:
            # the start_string must end with the delim
            if not self.start_string.endswith(self.config.field_delimiter):
                raise GenerationError(f"start_string must end with the specified field delimiter: {self.config.field_delimiter}")  # noqa
            self.start_string = self.start_string.replace(
                self.config.field_delimiter,
                self.config.field_delimiter_token
            )


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

    NOTE:
        If the ``settings`` object has a non-default ``start_string`` set, then that ``start_string`` MUST have
        already had special tokens inserted. This should generally be handled during the construction of the Settings
        object.
    """
    settings: Settings
    model: tf.keras.Sequential
    sp: spm.SentencePieceProcessor
    delim: str
    total_invalid: int = 0
    total_generated: int = 0
    _predictions: GeneratorType[PredString, None, None]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sp, self.model = _load_model(settings.config)
        self.delim = settings.config.field_delimiter
        self._predictions = self._predict_forever()

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

        while valid_lines_generated < num_lines and (hard_limit is None or total_lines_generated < hard_limit):
            rec = next(self._predictions).data
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

    def _predict_forever(self) -> GeneratorType[PredString, None, None]:
        """
        Returns a generator infinitely producing prediction strings.

        Returns:
            A generator producing an infinite sequence of ``PredString``s.
        """
        @tf.function
        def compiled_predict_and_sample(input_eval):
            return _predict_and_sample(self.model, input_eval, self.settings.config.gen_temp)

        while True:
            yield from _predict_chars(
                self.model, self.sp, self.settings.start_string, self.settings.config,
                compiled_predict_and_sample)


def _replace_decoded_tokens(batch_decoded, store: BaseConfig, prefix: str = None) -> List[Tuple[int, str]]:
    """Given a decoded predicted string, that contains special tokens for things like field
    delimiters, we restore those tokens back to the original char they were previously.

    Additionally, if a ``start_string`` was provided to seed the generation, we need to restore
    the delim tokens in that start string and preprend it to the predicted string.
    """
    out = []
    for i, decoded in batch_decoded:
        if store.field_delimiter is not None:
            decoded = decoded.replace(store.field_delimiter_token, store.field_delimiter)
        if prefix is not None:
            decoded = "".join([prefix, decoded])
        out.append((i, decoded))
    return out


def _predict_chars(
    model: tf.keras.Sequential,
    sp: spm.SentencePieceProcessor,
    start_string: str,
    store: BaseConfig,
    predict_and_sample: Optional[Callable] = None,
) -> GeneratorType[PredString, None, None]:
    """
    Evaluation step (generating text using the learned model).

    Args:
        model: tf.keras.Sequential model
        sp: SentencePiece tokenizer
        start_string: string to bootstrap model. NOTE: this string MUST already have had special tokens
            inserted (i.e. <d>)
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

    # if the start string is not the default newline, then we create a prefix string
    # that we will append to each decoded prediction
    prediction_prefix = None
    if start_string != NEWLINE:
        if store.field_delimiter is not None:
            prediction_prefix = start_string.replace(store.field_delimiter_token, store.field_delimiter)
        else:
            prediction_prefix = start_string

    while not_done:
        input_eval = predict_and_sample(input_eval)
        for i in not_done:
            batch_sentence_ids[i].append(int(input_eval[i, 0].numpy()))

        batch_decoded = [(i, sp.DecodeIds(batch_sentence_ids[i])) for i in not_done]
        batch_decoded = _replace_decoded_tokens(batch_decoded, store, prediction_prefix)

        for i, decoded in batch_decoded:
            end_idx = decoded.find(NEWLINE)
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
