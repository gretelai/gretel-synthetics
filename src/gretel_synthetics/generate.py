"""
This module provides the functionality to generate synthetic records.

Before using this module you must have already:

    - Created a config
    - Trained a model
"""
import logging
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import Tuple, TYPE_CHECKING, List, Callable

import sentencepiece as spm
import tensorflow as tf

from gretel_synthetics.model import _build_sequential_model

if TYPE_CHECKING:  # pragma: no cover
    from gretel_synthetics.config import _BaseConfig

PredString = namedtuple("pred_string", ["data"])


@dataclass
class gen_text:
    """
    A record that is yielded from the ``generate_text`` generator.

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

    def values_as_list(self) -> List[str]:
        """Attempt to split the generated text on the provided delimiter

        Returns:
            A list of values that are separated by the object's delimiter or None is there
            is no delimiter in the text
        """
        if self.delimiter is not None:
            tmp = self.text.rstrip(self.delimiter)
            return tmp.split(self.delimiter)
        return None


logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


def _load_tokenizer(store: "_BaseConfig") -> spm.SentencePieceProcessor:
    logging.info("Loading SentencePiece tokenizer")
    sp = spm.SentencePieceProcessor()
    sp.Load(store.tokenizer_model)
    return sp


def _prepare_model(
    sp: spm, batch_size: int, store: "_BaseConfig"
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
    store: "_BaseConfig",
) -> Tuple[spm.SentencePieceProcessor, tf.keras.Sequential]:
    sp = _load_tokenizer(store)
    model = _prepare_model(sp, 1, store)
    return sp, model


def generate_text(
    config: "_BaseConfig",
    start_string: str = "<n>",
    line_validator: Callable = None,
    max_invalid: int = 1000,
):
    """A generator that will load a model and start creating records.

    Args:
        store: A configuration object, which you must have created previously
        start_string:  A prefix string that is used to seed the record generation.
            By default we use a newline, but you may substitue any initial value here
            which will influence how the generator predicts what to generate.
        line_validator: An optional callback validator function that will take
            the raw string value from the generator as a single argument. This validator
            can executue arbitrary code with the raw string value. The validator function
            may return a bool to indicate line validity. This boolean value will be set
            on the yielded ``gen_text`` object. Additionally, if the validator throws an
            exception, the ``gen_text`` object will be set with a failed validation. If the
            validator returns None, we will assume successful validation.
        max_invalid: If using a ``line_validator``, this is the maximum number of invalid
            lines to generate. If the number of invalid lines exceeds this value a ``RunTimeError``
            will be raised.

    Simple validator example::

        def my_validator(raw_line: str):
            parts = raw_line.split(',')
            if len(parts) != 5:
                raise Exception('record does not have 5 fields')

    NOTE:
        ``gen_lines`` from the ``config`` is important for this function. If a line validator is not provided,
        each line will count towards the number of total generated lines. When the total lines generated is >=
        ``gen_lines`` we stop. If a line validator is provided, only *valid* lines will count towards
        the total number of lines generated. When the total number of valid lines generated is >= ``gen_lines``,
        we stop.

    NOTE:
        ``gen_chars``, controls the possible maximum number of characters a single
        generated line can have. If a newline character has not been generated before reaching
        this number, then the line will be returned. For example if ``gen_chars`` is 180 and a
        newline has not been generated, once 180 chars have been created, the line will be returned
        no matter what. As a note, if this value is 0, then each line will generate until
        a newline is observed.

    Yields:
        A ``gen_text`` object for each record that is generated. The generator
        will stop after the max number of lines is reached (based on your config).

    """
    logging.info(
        f"Latest checkpoint: {tf.train.latest_checkpoint(config.checkpoint_dir)}"
    )  # noqa

    sp, model = _load_model(config)
    lines_generated = 0
    delim = config.field_delimiter
    invalid = 0

    while True:
        rec = _predict_chars(model, sp, start_string, config).data
        _valid = None
        try:
            if not line_validator:
                yield gen_text(text=rec, valid=None, explain=None, delimiter=delim)
            else:
                check = line_validator(rec)
                if check is False:
                    _valid = False
                    invalid += 1
                else:
                    _valid = True
                yield gen_text(text=rec, valid=_valid, explain=None, delimiter=delim)
        except Exception as err:
            # logging.warning(f'Line failed validation: {rec} errored with {str(err)}')
            invalid += 1
            yield gen_text(text=rec, valid=False, explain=str(err), delimiter=delim)
        else:
            if line_validator and _valid:
                lines_generated += 1
            elif not line_validator:
                lines_generated += 1
            else:
                ...

        if invalid > max_invalid:
            raise RuntimeError("Maximum number of invalid lines reached!")

        if lines_generated >= config.gen_lines:
            break


def _predict_chars(
    model: tf.keras.Sequential,
    sp: spm.SentencePieceProcessor,
    start_string: str,
    store: "_BaseConfig",
) -> str:
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

    # Empty string to store each line
    sentence_ids = []

    # Here batch size == 1
    model.reset_states()

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
