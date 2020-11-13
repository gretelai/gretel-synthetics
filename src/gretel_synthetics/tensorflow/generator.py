from typing import (
    TYPE_CHECKING,
    Callable,
    Generator as GeneratorType,
    List,
    Iterable,
    Optional,
    Tuple,
)

import tensorflow as tf

from gretel_synthetics.tensorflow.model import load_model
from gretel_synthetics.generate import PredString, GenText, Settings, BaseGenerator
from gretel_synthetics.errors import TooManyInvalidError

if TYPE_CHECKING:
    from gretel_synthetics.config import TensorFlowConfig
    from gretel_synthetics.tokenizers import BaseTokenizer
else:
    TensorFlowConfig = None
    BaseTokenizer = None


class TensorFlowGenerator(BaseGenerator):
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
    delim: str
    total_invalid: int = 0
    total_generated: int = 0
    _predictions: GeneratorType[PredString, None, None]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = load_model(settings.config, self.settings.tokenizer)
        self.delim = settings.config.field_delimiter
        self._predictions = self._predict_forever()

    def generate_next(
        self, num_lines: int, hard_limit: Optional[int] = None
    ) -> Iterable[GenText]:
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

        while valid_lines_generated < num_lines and (
            hard_limit is None or total_lines_generated < hard_limit
        ):
            rec = next(self._predictions).data
            total_lines_generated += 1
            _valid = None
            try:
                if not self.settings.line_validator:
                    yield GenText(
                        text=rec, valid=None, explain=None, delimiter=self.delim
                    )
                else:
                    check = self.settings.line_validator(rec)
                    if check is False:
                        _valid = False
                        self.total_invalid += 1
                    else:
                        _valid = True
                    yield GenText(
                        text=rec, valid=_valid, explain=None, delimiter=self.delim
                    )
            except Exception as err:
                # NOTE: this catches any exception raised by the line validator, which
                # also creates an invalid record
                self.total_invalid += 1
                yield GenText(
                    text=rec, valid=False, explain=str(err), delimiter=self.delim
                )
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
            return _predict_and_sample(
                self.model, input_eval, self.settings.config.gen_temp
            )

        while True:
            yield from _predict_chars(
                self.model,
                self.settings.tokenizer,
                self.settings.start_string,
                self.settings.config,
                compiled_predict_and_sample,
            )


def _replace_prefix(
    batch_decoded, prefix: str = None
) -> List[Tuple[int, str]]:
    """Given a decoded predicted string, that contains special tokens for things like field
    delimiters, we restore those tokens back to the original char they were previously.

    Additionally, if a ``start_string`` was provided to seed the generation, we need to restore
    the delim tokens in that start string and preprend it to the predicted string.
    """
    out = []
    for i, decoded in batch_decoded:
        if prefix is not None:
            decoded = "".join([prefix, decoded])
        out.append((i, decoded))
    return out


def _predict_chars(
    model: tf.keras.Sequential,
    tokenizer: BaseTokenizer,
    start_string: str,
    store: TensorFlowConfig,
    predict_and_sample: Optional[Callable] = None,
) -> GeneratorType[PredString, None, None]:
    """
    Evaluation step (generating text using the learned model).

    Args:
        model: tf.keras.Sequential model
        tokenizer: A subclass of BaseTokenizer
        start_string: string to bootstrap model. NOTE: this string MUST already have had special tokens
            inserted (i.e. <d>)
        store: our config object
    Returns:
        Yields line of text per iteration
    """

    # Converting our start string to numbers (vectorizing)
    start_vec = tokenizer.encode_to_ids(start_string)
    input_eval = tf.constant([start_vec for _ in range(store.predict_batch_size)])

    if predict_and_sample is None:

        def predict_and_sample(this_input):
            return _predict_and_sample(model, this_input, store.gen_temp)

    # Batch prediction
    batch_sentence_ids = [[] for _ in range(store.predict_batch_size)]
    not_done = set(i for i in range(store.predict_batch_size))

    if store.reset_states:
        # Reset RNN model states between each record created
        # guarantees more consistent record creation over time, at the
        # expense of model accuracy
        model.reset_states()

    # if the start string is not the default newline, then we create a prefix string
    # that we will append to each decoded prediction
    prediction_prefix = None
    if start_string != tokenizer.newline_str:
        if store.field_delimiter is not None:
            prediction_prefix = start_string.replace(
                store.field_delimiter_token, store.field_delimiter
            )
        else:
            prediction_prefix = start_string

    while not_done:
        input_eval = predict_and_sample(input_eval)
        for i in not_done:
            batch_sentence_ids[i].append(int(input_eval[i, 0].numpy()))

        batch_decoded = [
            (i, tokenizer.decode_from_ids(batch_sentence_ids[i])) for i in not_done
        ]
        batch_decoded = _replace_prefix(batch_decoded, prediction_prefix)

        for i, decoded in batch_decoded:
            end_idx = decoded.find(tokenizer.newline_str)
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
