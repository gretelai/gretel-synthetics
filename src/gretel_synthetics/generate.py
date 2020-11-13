"""
Abstract module for generating data.  The ``generate_text`` function is the primary entrypoint for
creating text.
"""
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Optional, Callable, List, Iterable, Iterator
from abc import ABC, abstractmethod

from gretel_synthetics.generate_parallel import get_num_workers, generate_parallel
from gretel_synthetics.errors import GenerationError
from gretel_synthetics.tokenizers import BaseTokenizer, tokenizer_from_model_dir

if TYPE_CHECKING:
    from gretel_synthetics.config import BaseConfig
else:
    BaseConfig = None

PredString = namedtuple("pred_string", ["data"])


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


# NOTE(jm): Better naming of the class, but keeping the old one around
# for backwards compat for now
@dataclass
class GenText(gen_text):
    pass


class BaseGenerator(ABC):
    """Specific generation modules should have a
    subclass of this ABC that implements the core logic
    for generating data
    """

    @abstractmethod
    def generate_next(self, num_lines: int, hard_limit: Optional[int] = None) -> Iterable[GenText]:
        pass


@dataclass
class Settings:
    """
    Arguments for a generator generating lines of text.

    This class contains basic settings for a generation process. It is separated from the Generator class
    for ensuring reliable serializability without an excess amount of code tied to it.

    This class also will take a provided start string and validate that it can be utilized for text
    generation. If the ``start_string`` is something other than the default, we have to do a couple things:

    1) If the config utilizes a field delimiter, the ``start_string`` MUST end with that delimiter
    2) Convert the user-facing delim char into the special delim token specified in the config
    """

    config: BaseConfig
    start_string: Optional[str] = None
    line_validator: Optional[Callable] = None
    max_invalid: int = 1000
    generator: BaseGenerator = None
    tokenizer: BaseTokenizer = None

    def __post_init__(self):
        if self.start_string is not None:
            self._process_start_string()
        else:
            self.start_string = self.tokenizer.newline_str

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


def generate_text(
    config: BaseConfig,
    start_string: Optional[str] = None,
    line_validator: Optional[Callable] = None,
    max_invalid: int = 1000,
    num_lines: Optional[int] = None,
    parallelism: int = 0
) -> Iterator[GenText]:
    """A generator that will load a model and start creating records.

    Args:
        config: A configuration object, which you must have created previously
        start_string:  A prefix string that is used to seed the record generation.
            By default we use a newline, but you may substitue any initial value here
            which will influence how the generator predicts what to generate. If you
            are working with a field delimiter, and you want to seed more than one column
            value, then you MUST utilize the field delimiter specified in your config.
            An example would be "foo,bar,baz,". Also, if using a field delimiter, the string
            MUST end with the delimiter value.
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
        num_lines: If not ``None``, this will override the ``gen_lines`` value that is provided in the ``config``
        parallelism: The number of concurrent workers to use. ``1`` (the default) disables parallelization,
            while a non-positive value means "number of CPUs + x" (i.e., use ``0`` for using as many workers
            as there are CPUs). A floating-point value is interpreted as a fraction of the available CPUs,
            rounded down.

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
        A ``GenText`` object for each record that is generated. The generator
        will stop after the max number of lines is reached (based on your config).

    Raises:
        A  ``RunTimeError`` if the ``max_invalid`` number of lines is generated

    """

    generator_class = config.get_generator_class()
    tokenizer = tokenizer_from_model_dir(config.checkpoint_dir)

    settings = Settings(
        config=config,
        start_string=start_string,
        line_validator=line_validator,
        max_invalid=max_invalid,
        generator=generator_class,
        tokenizer=tokenizer
    )

    if num_lines is not None:
        _line_count = num_lines
    else:
        _line_count = config.gen_lines

    num_workers = get_num_workers(parallelism, _line_count, chunk_size=5)
    if num_workers == 1:
        gen = generator_class(settings)
        yield from gen.generate_next(_line_count)
    else:
        yield from generate_parallel(
            settings,
            _line_count,
            num_workers,
            chunk_size=5
        )
