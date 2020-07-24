"""
This module provides the functionality to generate synthetic records.

Before using this module you must have already:

    - Created a config
    - Trained a model
"""
import logging
from typing import TYPE_CHECKING, Callable

import tensorflow as tf

from gretel_synthetics.generator import Generator, Settings
from gretel_synthetics.generator import gen_text, PredString  # noqa # pylint: disable=unused-import
from gretel_synthetics.generate_parallel import split_work, generate_parallel

if TYPE_CHECKING:  # pragma: no cover
    from gretel_synthetics.config import LocalConfig
else:
    LocalConfig = None


logging.basicConfig(
    format="%(asctime)s : %(threadName)s : %(levelname)s : %(message)s",
    level=logging.INFO,
)


def generate_text(
    config: LocalConfig,
    start_string: str = "<n>",
    line_validator: Callable = None,
    max_invalid: int = 1000,
    num_lines: int = None,
    parallelism: int = 0,
):
    """A generator that will load a model and start creating records.

    Args:
        config: A configuration object, which you must have created previously
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
        A ``gen_text`` object for each record that is generated. The generator
        will stop after the max number of lines is reached (based on your config).

    Raises:
        A  ``RunTimeError`` if the ``max_invalid`` number of lines is generated

    """
    logging.info(
        f"Latest checkpoint: {tf.train.latest_checkpoint(config.checkpoint_dir)}"
    )  # noqa

    settings = Settings(
        config=config,
        start_string=start_string,
        line_validator=line_validator,
        max_invalid=max_invalid,
    )

    if num_lines is not None:
        _line_count = num_lines
    else:
        _line_count = config.gen_lines

    num_workers, chunks = split_work(parallelism, _line_count)

    if num_workers == 1:  # Sequential operation
        gen = Generator(settings)
        yield from gen.generate_next(_line_count)
    else:
        yield from generate_parallel(settings, num_workers, chunks)
