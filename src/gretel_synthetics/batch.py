"""
This module allows automatic splitting of a DataFrame
into smaller DataFrames (by clusters of columns) and doing
model training and text generation on each sub-DF independently.

Then we can concat each sub-DF back into one final synthetic dataset.

For example usage, please see our Jupyter Notebook.
"""
from dataclasses import dataclass, field
from pathlib import Path
import gzip
from math import ceil
from typing import List, Type, Callable, Dict, Union, Iterator as IteratorType, Optional
from copy import deepcopy
import logging
import io
import json
import glob
import shutil
import time
import abc
import threading
from itertools import zip_longest

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import cloudpickle

from gretel_synthetics.config import (
    LocalConfig,
    BaseConfig,
    config_from_model_dir,
    CONFIG_MAP,
)
import gretel_synthetics.const as const
from gretel_synthetics.generate import GenText, generate_text, SeedingGenerator
from gretel_synthetics.errors import TooManyInvalidError
from gretel_synthetics.train import train
from gretel_synthetics.tokenizers import BaseTokenizerTrainer


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_INVALID = 1000
BATCH_SIZE = 15
FIELD_DELIM = "field_delimiter"
GEN_LINES = "gen_lines"
READ = "read"
WRITE = "write"
HEADER_FILE = "headers.json"
ORIG_HEADERS = "original_headers.json"
CHECKPOINT_DIR = "checkpoint_dir"
CONFIG_FILE = "model_params.json"
TRAIN_FILE = "train.csv"
PATH_HOLDER = "___path_holder___"


@dataclass
class GenerationSummary:
    """A class to capture the summary data after synthetic data is generated."""

    valid_lines: int = 0
    invalid_lines: int = 0
    is_valid: bool = False


class _BatchEpochCallback:
    """
    Wrapper class to take a user supplied callback and inject the batch number.  The batch number
    is then available in the EpochState object when it is supplied to the callback.
    """

    def __init__(self, user_callback: callable, batch_number: int):
        self._batch_number = batch_number
        self._user_callback = user_callback

    def callback(self, epoch_state):
        epoch_state.batch = self._batch_number
        self._user_callback(epoch_state)


@dataclass
class Batch:
    """A representation of a synthetic data workflow.  It should not be used
    directly. This object is created automatically by the primary batch handler,
    such as ``DataFrameBatch``.  This class holds all of the necessary information
    for training, data generation and DataFrame re-assembly.
    """

    checkpoint_dir: str
    input_data_path: str
    headers: List[str]
    config: LocalConfig
    gen_data_count: int = 0

    training_df: Type[pd.DataFrame] = field(default_factory=lambda: None, init=False)
    gen_data_stream: io.StringIO = field(default_factory=io.StringIO, init=False)
    gen_data_invalid: List[GenText] = field(default_factory=list, init=False)
    validator: Callable = field(default_factory=lambda: None, init=False)

    def __post_init__(self):
        self.reset_gen_data()

    @property
    def synthetic_df(self) -> pd.DataFrame:
        """Get a DataFrame constructed from the generated lines"""
        if not self.gen_data_stream.getvalue():  # pragma: no cover
            return pd.DataFrame()
        self.gen_data_stream.seek(0)
        return pd.read_csv(self.gen_data_stream, sep=self.config.field_delimiter)

    def set_validator(self, fn: Callable, save=True):
        """Assign a validation callable to this batch. Optionally
        pickling and saving the validator for loading later
        """
        self.validator = fn
        if save:
            p = Path(self.checkpoint_dir) / "validator.p.gz"
            with gzip.open(p, "w") as fout:
                fout.write(cloudpickle.dumps(fn))

    def load_validator_from_file(self):
        """Load a saved validation object if it exists"""
        p = Path(self.checkpoint_dir) / "validator.p.gz"
        if p.exists():
            with gzip.open(p, "r") as fin:
                self.validator = cloudpickle.loads(fin.read())

    def reset_gen_data(self):
        """Reset all objects that accumulate or track synthetic
        data generation
        """
        self.gen_data_invalid = []
        self.gen_data_stream = io.StringIO()
        self.gen_data_stream.write(
            self.config.field_delimiter.join(self.headers) + "\n"
        )
        self.gen_data_count = 0

    def add_valid_data(self, data: GenText):
        """Take a ``gen_text`` object and add the generated
        line to the generated data stream
        """
        self.gen_data_stream.write(data.text + "\n")
        self.gen_data_count += 1

    def _basic_validator(self, raw_line: str):  # pragma: no cover
        return len(raw_line.split(self.config.field_delimiter)) == len(self.headers)

    def get_validator(self):
        """If a custom validator is set, we return that. Otherwise,
        we return the built-in validator, which simply checks if a generated
        line has the right number of values based on the number of headers
        for this batch.

        This at least makes sure the resulting DataFrame will be the right
        shape
        """
        if self.validator is not None:
            return self.validator

        return self._basic_validator


def _create_batch_from_dir(batch_dir: str):
    path = Path(batch_dir)
    if not path.is_dir():  # pragma: no cover
        raise ValueError("%s is not a directory" % batch_dir)

    if not (path / HEADER_FILE).is_file():  # pragma: no cover
        raise ValueError("missing headers")
    headers = json.loads(open(path / HEADER_FILE).read())

    if not (path / CONFIG_FILE).is_file():  # pragma: no cover
        raise ValueError("missing model param file")

    config = config_from_model_dir(batch_dir)

    # training path can be empty, since we will not need access
    # to training data simply for read-only data generation
    train_path = ""

    # Wrap the user supplied callback with a _BatchEpochCallback so we have the batch number too.
    if config.epoch_callback is not None:
        batch_count = int(Path(batch_dir).name.split("_")[-1])
        config.epoch_callback = _BatchEpochCallback(
            config.epoch_callback, batch_count
        ).callback

    batch = Batch(
        checkpoint_dir=batch_dir,
        input_data_path=train_path,
        headers=headers,
        config=config,
    )

    batch.load_validator_from_file()

    return batch


def _crawl_checkpoint_for_batches(checkpoint_dir: str):
    logger.info("Looking for and loading batch data...")
    matching_dirs = glob.glob(str(Path(checkpoint_dir) / "batch_*"))
    if not matching_dirs:
        raise ValueError(
            "checkpoint directory does not exist or does not contain batch data"
        )

    batches = []
    for batch_dir in matching_dirs:
        idx = int(Path(batch_dir).name.split("_")[-1])
        batches.append((idx, _create_batch_from_dir(batch_dir)))

    logger.info("Found and loaded %d batches", len(batches))
    return dict(sorted(batches, key=lambda b: b[0]))


def _build_batch_dirs(
    base_ckpoint: str, headers: List[List[str]], config: dict
) -> dict:
    """Return a mapping of batch number => ``Batch`` object"""
    out = {}
    logger.info("Creating directory structure for batch jobs...")
    base_path = Path(config["checkpoint_dir"])
    if not base_path.is_dir():
        base_path.mkdir()
    for i, headers in enumerate(headers):
        ckpoint = Path(base_ckpoint) / f"batch_{i}"
        if not ckpoint.is_dir():
            ckpoint.mkdir()
        checkpoint_dir = str(ckpoint)
        input_data_path = str(ckpoint / "train.csv")
        new_config = deepcopy(config)
        new_config.update(
            {"checkpoint_dir": checkpoint_dir, "input_data_path": input_data_path}
        )

        # Determine what BaseConfig subclass to use, if the config template does
        # not have a model type then we'll default to using a LocalConfig which gives
        # us backwards compat to 0.14.0
        config_class_str = new_config.get(const.MODEL_TYPE, None)
        if config_class_str is None:
            config_class = LocalConfig
        else:
            config_class = CONFIG_MAP[config_class_str]

        # Wrap the user supplied callback with a _BatchEpochCallback so we have the batch number too.
        if new_config.get("epoch_callback") is not None:
            new_config["epoch_callback"] = _BatchEpochCallback(
                new_config.get("epoch_callback"), i
            ).callback

        out[i] = Batch(
            checkpoint_dir=checkpoint_dir,
            input_data_path=input_data_path,
            headers=headers,
            config=config_class(**new_config),
        )
        # try and load any previously saved validators
        out[i].load_validator_from_file()

        # we write the headers out as well incase we load these
        # batches back in via "read" mode only later...it's the only
        # way to get the header names back
        with open(ckpoint / HEADER_FILE, "w") as fout:
            fout.write(json.dumps(headers))

    return out


def _validate_batch_seed_values(
    batch: Batch, seed_values: Union[dict, List[dict]]
) -> Union[str, List[str]]:
    """Validate that seed values line up with the first N columns in a batch. Also construct
    an appropiate seed string based on the values in the batch
    """
    ret_str = True
    if isinstance(seed_values, dict):
        seed_values = [seed_values]
    elif isinstance(seed_values, list):
        ret_str = False
    else:
        raise TypeError("seed_values should be a dict or list of dicts")

    seed_strings = []

    for seed in seed_values:
        if len(seed) > len(batch.headers):
            raise RuntimeError(
                "The number of seed fields is greater than the number of columns in the first batch"
            )

        headers_to_seed = batch.headers[: len(seed)]
        tmp = []
        for header in headers_to_seed:
            value = seed.get(header)
            if value is None:
                raise RuntimeError(
                    f"The header: {header} is not in the seed values mapping"
                )  # noqa
            tmp.append(str(value))

        seed_strings.append(
            batch.config.field_delimiter.join(tmp) + batch.config.field_delimiter
        )

    if ret_str:
        return seed_strings[0]
    else:
        return seed_strings


class _BufferedRecords(abc.ABC):
    """Base class for all buffers used when
    generating records
    """

    @abc.abstractmethod
    def add(self, record: dict):
        ...

    @abc.abstractmethod
    def get_records(self):
        ...


class _BufferedDicts(_BufferedRecords):

    _records: List[dict]

    def __init__(self):
        self._records = []

    def add(self, record: dict):
        self._records.append(record)

    def get_records(self):
        return self._records


class _BufferedDataFrame(_BufferedRecords):
    """Buffer dictionaries into a memory stream, then
    load it as a DataFrame and set the column order
    based on the provided list. This allows
    datatypes to be inferred as if the values were
    being read from a CSV on disk.
    """

    def __init__(self, delim: str, columns: List[str]):
        self.delim = delim
        self.columns = columns
        self.headers_set = False
        self.buffer = io.StringIO()

    def add(self, record: dict):
        # write the columns names into the buffer, we
        # use the first dict to specify the order and
        # assume subsequent dicts have the same order
        if not self.headers_set:
            _columns = self.delim.join(record.keys())
            self.buffer.write(_columns + "\n")
            self.headers_set = True
        _row = self.delim.join(record.values())
        self.buffer.write(_row + "\n")

    @property
    def df(self) -> pd.DataFrame:
        self.buffer.seek(0)
        return pd.read_csv(self.buffer, sep=self.delim)[self.columns]

    def get_records(self) -> pd.DataFrame:
        return self.df


@dataclass
class GenerationProgress:
    """
    This class should not have to be used directly.

    It is used to communicate the current progress of record generation.

    When a callback function is passed to the ``RecordFactory.generate_all()`` method,
    each time the callback is called an instance of this class will be passed
    as the single argument::

        def my_callback(data: GenerationProgress):
            ...

        factory: RecordFactory
        df = factory.generate_all(output="df", callback=my_callback)

    This class is used to periodically communicate progress of generation to the user,
    through a callback that can be passed to ``RecordFactory.generate_all()`` method.
    """

    current_valid_count: int = 0
    """The number of valid lines/records that
    were generated so far.
    """

    current_invalid_count: int = 0
    """The number of invalid lines/records that
    were generated so far.
    """

    new_valid_count: int = 0
    """The number of new valid lines/records that
    were generated since the last progress callback.
    """

    new_invalid_count: int = 0
    """The number of new valid lines/records that
    were generated since the last progress callback.
    """

    completion_percent: float = 0.0
    """The percentage of valid lines/records that have been generated."""

    timestamp: float = field(default_factory=time.time)
    """The timestamp from when the information in this object has been captured."""


class _GenerationCallback:
    """
    Wrapper around a callback function that is sending progress updates only once
    per configured time period (``update_interval``).

    Args:
        callback_fn: Callback function to be invoked with current progress.
        update_interval: Number of seconds to wait between sending progress update.
    """

    def __init__(self, callback_fn: callable, update_interval: int = 30):
        self._callback_fn = callback_fn
        self._update_interval = update_interval

        self._last_update_time = int(time.monotonic())
        self._last_progress = GenerationProgress()

    def update_progress(
        self,
        num_lines: int,
        valid_count: int,
        invalid_count: int,
        *,
        force_update=False,
    ):

        """
        Method that's being called from the generator with a progress update.

        Args:
            num_lines: Total number of lines to be generated.
            valid_count: Number of valid lines that were generated so far.
            invalid_count: Number of invalid lines that were generated so far.
            final_update:
                Is this the final update? It is ``True`` when sending last update, after the
                whole generation was complete.
        """
        now = int(time.monotonic())

        if now - self._last_update_time >= self._update_interval or force_update:
            current_progress = GenerationProgress(
                current_valid_count=valid_count,
                current_invalid_count=invalid_count,
                new_valid_count=valid_count - self._last_progress.current_valid_count,
                new_invalid_count=invalid_count
                - self._last_progress.current_invalid_count,
                completion_percent=0
                if num_lines == 0
                else round(valid_count / num_lines * 100, 2),
            )

            self._callback_fn(current_progress)
            self._last_update_time = now
            self._last_progress = current_progress


@dataclass
class _FactoryCounter:
    num_lines: int = 0
    """The target number of lines to generate when
    iterating or generating all records.
    """

    max_invalid: int = MAX_INVALID
    """The number of max invalid lines to tolerate before
    stopping generation and raising a ``RunTimeError.``
    """

    valid_count: int = 0
    """The number of valid records / lines that have been generated
    """

    invalid_count: int = 0
    """The number of invalid records / lines that were generated
    """


def _threading_generation_callback(
    counter: _FactoryCounter, callback: _GenerationCallback, event: threading.Event
):
    while not event.is_set():
        try:
            callback.update_progress(counter.num_lines, counter.valid_count, counter.invalid_count)
        except Exception:
            event.set()
            break
        time.sleep(1)


class RecordFactory:
    """A stateful factory that can be used to generate and validate entire
    records, regardless of the number of underlying header clusters that were
    used to build multiple sub-models.

    Instances of this class should be created by calling the appropiate method
    of the ``DataFrameBatch`` instance. This class should not have to
    be used directly. You should be able to create an instance like so::

        factory = batcher.create_record_factory(num_lines=50)

    The class is init'd with default capacity and limits as specified
    by the ``num_lines`` and ``max_invalid`` attributes.  At any time,
    you can inspect the state of the instance by doing::

        factory.summary

    The factory instance can be used one of two ways: buffered or unbuffered.

    For unbuffered mode, the entire instance can be used as an iterator to
    create synthetic records. Each record will be a dictionary.

    NOTE:
        All values in the generated dictionaries will be strings.

    The ``valid_count`` and ``invalid_count`` counters will update as
    records are generated.

    When creating the record factory, you may also provide an entire
    record validator::

        def validator(rec: dict):
            ...

        factory = batcher.create_record_factory(num_lines=50, validator=validator)

    Each generated record dict will be passed to the validator. This validator may either
    return False or raise an exception to mark a record as invalid.

    At any point, you may reset the state of the factory by calling::

        factory.reset()

    This will reset all counters and allow you to keep generating records.

    Finally, you can generate records in buffered mode, where generated records
    will be buffered in memory and returned as one collection. By default, a list
    of dicts will be returned::

        factory.generate_all()

    You may request the records to be returned as a DataFrame.  The dtypes will
    be inferred as if you were reading the data from a CSV::

        factory.generate_all(output="df")

    NOTE:
        When using ``generate_all``, the factory states will be reset automatically.
    """

    validator: Callable
    """An optional callable that will receive a fully constructed record for one
    final validation before returning or yielding a single record. Records that
    do not pass this validation will also increment the ``invalid_count.``
    """

    _batches: Dict[int, Batch]
    _header_list: List[str]
    _seed_fields: Union[str, List[str]]
    _record_generator: IteratorType[dict]
    _delimiter: str
    _parallelism: int
    _counter = _FactoryCounter
    _invalid_cache_size: int
    _thread_event: threading.Event = None

    invalid_cache: List[dict]

    def __init__(
        self,
        *,
        num_lines: int,
        batches: dict,
        header_list: list,
        delimiter: str,
        seed_fields: Union[dict, list] = None,
        max_invalid=MAX_INVALID,
        validator: Optional[Callable] = None,
        parallelism: int = 4,
        invalid_cache_size: int = 100
    ):
        self._counter = _FactoryCounter()
        self._counter.num_lines = num_lines
        self.max_invalid = max_invalid
        self._batches = batches
        self._header_list = header_list
        self._seed_fields = seed_fields
        self._delimiter = delimiter
        self._parallelism = parallelism
        self.validator = validator
        self._invalid_cache_size = invalid_cache_size
        self.reset()

        if self._seed_fields is not None:
            self._seed_fields = _validate_batch_seed_values(
                self._batches[0], self._seed_fields
            )

        if isinstance(self._seed_fields, list):
            logger.info(
                "Adjusting num_lines and parallelism because seed_fields is a list, will only target %d lines",
                len(self._seed_fields),
            )  # noqa
            self._parallelism = 1
            self._counter.num_lines = len(self._seed_fields)

    def _cache_invalid(self, line: GenText):
        self.invalid_cache.append(line.as_dict())
        self.invalid_cache = self.invalid_cache[:self._invalid_cache_size]

    def _get_record(self) -> IteratorType[dict]:
        # our actual batch line generators
        generators = []

        # if we have a list of seed fields, we do special
        # handling to create the proper generator
        seed_generator = None  # assume no seeds to start
        if isinstance(self._seed_fields, list):
            seed_generator = SeedingGenerator(
                self._batches[0].config,
                seed_list=self._seed_fields,
                line_validator=self._batches[0].get_validator(),
                max_invalid=self.max_invalid * 10000,
            )
            generators.append((self._batches[0], seed_generator))

        for idx, batch in self._batches.items():
            start_string = None
            if idx == 0 and seed_generator:
                # We've already added the first batch's generator to the list
                # so we just continue on to the next one
                continue
            if idx == 0:
                # In the event we have seeds that aren't a list, (i.e. static seeds)
                start_string = self._seed_fields
            generators.append(
                (
                    batch,
                    # We seed the low level API with much higher limits on
                    # valid / invalid generation because we will enforce
                    # those limits in this high level instance.
                    generate_text(
                        batch.config,
                        line_validator=batch.get_validator(),
                        max_invalid=self.max_invalid * 10000,
                        num_lines=self._counter.num_lines * 10000,
                        start_string=start_string,
                        parallelism=self._parallelism,
                    ),
                )
            )

        # At this point, we've created our list of generators. Below here
        # is what gets run on every next() call, which tries to construct
        # a full record from all the underlying batches.

        # keep looping as long as our target line count is less than
        # our total line count
        while self._counter.valid_count < self._counter.num_lines:
            # loop over each batch line generater and attempt
            # to construct a full line, we'll only count a
            # full line once we get through each generator

            # if we are using a watchdog thread to monitor generation
            # and it throws an exception, a threading event will be set
            # that signals generation should stop
            if self._thread_event and self._thread_event.is_set():
                break

            if self._counter.invalid_count >= self.max_invalid:
                raise RuntimeError("Invalid record count exceeded during generation")

            seed_cache = None
            if seed_generator:
                # If we're using a seeding generator (from a list of seeds)
                # we cache the next seed we are about to use to generate
                # the next record.
                seed_cache = seed_generator.settings.start_string[0]

            record = {}
            batch: Batch
            for batch, gen in generators:
                while True:

                    # see above usage for watchdog thread exception handling
                    if self._thread_event and self._thread_event.is_set():
                        break

                    line = next(gen)  # type:  GenText
                    if line.valid is False:
                        self._cache_invalid(line)
                        self._counter.invalid_count += 1
                        if self._counter.invalid_count > self.max_invalid:
                            raise RuntimeError(
                                "Invalid record count exceeded during generation"
                            )
                        continue
                    partial_rec = dict(zip_longest(batch.headers, line.values_as_list(), fillvalue=""))
                    record.update(partial_rec)
                    break

            # Do a final validation, if configured, on the fully constructed
            # record, if this validation fails, we'll still increment our
            # invalid count.

            valid = True  # assume we have a valid record

            if self.validator is not None:
                try:
                    _valid = self.validator(record)
                    if _valid is False:
                        valid = False
                except Exception:
                    valid = False

            if not valid:
                self._counter.invalid_count += 1
                if seed_cache:
                    seed_generator.settings.start_string.insert(0, seed_cache)
                continue  # back to the while start

            self._counter.valid_count += 1
            yield record

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._record_generator)

    def reset(self):
        self._counter.valid_count = 0
        self._counter.invalid_count = 0
        self._record_generator = self._get_record()
        self._thread_event = None
        self.invalid_cache = []

    def generate_all(
        self,
        output: Optional[str] = None,
        callback: Optional[callable] = None,
        callback_interval: int = 30,
        callback_threading: bool = False
    ):
        """Attempt to generate the full number of records that was set when
        creating the ``RecordFactory.``  This method will create a buffer
        that holds all records and then returns the the buffer once
        generation is complete.

        Args:
            output: How the records should be returned. If ``None``, which is the
                default, then a list of record dicts will be returned. Other options
                that are supported are: 'df' for a DataFrame.
            callback: An optional callable that will periodically be called with
                a ``GenerationProgress`` instance as the single argument while
                records are being generated.
            callback_interval: If using a callback, the minimum number of seconds that
                should occur between callbacks.
            callback_threading: If enabled, a watchdog thread will be used to execute
                the callback. This will ensure that the callback is called regardless
                of invalid or valid counts. If callback threading is disabled, the callback
                will only be called after valid records are generated. If the callback
                raises and exception, then a threading event will be set which will trigger
                the stopping of generation.

        Returns:
            Generated records in an object that is dependent on the ``output`` param.  By default
            this will be a list of dicts.
        """
        progress_callback = None
        if callback:
            progress_callback = _GenerationCallback(callback, callback_interval)

        self.reset()
        if output is not None and output not in ("df",):
            raise ValueError("invalid output type")

        _iter = tqdm(self._record_generator, total=self._counter.num_lines)

        buffer = None  # type: _BufferedRecords

        if output == "df":
            buffer = _BufferedDataFrame(self._delimiter, self._header_list)

        if not buffer:
            buffer = _BufferedDicts()

        callback_thread = None
        if callback_threading:
            if not progress_callback:
                raise ValueError("Cannot use callback_threading without a progress callback")
            self._thread_event = threading.Event()
            callback_thread = threading.Thread(
                target=_threading_generation_callback,
                args=(self._counter, progress_callback, self._thread_event)
            )
            callback_thread.start()

        try:
            for rec in _iter:
                # NOTE: This iterator will block while no records are being
                # succesfully generated. If callbacks need to occur in this
                # situation, ensure the callback threading option is enabled
                #
                # If threading is enabled, and the callback encounters an exception,
                # a threading event will be set and the generator will break out of its
                # loop and generation will cease.
                buffer.add(rec)

                if progress_callback and not callback_threading:
                    progress_callback.update_progress(
                        self._counter.num_lines,
                        self._counter.valid_count,
                        self._counter.invalid_count,
                    )

        except (RuntimeError, StopIteration) as err:
            logger.warning(
                f"Runtime error on iteration, returning current buffer, {str(err)}"
            )
        finally:
            if callback_threading:
                self._thread_event.set()
                callback_thread.join()

        # send final progress update
        if progress_callback:
            progress_callback.update_progress(
                self._counter.num_lines,
                self._counter.valid_count,
                self._counter.invalid_count,
                force_update=True,
            )

        return buffer.get_records()

    @property
    def summary(self):
        return {
            "num_lines": self._counter.num_lines,
            "max_invalid": self._counter.max_invalid,
            "valid_count": self._counter.valid_count,
            "invalid_count": self._counter.invalid_count,
        }


class DataFrameBatch:
    """Create a multi-batch trainer / generator. When created, the directory
    structure to store models and training data will automatically be created.
    The directory structure will be created under the "checkpoint_dir" location
    provided in the ``config`` template. There will be one directory per batch,
    where each directory will be called "batch_N" where N is the batch number, starting
    from 0.

    Training and generating can happen per-batch or we can loop over all batches to
    do both train / generation functions.

    Example:
        When creating this object, you must explicitly create the training data
        from the input DataFrame before training models::

            my_batch = DataFrameBatch(df=my_df, config=my_config)
            my_batch.create_training_data()
            my_batch.train_all_batches()

    Args:
        df: The input, source DataFrame
        batch_size:  If ``batch_headers`` is not provided we automatically break up
            the number of columns in the source DataFrame into batches of N columns.
        batch_headers:  A list of lists of strings can be provided which will control
            the number of batches. The number of inner lists is the number of batches, and each
            inner list represents the columns that belong to that batch
        config: A template training config to use, this will be used as kwargs for each Batch's
            synthetic configuration. This may also be a sucblass of ``BaseConfig``. If this is used,
            you can set the ``input_data_path`` param to the constant ``PATH_HOLDER`` as it does not
            really matter
        tokenizer_class:  An optional ``BaseTokenizerTrainer`` subclass. If not provided the default
            tokenizer will be used for the underlying ML engine.

    NOTE:
        When providing a config, the source of training data is not necessary, only the
        ``checkpoint_dir`` is needed. Each batch will control its input training data path
        after it creates the training dataset.
    """

    batches: Dict[int, Batch]
    """A mapping of ``Batch`` objects to a batch number. The batch number (key)
    increments from 0..N where N is the number of batches being used.
    """

    batch_size: int
    """The max number of columns allowed for a single DF batch
    """

    # NOTE: Allowing a dict is for backwards compat
    config: Union[dict, BaseConfig]
    """The template config that will be used for all batches. If a dict
    is provided we default to a TensorFlowConfig.
    """

    mode: Union[WRITE, READ]

    master_header_list: List[str]
    """During training, this is the original column order. When reading from
    disk, we concatenate all headers from all batches together. This list is not
    guaranteed to preserve the original header order.
    """

    original_headers: List[str]
    """Stores the original header list / order from the original training data that was used.
    This is written out to the model directory during training and loaded back in when
    using read-only mode.
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame = None,
        batch_size: int = BATCH_SIZE,
        batch_headers: List[List[str]] = None,
        config: Union[dict, BaseConfig] = None,
        tokenizer: BaseTokenizerTrainer = None,
        mode: str = WRITE,
        checkpoint_dir: str = None,
    ):

        if mode not in (WRITE, READ):  # pragma: no cover
            raise ValueError("mode must be read or write")

        self.mode = mode

        # If the config was a subclass of BaseConfig, then we convert
        # it to a dict and utilize that dict as our template. We do this
        # because when we re-create the batches we want to utilize the
        # Config constructors to set some attrs for us
        if isinstance(config, BaseConfig):
            config = config.as_dict()

        self.tokenizer = tokenizer

        self.original_headers = None

        if self.mode == READ:
            if isinstance(config, dict):
                _ckpoint_dir = config.get("checkpoint_dir")
            else:
                _ckpoint_dir = checkpoint_dir

            if _ckpoint_dir is None:
                raise ValueError("checkpoint_dir required for read mode")
            else:
                self._read_checkpoint_dir = _ckpoint_dir

        if self.mode == WRITE:
            if not config:
                raise ValueError("config is required!")

            checkpoint_path = Path(config[CHECKPOINT_DIR])
            overwrite = config.get("overwrite", False)
            if (
                not overwrite
                and checkpoint_path.is_dir()
                and any(checkpoint_path.iterdir())
            ):
                raise RuntimeError(
                    "checkpoint_dir already exists and is non-empty, set overwrite on config or remove model directory!"
                )  # noqa

            if overwrite and checkpoint_path.is_dir():
                shutil.rmtree(checkpoint_path)

            if not isinstance(df, pd.DataFrame):
                raise ValueError("df must be a DataFrame in write mode")

            if FIELD_DELIM not in config:
                raise ValueError("field_delimiter must be in config")

            if GEN_LINES not in config:
                config[GEN_LINES] = df.shape[0]

            self._source_df = df
            self.batch_size = batch_size
            self.config = config
            self._source_df.fillna("", inplace=True)
            self.master_header_list = list(self._source_df.columns)

            if not batch_headers:
                self.batch_headers = self._create_header_batches()
            else:  # pragma: no cover
                self.batch_headers = batch_headers

            self.batches = _build_batch_dirs(
                self.config["checkpoint_dir"], self.batch_headers, self.config
            )

            # Preserve the original order of the DF headers
            self.original_headers = list(self._source_df)
            with open(Path(self.config[CHECKPOINT_DIR]) / ORIG_HEADERS, "w") as fout:
                fout.write(json.dumps(list(self.original_headers)))
        else:
            self.batches = _crawl_checkpoint_for_batches(self._read_checkpoint_dir)
            self.master_header_list = []
            for batch in self.batches.values():
                self.master_header_list.extend(batch.headers)

            try:
                self.original_headers = json.loads(
                    open(Path(self._read_checkpoint_dir) / ORIG_HEADERS).read()
                )
            except FileNotFoundError:
                self.original_headers = None

            logger.info("Validating underlying models exist via generation test...")
            try:
                self.generate_all_batch_lines(parallelism=1, num_lines=1)
            except Exception as err:
                raise RuntimeError(
                    "Error testing generation during model load"
                ) from err

    def _create_header_batches(self):
        num_batches = ceil(len(self._source_df.columns) / self.batch_size)
        tmp = np.array_split(list(self._source_df.columns), num_batches)
        return [list(row) for row in tmp]

    def create_training_data(self):
        """Split the original DataFrame into N smaller DataFrames. Each
        smaller DataFrame will have the same number of rows, but a subset
        of the columns from the original DataFrame.

        This method iterates over each ``Batch`` object and assigns
        a smaller training DataFrame to the ``training_df`` attribute
        of the object.

        Finally, a training CSV is written to disk in the specific
        batch directory
        """
        if self.mode == READ:  # pragma: no cover
            raise RuntimeError("Method cannot be used in read-only mode")
        for i, batch in self.batches.items():
            logger.info(f"Generating training DF and CSV for batch {i}")
            out_df = self._source_df[batch.headers]
            batch.training_df = out_df.copy(deep=True)
            out_df.to_csv(
                batch.input_data_path,
                header=False,
                index=False,
                sep=self.config[FIELD_DELIM],
            )

    def train_batch(self, batch_idx: int):
        """Train a model for a single batch. All model information will
        be written into that batch's directory.

        Args:
            batch_idx: The index of the batch, from the ``batches`` dictionary
        """
        if self.tokenizer is not None:
            _tokenizer = deepcopy(self.tokenizer)
            _tokenizer.config = self.batches[batch_idx].config
        else:
            _tokenizer = None

        if self.mode == READ:  # pragma: no cover
            raise RuntimeError("Method cannot be used in read-only mode")
        try:
            train(self.batches[batch_idx].config, _tokenizer)
        except KeyError:
            raise ValueError("batch_idx is invalid")

    def train_all_batches(self):
        """Train a model for each batch."""
        if self.mode == READ:  # pragma: no cover
            raise RuntimeError("Method cannot be used in read-only mode")
        for idx in self.batches.keys():
            self.train_batch(idx)

    def set_batch_validator(self, batch_idx: int, validator: Callable):
        """Set a validator for a specific batch. If a validator is configured
        for a batch, each generated record from that batch will be sent
        to the validator.

        Args:
            batch_idx: The batch number .
            validator: A callable that should take exactly one argument,
                which will be the raw line generated from the ``generate_text``
                function.
        """
        if self.mode == READ:  # pragma: no cover
            raise RuntimeError("Method cannot be used in read-only mode")
        if not callable(validator):
            raise ValueError("validator must be callable!")
        try:
            self.batches[batch_idx].set_validator(validator)
        except KeyError:
            raise ValueError("invalid batch number!")

    def generate_batch_lines(
        self,
        batch_idx: int,
        max_invalid=MAX_INVALID,
        raise_on_exceed_invalid: bool = False,
        num_lines: int = None,
        seed_fields: Union[dict, List[dict]] = None,
        parallelism: int = 0,
    ) -> GenerationSummary:
        """Generate lines for a single batch. Lines generated are added
        to the underlying ``Batch`` object for each batch. The lines
        can be accessed after generation and re-assembled into a DataFrame.

        Args:
            batch_idx: The batch number
            max_invalid: The max number of invalid lines that can be generated, if
                this is exceeded, generation will stop
            raise_on_exceed_invalid: If true and if the number of lines generated exceeds the ``max_invalid``
                amount, we will re-raise the error thrown by the generation module which will interrupt
                the running process. Otherwise, we will not raise the caught exception and just return ``False``
                indicating that the batch failed to generate all lines.
            num_lines: The number of lines to generate, if ``None``, then we use the number from the
                batch's config
            seed_fields: A dictionary that maps field/column names to initial seed values for those columns. This seed
                will only apply to the first batch that gets trained and generated. Additionally, the fields provided
                in the mapping MUST exist at the front of the first batch.

                NOTE:
                    This param may also be a list of dicts. If this is the case, then ``num_lines`` will automatically
                    be set to the list length downstream, and a 1:1 ratio will be used for generating valid lines for
                    each prefix.
            parallelism: The number of concurrent workers to use. ``1`` (the default) disables parallelization,
                while a non-positive value means "number of CPUs + x" (i.e., use ``0`` for using as many workers
                as there are CPUs). A floating-point value is interpreted as a fraction of the available CPUs,
                rounded down.
        """
        try:
            batch = self.batches[batch_idx]
        except KeyError:  # pragma: no cover
            raise ValueError("invalid batch index")

        seed_string = None

        # If we are on batch 0 and we have seed values, we want to validate that
        # the seed values line up properly with the first N columns.
        if batch_idx == 0 and seed_fields is not None:
            seed_string = _validate_batch_seed_values(batch, seed_fields)

        batch: Batch
        batch.reset_gen_data()
        validator = batch.get_validator()
        if num_lines is None:
            num_lines = batch.config.gen_lines

        if isinstance(seed_fields, list):
            num_lines = len(seed_fields)

        t = tqdm(total=num_lines, desc="Valid record count ")
        t2 = tqdm(total=max_invalid, desc="Invalid record count ")
        line: GenText
        summary = GenerationSummary()
        try:
            for line in generate_text(
                batch.config,
                line_validator=validator,
                max_invalid=max_invalid,
                num_lines=num_lines,
                start_string=seed_string,
                parallelism=parallelism,
            ):
                if line.valid is None or line.valid is True:
                    batch.add_valid_data(line)
                    t.update(1)
                    summary.valid_lines += 1
                else:
                    t2.update(1)
                    batch.gen_data_invalid.append(line)
                    summary.invalid_lines += 1
        except TooManyInvalidError:
            if raise_on_exceed_invalid:
                raise
            else:
                return summary
        t.close()
        t2.close()
        summary.is_valid = batch.gen_data_count >= num_lines
        return summary

    def create_record_factory(
        self,
        *,
        num_lines: int,
        max_invalid: int = MAX_INVALID,
        validator: Callable = None,
        seed_fields: Union[dict, List[dict]] = None,
        parallellism: int = 4,
        **kwargs
    ) -> RecordFactory:
        if validator is not None:
            if not callable(validator):
                raise ValueError("validator must be callable")
        return RecordFactory(
            num_lines=num_lines,
            batches=self.batches,
            delimiter=self.batches[0].config.field_delimiter,
            header_list=self.original_headers or self.master_header_list,
            seed_fields=seed_fields,
            max_invalid=max_invalid,
            validator=validator,
            parallelism=parallellism,
            **kwargs
        )

    def generate_all_batch_lines(
        self,
        max_invalid=MAX_INVALID,
        raise_on_failed_batch: bool = False,
        num_lines: int = None,
        seed_fields: Union[dict, List[dict]] = None,
        parallelism: int = 0,
    ) -> Dict[int, GenerationSummary]:
        """Generate synthetic lines for all batches. Lines for each batch
        are added to the individual ``Batch`` objects. Once generateion is
        done, you may re-assemble the dataset into a DataFrame.

        Example::

            my_batch.generate_all_batch_lines()
            # Wait for all generation to complete
            synthetic_df = my_batch.batches_to_df()

        Args:
            max_invalid: The number of invalid lines, per batch. If this number
                is exceeded for any batch, generation will stop.
            raise_on_failed_batch: If True, then an exception will be raised if any single batch
                fails to generate the requested number of lines. If False, then the failed batch
                will be set to ``False`` in the result dictionary from this method.
            num_lines: The number of lines to create from each batch.  If ``None`` then the value
                from the config template will be used.

                NOTE:
                    Will be overridden / ignored if ``seed_fields`` is a list. Will be set to the len of the list.
            seed_fields: A dictionary that maps field/column names to initial seed values for those columns. This seed
                will only apply to the first batch that gets trained and generated. Additionally, the fields provided
                in the mapping MUST exist at the front of the first batch.

                NOTE:
                    This param may also be a list of dicts. If this is the case, then ``num_lines`` will automatically
                    be set to the list length downstream, and a 1:1 ratio will be used for generating valid lines for
                    each prefix.
            parallelism: The number of concurrent workers to use. ``1`` (the default) disables parallelization,
                while a non-positive value means "number of CPUs + x" (i.e., use ``0`` for using as many workers
                as there are CPUs). A floating-point value is interpreted as a fraction of the available CPUs,
                rounded down.

        Returns:
            A dictionary of batch number to a dictionary that reports the number of valid, invalid lines and bool value
            that shows if each batch was able to generate the full number of requested lines::

                {
                    0: GenerationSummary(valid_lines=1000, invalid_lines=10, is_valid=True),
                    1: GenerationSummary(valid_lines=500, invalid_lines=5, is_valid=True)
                }
        """
        batch_status = {}
        for idx in self.batches.keys():
            batch_status[idx] = self.generate_batch_lines(
                idx,
                max_invalid=max_invalid,
                raise_on_exceed_invalid=raise_on_failed_batch,
                num_lines=num_lines,
                seed_fields=seed_fields,
                parallelism=parallelism,
            )
        return batch_status

    def batch_to_df(self, batch_idx: int) -> pd.DataFrame:  # pragma: no cover
        """Extract a synthetic data DataFrame from a single batch.

        Args:
            batch_idx: The batch number

        Returns:
            A DataFrame with synthetic data
        """
        try:
            return self.batches[batch_idx].synthetic_df
        except KeyError:
            raise ValueError("batch_idx is invalid!")

    def batches_to_df(self) -> pd.DataFrame:
        """Convert all batches to a single synthetic data DataFrame.

        Returns:
            A single DataFrame that is the concatenation of all the
            batch DataFrames.
        """
        batch_iter = iter(self.batches.values())
        base_batch = next(batch_iter)
        accum_df = base_batch.synthetic_df

        for batch in batch_iter:
            accum_df = pd.concat([accum_df, batch.synthetic_df], axis=1)

        return accum_df[self.original_headers or self.master_header_list]
