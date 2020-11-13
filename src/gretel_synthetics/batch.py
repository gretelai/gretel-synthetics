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
from typing import List, Type, Callable, Dict, Union
from copy import deepcopy
import logging
import io
import json
import glob

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import cloudpickle

from gretel_synthetics.config import LocalConfig, BaseConfig, config_from_model_dir, CONFIG_MAP
import gretel_synthetics.const as const
from gretel_synthetics.generate import GenText, generate_text
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
CONFIG_FILE = "model_params.json"
TRAIN_FILE = "train.csv"
PATH_HOLDER = "___path_holder___"


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
        """Get a DataFrame constructed from the generated lines """
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
        """Load a saved validation object if it exists """
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
    """Return a mapping of batch number => ``Batch`` object
    """
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
        with open(ckpoint / "headers.json", "w") as fout:
            fout.write(json.dumps(headers))

    return out


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
        else:
            self.batches = _crawl_checkpoint_for_batches(self._read_checkpoint_dir)
            self.master_header_list = []
            for batch in self.batches.values():
                self.master_header_list.extend(batch.headers)

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
        """Train a model for each batch.
        """
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

    def _validate_batch_seed_values(self, batch: Batch, seed_values: dict) -> str:
        """Validate that seed values line up with the first N columns in a batch. Also construct
        an appropiate seed string based on the values in the batch
        """
        if len(seed_values) > len(batch.headers):
            raise RuntimeError("The number of seed fields is greater than the number of columns in the first batch")

        headers_to_seed = batch.headers[:len(seed_values)]
        tmp = []
        for header in headers_to_seed:
            value = seed_values.get(header)
            if value is None:
                raise RuntimeError(f"The header: {header} is not in the seed values mapping")  # noqa
            tmp.append(str(value))

        return batch.config.field_delimiter.join(tmp) + batch.config.field_delimiter

    def generate_batch_lines(
        self,
        batch_idx: int,
        max_invalid=MAX_INVALID,
        raise_on_exceed_invalid: bool = False,
        num_lines: int = None,
        seed_fields: dict = None,
        parallelism: int = 0,
    ) -> bool:
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
            seed_string = self._validate_batch_seed_values(batch, seed_fields)

        batch: Batch
        batch.reset_gen_data()
        validator = batch.get_validator()
        if num_lines is None:
            num_lines = batch.config.gen_lines
        t = tqdm(total=num_lines, desc="Valid record count ")
        t2 = tqdm(total=max_invalid, desc="Invalid record count ")
        line: GenText
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
                else:
                    t2.update(1)
                    batch.gen_data_invalid.append(line)
        except TooManyInvalidError:
            if raise_on_exceed_invalid:
                raise
            else:
                return False
        t.close()
        t2.close()
        return batch.gen_data_count >= num_lines

    def generate_all_batch_lines(
        self,
        max_invalid=MAX_INVALID,
        raise_on_failed_batch: bool = False,
        num_lines: int = None,
        seed_fields: dict = None,
        parallelism: int = 0,
    ) -> dict:
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
            seed_fields: A dictionary that maps field/column names to initial seed values for those columns. This seed
                will only apply to the first batch that gets trained and generated. Additionally, the fields provided
                in the mapping MUST exist at the front of the first batch.
            parallelism: The number of concurrent workers to use. ``1`` (the default) disables parallelization,
                while a non-positive value means "number of CPUs + x" (i.e., use ``0`` for using as many workers
                as there are CPUs). A floating-point value is interpreted as a fraction of the available CPUs,
                rounded down.

        Returns:
            A dictionary of batch number to a bool value that shows if each batch
            was able to generate the full number of requested lines::

                {
                    0: True,
                    1: True
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

        return accum_df[self.master_header_list]
