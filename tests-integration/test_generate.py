"""
E2E Tests for Generating Data. These tests make use of pre-created models that can
be downloaded from S3.  We utilize a generation utility that will automatically determine
if we are using a simple model or a DF Batch model.

When adding a new model to test, the model filename should conform to:

description-MODE-TOK-major-minor.tar.gz

So for example:

    safecast-batch-sp-0-14.tar.gz -- would be a model built on Safecast data in DF batch mode
        using version 0.14.x of synthetics with a SentencePiece tokenizer

MODES:
    - simple
    - batch

TOK:
    - char
    - sp
"""
from unittest.mock import Mock

import pytest
import pandas as pd
import gzip
import tarfile
import tempfile
import random

from smart_open import open as smart_open

from gretel_synthetics.generate_utils import DataFileGenerator
from gretel_synthetics.batch import DataFrameBatch, GenerationProgress

BATCH_MODELS = [
    "https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/safecast-batch-sp-0-14.tar.gz",
]


def _unpack_to_dir(source: str, target: str):
    with smart_open(source, "rb", ignore_ext=True) as fin:
        with gzip.open(fin) as gzip_in:
            with tarfile.open(fileobj=gzip_in, mode="r:gz") as tar_in:
                tar_in.extractall(target)


def _unpack_to_dir_nogz(source: str, target: str):
    with smart_open(source, "rb", ignore_ext=True) as fin:
        with tarfile.open(fileobj=fin, mode="r:gz") as tar_in:
            tar_in.extractall(target)


@pytest.fixture(scope="module")
def safecast_model_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        _unpack_to_dir(BATCH_MODELS[0], tmp_dir)
        yield tmp_dir


@pytest.fixture(scope="module")
def hr_model_dir():
    archive = "https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/hr-3batch-sp.tar.gz"
    with tempfile.TemporaryDirectory() as tmp_dir:
        _unpack_to_dir_nogz(archive, tmp_dir)
        yield tmp_dir


def test_record_factory_single_line(safecast_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)
    factory = batcher.create_record_factory(num_lines=10)
    record = next(factory)
    assert "payload.service_handler" in record
    assert factory._counter.valid_count == 1


def test_record_factory_exhaust_iter(safecast_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)
    factory = batcher.create_record_factory(num_lines=10)
    records = list(factory)
    assert len(records) == 10
    assert factory._counter.valid_count == 10
    summary = factory.summary
    assert summary["num_lines"] == 10
    assert summary["max_invalid"] == 1000
    assert summary["valid_count"] == 10


def test_record_factory_generate_all(safecast_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)

    def _validator(rec: dict):
        assert float(rec["payload.loc_lat"])

    factory = batcher.create_record_factory(num_lines=10, validator=_validator)
    next(factory)
    next(factory)

    # generate_all should reset our iterator for the full 10 records
    assert len(factory.generate_all()) == 10

    df = factory.generate_all(output="df")
    assert df.shape == (10, 16)
    assert str(df["payload.loc_lat"].dtype) == "float64"


@pytest.mark.parametrize("threading", [True, False])
def test_record_factory_generate_all_with_callback(safecast_model_dir, threading):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)

    def _validator(rec: dict):
        assert float(rec["payload.loc_lat"])

    factory = batcher.create_record_factory(
        num_lines=1000,
        validator=_validator,
        invalid_cache_size=5
    )

    callback_fn = Mock()

    df = factory.generate_all(output="df", callback=callback_fn, callback_interval=1, callback_threading=threading)
    assert df.shape == (1000, 16)

    # assuming we get at least 5 bad records
    assert len(factory.invalid_cache) == 5
    assert str(df["payload.loc_lat"].dtype) == "float64"

    assert callback_fn.call_count >= 2
    # at least 1 call during generation and another one with final update
    assert callback_fn.call_count < 1000, "Progress update should be only called periodically"

    args, _ = callback_fn.call_args  # pylint: disable=unpacking-non-sequence
    last_update: GenerationProgress = args[0]
    assert last_update.current_valid_count == 1000
    assert last_update.completion_percent == 100

    # calculate sum from all updates
    valid_total_count = 0
    for call_args in callback_fn.call_args_list:
        args, _ = call_args
        valid_total_count += args[0].new_valid_count

    assert valid_total_count == 1000


def test_record_factory_bad_validator(safecast_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)
    with pytest.raises(ValueError) as err:
        batcher.create_record_factory(num_lines=10, validator="foo")
    assert "must be callable" in str(err)


def test_record_factory_simple_validator(safecast_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)

    def _validator(rec: dict):
        # ensure returning None and True works
        which = random.randint(0, 1)
        if which:
            assert float(rec["payload.loc_lat"])
        else:
            return True

    factory = batcher.create_record_factory(num_lines=10, validator=_validator)
    assert len(list(factory)) == 10


def test_record_factory_bool_fail_validator(safecast_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)

    def _validator(rec: dict):
        return False

    factory = batcher.create_record_factory(num_lines=10, validator=_validator, max_invalid=10)
    with pytest.raises(RuntimeError):
        list(factory)


def test_record_factory_exc_fail_validator(safecast_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)

    def _validator(rec: dict):
        raise ValueError

    factory = batcher.create_record_factory(num_lines=10, validator=_validator, max_invalid=10)
    with pytest.raises(RuntimeError):
        list(factory)


@pytest.mark.parametrize(
    "model_path", BATCH_MODELS
)
def test_generate_batch(model_path, tmp_path):
    gen = DataFileGenerator(model_path)
    out_file = str(tmp_path / "outdata")
    fname = gen.generate(100, out_file)
    count = 0
    with open(fname) as fin:
        for _ in fin:
            count += 1
    # account for the header
    assert count-1 == 100


def scooter_val(line):
    rec = line.split(", ")
    if len(rec) == 6:
        float(rec[5])
        float(rec[4])
        float(rec[3])
        float(rec[2])
        int(rec[0])
    else:
        raise Exception('record not 6 parts')


SIMPLE_MODELS = [
    ("https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/scooter-simple-sp-0-14.tar.gz", scooter_val),  # noqa
    ("https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/scooter-simple-char-0-15.tar.gz", scooter_val)  # noqa
]


@pytest.mark.parametrize(
    "model_path,validator_fn", SIMPLE_MODELS
)
def test_generate_simple(model_path, validator_fn, tmp_path):
    gen = DataFileGenerator(model_path)
    out_file = str(tmp_path / "outdata")
    fname = gen.generate(100, out_file, validator=validator_fn)
    count = 0
    with open(fname) as fin:
        for _ in fin:
            count += 1
    assert count == 100


@pytest.mark.parametrize(
    "model_path,seed", [
        ("https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/safecast-batch-sp-0-14.tar.gz", {"payload.service_handler": "i-051a2a353509414f0"})  # noqa
    ]
)
def test_generate_batch_smart_seed(model_path, seed, tmp_path):
    gen = DataFileGenerator(model_path)
    out_file = str(tmp_path / "outdata")
    fname = gen.generate(100, out_file, seed=seed)
    df = pd.read_csv(fname)
    for _, row in df.iterrows():
        row = dict(row)
        for k, v in seed.items():
            assert row[k] == v


@pytest.mark.parametrize(
    "model_path,seed", [
        ("https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/safecast-batch-sp-0-14.tar.gz",  # noqa
            [{"payload.service_handler": "i-051a2a353509414f0"},
             {"payload.service_handler": "i-051a2a353509414f1"},
             {"payload.service_handler": "i-051a2a353509414f2"},
             {"payload.service_handler": "i-051a2a353509414f3"}])  # noqa
    ]
)
def test_generate_batch_smart_seed_multi(model_path, seed, tmp_path):
    gen = DataFileGenerator(model_path)
    out_file = str(tmp_path / "outdata")
    fname = gen.generate(100, out_file, seed=seed)
    df = pd.read_csv(fname)
    assert list(df["payload.service_handler"]) == list(pd.DataFrame(seed)["payload.service_handler"])


def test_record_factory_smart_seed(safecast_model_dir):
    seeds = [{"payload.service_handler": "i-051a2a353509414f0"},
             {"payload.service_handler": "i-051a2a353509414f1"},
             {"payload.service_handler": "i-051a2a353509414f2"},
             {"payload.service_handler": "i-051a2a353509414f3"}]

    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)
    factory = batcher.create_record_factory(
        num_lines=1000,
        seed_fields=seeds*1000
    )

    # list of seeds should reset num_lines
    assert factory._counter.num_lines == len(seeds) * 1000

    for seed, record in zip(seeds, factory):
        assert seed["payload.service_handler"] == record["payload.service_handler"]


class MyValidator:

    counter = 0

    def __call__(self, _):
        if self.counter:
            self.counter = 0
            return False
        self.counter = 1


def test_record_factory_smart_seed_buffer(safecast_model_dir):
    seeds = [{"payload.service_handler": "i-051a2a353509414f0"},
             {"payload.service_handler": "i-051a2a353509414f1"},
             {"payload.service_handler": "i-051a2a353509414f2"},
             {"payload.service_handler": "i-051a2a353509414f3"}] * 2

    batcher = DataFrameBatch(mode="read", checkpoint_dir=safecast_model_dir)

    factory = batcher.create_record_factory(
        num_lines=100,  # doesn't matter w/ smart seed
        seed_fields=seeds,
        validator=MyValidator(),
        max_invalid=5000
    )

    df = factory.generate_all(output="df")
    assert len(df) == 8
    assert factory.summary["num_lines"] == 8
    assert factory.summary["valid_count"] == 8


def test_record_factory_multi_batch(hr_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=hr_model_dir)

    factory = batcher.create_record_factory(
        num_lines=50,
        max_invalid=5000
    )

    df = factory.generate_all(output="df")
    assert len(df) == 50


def test_record_factory_multi_batch_seed_list(hr_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=hr_model_dir)

    age_seeds = [{"age": i} for i in range(1, 51)]

    factory = batcher.create_record_factory(
        num_lines=50,  # doesn't matter w/ smart seed
        max_invalid=5000,
        seed_fields=age_seeds,
        validator=MyValidator()
    )

    df = factory.generate_all(output="df")
    assert len(df) == 50
    assert df["age"].nunique() == 50


def test_record_factory_multi_batch_seed_static(hr_model_dir):
    batcher = DataFrameBatch(mode="read", checkpoint_dir=hr_model_dir)

    factory = batcher.create_record_factory(
        num_lines=10,
        max_invalid=5000,
        seed_fields={"age": 5},
        validator=MyValidator()
    )

    df = factory.generate_all(output="df")
    assert len(df) == 10
    assert df["age"].nunique() == 1
    assert df.iloc[0]["age"] == 5
