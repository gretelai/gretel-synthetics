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
import pytest
import pandas as pd

from gretel_synthetics.generate_utils import DataFileGenerator


BATCH_MODELS = [
    "https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/safecast-batch-sp-0-14.tar.gz"
]


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
