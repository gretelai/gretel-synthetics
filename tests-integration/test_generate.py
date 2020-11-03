"""
E2E Tests for Generating Data. These tests make use of pre-created models that can
be downloaded from S3.  We utilize a generation utility that will automatically determine
if we are using a simple model or a DF Batch model. 

When adding a new model to test, the model filename should conform to:

description-batch|simple-major-minor.tar.gz

So for example:

    safecast-batch-0-14.tar.gz -- would be a model built on Safecast data in DF batch mode
    using version 0.14.x of synthetics.
"""
import pytest

from gretel_synthetics.generate_utils import DataFileGenerator


BATCH_MODELS = [
    "https://gretel-public-website.s3-us-west-2.amazonaws.com/tests/synthetics/models/safecast-batch-0-14.tar.gz"
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
    assert count-1 == 100
