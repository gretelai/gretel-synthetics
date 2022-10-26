import shutil

from pathlib import Path

import pytest

from gretel_synthetics.config import TensorFlowConfig
from gretel_synthetics.tokenizers import tokenizer_from_model_dir
from gretel_synthetics.train import _create_default_tokenizer, TrainingParams

test_data_dir = Path(__file__).parent


@pytest.fixture(scope="session")
def tf_config():
    target = test_data_dir / "ckpoint"
    input_data = test_data_dir / "data" / "smol.txt"
    if not target.exists():
        target.mkdir()
    config = TensorFlowConfig(
        checkpoint_dir=target.as_posix(),
        input_data_path=input_data.as_posix(),
        field_delimiter=",",
        predict_batch_size=1,
        overwrite=True,
    )
    yield config
    shutil.rmtree(target)


@pytest.fixture(scope="session")
def training_params(tf_config):
    tokenizer_trainer = _create_default_tokenizer(tf_config)
    tokenizer_trainer.annotate_data()
    tokenizer_trainer.train()
    tokenizer = tokenizer_from_model_dir(tf_config.checkpoint_dir)
    training_params = TrainingParams(
        tokenizer_trainer=tokenizer_trainer, tokenizer=tokenizer, config=tf_config
    )
    yield training_params
