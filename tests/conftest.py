import pytest
from pathlib import Path
import shutil

from gretel_synthetics.config import LocalConfig
from gretel_synthetics.train import annotate_training_data

test_data_dir = Path(__file__).parent


@pytest.fixture(scope='session')
def global_local_config():
    target = test_data_dir / 'ckpoint'
    input_data = test_data_dir / 'data' / 'smol.txt'
    if not target.exists():
        target.mkdir()
    config = LocalConfig(checkpoint_dir=target, input_data=input_data)
    annotate_training_data(config)
    yield config
    shutil.rmtree(target)
