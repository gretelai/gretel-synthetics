from unittest.mock import patch
import pytest
import os
from pathlib import Path
import shutil

from gretel_synthetics.config import LocalConfig
from gretel_synthetics.train import annotate_training_data

test_data_dir = os.path.join(str(Path(__file__).parent.resolve()), 'data')


@pytest.fixture(scope='session')
def global_local_config():
    target = os.path.join(test_data_dir, 'ckpoint')
    if not os.path.exists(target):
        os.makedirs(target)
    config = LocalConfig(checkpoint_dir=target, input_data=os.path.join(test_data_dir, 'smol.txt'))
    annotate_training_data(config)
    yield config
    shutil.rmtree(target)
