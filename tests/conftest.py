from unittest.mock import patch
import pytest
import os

from gretel_synthetics.config import LocalConfig
from gretel_synthetics.train import annotate_training_data


@pytest.fixture
def global_local_config():
    target = 'ckpoint'
    print(os.getcwd())
    if not os.path.exists(target):
        os.makedirs(target)
    config = LocalConfig(checkpoint_dir=target, input_data=os.path.join(os.getcwd(), 'tests', 'data', 'smol.txt'))
    annotate_training_data(config)
    return config
