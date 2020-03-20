from unittest.mock import patch
import pytest
import os

from gretel_synthetics.config import LocalConfig


@pytest.fixture
def global_local_config():
    target = 'ckpoint'
    os.mkdir(target)
    return LocalConfig(checkpoint_dir=target, input_data='blah')
