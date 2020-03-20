from unittest.mock import patch
import pytest

from gretel_synthetics.config import LocalConfig


@pytest.fixture
def global_local_config():
    with patch('os.mkdir'):
        return LocalConfig(checkpoint_dir='ckpoint_dir', training_data='smol.txt')
